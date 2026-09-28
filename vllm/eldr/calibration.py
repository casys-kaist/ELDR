# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, request-aligned MoE calibration. Never imported by serving workers."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class Capture:
    """Accumulate actual gate selections; reject gaps, recomputation and bad IDs."""

    def __init__(self, layers, experts):
        self.layers, self.experts = layers, experts
        self.rows = {}
        self.topk = None

    def add(self, request_id, prompt_ids, computed, ids, probabilities):
        if (
            ids.ndim != 3
            or ids.shape[0] != self.layers
            or not np.issubdtype(ids.dtype, np.integer)
            or not 0 < ids.shape[2] <= self.experts
            or np.any(ids < 0)
            or np.any(ids >= self.experts)
        ):
            raise ValueError("Invalid actual expert IDs")
        tokens, topk = ids.shape[1:]
        if not tokens or self.topk not in (None, topk):
            raise ValueError("Inconsistent gate top-k")
        self.topk = topk
        prompt_ids = tuple(prompt_ids)
        if request_id not in self.rows:
            if computed != 0 or not prompt_ids:
                raise ValueError("Capture must start at the first prompt token")
            self.rows[request_id] = dict(
                prompt_ids=prompt_ids,
                computed=0,
                prefill=np.zeros((self.layers, self.experts), dtype=np.int32),
                decode=np.zeros((self.layers, self.experts), dtype=np.int32),
                probability=np.zeros((self.layers, self.experts), dtype=np.float32),
            )
        row = self.rows[request_id]
        if row["prompt_ids"] != prompt_ids or computed != row["computed"]:
            raise ValueError("Prompt changed, tokens missing, or request recomputed")
        prefill = computed < len(prompt_ids)
        if prefill:
            if computed + tokens > len(prompt_ids):
                raise ValueError("Mixed prefill/decode chunk is unsupported")
            if (
                probabilities is None
                or probabilities.shape != (self.layers, self.experts)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(-1), tokens, rtol=1e-4, atol=1e-3)
            ):
                raise ValueError("Invalid prefill probability sums")
            row["probability"] += probabilities
        elif probabilities is not None:
            raise ValueError("Decode probabilities are not calibration inputs")
        offsets = np.arange(self.layers)[:, None, None] * self.experts
        counts = np.bincount(
            (ids + offsets).reshape(-1), minlength=self.layers * self.experts
        ).reshape(self.layers, self.experts)
        row["prefill" if prefill else "decode"] += counts.astype(np.int32)
        row["computed"] += tokens

    def arrays(self, requests, output_tokens):
        """Join on request ID AND exact prompt tokens, never on completion order."""
        if len(requests) != len(self.rows) or output_tokens < 2:
            raise ValueError("Incomplete calibration request set")
        by_external = {}
        for internal, row in self.rows.items():
            # The pinned vLLM adds an eight-character random suffix internally.
            external, separator, suffix = internal.rpartition("-")
            if not separator or len(suffix) != 8 or external in by_external:
                raise ValueError("Unexpected or duplicated vLLM request ID")
            by_external[external] = row
        ordered = []
        for request in requests:
            row = by_external.pop(request["id"])
            if (
                row["prompt_ids"] != tuple(request["prompt_ids"])
                or row["computed"] != len(row["prompt_ids"]) + output_tokens - 1
                or len(request["output_ids"]) != output_tokens
            ):
                raise ValueError("Prompt/token mismatch or incomplete decode capture")
            ordered.append(row)
        if by_external:
            raise ValueError("Unexpected calibration requests")
        return dict(
            request_ids=np.asarray([r["prompt_sha256"] for r in requests]),
            prefill_counts=np.stack([r["prefill"] for r in ordered]),
            gate_probabilities=np.stack([r["probability"] for r in ordered]),
            decode_counts=np.stack([r["decode"] for r in ordered]),
            prompt_tokens=np.asarray([len(r["prompt_ids"]) for r in ordered]),
            decode_tokens=np.full(len(ordered), output_tokens - 1),
        )


class CalibrationWorker:
    """vLLM worker extension; RPCs use names, not serialized executable code."""

    def install_calibration(worker: Any):
        """Use vLLM's worker RPC and actual gate callbacks, only in this process."""
        import torch

        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        runner = worker.model_runner
        config = worker.vllm_config
        if (
            not config.model_config.enforce_eager
            or config.scheduler_config.async_scheduling
            or config.cache_config.enable_prefix_caching
            or config.kv_transfer_config is not None
            or config.speculative_config is not None
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
            or hasattr(runner, "_eldr_calibration")
        ):
            raise ValueError(
                "Calibration requires a fresh synchronous eager TP=PP=1 engine"
            )
        layers = [m for m in runner.model.modules() if isinstance(m, FusedMoE)]
        if not layers or sorted(m.moe_layer_id for m in layers) != list(
            range(len(layers))
        ):
            raise ValueError("Contiguous MoE layer IDs required")
        recorder = Capture(len(layers), layers[0].global_num_experts)
        runner._eldr_calibration = recorder
        ids, logits = [None] * len(layers), [None] * len(layers)
        active = False
        need_probabilities = False

        def record(index, value):
            if active:
                if ids[index] is not None:
                    raise ValueError("Gate invoked twice for one layer")
                ids[index] = value

        for layer in layers:
            if layer.global_num_experts != recorder.experts or layer.router.capture_fn:
                raise ValueError(
                    "Inconsistent expert geometry or another capture installed"
                )
            layer.router.set_capture_fn(
                lambda value, index=layer.moe_layer_id: record(index, value)
            )
            if layer.runner.gate is not None:
                if layer.runner._fse_fuse_gate:
                    raise ValueError(
                        "Calibration requires an observable gate projection"
                    )

                def gate_hook(module, args, output, index=layer.moe_layer_id):
                    if active and need_probabilities:
                        logits[index] = (
                            output[0] if isinstance(output, tuple) else output
                        )

                layer.runner.gate.register_forward_hook(gate_hook)
            else:

                def expert_hook(module, args, kwargs, index=layer.moe_layer_id):
                    if active and need_probabilities:
                        logits[index] = kwargs.get(
                            "router_logits", args[1] if len(args) > 1 else None
                        )

                layer.register_forward_pre_hook(expert_hook, with_kwargs=True)

        original = runner._model_forward

        def forward(*args, **kwargs):
            nonlocal active, need_probabilities
            request_ids = list(runner.input_batch.req_ids)
            starts = runner.query_start_loc.np[: len(request_ids) + 1].copy()
            computed = runner.input_batch.num_computed_tokens_cpu[
                : len(request_ids)
            ].copy()
            prompts = [tuple(runner.requests[r].prompt_token_ids) for r in request_ids]
            need_probabilities = any(c < len(p) for c, p in zip(computed, prompts))
            ids[:] = [None] * len(layers)
            logits[:] = [None] * len(layers)
            active = True
            try:
                result = original(*args, **kwargs)
                if any(value is None for value in ids):
                    raise ValueError("Missing actual gate IDs")
                observed = torch.stack(ids)[:, : starts[-1]].cpu().numpy()
                probability = None
                if need_probabilities:
                    if any(value is None for value in logits):
                        raise ValueError("Missing actual prefill gate logits")
                    probability = (
                        torch.stack(logits)[:, : starts[-1]].float().softmax(-1)
                    )
                for i, request_id in enumerate(request_ids):
                    start, end = int(starts[i]), int(starts[i + 1])
                    sums = None
                    if computed[i] < len(prompts[i]):
                        assert probability is not None
                        sums = probability[:, start:end].sum(1).cpu().numpy()
                    recorder.add(
                        request_id,
                        prompts[i],
                        int(computed[i]),
                        observed[:, start:end],
                        sums,
                    )
                return result
            finally:
                active = False
                ids[:] = [None] * len(layers)
                logits[:] = [None] * len(layers)

        runner._model_forward = forward
        return dict(layers=recorder.layers, experts=recorder.experts)

    def save_calibration(worker: Any, path, requests, metadata):
        recorder = worker.model_runner._eldr_calibration
        arrays = recorder.arrays(requests, metadata["output_tokens"])
        metadata = dict(metadata, top_k=recorder.topk, format="eldr-calibration-v1")
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream, **arrays, metadata=np.asarray(json.dumps(metadata))
            )
        return dict(
            requests=len(requests), layers=recorder.layers, experts=recorder.experts
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("qwen", "gptoss", "gemma"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    raw = args.prompts.read_bytes()
    pairs = json.loads(raw)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        quantization="mxfp4" if args.model == "gptoss" else None,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        distributed_executor_backend="uni",
        worker_extension_cls="vllm.eldr.calibration.CalibrationWorker",
        disable_hybrid_kv_cache_manager=True,
        max_model_len=2048,
        max_num_seqs=48,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
        seed=1,
    )
    llm.collective_rpc("install_calibration")
    params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, seed=1)
    outputs = llm.generate([p for p, _ in pairs], params, use_tqdm=True)
    if len(outputs) != len(pairs):
        raise ValueError("Incomplete calibration generation")
    requests = []
    for index, (output, (prompt, _)) in enumerate(zip(outputs, pairs)):
        if (
            output.request_id != str(index)
            or not output.finished
            or len(output.outputs) != 1
        ):
            raise ValueError("Unexpected calibration request identity/completion")
        requests.append(
            dict(
                id=output.request_id,
                prompt_ids=output.prompt_token_ids,
                output_ids=output.outputs[0].token_ids,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            )
        )
    metadata = dict(
        model=args.model,
        training_sha256=hashlib.sha256(raw).hexdigest(),
        output_tokens=128,
        temperature=0,
        seed=1,
    )
    print(
        llm.collective_rpc(
            "save_calibration", args=(str(args.output), requests, metadata)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
