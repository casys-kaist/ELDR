# ELDR Artifact Evaluation

Artifact for “ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated
MoE Serving” (ACM ATC 2026). This repository provides six experiments covering
Task and Language performance, expert signatures, balanced clustering,
locality-band routing, and prefix-cache compatibility.

## Run

Clone the repository on the controller node of a prepared GPU cluster.
Prerequisites are `uv`, Docker, `rsync`, configured GPU/RDMA drivers, and
persistent SSH connections to worker nodes.

```bash
git clone https://github.com/casys-kaist/ELDR.git eldr
cd eldr
```

Prepare inputs, then run all six experiments and generate their plots:

```bash
bash eldr/scripts/prepare_data.sh
bash eldr/scripts/run_all.sh
```

The scripts install CPU dependencies, verify inputs, and automatically synchronize
serving code across worker nodes. Allow 1–2 days for the full suite on the reference
cluster, including model loading and compilation.

## Prepare inputs

`prepare_data.sh` downloads pinned public datasets, deduplicates and filters
prompts, and creates disjoint calibration and evaluation splits. It then collects
expert activations to fit the signature transform and centroids.

Newly generated datasets, calibration activations, and fitted centroids are saved
in `eldr/inputs/`; preparation logs are saved in `eldr/artifacts/preparation/`.
Preparation requires internet access and the configured GPUs and models.

Rerunning `prepare_data.sh` verifies and reuses completed inputs. If preparation is
interrupted, rerun the same command.

## Experiments

Each experiment has a script in `eldr/scripts/` that runs the experiment and
generates its plots. Main experiments use all three models. Signature,
cluster-balance, and locality-band experiments cover both Task and Language;
prefix cache uses GPT-OSS Task.

| Figure | Script | Requests/s | Estimated traffic + warmup |
| --- | --- | --- | --- |
| 10: Main Task | `fig10_main_task.sh` | 20, 40, 60, 80, 100 | 4.6 h |
| 11: Main Language | `fig11_main_language.sh` | 20, 40, 60, 80, 100 | 4.6 h |
| 13: Signature | `fig13_signature.sh` | 60 | 1.8 h |
| 14: Cluster balance | `fig14_cluster_balance.sh` | 20, 40, 60, 80, 100 | 4.8 h |
| 15: Locality band | `fig15_locality_band.sh` | 20, 40, 60, 80, 100 | 7.8 h |
| 16: Prefix cache | `fig16_prefix_cache.sh` | 100 | 18 min |

To run a single experiment, use its script, for example:

```bash
bash eldr/scripts/fig10_main_task.sh
```

Default settings are counts × IDF, a fixed layer mask, global L2 normalization,
static centroids, and locality-band JSQ (τ=0.1). Fitting uses seed 1; traffic uses
seed 1234 and 512 output tokens. Prefill routing is prefix-hash based. Deployments
use 8P16D, except prefix cache (1P16D).

Use `--output NEW_DIR` to choose an output directory and `--plan` to check the
experiment plan without launching GPU experiments. For another cluster, adapt
`eldr/experiments/cluster.json` and pass `--config CLUSTER_JSON` to both preparation
and experiment scripts. Models, the serving image, and worker SSH access must be
prepared. Each script supports `--help`.

## Results

Results are saved in `eldr/artifacts/runs/<run>/`; the script prints the exact path.
Each experiment produces CSV/JSON summaries and PDF/PNG plots. Raw request
measurements are saved in `<worker-group>/run/<trial>/measure/raw.json`.

After resolving an error, resume using the original run directory:

```bash
bash eldr/scripts/run_all.sh --resume RUN_DIR
```

Completed experiments are skipped; the interrupted experiment restarts from
warmup. Resume requires unchanged code and inputs and successful worker cleanup.

Metrics are TPOT P50/P95/P99 and TTFT P50. TTFT ends when the prefill-generated
first token reaches the client. Tail latency may vary across runs and system
conditions.

Signature, cluster-balance, and locality-band plots show mean per-rate percentage
changes relative to RR; negative values indicate lower latency.

## Code and tests

```text
eldr/scripts/       Experiment and plotting entry points
eldr/serving/       Proxy, signatures, clustering and routing
eldr/experiments/   Experiment definitions, datasets and shared runner
eldr/tests/         CPU and opt-in GPU tests
vllm/eldr/          Gate capture, block counts and signature transport
```

Other source directories are from upstream vLLM.

To run CPU tests without GPUs or generated inputs:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r eldr/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s eldr/tests -q
```

## Environment and license

The supplied cluster uses 24 AMD MI300X GPUs across three worker nodes, plus a
controller: Ubuntu 22.04.5, kernel 5.15.0-1041-azure, Intel Xeon 8480C, 96 logical
CPUs and two NUMA nodes per host. Node3 handles prefill; node4 and node2 handle
decode. Models are Qwen3-30B-A3B, GPT-OSS-120B and Gemma-4-26B-A4B, with TP=PP=1.

The serving image contains ROCm 7.2.2, Python 3.12.13, PyTorch 2.11.0+git96bfee1,
and vLLM 0.21.0rc1+rocm722. Experiments use the ELDR-modified vLLM source from this
repository.

Serving image:

```text
rocm/vllm-dev@sha256:411f7ca9abeb57f69064e8bf60f46b5cd0b67f26264af0fc831f124fbc22f4b1
```

Based on [vLLM](https://github.com/vllm-project/vllm) commit
`d801ae8c2650b590438f3d9794dd7a47abd86c9a`, under [Apache 2.0](LICENSE).
Original license and copyright notices are retained; modified upstream files are
marked. Please also cite the [vLLM paper](https://arxiv.org/abs/2309.06180).
Models and datasets retain their own licenses. Dataset sources, pinned revisions,
and preprocessing rules are documented in
[eldr/experiments/datasets/](eldr/experiments/datasets/).
