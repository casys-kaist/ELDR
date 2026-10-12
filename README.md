# ELDR Artifact Evaluation

Artifact for “ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated
MoE Serving” (ACM ATC 2026). This repository provides eight experiments covering
Task and Language performance, expert signatures, balanced clustering,
locality-band routing, prefix-cache compatibility, single-category workloads,
and workload drift.

## Run

Clone the repository on the controller node of a prepared GPU cluster.
Prerequisites are `uv`, Docker, `rsync`, configured GPU/RDMA drivers, and
persistent SSH connections to worker nodes.

```bash
git clone https://github.com/casys-kaist/ELDR.git eldr
cd eldr
```

Prepare inputs, then run all eight experiments and generate their figures/tables:

```bash
bash eldr/scripts/prepare_data.sh
bash eldr/scripts/run_all.sh
```

The scripts install CPU dependencies, verify inputs, and automatically synchronize
serving code across worker nodes. Runtime depends on model loading, compilation,
and request latency in addition to the measurement and warmup traffic.

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
generates its figures or table. Main experiments use all three models. Signature,
cluster-balance, and locality-band experiments cover both Task and Language;
prefix cache uses GPT-OSS Task. Single-category and drift use Qwen.

| Figure | Script | Requests/s | Measurement cells |
| --- | --- | --- | --- |
| 10: Main Task | `fig10_main_task.sh` | 20, 40, 60, 80, 100 | 90 |
| 11: Main Language | `fig11_main_language.sh` | 20, 40, 60, 80, 100 | 90 |
| 13: Signature | `fig13_signature.sh` | 20, 40, 60, 80, 100 | 120 |
| 14: Cluster balance | `fig14_cluster_balance.sh` | 20, 40, 60, 80, 100 | 90 |
| 15: Locality band | `fig15_locality_band.sh` | 20, 40, 60, 80, 100 | 150 |
| 16: Prefix cache | `fig16_prefix_cache.sh` | 100 | 4 |
| Table 2: Math-only / English-only | `tab2_single_category.sh` | 20, 40, 60, 80, 100 | 20 |
| 17: Workload drift | `fig17_workload_drift.sh` | 100 | 3 |

To run a single experiment, use its script, for example:

```bash
bash eldr/scripts/fig10_main_task.sh
```

Default settings are counts × IDF, a fixed layer mask, global L2 normalization,
online centroid refitting with a 5-second history and a 1-second interval, and
locality-band JSQ (τ=0.1). IDF weights and the layer mask remain fixed during
serving. Both cluster-balance variants retain their initial clustering algorithm
during refits. Fitting uses seed 1; traffic uses seed 1234 and 512 output tokens.
Prefill routing is prefix-hash based. Deployments use 8P16D, except prefix cache
(1P16D); the reference proxy uses CPU cores 4–5.

Each rate measures 120 seconds of offered traffic after warmup: 14,400 requests
for the first trial in each worker group, then 3,840 per trial, at 64 req/s.
Signature ablation keeps a separate matched RR control for each signature
representation. Prefix-cache trials each use 12,000 requests at 100 req/s from
the same seeded 2,000-prompt pool. `eldr-static` remains available as an explicit
fixed-centroid control; it is not the default ELDR configuration.

Single-category performs two fresh category-only calibration captures, separate
from its 20 measurements. Calibration and evaluation prompts are disjoint. The
Full signature's IDF weights and layer mask remain fixed; only centroids are
fitted to each category. The output is a native LaTeX table, plus CSV/JSON with
every rate retained, including regressions.

Drift compares RR, static ELDR, and online ELDR from the same initial centroids.
It changes the combined code/math share from 49% to 80% and back, with 10-second
ramps. Each trial sends 15,000 ordered requests: 120 seconds of analysis plus
15-second guards on each side. Whole-request TPOT is grouped into 5-second bins
using actual request arrival times; the saved comparison retains each policy's
observed domain mix. Online refitting uses the same 5-second history/1-second
interval as the other experiments.

The final paper's active-expert-count, topology, and 235B experiments are not
covered by these scripts.

Use `--output NEW_DIR` to choose an output directory and `--plan` to check the
experiment plan without launching GPU experiments. For another cluster, adapt
`eldr/experiments/cluster.json` and pass `--config CLUSTER_JSON` to both preparation
and experiment scripts. Models, the serving image, and worker SSH access must be
prepared. Each script supports `--help`.

`--profile smoke` reduces traffic and output lengths; it is not a paper result.
Drift smoke still covers the full 120-second analysis interval (300 requests at
2 req/s including guards). Single-category still needs both fresh calibration
captures. Execution takes the shared `/tmp/eldr-ae.lock` and stops on failure;
`--plan` does not deploy code or launch GPU work.

## Results

Results are saved in `eldr/artifacts/runs/<run>/`; the script prints the exact path.
Each experiment produces CSV/JSON summaries and PDF/PNG plots, except Table 2,
which produces `.tex` instead of an image. Raw request
measurements are saved in `<worker-group>/run/<trial>/measure/raw.json`.

After resolving an error, resume using the original run directory:

```bash
bash eldr/scripts/run_all.sh --resume RUN_DIR
```

Completed experiments are skipped; the interrupted experiment restarts from
warmup. Resume requires unchanged code and inputs and successful worker cleanup.
Use the original checkout to resume an older run; updated protocols require a
new output directory. Plot replay still accepts completed static-centroid runs
and preserves their saved measurements.

Metrics are TPOT P50/P95/P99 and TTFT P50. TTFT ends when the prefill-generated
first token reaches the client. Tail latency may vary across runs and system
conditions.

Signature, cluster-balance, and locality-band plots show mean per-rate percentage
changes relative to RR; negative values indicate lower latency.

Redraw saved results without GPU access:

```bash
bash eldr/scripts/plot_all.sh RUN_DIR --output NEW_PLOT_DIR
bash eldr/scripts/plot_tab2_single_category.sh RUN_DIR/tab2_single_category --output NEW_TABLE_DIR
bash eldr/scripts/plot_fig17_workload_drift.sh RUN_DIR/fig17_workload_drift --output NEW_DRIFT_DIR
```

`plot_all.sh` requires all eight experiments by default. For a partial or older
six-experiment suite, pass `--available` explicitly; this validates only the
recognized experiments present in `RUN_DIR`. An incomplete experiment directory
still fails validation. Replay recomputes metrics from raw
requests and checks them against saved summaries. Generalization additionally
checks calibration/input/source provenance, routing/refit/cache flags, cleanup,
and the saved comparison. Keep the original input bundle and source snapshots
alongside results for these checks. Outputs are never overwritten.

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
