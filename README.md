# ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving [SIGOPS ATC 2026]

ELDR is a lossless, expert-locality-aware decode router for prefill–decode (PD)
disaggregated MoE serving. Equally loaded decode workers can have different
latency because their requests activate different numbers of distinct experts.
ELDR builds expert signatures from prefill activations to predict decode-time
expert usage, then combines balanced K-means with locality-band routing to group
similar requests while balancing load. Centroids are refitted in the background,
and a block-level signature cache keeps signatures exact under prefix caching.
ELDR changes only request placement, preserving model weights, token-to-expert
assignments, and model outputs.

## Requirements

- Three GPU nodes, each with eight AMD MI300X GPUs (192 GB HBM per GPU),
  connected via RDMA-capable InfiniBand. The default 8P16D configuration uses one
  node for prefill and two for decode, with one GPU per worker. Nodes must be idle
  and dedicated to the experiment, with GPU and RDMA drivers configured.
- A Linux host for running the experiment scripts, routing proxy, and load
  generator, with `uv`, Docker, `rsync`, and `flock` installed. This host is
  referred to as the controller below.
- The [pinned serving image](#tested-environment) on the controller and workers,
  and model weights at the paths specified in the cluster configuration.
- SSH host aliases with existing persistent connections to the workers. The
  runner reuses these connections; it does not establish or repair them.
- Internet access for dataset preparation and dependency installation.

GPU calibration is part of input preparation. The CPU test environment alone
cannot run or validate the serving experiments.

## Quick start

Run the following commands from the repository root on the controller.

### 1. Clone and configure

```bash
git clone https://github.com/casys-kaist/ELDR.git eldr
cd eldr
```

Update [eldr/experiments/cluster.json](eldr/experiments/cluster.json) for your
cluster before running anything. Check the controller address, model and tokenizer
cache paths, worker SSH aliases and addresses, GPU IDs, CPU masks, RDMA interfaces,
and ports. The supplied values describe the reference cluster, not an automatic
hardware discovery configuration.

To keep your configuration in a separate file, pass `--config /path/to/cluster.json`
to both preparation and experiment scripts.

### 2. Prepare data and calibration

```bash
bash eldr/scripts/prepare_data.sh
```

This installs dependencies, downloads pinned public datasets, creates disjoint
calibration/evaluation splits, and collects GPU activations to fit the signature
transform and centroids. Serving sources are synchronized to worker nodes.

Inputs are saved in `eldr/inputs/`; preparation logs are saved in
`eldr/artifacts/preparation/`. Rerunning the command verifies and reuses completed
inputs. After an interrupted capture, resolve the error and confirm worker cleanup
before rerunning; keep the previous attempt's logs.

### 3. Check the plan and run

```bash
bash eldr/scripts/run_all.sh --plan
bash eldr/scripts/run_all.sh
```

`--plan` checks the prepared inputs and writes configurations and experiment plans
without deploying sources or launching GPU work. It requires all six shared
model/workload input sets from Step 2, even when selecting one experiment. It is
not a GPU validation run.

The second command runs all eight experiments and generates their figures and
table. It prints the output directory and log paths. Use `--output NEW_DIR` to
choose a new destination; an existing directory is not overwritten.

Preparation and execution share `/tmp/eldr-ae.lock` to prevent overlapping jobs on
the controller. Execution stops on failure. Runtime includes model loading,
compilation, warmup, and request completion, not just measurement traffic.

## Experiments

Each script runs one study and generates its figures or table. The main and three
design-ablation studies cover Qwen3-30B-A3B, GPT-OSS-120B, and Gemma-4-26B-A4B.
The ablations cover both Task and Language workloads. Prefix cache uses GPT-OSS
Task; single-category and drift use Qwen. Run counts exclude warmup and calibration.

| Paper item | Script | Request rates (req/s) | Measured runs |
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

Each script supports `--help`. For a shorter end-to-end check, use
`--profile smoke`; it reduces request rates/counts and output lengths but still
requires prepared inputs and GPUs. Drift smoke retains the 120-second analysis
interval, and single-category still performs both fresh calibration captures.
Smoke results are not paper measurements.

### Default protocol

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

## Results and replay

Results are saved in `eldr/artifacts/runs/<run>/`; the script prints the exact path.
Each experiment produces CSV/JSON summaries and PDF/PNG plots, except Table 2,
which produces native LaTeX (`.tex`) instead of an image. Raw request measurements
are saved in `<worker-group>/run/<trial>/measure/raw.json`.

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

Replace `RUN_DIR` with the path printed by the experiment runner. Plotting uses
the CPU environment below and does not launch serving workers.

`plot_all.sh` requires all eight experiments by default. For a partial or older
six-experiment suite, add `--available`; missing studies are skipped, but an
incomplete study still fails validation. Replay recomputes metrics from raw
requests and checks saved summaries. Single-category and drift also validate
input/source provenance, routing/refit/cache settings, cleanup, and saved
comparisons. Keep the input bundle and source snapshots with the raw results.
Completed static-centroid runs remain replayable with their original measurements.

### Resume after a failure

Resolve the error and confirm worker cleanup before resuming:

```bash
bash eldr/scripts/run_all.sh --resume RUN_DIR
```

Resume validates the saved configuration, code, and inputs. Completed experiments
are skipped; an incomplete experiment is archived and restarted from warmup.
This is experiment-level recovery, not continuation from an individual measured
request or trial. Use the original checkout for an older run; a changed protocol
or source requires a new run directory. Do not delete raw results or bypass a
failed cleanup/provenance check to force a retry.

## Code and CPU tests

```text
eldr/
  serving/              Routing proxy, clustering and centroid refitting
  experiments/          Experiment definitions and figure/table generation
    datasets/           Public-dataset preprocessing
    runner/             Shared configuration, execution and result validation
  scripts/              Shell entry points for preparation, runs and replay
  tests/                CPU tests and opt-in GPU integration checks
vllm/eldr/              Gate capture, block counts and signature transport
```

The repository retains the upstream vLLM layout. Engine integration hooks call
the helpers in `vllm/eldr/`; experiment orchestration and routing live under
`eldr/`.

To run CPU tests without GPUs or generated inputs:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r eldr/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s eldr/tests -q
```

This environment supports CPU tests and result replay, not GPU serving. The GPU
runner uses the pinned container and this repository's ELDR-modified vLLM source.

## Tested environment

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
`d801ae8c2650b590438f3d9794dd7a47abd86c9a`.

## License

The code is distributed under [Apache 2.0](LICENSE).
Original license and copyright notices are retained; modified upstream files are
marked. Please also cite the [vLLM paper](https://arxiv.org/abs/2309.06180).
Models and datasets retain their own licenses. Dataset sources, pinned revisions,
and preprocessing rules are documented in
[eldr/experiments/datasets/](eldr/experiments/datasets/).
