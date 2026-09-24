# ELDR Artifact Evaluation

Artifact for **ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated
MoE Serving** (ACM ATC 2026). ELDR routes decode requests using prefill expert
activations and worker load, without changing expert selection.

## Run

Ask the authors for SSH access and the input/calibration bundle. On the supplied
controller (**node1**), clone into your own directory; replace `ReviewerA` with
your reviewer name:

```bash
cd /mnt/md0
mkdir ReviewerA
cd ReviewerA
git clone https://github.com/casys-kaist/ELDR.git eldr
cd eldr
```

Place the supplied inputs, `manifest.json` and `cluster.json` in `eldr/inputs/`,
then run all six experiments and generate their plots:

```bash
bash eldr/scripts/run_all.sh
```

Scripts install CPU dependencies, verify inputs, and automatically synchronize
serving code to **node2, node3 and node4**. Models and inputs are not in Git.
The authors prepare the models, serving image, Docker/ROCm/RDMA, uv, rsync and
SSH control connections. Reviewers share the cluster and must run one at a time.
Allow **1–2 days** for the full suite, including model loading and compilation.

## Experiments

Run scripts are in [eldr/scripts/](eldr/scripts/); each also generates its plots.
Main experiments use all three models. Signature, cluster-balance and locality-band
experiments cover both Task and Language; prefix cache uses GPT-OSS Task.

| Figure | Script | Requests/s | Estimated traffic + warmup |
| --- | --- | --- | --- |
| 10: Main Task | `fig10_main_task.sh` | 20, 40, 60, 80, 100 | 4.6 h |
| 11: Main Language | `fig11_main_language.sh` | 20, 40, 60, 80, 100 | 4.6 h |
| 13: Signature | `fig13_signature.sh` | 60 | 1.8 h |
| 14: Cluster balance | `fig14_cluster_balance.sh` | 20, 40, 60, 80, 100 | 4.8 h |
| 15: Locality band | `fig15_locality_band.sh` | 20, 40, 60, 80, 100 | 7.8 h |
| 16: Prefix cache | `fig16_prefix_cache.sh` | 100 | 18 min |

These six experiments cover the main performance results and core design choices.
For a single experiment, run its script, for example:

```bash
bash eldr/scripts/fig10_main_task.sh
```

The default is counts × IDF, a fixed layer mask, global L2 normalization, static
centroids and locality-band JSQ with τ=0.1. A singleton band adds the second-nearest
decoder; τ=0 retains pure locality without expansion. Fitting uses seed 1; traffic
uses seed 1234 and 512 output tokens. Prefill routing is prefix-hash based.
Deployments use 8P16D, except prefix cache (1P16D, four conditions, 12,000 requests
each from a 2,000-prompt pool).

Use `--output NEW_DIR` to choose a new result directory, `--plan` to validate
without starting GPU experiments, or `--config CLUSTER_JSON` for another cluster.
Update the supplied `cluster.json` with that cluster's hosts, model paths,
GPU/CPU/NIC mappings and ports. Required models, image and SSH connections must
be prepared there too. Each script supports `--help`.

## Results

Results are saved under `eldr/artifacts/runs/<run>/`; the script prints the path.
Each experiment produces CSV/JSON summaries and PDF/PNG plots. Raw measurements
are in `<worker-group>/run/<trial>/measure/raw.json`. Runs stop on error and never
overwrite earlier results. Send the printed log and `cleanup.json` to the authors
if a run fails; do not reconnect worker SSH sessions or kill shared jobs.

Metrics are TPOT P50/P95/P99 and TTFT P50. TTFT ends when the prefill-generated
first token reaches the client. Tail latency can vary across runs; report
substantial or persistent discrepancies to the authors.

To redraw all completed experiments without GPUs:

```bash
bash eldr/scripts/plot_all.sh RUN_DIR
```

Plots go to `eldr/artifacts/figures/all-<timestamp>/`. For one experiment, use its
matching `plot_fig*.sh` script with that experiment's result directory.

## Code and tests

```text
eldr/scripts/       Experiment and plotting entry points
eldr/serving/       Proxy, signatures, clustering and routing
eldr/experiments/   Experiment definitions, datasets and shared runner
eldr/tests/         CPU and opt-in GPU tests
vllm/eldr/          Gate capture, block counts and signature transport
```

Other source directories are upstream vLLM. `eldr/inputs/` contains separately
supplied data; `eldr/artifacts/` holds generated results and caches.
CPU tests require neither inputs nor GPUs:

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
The serving image uses ROCm 7.2.2 (host ROCm: 6.2.0):

```text
rocm/vllm-dev@sha256:411f7ca9abeb57f69064e8bf60f46b5cd0b67f26264af0fc831f124fbc22f4b1
```

Based on [vLLM](https://github.com/vllm-project/vllm) commit
`d801ae8c2650b590438f3d9794dd7a47abd86c9a`, under [Apache 2.0](LICENSE).
Modified upstream files are marked; original license and copyright notices are
retained. Please also cite the [vLLM paper](https://arxiv.org/abs/2309.06180).
Models and datasets retain their own licenses; see the supplied input bundle's
`README.md` for provenance and redistribution restrictions.
