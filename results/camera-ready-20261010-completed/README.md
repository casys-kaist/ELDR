# Completed camera-ready experiment previews — 2026-10-10

Fresh measurements from `run-2cpu-rr-eldr` and its matching
`run-2cpu-baselines` supplement, using two proxy CPU cores.
Unless varied by an ablation, ELDR uses Full signatures (counts × IDF,
calibrated layer mask, L2 normalization), locality-band JSQ (tau = 0.1),
and online centroid refitting with a 5-second history and 1-second interval.
Outputs are capped at 512 tokens.
Raw requests, input/source hashes and worker cleanup were checked before plotting.

The existing paper-style renderers were reused. The prefix-cache TTFT axis is
0–160 ms to fit these measurements; no data are clipped. These are result previews;
the paper and the artifact's main branch are unchanged.

| Figure | Scope | PDF | PNG |
| --- | --- | --- | --- |
| Main Task | Three models; all six policies, 90 measurements | [PDF](main_task_all_policies.pdf) | [PNG](main_task_all_policies.png) |
| Main Task, RR/ELDR preview | Three models; RR and ELDR only | [PDF](fig10_main_task.pdf) | [PNG](fig10_main_task.png) |
| Main Language | Three models; RR and ELDR only | [PDF](fig11_main_language.pdf) | [PNG](fig11_main_language.png) |
| Qwen Task | All six policies, completed | [PDF](main_task_qwen_all_policies.pdf) | [PNG](main_task_qwen_all_policies.png) |
| Signature | Task and Language | [PDF](fig13_signature.pdf) | [PNG](fig13_signature.png) |
| Cluster balance | Task and Language | [PDF](fig14_cluster_balance.pdf) | [PNG](fig14_cluster_balance.png) |
| Locality band | Task and Language | [PDF](fig15_locality_band.pdf) | [PNG](fig15_locality_band.png) |
| Prefix cache | GPT-OSS Task, cache off/on | [PDF](fig16_prefix_cache.pdf) | [PNG](fig16_prefix_cache.png) |

Main figures use 20/40/60/80/100 requests/s on 8P16D. Ablations use 60 requests/s;
prefix cache uses 1P16D. Main plots show TPOT P50/P99 and TTFT P50. Ablation bars
and heatmaps show latency change relative to matched RR: **negative is better**.
All results are single measurements per configuration, not confidence intervals.

The Main Task six-policy figure combines 30 completed RR/ELDR measurements with
60 supplementary measurements from `run-2cpu-baselines`, using identical source,
inputs, topology, CPU allocation and traffic settings. Its CSV explicitly identifies
both source runs. The earlier Qwen-only preview is retained above. The complete
Main Language baseline supplement is still running and is not plotted here.
No earlier paused run or reviewer results are used.

Adjacent CSV files contain the plotted measurements, TPOT P95, and raw-result
SHA-256 hashes. Task/Language ablation summaries are separate CSV files.
