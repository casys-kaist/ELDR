# Paper figure sources

Plotting code and compact inputs for all 19 figures included in `main.tex`.
The scripts write directly to `../figure/`. They do not run GPU experiments,
launch measurements, convert PDFs, or outline text.

With Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Pango installed:

```bash
bash plots/redraw.sh
```

Compile `main.tex` with pdfLaTeX (locally: `latexmk -pdf main.tex`),
then audit the resulting PDF:

```bash
uv run --no-project --with pymupdf==1.28.2 python plots/check_fonts.py main.pdf
```

The script pins Matplotlib 3.10.9. Plots and HTML diagrams use the same bundled
DejaVu Sans family, including bold and italic faces, with embedded TrueType fonts.
Diagrams are rendered directly from HTML with WeasyPrint 70.0.
Overleaf compiles the generated PDFs; Python runs locally, not inside Overleaf.

## Shared presentation

`figstyle.py` defines typography, palette and legend styling. RR is blue
(`#1f77b4`) and ELDR is red (`#d62728`) in policy comparisons. P2C is orange
(`#ff7f0e`) in the main figures so each policy has a distinct color. Blue
and red distinguish ablation alternatives and the ELDR configuration,
respectively. ELDR bars and legend entries are on the right in the signature
and cluster-balance comparisons, with configuration-only legend labels;
the signature-quality chart also places
ELDR's count·idf transform last. Blue/red cell colors in the locality
heatmap denote signed improvement/regression, not policy identity; all
band-width ticks use the same neutral styling. Task/Language and domain colors
retain their separate meanings. Legends
use square light-gray frames. Dense multipanel figures retain compact text
sizes, while titles, labels and ticks keep a consistent hierarchy. The main
figures retain their approved shared P50/P99 axes and integer TTFT ticks.
RR curves and markers are drawn above the other policies without changing
the legend order, line widths, colors or data.
The batch-locality figure uses blue for mixed-domain batches and red for
same-domain batches. The overview separates initialization, the numbered
request path and asynchronous background refitting; its router box is
labeled only "ELDR Router". IDF and the fixed layer mask are described in
the paper text rather than inside that box.
ACM class settings and document float/caption spacing are not changed.

| Figure content | Source | Input in `data/` |
| --- | --- | --- |
| Domain heatmap; prefill/decode scatter | `plot_motivation.py` | `motivation.json` |
| MoE latency | `plot_moe_latency.py` | `u2_*.log` |
| Active experts vs. batch size | `plot_active_experts.py` | `opportunity_*.json` |
| Language distribution | `plot_language_mix.py` | `wc_langdist.json` |
| Signature transform | `plot_sig_transform.py` | `rho_transform.json` |
| Layer selection | `plot_sig_layers.py` | `sig_layers.json` |
| Signature PCA | `plot_cluster_pca.py` | `cluster_pca.json` |
| Main Task/Language | `plot_main.py` | `main_task.csv`, `main_lang.csv` |
| Expert reduction | `plot_expert_reduction.py` | `u_dumps/` |
| Signature ablation | `plot_signature_ablation.py` | `signature_ablation.json` |
| Cluster balance | `plot_cluster_balance.py` | `cluster_balance_five_rates-{task,language}.json` |
| Locality band | `plot_locality_band.py` | `locality_band_five_rates-{task,language}.json` |
| Prefix cache | `plot_prefix_cache.py` | `prefix_cache_100reqs.csv` |
| Workload drift | `plot_workload_drift.py` | `workload_drift.json` |
| Larger model | `plot_qwen235.py` | `qwen235.json` |
| Overview; signature-cache diagram | `plot_diagrams.py` | `overview.html`, `prefix_coherence.html` |

## Motivation inputs

The heatmap and prefill/decode scatter use the October 1, 2026 captures
(128 output tokens, temperature 0, seed 1), not the unavailable May 29 traces.
`motivation.json` contains the derived heatmap matrices and scatter coordinates,
plus capture settings and input hashes. It contains no request text.
The measured arrays and aggregation rules are unchanged; all figures redraw
from the bundled inputs, without GPUs or access to the raw capture directory.

To preview new captures and export their compact plot data separately:

```bash
python plots/plot_motivation.py /path/to/fresh-run --fresh --output /tmp/eldr-preview --save-data /tmp/motivation.json
python plots/test_motivation.py
```

Use the pinned environment above. The fresh run contains `inputs/{task,language}.json`
and six `results/<model>-<workload>-full.npz` files. Request IDs and input hashes
must match. The original plotting rules are unchanged; fresh measurements are
identified in the generated `summary.json`, not presented as recovered originals.

## Input provenance

Except for the captures and experiments identified here, measurement inputs
come from the original ELDR figure sources and archived benchmark outputs.
`data/benchmark_sources.json` records the
historical benchmark filenames and hashes. Main Task and Language use all 90
validated October cells per workload: 30 RR/online-ELDR plus 60 matched
Random/JSQ/P2C/Domain cells. Cluster balance uses 90 cells; locality band uses
150 cells. Both ablations combine the preserved 60 req/s cells with the fresh
20/40/80/100 req/s cells, with an equal-weight mean of per-rate matched
RR-relative percentage changes. They never pool requests across rates.
The data retain each row's `source_run` and raw-result hash; companion
`*_source_runs.json` and `*_validation.json` files record validation provenance.
Reviewer, paused-run and instrumented diagnostic measurements are not used.

Layer curves were recomputed with the original `plot_sig_layers_paper.py`
analysis (50,000 pairs, seed 0). PCA coordinates use the archived June 3
captures, original forward-greedy masks (30,000 pairs, seed 0), and the
clustering implementation at `bb87b98e0`. The arbitrary PCA sign is aligned
with the original figure. These compact arrays avoid bundling large captures.

Prefix cache uses the four validated October 100 req/s online-ELDR/RR cells.
The signature-ablation figure retains the existing 60 req/s measurements;
its pending five-rate sweep is not substituted. Drift and other unchanged
experiments retain their existing measurements. In particular, the drift
figure is the previously selected round-1 RR/static/online comparison, not
the queued new drift experiment. Old unused compact inputs are retained
only as historical evidence and are not read by the updated renderers.
DejaVu font copyright and redistribution terms are preserved in
`fonts/LICENSE.txt`.
