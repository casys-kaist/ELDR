#!/usr/bin/env bash
# Redraw from source, never by converting existing PDFs. Requires uv and Pango.
set -euo pipefail
if [ "$#" -ne 0 ]; then
    printf '%s\n' 'Usage: bash plots/redraw.sh' >&2
    exit 2
fi
cd "$(dirname "$0")/.."

for name in motivation moe_latency active_experts language_mix sig_transform sig_layers \
            cluster_pca main expert_reduction signature_ablation cluster_balance \
            locality_band qwen235 prefix_cache workload_drift; do
    uv run --no-project --python 3.12 --with matplotlib==3.10.9 \
        --with numpy==2.5.3 python -B "plots/plot_${name}.py"
done
uv run --no-project --python 3.12 --with weasyprint==70.0 \
    python -B plots/plot_diagrams.py
