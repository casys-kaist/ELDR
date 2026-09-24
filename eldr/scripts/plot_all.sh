#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s RUN_DIR [--output NEW_DIR]\n' "$0"
}
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
if [[ $# != 1 && $# != 3 ]] || [[ $# == 3 && $2 != --output ]]; then
    usage >&2; exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source=$1
output=${3:-eldr/artifacts/figures/all-$(date -u +%Y%m%dT%H%M%S)-$BASHPID}
if [[ ! -d $source || -z $output || -e $output || -L $output ]]; then
    usage >&2
    printf 'Use a completed run directory and a new output directory.\n' >&2
    exit 2
fi
entries=(
    fig10_main_task fig11_main_language
    fig13_signature/task fig13_signature/language
    fig14_cluster_balance/task fig14_cluster_balance/language
    fig15_locality_band/task fig15_locality_band/language
    fig16_prefix_cache
)
for entry in "${entries[@]}"; do
    if [[ ! -f $source/$entry/complete.json ]]; then
        printf 'Missing completed experiment: %s/%s\n' "$source" "$entry" >&2
        exit 1
    fi
done
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
for entry in "${entries[@]}"; do
    printf 'Plotting %s\n' "$entry"
    bash "eldr/scripts/plot_${entry%%/*}.sh" "$source/$entry" --output "$output/$entry"
done
printf 'All plots and summaries: %s\n' "$(cd -- "$output" && pwd)"
