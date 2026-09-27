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
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PWD/eldr/artifacts/cache/matplotlib}
exec .venv/bin/python -m eldr.experiments.runner.results "$source" --output "$output"
