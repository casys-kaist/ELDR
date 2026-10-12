#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s RUN_DIR [--output NEW_DIR] [--available]\n' "$0"
}
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
if (($# == 0)); then usage >&2; exit 2; fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source=$1
shift
output=eldr/artifacts/figures/all-$(date -u +%Y%m%dT%H%M%S)-$BASHPID
options=()
while (($#)); do
    case "$1" in
        --available) options+=(--available); shift ;;
        --output)
            if (($# < 2)) || [[ -z $2 || $2 == --* ]]; then usage >&2; exit 2; fi
            output=$2; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! -d $source || -z $output || -e $output || -L $output ]]; then
    usage >&2
    printf 'Use a completed run directory and a new output directory.\n' >&2
    exit 2
fi
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PWD/eldr/artifacts/cache/matplotlib}
exec .venv/bin/python -m eldr.experiments.runner.results "$source" --output "$output" "${options[@]}"
