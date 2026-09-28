#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s [--config CLUSTER_JSON] [--output INPUT_DIR] [--setting MODEL-WORKLOAD]\n' "$0"
}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
config=eldr/experiments/cluster.json
arguments=()
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --config|--output|--setting)
            if (($# < 2)) || [[ -z $2 || $2 == --* ]]; then usage >&2; exit 2; fi
            if [[ $1 == --config ]]; then config=$2; else arguments+=("$1" "$2"); fi
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! -f $config ]]; then usage >&2; exit 2; fi
exec 9>>/tmp/eldr-ae.lock
if ! flock -n 9; then printf 'Another ELDR preparation/experiment is running.\n' >&2; exit 1; fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/eldr/artifacts/cache/uv}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r eldr/experiments/datasets/requirements.txt
exec .venv/bin/python -m eldr.experiments.prepare_data --config "$config" "${arguments[@]}"
