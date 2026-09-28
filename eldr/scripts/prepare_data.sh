#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s {task|language} --tokenizer LOCAL_DIR --output NEW_DIR\n' "$0"
}
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    task|language) workload=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac
if (($# == 0)); then usage >&2; exit 2; fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/eldr/artifacts/cache/uv}
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r eldr/experiments/datasets/requirements.txt
exec .venv/bin/python -m "eldr.experiments.datasets.build_$workload" "$@"
