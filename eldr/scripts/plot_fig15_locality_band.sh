#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  -h|--help) printf 'usage: %s RUN_DIR --output NEW_DIR\n' "$0"; exit 0 ;;
esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec .venv/bin/python -m eldr.experiments.locality_band --replay "$@"
