"""Redraw the validated figure from the adjacent data, without raw requests."""

import argparse
import json
from pathlib import Path

from renderer.plots import draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='A new directory for the PDF and PNG.')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    panels = {}
    expected = {(model, variant, rate) for model in ('qwen', 'gptoss', 'gemma')
                for variant in ('rr', 'balanced', 'vanilla')
                for rate in (20, 40, 60, 80, 100)}
    for workload in ('task', 'language'):
        rows = json.loads((folder / f'cluster_balance_five_rates-{workload}.json').read_text())
        assert len(rows) == 45
        assert {(r['setting'], r['variant'], r['rate']) for r in rows} == expected
        panels[workload] = rows
    args.output.mkdir(parents=True, exist_ok=False)
    draw(panels, args.output / 'cluster_balance_five_rates', 'cluster_balance')


if __name__ == '__main__':
    main()
