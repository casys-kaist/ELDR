"""Render validated five-rate ablations using the approved paper renderers.

Run with the existing plotting venv; no experiment or paper files are modified.
Negative deltas mean lower latency. Each variant uses its matched RR at each
rate, then the five percentages are averaged (never pooled requests).
"""
import contextlib
import csv
import hashlib
import io
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = Path('/mnt/md0/eldr-camera-ready-20261008.mXxy7s/run-2cpu-ablations-rates/figures-five-rates')
PAPER = Path('/mnt/md0/eldr-paper-language-20261010.LGCjeK/plots')
RATES = [20, 40, 60, 80, 100]
MODELS = ['qwen', 'gptoss', 'gemma']
METRICS = ['tpot50_ms', 'tpot95_ms', 'tpot99_ms', 'ttft50_ms']
SPECS = {
    'signature': ('fig13_signature', ['count_idf', 'gate_prob_all'], 60),
    'cluster_balance': ('fig14_cluster_balance', ['balanced', 'vanilla'], 45),
    'locality_band': ('fig15_locality_band', ['tau-0', 'tau-0.1', 'tau-0.2', 'tau-0.3'], 75),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def main():
    assert json.loads((SOURCE / 'complete.json').read_text()) == {
        'measurements': 360, 'added_measurements': 288,
    }
    provenance = json.loads((SOURCE / 'source_runs.json').read_text())
    assert provenance['rates'] == RATES
    data_dir = OUT / 'data'
    renderer_dir = OUT / 'renderer'
    data_dir.mkdir()
    renderer_dir.mkdir()
    frozen = {}
    for name in ['figstyle.py', 'plot_signature_ablation.py',
                 'plot_cluster_balance.py', 'plot_locality_band.py']:
        shutil.copy2(PAPER / name, renderer_dir / name)
        frozen[str(PAPER / name)] = digest(renderer_dir / name)
    for name in ['complete.json', 'source_runs.json']:
        shutil.copy2(SOURCE / name, data_dir / ('source_' + name))
        frozen[str(SOURCE / name)] = digest(SOURCE / name)

    mean_rows, per_rate_rows = [], []
    means = {}
    total = 0
    for study, (figure, variants, count) in SPECS.items():
        for workload in ['task', 'language']:
            path = SOURCE / f'{figure}-{workload}.json'
            rows = json.loads(path.read_text())
            table = {(r['setting'], r['variant'], r['rate']): r for r in rows}
            assert len(rows) == len(table) == count, path
            expected = set()
            for model in MODELS:
                for variant in variants:
                    setting = f'{model}-{variant}' if study == 'signature' else model
                    for rate in RATES:
                        expected.update([(setting, variant, rate), (setting, 'rr', rate)])
                    changes = []
                    for rate in RATES:
                        candidate, rr = table[setting, variant, rate], table[setting, 'rr', rate]
                        delta = {m: 100 * (candidate[m] / rr[m] - 1) for m in METRICS}
                        changes.append(delta)
                        per_rate_rows.append(dict(
                            study=study, workload=workload, model=model, variant=variant,
                            rate=rate, **delta, source_run=candidate['source_run'],
                            raw=candidate['raw'], rr_raw=rr['raw'],
                            raw_sha256=candidate['sha256'], rr_sha256=rr['sha256'],
                        ))
                    avg = {m: float(np.mean([d[m] for d in changes])) for m in METRICS}
                    means[study, workload, model, variant] = avg
                    mean_rows.append(dict(study=study, workload=workload,
                                          model=model, variant=variant, **avg))
            assert set(table) == expected, path
            for row in rows:
                run = Path(row['source_run'])
                assert str(run) in provenance['sources'][f'{figure}/{workload}']
                original = json.loads((run / 'summary.json').read_text())
                match = [r for r in original if (r['setting'], r['variant'], r['rate']) ==
                         (row['setting'], row['variant'], row['rate'])]
                assert len(match) == 1
                check = dict(match[0], source_run=str(run), raw=str(run / match[0]['raw']))
                assert row == check
                assert row['requests'] == 120 * row['rate']
            # The previous monitoring pass independently recomputed all 360 raw
            # results. Preserve those raw hashes and the exact source paths here.
            filename = f'{study}_five_rates-{workload}.json'
            write_json(data_dir / filename, rows)
            frozen[str(path)] = digest(path)
            total += len(rows)
    assert total == 360
    assert len(per_rate_rows) == 240 and len(mean_rows) == 48

    signature = {d: {m: [[means['signature', w, m, v][metric]
                           for metric in ['tpot50_ms', 'tpot99_ms']]
                          for v in SPECS['signature'][1]] for m in MODELS}
                 for d, w in [('task', 'task'), ('lang', 'language')]}
    write_json(data_dir / 'signature_ablation.json', signature)
    for name, rows in [('per_rate_changes.csv', per_rate_rows), ('mean_changes.csv', mean_rows)]:
        with (OUT / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    sys.path.insert(0, str(renderer_dir))
    import matplotlib.pyplot as plt
    import figstyle
    import plot_signature_ablation as signature_renderer
    import plot_cluster_balance as cluster_renderer
    import plot_locality_band as locality_renderer

    layout_checks = []

    def save_bars(fig, unused, *, study):
        # Keep the approved colors/order and font family; widen this review
        # preview slightly so five-rate values cannot overlap their neighbours.
        fig.set_size_inches(9.0, 5.4)
        for ax in fig.axes:
            for annotation in ax.texts:
                annotation.set_fontsize(11)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, (workload, model) in zip(fig.axes, itertools.product(['task', 'language'], MODELS)):
            variants = SPECS[study][1]
            for container, variant in zip(ax.containers, variants):
                actual = [rect.get_height() for rect in container.patches]
                expected = [means[study, workload, model, variant][m]
                            for m in ['tpot50_ms', 'tpot99_ms']]
                assert np.allclose(actual, expected, atol=1e-12)
            for a, b in itertools.combinations(ax.texts, 2):
                assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), (
                    study, workload, model, 'overlapping value labels', a.get_text(), b.get_text())
            for annotation in ax.texts:
                box = annotation.get_window_extent(renderer)
                assert box.y0 >= ax.bbox.y0 and box.y1 <= ax.bbox.y1, (study, 'value outside panel')
        layout_checks.append(study + ': exact bar heights; no overlapping or clipped value labels')
        figstyle.save(fig, OUT / study)
        plt.close(fig)

    for study, module in [('signature', signature_renderer), ('cluster_balance', cluster_renderer)]:
        module.DATA = data_dir
        module.OUTPUT = OUT
        module.save = lambda fig, path, study=study: save_bars(fig, path, study=study)
        # The reused signature renderer's console-only text still says r=60;
        # its data above are the five-rate means, checked against every bar.
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
        print('Rendered', study, 'five-rate means', flush=True)

    locality = {}
    for workload in ['task', 'language']:
        for model in MODELS:
            key = model + ('_task' if workload == 'task' else '_lang')
            locality[key] = {m: {tau: means['locality_band', workload, model, f'tau-{tau:g}'][m]
                                for tau in locality_renderer.TAUS}
                             for m, _ in locality_renderer.METRICS}
    locality_renderer.plot(locality, OUT / 'locality_band')
    write_json(OUT / 'validation.json', dict(
        measurements=360, added_measurements=288, preserved_rate60_measurements=72,
        rates=RATES, per_rate_comparisons=240, mean_comparisons=48,
        aggregation='mean over five matched per-rate 100*(candidate/RR - 1); negative is faster',
        prior_raw_revalidation='All 360 raw hashes, counts and metrics were independently rechecked at 2026-10-11 22:45 KST.',
        input_sha256=frozen, source_runs=provenance, layout_checks=layout_checks,
        paper_modified=False, github_pushed=False,
    ))
    print('Validated', total, 'source rows; 240 per-rate comparisons; 48 five-rate means.')


if __name__ == '__main__':
    main()
