"""Read-only validation of the two approved runs, then render their five-rate plot."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from eldr.experiments.runner.config import file_sha256
from eldr.experiments.runner.experiment import check_resume
from eldr.experiments.runner.results import write_report
from eldr.experiments.runner.workers import write_new_json

ROOT = Path('/mnt/md0/eldr-camera-ready-20261008.mXxy7s')
OUTPUT = Path(__file__).resolve().parent / 'figures'
RATES = (20, 40, 60, 80, 100)
MODELS = ('qwen', 'gptoss', 'gemma')
METRICS = ('tpot50_ms', 'tpot95_ms', 'tpot99_ms', 'ttft50_ms')


def load(path):
    return json.loads(path.read_text())


def validate_runtime(folder, plan):
    for group in plan['fleets']:
        run = folder / group['key'] / 'run'
        prepared = load(run / 'prepared.json')
        site = prepared['site']
        assert site['proxy_cpus'] == '4-5' and site['prefix_cache']
        assert len(site['prefills']) == 8 and len(site['decoders']) == 16
        for path, digest in load(run / 'trial_inputs.json').items():
            assert file_sha256(path) == digest, path
        saved = load(run / 'results.json')['results']
        assert len(saved) == len(group['trials'])
        for (label, policy, protocol), metrics in zip(group['trials'], saved):
            assert metrics == load(run / label / 'measure' / 'metrics.json')
            assert metrics['protocol'] == protocol
            assert protocol['seed'] == 1234 and protocol['output_tokens'] == 512
            assert protocol['requests'] == protocol['request_rate'] * 120
            assert protocol['window_seconds'] == 5 and protocol['refit_seconds'] == 1
            for phase in ('warmup', 'measure'):
                capture = load(run / label / phase / 'capture.json')
                assert len(capture) == 8
                for state in capture:
                    assert all(state[key] for key in (
                        'prefix_cache_enabled', 'resident_signature_cache',
                        'generation_empty', 'same_process_checked'))
                    assert state['enabled'] == (policy == 'eldr')
                    assert state['transport_enabled'] == (policy == 'eldr')
                if policy == 'eldr':
                    route = load(run / label / phase / 'routing.json')
                    assert route['rule'] == 'locality_band_jsq_v1'
                    assert route['selector'] == 'jsq' and route['tau'] == 0.1
                    assert route['signature'] == 'count_idf'
                    assert route['centroid_mode'] == 'online'
                    assert not any(route['inflight_by_worker'])
                    refit = route['refit']
                    assert refit['balanced'] == (protocol['variant'] == 'balanced')
                    assert refit['window_seconds'] == 5 and refit['refit_seconds'] == 1
                    assert refit['refits_completed'] > 0 and refit['last_error'] is None
        for role, count in (('prefills', 8), ('decoders', 16)):
            logs = list(run.glob(f'eldr-*-{role}-*.log'))
            assert len(logs) == count
            for log in logs:
                args = next(line for line in log.read_text().splitlines()
                            if 'non-default args:' in line)
                assert "'enable_prefix_caching': True" in args
                assert f"'async_scheduling': {role == 'decoders'}" in args
                assert 'FULL_DECODE_ONLY' in args
                if role == 'prefills':
                    assert "'enforce_eager': True" in args


def main():
    assert not OUTPUT.exists(), 'Use a fresh output directory; never replace results.'
    panels, provenance = {}, {}
    for workload in ('task', 'language'):
        combined = []
        for suite, expected in (('run-2cpu-rr-eldr', 9),
                                ('run-2cpu-ablations-rates', 36)):
            folder = ROOT / suite / 'fig14_cluster_balance' / workload
            plan = load(folder / 'plan.json')
            assert plan['study'] == 'cluster_balance'
            for path, digest in plan['inputs'].items():
                assert file_sha256(path) == digest, path
            assert check_resume(plan, folder)
            validate_runtime(folder, plan)
            rows = load(folder / 'summary.json')
            assert len(rows) == expected
            provenance[str(folder)] = {
                'measurements': expected,
                'rates': sorted({row['rate'] for row in rows}),
                'inputs': plan['inputs'],
                'manifest_sha256': {name: file_sha256(folder / name) for name in
                                    ('plan.json', 'sources.json', 'summary.json', 'complete.json')},
            }
            for row in rows:
                row['source_run'] = str(folder)
                row['raw'] = str(folder / row['raw'])
                combined.append(row)
            print(f'Validated {workload}/{suite}: {expected} cells', flush=True)
        expected_cells = {(model, variant, rate) for model in MODELS
                          for variant in ('rr', 'balanced', 'vanilla') for rate in RATES}
        assert len(combined) == 45
        assert {(r['setting'], r['variant'], r['rate']) for r in combined} == expected_cells
        panels[workload] = combined

    per_rate, averages = [], []
    for workload, rows in panels.items():
        table = {(r['setting'], r['variant'], r['rate']): r for r in rows}
        for model in MODELS:
            for variant in ('balanced', 'vanilla'):
                changes = []
                for rate in RATES:
                    rr, candidate = table[model, 'rr', rate], table[model, variant, rate]
                    values = {metric: 100 * (candidate[metric] / rr[metric] - 1)
                              for metric in METRICS}
                    changes.append(values)
                    per_rate.append(dict(workload=workload, model=model, variant=variant,
                                         rate=rate, **values))
                averages.append(dict(workload=workload, model=model, variant=variant,
                                     **{m: mean(v[m] for v in changes) for m in METRICS}))
    assert len(per_rate) == 60 and len(averages) == 12
    OUTPUT.mkdir()
    write_report(panels, OUTPUT, 'cluster_balance', stem='cluster_balance_five_rates')
    for name, rows in (('per_rate_changes', per_rate), ('mean_changes', averages)):
        write_new_json(OUTPUT / f'{name}.json', rows)
        with (OUTPUT / f'{name}.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_new_json(OUTPUT / 'source_runs.json', {
        'sources': provenance, 'rates': RATES, 'measurements': 90,
        'aggregation': 'Equal-weight mean of per-rate matched RR-relative percentage changes; no pooled requests.',
        'renderer_sha256': {name: file_sha256(ROOT / 'eldr/eldr/experiments' / name)
                            for name in ('plots.py', 'plot_style.py')},
    })
    write_new_json(OUTPUT / 'validation.json', {
        'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'measurements': 90, 'preserved_60_req_s': 18, 'fresh_other_rates': 72,
        'raw_counts_metrics_hashes_sources_inputs_runtime_flags_cleanup': 'passed',
        'validation_script_sha256': file_sha256(Path(__file__)),
    })
    print(json.dumps(averages, indent=2), flush=True)
    print(f'Output: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
