"""CPU-only analysis of the adaptive same-program wake-up follow-up.

The first same-program experiment remains frozen. Its estimator is loaded
in a separate module namespace, with only the new condition names changed.
Every additional contrast reuses its shared worker/visit bootstrap draws.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve()
BASE_SOURCE = SOURCE.with_name('analysis.py')
CONDITIONS = ('fast_gap', 'idle_20ms', 'idle_then_same_prime',
    'idle_then_disjoint_prime')


def _load_base():
    spec = importlib.util.spec_from_file_location(
        '_layout_coordination_wake_base_analysis', BASE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONDITIONS = CONDITIONS
    return module


base = _load_base()
encode = base.encode
_csv_bytes = base._csv_bytes


def analyze(protocol, raw_records):
    """Keep the original analysis untouched and add paired wake-up contrasts."""
    base._require(protocol['stage'] == 'same_program_wake_followup',
        'Unexpected follow-up stage')
    result = base.analyze(protocol, raw_records)
    timing = protocol['timing']
    workers, visits, reps = [int(timing[key]) for key in
        ('workers', 'visits', 'bootstrap_replicates')]
    rng = np.random.default_rng(int(timing['seed']))
    wi = rng.integers(0, workers, (reps, workers))
    vi = rng.integers(0, visits, (reps, workers, visits))
    base._require(base.sha(wi.tobytes() + vi.tobytes()) == result['bootstrap']['draw_SHA256'],
        'Additional contrasts must reuse the same bootstrap draws')
    index = {(row['condition'], row['variant']):
        np.asarray(row['worker_visit_medians_us']) for row in result['rows']}
    pooled = {condition: (index[(condition, 'A')] + index[(condition, 'B')]) / 2
        for condition in CONDITIONS}
    boot = {condition: values[wi[:, :, None], vi].mean((1, 2))
        for condition, values in pooled.items()}
    means = {condition: float(values.mean()) for condition, values in pooled.items()}
    threshold = result['practical_threshold_percent']
    contrasts = []
    for condition, reference in (
            ('idle_then_same_prime', 'idle_20ms'),
            ('idle_then_disjoint_prime', 'idle_20ms'),
            ('idle_then_same_prime', 'idle_then_disjoint_prime')):
        workers_out = []
        for worker_index, worker in enumerate(result['audit']['workers']):
            values = pooled[condition][worker_index] - pooled[reference][worker_index]
            workers_out.append(dict(invocation=worker, difference_us=float(values.mean()),
                paired_visit_differences_us=values.tolist()))
        contrasts.append(dict(condition=condition, reference=reference,
            time_difference_us=base._estimate(means[condition] - means[reference],
                boot[condition] - boot[reference]),
            time_reduction_percent=base._estimate(
                100 * (means[reference] - means[condition]) / means[reference],
                100 * (boot[reference] - boot[condition]) / boot[reference], threshold),
            matched_worker_effects=workers_out))
    excess = means['idle_20ms'] - means['fast_gap']
    excess_boot = boot['idle_20ms'] - boot['fast_gap']
    recovery = []
    for condition in CONDITIONS[2:]:
        recovered = means['idle_20ms'] - means[condition]
        recovered_boot = boot['idle_20ms'] - boot[condition]
        row = dict(condition=condition,
            definition='100*(T_idle-T_primed)/(T_idle-T_fast)',
            idle_excess_us=base._estimate(excess, excess_boot),
            recovered_time_us=base._estimate(recovered, recovered_boot),
            positive_denominator_bootstrap_fraction=float(np.mean(excess_boot > 0)),
            zero_denominator_bootstrap_count=int(np.sum(excess_boot == 0)),
            estimate_percent=float(100 * recovered / excess) if excess != 0 else None,
            CI95_percent=None, discarded_bootstrap_replicates=0)
        if np.all(excess_boot > 0):
            row['CI95_percent'] = np.percentile(100 * recovered_boot / excess_boot,
                [2.5, 97.5]).tolist()
            row['status'] = 'positive_idle_excess_in_all_bootstrap_draws'
        else:
            row['status'] = 'unstable_or_nonpositive_idle_excess_no_recovery_interval'
        recovery.append(row)
    result.update(wake_contrasts=contrasts, recovery_diagnostics=recovery,
        wake_interpretation='The timed target is identical in all observations. Priming kernels execute outside target events. Recovery after a disjoint-buffer, different-CUBIN primer would show that reuse of the exact target data is not necessary; it does not uniquely identify frequency, clock gating, scheduling, event overhead or another GPU-state mechanism.',
        adaptive_scope='This follow-up was designed after seeing the first unlocked same-program experiment. It is a new adaptive diagnostic; first-run raw samples and estimates are not pooled or replaced.')
    result['limitations'].extend([
        'The disjoint primer has independent allocations and a different frozen CUBIN but exercises similar memory/computation resources. It can affect shared cache state as well as GPU execution state; it is not a cache-free intervention.',
        'A recovery percentage is an unstable ratio when idle-minus-fast approaches or crosses zero. Its interval is omitted unless every bootstrap denominator is positive; no draws are discarded and no recovery outcome is required.'])
    return result


def _recovery_text(value):
    if value['estimate_percent'] is None:
        return 'undefined'
    point = f'{value["estimate_percent"]:.3f}'
    if value['CI95_percent'] is None:
        return point + ' [interval omitted: unstable denominator]'
    lo, hi = value['CI95_percent']
    return f'{point} [{lo:.3f}, {hi:.3f}]'


def render(result):
    text = base.render(result)
    lines = ['', '## 自适应 wake-up 跟进的额外配对对照', '',
        '这一轮在首轮结果之后设计，独立采集，不与首轮样本混算。四种条件均计时同一目标 graph；primer 位于目标事件外。same-prime 使用目标程序和目标数据，disjoint-prime 使用另一冻结 CUBIN 与独立分配的数据。', '',
        '| Condition | Reference | condition−reference µs [95% CI] | 时间降低 % [95% CI] |',
        '|---|---|---:|---:|']
    for row in result['wake_contrasts']:
        lines.append(f'| {row["condition"]} | {row["reference"]} | '
            f'{base._cell(row["time_difference_us"])} | '
            f'{base._cell(row["time_reduction_percent"])} |')
    lines += ['', '恢复比例为 100×(idle−primed)/(idle−fast)，允许小于 0% 或超过 100%；它不是额外的独立观察。所有分子与分母直接共用同一 bootstrap 重采样。若任何重采样的 idle−fast 分母不为正，保留点值与分母诊断，但不报告恢复比例区间，也不删除这些抽样。', '',
        '| Primer condition | 恢复比例 % [95% CI] | 正分母抽样比例 |',
        '|---|---:|---:|']
    for row in result['recovery_diagnostics']:
        lines.append(f'| {row["condition"]} | {_recovery_text(row)} | '
            f'{row["positive_denominator_bootstrap_fraction"]:.6f} |')
    lines += ['', '若 disjoint-prime 也恢复目标时间，只能说明精确重用目标输入并非恢复的必要条件。两种 primer 都会改变 GPU 与缓存的运行历史，不能由这一对照独断地判定 DVFS、clock gating、调度或 event 时间路径中的哪一种是原因。采样时钟仍不能充当每个 kernel 的瞬时频率证明。', '']
    return text + '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / 'preregistration/protocol.json'
    raw_paths = sorted((root / 'timing').glob('invocation_*/raw.json'))
    base._require(bool(raw_paths), 'No timing/invocation_*/raw.json under root')
    protocol = json.loads(protocol_path.read_text())
    result = analyze(protocol, [json.loads(path.read_text()) for path in raw_paths])
    result['inputs'] = dict(protocol=dict(path=str(protocol_path),
        SHA256=base.sha(protocol_path.read_bytes())),
        raw=[dict(path=str(path), SHA256=base.sha(path.read_bytes())) for path in raw_paths],
        analysis_source=dict(path=str(SOURCE), SHA256=base.sha(SOURCE.read_bytes())),
        base_analysis_source=dict(path=str(BASE_SOURCE), SHA256=base.sha(BASE_SOURCE.read_bytes())))
    output = root / 'analysis'
    outputs = {'analysis.json': base.encode(result),
        'timing.csv': base._csv_bytes(result), 'REPORT.md': render(result).encode()}
    base._require(not any((output / name).exists() for name in outputs),
        'Refusing to overwrite an existing follow-up analysis artifact')
    output.mkdir(parents=True, exist_ok=True)
    for name, blob in outputs.items():
        with (output / name).open('xb') as handle:
            handle.write(blob)
    print(json.dumps(dict(status=result['status'], raw_samples=result['audit']['raw_sample_count'],
        output=str(output)), sort_keys=True))


if __name__ == '__main__':
    main()
