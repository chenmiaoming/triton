"""CPU-only paired analysis: same CUBIN, shape, inputs and loaded graph.

Only the between-sample conditions and independent A/B observation labels
differ. All eight cells share hierarchical bootstrap draws. No old evidence
or compiler artifacts are rewritten.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


VARIANTS = ('A', 'B')
CONDITIONS = ('fast_gap', 'legacy_gap', 'idle_20ms', 'gpu_preload')
CASE = 'N02_k1024_p128_g128'
NVML_FIELDS = ('graphics_clock_mhz', 'sm_clock_mhz', 'memory_clock_mhz',
    'power_usage_mw', 'power_limit_mw', 'temperature_c',
    'gpu_utilization_percent', 'memory_utilization_percent', 'pstate',
    'clock_event_reasons')


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _require(test, message):
    if not test:
        raise ValueError(message)


def _ids(value):
    if isinstance(value, dict):
        return list(value)
    return [row['id'] if isinstance(row, dict) else row for row in value]


def _summary(values):
    a = np.asarray(values, dtype=float)
    _require(a.ndim == 1 and np.all(np.isfinite(a)), 'Invalid summary observations')
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), min=float(a.min()), p05=float(np.percentile(a, 5)),
        median=float(np.median(a)), mean=float(a.mean()),
        p95=float(np.percentile(a, 95)), max=float(a.max()))


def _direction(ci, threshold):
    if ci[0] > threshold:
        return 'practical_improvement'
    if ci[1] < -threshold:
        return 'practical_regression'
    if ci[0] >= -threshold and ci[1] <= threshold:
        return 'practically_equivalent'
    return 'inconclusive'


def _estimate(point, draws, threshold=None):
    ci = np.percentile(draws, [2.5, 97.5]).tolist()
    result = dict(estimate=float(point), CI95=ci)
    if threshold is not None:
        result['classification'] = _direction(ci, threshold)
    return result


def _telemetry(raw_records):
    """Summarize sampled telemetry without assigning a clock to a kernel."""
    groups = defaultdict(list)
    for raw in raw_records:
        for row in raw.get('telemetry', []):
            context = dict(row.get('context') or {})
            context.update({key: row[key] for key in ('phase', 'case', 'condition') if key in row})
            key = (str(raw['invocation']), str(context.get('phase', 'unlabelled')),
                context.get('case'), context.get('condition'))
            groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        worker, phase, case, condition = key
        entry = dict(invocation=worker, phase=phase, case=case, condition=condition,
            sample_count=len(rows), metrics={})
        for name in NVML_FIELDS:
            values, rc, errors = [], Counter(), Counter()
            for row in rows:
                field = row.get(name)
                if not isinstance(field, dict):
                    rc['missing'] += 1
                    continue
                code, value = field.get('rc'), field.get('value')
                rc[str(code)] += 1
                if field.get('error'):
                    errors[str(field['error'])] += 1
                if code == 0 and isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(value)
            metric = dict(return_code_counts=dict(rc), errors=dict(errors),
                successful_value_count=len(values))
            if name == 'clock_event_reasons':
                masks = Counter(hex(int(value)) for value in values)
                bits = Counter(hex(1 << bit) for value in values for bit in range(64)
                    if int(value) & (1 << bit))
                metric.update(mask_counts=dict(masks), active_bit_counts=dict(bits),
                    nonzero_mask_count=sum(int(value) != 0 for value in values))
            else:
                metric['summary'] = _summary(values)
            entry['metrics'][name] = metric
        timestamps = [row['monotonic_ns'] for row in rows if 'monotonic_ns' in row]
        entry['sample_monotonic_span_ns'] = [min(timestamps), max(timestamps)] if timestamps else None
        entry['query_duration_ns'] = _summary([row['query_duration_ns'] for row in rows
            if isinstance(row.get('query_duration_ns'), (int, float))])
        output.append(entry)
    return dict(groups=output,
        scope='NVML samples grouped by worker/phase/case/condition. Context labels may span host gaps and multiple kernels; no per-kernel clock attribution, clock locking, or causal DVFS conclusion follows.',
        mask_scope='clock_event_reasons is a sampled bitmask, not elapsed throttling time; unavailable getters retain their return codes and errors.')


def _collect(protocol, raw_records):
    timing = protocol['timing']
    workers, visits, samples, reps = [int(timing[key]) for key in
        ('workers', 'visits', 'samples', 'bootstrap_replicates')]
    _require((workers, visits, samples, reps) == (3, 12, 31, 8192),
        'Expected preregistered 3 workers / 12 visits / 31 samples / 8192 replicates')
    cases = _ids(protocol['cases'])
    conditions = _ids(protocol.get('conditions', CONDITIONS))
    variants = _ids(protocol.get('variants', VARIANTS))
    _require(cases == [CASE], 'Expected one frozen same-shape AWQ case')
    _require(len(conditions) == 4 and set(conditions) == set(CONDITIONS), 'Unexpected conditions')
    _require(len(variants) == 2 and set(variants) == set(VARIANTS), 'Unexpected observation aliases')
    expected = {(case, condition, variant) for case in cases
        for condition in conditions for variant in variants}
    protocol_sha = sha(encode(protocol))
    _require(len(raw_records) == workers, 'Incomplete worker count')
    index, visit_ids, environments, checks, metadata = {}, {}, [], [], []
    raw_summaries, worker_summaries, windows = [], [], {}
    raw_count = 0
    for raw in raw_records:
        worker = str(raw['invocation'])
        _require(worker not in index, 'Duplicate worker ' + worker)
        _require(raw['status'] == 'OK', 'Worker failed: ' + worker)
        _require(raw['clock_control'] == 'unlocked', 'This analysis is for unlocked measurements')
        _require(raw['protocol_SHA256'] == protocol_sha, 'Protocol SHA mismatch: ' + worker)
        _require(raw['payload_SHA256'] == protocol['payload_SHA256'], 'Payload SHA mismatch: ' + worker)
        _require(bool(raw.get('checks')) and bool(raw.get('bindings')) and bool(raw.get('orders')),
            'Missing checks, bindings, or orders: ' + worker)
        for check in raw['checks']:
            _require(not isinstance(check, dict) or check.get('pass') is True,
                'Failed/missing correctness result: ' + worker)
        logical, common_ids = {}, None
        for cell in raw['cells']:
            key = (cell['case'], cell['condition'], cell['variant'])
            _require(key in expected and key not in logical, 'Unexpected or duplicate cell: ' + str(key))
            entries = cell['visits']
            by_visit = {str(entry['visit']): entry for entry in entries}
            _require(len(entries) == len(by_visit) == visits, 'Incomplete/duplicate visits: ' + str(key))
            ids = sorted(by_visit, key=lambda item: (len(item), item))
            if common_ids is None:
                common_ids = ids
            _require(ids == common_ids, 'Unpaired visit identifiers: ' + str(key))
            medians, all_samples = [], []
            for visit in ids:
                entry = by_visit[visit]
                vector = np.asarray(entry['samples_us'], dtype=float)
                _require(vector.shape == (samples,) and np.all(np.isfinite(vector))
                    and np.all(vector > 0), 'Invalid samples: ' + str((worker, key, visit)))
                _require(isinstance(entry.get('start_ns'), (int, float))
                    and isinstance(entry.get('end_ns'), (int, float))
                    and entry['end_ns'] > entry['start_ns'], 'Invalid visit window: ' + str(key))
                medians.append(float(np.median(vector)))
                all_samples.extend(vector.tolist())
                raw_count += len(vector)
                raw_summaries.append(dict(invocation=worker, case=key[0], condition=key[1],
                    variant=key[2], visit=visit, samples_us=_summary(vector)))
                detail = dict(invocation=worker, case=key[0], condition=key[1],
                    variant=key[2], visit=visit, start_ns=entry['start_ns'], end_ns=entry['end_ns'])
                for field in ('host_gaps_ns', 'guard_duration_ns', 'launch_ns', 'completion_ns'):
                    values = entry.get(field)
                    _require(values is not None, 'Missing ' + field + ': ' + str(key))
                    values = np.asarray(values, dtype=float)
                    _require(values.shape == (samples,) and np.all(np.isfinite(values)) and np.all(values >= 0),
                        'Invalid ' + field + ': ' + str(key))
                    detail[field] = _summary(values)
                launch = np.asarray(entry['launch_ns'])
                completion = np.asarray(entry['completion_ns'])
                _require(np.all(launch >= entry['start_ns']) and np.all(completion <= entry['end_ns'])
                    and np.all(completion > launch) and np.all(np.diff(launch) > 0),
                    'Invalid launch/completion ordering: ' + str(key))
                detail['node_type_counts'] = dict(Counter(str(x) for x in entry.get('node_types', [])))
                metadata.append(detail)
                windows[(worker, key[1], key[2], visit)] = (entry['start_ns'], entry['end_ns'])
            logical[key] = medians
            worker_summaries.append(dict(invocation=worker, case=key[0], condition=key[1],
                variant=key[2], raw_samples_us=_summary(all_samples), visit_medians_us=_summary(medians)))
        _require(set(logical) == expected, 'Incomplete cell matrix: ' + worker)
        index[worker] = logical
        visit_ids[worker] = common_ids
        environments.append(dict(invocation=worker, environment=raw['environment']))
        checks.append(dict(invocation=worker, count=len(raw['checks']), bindings=raw['bindings']))
    worker_ids = sorted(index, key=lambda item: (len(item), item))
    _require(all(visit_ids[worker] == visit_ids[worker_ids[0]] for worker in worker_ids),
        'Visit identifiers differ across workers')
    arrays = {key: np.asarray([index[worker][key] for worker in worker_ids]) for key in expected}
    audit = dict(workers=worker_ids, visits_by_worker=visit_ids,
        cells_per_worker=len(expected), physical_cells_all_workers=workers * len(expected),
        raw_sample_count=raw_count, environments=environments, checks=checks,
        visit_metadata_summaries=metadata, raw_sample_visit_summaries=raw_summaries,
        worker_variation=worker_summaries, clock_control='unlocked')
    return arrays, audit, windows


def _correlation(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _telemetry_associations(raw_records, windows, arrays, audit):
    """Visit-window associations are descriptive, with missing windows retained."""
    metrics = ('sm_clock_mhz', 'graphics_clock_mhz', 'memory_clock_mhz',
        'power_usage_mw', 'temperature_c', 'gpu_utilization_percent')
    matched, means = [], {}
    raw_index = {str(raw['invocation']): raw for raw in raw_records}
    for worker in audit['workers']:
        telemetry = raw_index[worker].get('telemetry', [])
        for condition in CONDITIONS:
            for variant in VARIANTS:
                for visit in audit['visits_by_worker'][worker]:
                    start, end = windows[(worker, condition, variant, visit)]
                    rows = [row for row in telemetry if start <= row.get('monotonic_ns', -1) <= end]
                    values = {}
                    for name in metrics:
                        numeric = [row[name]['value'] for row in rows
                            if isinstance(row.get(name), dict) and row[name].get('rc') == 0
                            and isinstance(row[name].get('value'), (int, float))
                            and np.isfinite(row[name]['value'])]
                        values[name] = dict(sample_count=len(numeric),
                            mean=float(np.mean(numeric)) if numeric else None)
                    means[(worker, condition, variant, visit)] = values
                    matched.append(dict(invocation=worker, condition=condition, variant=variant,
                        visit=visit, telemetry_sample_count=len(rows), metrics=values))
    associations = []
    for wi, worker in enumerate(audit['workers']):
        for variant in VARIANTS:
            for condition in CONDITIONS:
                if condition == 'fast_gap':
                    continue
                for name in metrics:
                    x, y = [], []
                    for vi, visit in enumerate(audit['visits_by_worker'][worker]):
                        a = means[(worker, condition, variant, visit)][name]['mean']
                        b = means[(worker, 'fast_gap', variant, visit)][name]['mean']
                        if a is not None and b is not None:
                            x.append(a - b)
                            y.append(float(arrays[(CASE, condition, variant)][wi, vi]
                                - arrays[(CASE, 'fast_gap', variant)][wi, vi]))
                    associations.append(dict(invocation=worker, variant=variant, condition=condition,
                        comparison='condition minus fast_gap at same visit', metric=name,
                        paired_visit_count=len(x), telemetry_difference=_summary(x),
                        time_difference_us=_summary(y), pearson_r=_correlation(x, y)))
    distinct = {(worker, start, end) for (worker, _, _, _), (start, end) in windows.items()}
    return dict(visit_windows=matched, distinct_recorded_windows=len(distinct),
        alias_windows_may_overlap=True,
        alias_window_scope='A/B are interleaved observations within the same condition/visit block; their identical telemetry windows are not independent telemetry observations.',
        paired_within_worker_associations=associations,
        scope='Descriptive correlations between matched visit-level time differences and NVML sample-mean differences. Only samples whose query start lies in each recorded visit window are included. Missing windows remain missing; no interpolation, per-kernel clock assignment or causal interpretation. Power/utilization hardware averaging can cross window boundaries.')


def analyze(protocol, raw_records):
    """Analyze loaded immutable JSON, without writing files or invoking GPUs."""
    arrays, audit, windows = _collect(protocol, raw_records)
    timing = protocol['timing']
    workers, visits, reps = [int(timing[key]) for key in
        ('workers', 'visits', 'bootstrap_replicates')]
    seed, threshold = int(timing['seed']), float(timing['practical_threshold_percent'])
    _require(threshold == 3, 'Expected preregistered 3% practical threshold')
    rng = np.random.default_rng(seed)
    wi = rng.integers(0, workers, (reps, workers))
    vi = rng.integers(0, visits, (reps, workers, visits))
    boot = {key: array[wi[:, :, None], vi].mean((1, 2)) for key, array in arrays.items()}
    means = {key: float(array.mean()) for key, array in arrays.items()}
    pooled = {condition: (arrays[(CASE, condition, 'A')] + arrays[(CASE, condition, 'B')]) / 2
        for condition in CONDITIONS}
    pooled_boot = {condition: (boot[(CASE, condition, 'A')] + boot[(CASE, condition, 'B')]) / 2
        for condition in CONDITIONS}
    rows, equivalence, effects = [], [], []
    for condition in CONDITIONS:
        for variant in VARIANTS:
            key = (CASE, condition, variant)
            rows.append(dict(case=CASE, condition=condition, variant=variant,
                time_us=_estimate(means[key], boot[key]),
                worker_visit_medians_us=arrays[key].tolist(),
                worker_means_us=arrays[key].mean(axis=1).tolist(),
                visit_medians_us=_summary(arrays[key].ravel()),
                worker_mean_variation_us=_summary(arrays[key].mean(axis=1))))
        a, b = (CASE, condition, 'A'), (CASE, condition, 'B')
        equivalence.append(dict(condition=condition, comparison='B minus A',
            independent_observations=True,
            time_difference_us=_estimate(means[b] - means[a], boot[b] - boot[a]),
            time_reduction_B_vs_A_percent=_estimate(100 * (means[a] - means[b]) / means[a],
                100 * (boot[a] - boot[b]) / boot[a], threshold)))
        reference = pooled['fast_gap']
        ref_boot = pooled_boot['fast_gap']
        worker_rows = []
        for worker_index, worker in enumerate(audit['workers']):
            difference = pooled[condition][worker_index] - reference[worker_index]
            worker_rows.append(dict(invocation=worker, condition_time_us=float(pooled[condition][worker_index].mean()),
                fast_gap_time_us=float(reference[worker_index].mean()),
                difference_us=float(difference.mean()), paired_visit_differences_us=difference.tolist(),
                paired_visit_difference_summary_us=_summary(difference),
                time_reduction_percent=float(100 * (reference[worker_index].mean()
                    - pooled[condition][worker_index].mean()) / reference[worker_index].mean())))
        effect = dict(condition=condition, reference='fast_gap',
            estimator='Equal-weight mean of independently measured A/B visit medians',
            time_us=_estimate(pooled[condition].mean(), pooled_boot[condition]),
            time_difference_us=_estimate(pooled[condition].mean() - reference.mean(),
                pooled_boot[condition] - ref_boot),
            time_reduction_vs_fast_percent=_estimate(100 * (reference.mean() - pooled[condition].mean()) / reference.mean(),
                100 * (ref_boot - pooled_boot[condition]) / ref_boot, threshold),
            matched_worker_effects=worker_rows, alias_specific_effects=[])
        for variant in VARIANTS:
            key, ref = (CASE, condition, variant), (CASE, 'fast_gap', variant)
            effect['alias_specific_effects'].append(dict(variant=variant,
                time_difference_us=_estimate(means[key] - means[ref], boot[key] - boot[ref]),
                time_reduction_percent=_estimate(100 * (means[ref] - means[key]) / means[ref],
                    100 * (boot[ref] - boot[key]) / boot[ref], threshold)))
        effects.append(effect)
    return dict(status='OK', stage=protocol.get('stage', protocol.get('phase', 'same_program_unlocked')),
        case=CASE, protocol_SHA256=sha(encode(protocol)), payload_SHA256=protocol['payload_SHA256'],
        audit=audit, rows=rows, alias_equivalence=equivalence, condition_effects=effects,
        telemetry=_telemetry(raw_records),
        telemetry_associations=_telemetry_associations(raw_records, windows, arrays, audit),
        estimator='Arithmetic mean of worker x visit medians. Primary condition estimates give independent A/B observations equal weight. One shared worker/visit hierarchical bootstrap over all eight cells; no within-visit sample resampling.',
        bootstrap=dict(replicates=reps, master_seed=seed, shared_draws_across_all_cells=True,
            draw_SHA256=sha(wi.tobytes() + vi.tobytes()), confidence_percent=95,
            intervals='Pointwise percentile; no multiplicity correction.'),
        practical_threshold_percent=threshold,
        interpretation='Positive time reduction means shorter kernel event time; it is not a layout gain. All labels execute the same frozen CUBIN, Loaded function, shape, parameters and graph. Equivalence is an observed classification, not an enforced result. All difference intervals use paired replicate differences.',
        limitations=[
            'All measurements are unlocked. Sampled NVML clocks do not establish fixed frequency or per-kernel frequency and cannot isolate a causal DVFS effect.',
            'NVML power/utilization readings have hardware averaging windows that may extend beyond a visit/context label; query timestamps do not make them instantaneous per-kernel measurements.',
            'Between-sample guard, idle and preload interventions may affect several mechanisms including cache state, temperature, frequency and scheduling. Kernel timing events exclude their host/preload duration.',
            'Three workers provide limited evidence about the platform population. Repeated within-worker observations are not independent GPU allocations.',
            'A failed A/B sham equivalence check remains a reported result; no worker or condition is dropped to obtain a favorable outcome.',
            'Raw-sample, visit and worker ranges/quantiles are descriptive variability measures; extreme-value ranges are not confidence intervals.'])


def _cell(value, digits=3):
    lo, hi = value['CI95']
    return f'{value["estimate"]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'


def render(result):
    """Return a self-contained report without modifying any evidence."""
    lines = ['# H100 同程序、同尺寸时间变化：未锁频诊断', '',
        '全部观测执行同一份冻结 CUBIN、同一 Loaded function、相同输入、参数、shape 和单目标 graph。A/B 只是独立随机观测标签。仅改变正式计时事件之外的 guard、空闲或预负载条件；本轮不比较布局收益或跨尺寸差异。', '',
        f'固定 case：`{result["case"]}`。所有 GPU 保持未锁频；本轮不能单独证明 DVFS 的因果作用。', '',
        f'完整样本：{len(result["audit"]["workers"])} workers，{result["audit"]["physical_cells_all_workers"]} cells，{result["audit"]["raw_sample_count"]:,} 个原始计时。',
        '每个 visit 先取 31 个样本的中位数，再对 worker × visit 均值。条件主估计对独立 A/B 观测等权平均。8192 次分层 bootstrap 在全部八个 cell 间共用重采样索引；区间为逐项 95% percentile CI。', '',
        '| Condition | Alias | 时间 µs [95% CI] | Worker 均值范围 µs | Visit 中位数范围 µs |',
        '|---|---|---:|---:|---:|']
    for row in result['rows']:
        workers, visits = row['worker_mean_variation_us'], row['visit_medians_us']
        lines.append(f'| {row["condition"]} | {row["variant"]} | {_cell(row["time_us"])} | [{workers["min"]:.3f}, {workers["max"]:.3f}] | [{visits["min"]:.3f}, {visits["max"]:.3f}] |')
    lines += ['', 'A/B sham：同一程序的两组独立观测。B 相对 A 时间降低为 100×(A−B)/A，正值代表 B 更短；practically_equivalent 要求整个区间落入 ±3%。分类名 practical_improvement/regression 仅表示时间方向，不能解释为代码或布局改进。', '',
        '| Condition | B−A µs [95% CI] | B 相对 A 时间降低 % [95% CI] | ±3% 分类 |',
        '|---|---:|---:|---|']
    for row in result['alias_equivalence']:
        reduction = row['time_reduction_B_vs_A_percent']
        lines.append(f'| {row["condition"]} | {_cell(row["time_difference_us"])} | {_cell(reduction)} | {reduction["classification"]} |')
    lines += ['', '下表比较同一程序在不同样本间条件下的 kernel event 时间。差分为 condition−fast_gap；时间降低为 100×(fast−condition)/fast。两种口径符号相反。', '',
        '| Condition | 时间 µs [95% CI] | condition−fast µs [95% CI] | 相对 fast 时间降低 % [95% CI] |',
        '|---|---:|---:|---:|']
    for row in result['condition_effects']:
        lines.append(f'| {row["condition"]} | {_cell(row["time_us"])} | {_cell(row["time_difference_us"])} | {_cell(row["time_reduction_vs_fast_percent"])} |')
    lines += ['', '全部区间直接来自共用重采样中的配对差分，未相减两个独立区间。以下保留每 worker 的匹配条件差分，不能当作额外独立 GPU。', '',
        '| Condition | Worker | condition−fast µs | 相对 fast 时间降低 % |',
        '|---|---|---:|---:|']
    for row in result['condition_effects']:
        for worker in row['matched_worker_effects']:
            lines.append(f'| {row["condition"]} | {worker["invocation"]} | {worker["difference_us"]:.3f} | {worker["time_reduction_percent"]:.3f} |')
    lines += ['', '每 worker / alias / condition 的原始样本 min/p05/median/p95/max 与 visit 中位数范围保存于 `analysis.json` 的 audit；guard、host gaps、launch/completion timestamp 和 graph node 类型单列，不加入 kernel event 时间。', '',
        '遥测按 worker / phase / case / condition 汇总。NVML 查询返回码、缺失值、采样跨度、SM/graphics/memory clocks、功率、温度、利用率与 clock-event mask 均保留。', '',
        '| Worker | Phase | Case | Condition | Samples | SM MHz median [min,max] |',
        '|---|---|---|---|---:|---:|']
    for group in result['telemetry']['groups']:
        clock = group['metrics']['sm_clock_mhz']['summary']
        text = f'{clock["median"]:.1f} [{clock["min"]:.1f},{clock["max"]:.1f}]' if clock['count'] else 'unavailable'
        lines.append(f'| {group["invocation"]} | {group["phase"]} | {group["case"] or "—"} | {group["condition"] or "—"} | {group["sample_count"]} | {text} |')
    lines += ['', '另在 `telemetry_associations` 按记录的 visit 时间窗匹配 NVML 样本，报告各 worker 内 condition−fast 时间差与遥测差的描述性关联。缺采样窗保留为空，不插值，不把相关性解释为 DVFS 因果证据。NVML telemetry 包含 host gaps，且功率/利用率自身具有硬件平均窗；clock-event mask 次数不是限频持续时间。', '']
    lines.extend('- ' + text for text in result['limitations'])
    return '\n'.join(lines) + '\n'


def _csv_bytes(result):
    stream = io.StringIO(newline='')
    fields = ('case', 'condition', 'variant', 'time_us', 'time_CI95',
        'worker_means_us', 'worker_visit_medians_us')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in result['rows']:
        value = dict(case=row['case'], condition=row['condition'], variant=row['variant'],
            time_us=row['time_us']['estimate'], time_CI95=row['time_us']['CI95'],
            worker_means_us=row['worker_means_us'],
            worker_visit_medians_us=row['worker_visit_medians_us'])
        writer.writerow({key: json.dumps(value[key]) if isinstance(value[key], list) else value[key]
            for key in fields})
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--timing-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.timing_dir.glob('invocation_*/raw.json'))
    _require(bool(paths), 'No invocation_*/raw.json under timing directory')
    protocol = json.loads(args.protocol.read_text())
    result = analyze(protocol, [json.loads(path.read_text()) for path in paths])
    result['inputs'] = dict(protocol=dict(path=str(args.protocol.resolve()),
        SHA256=sha(args.protocol.read_bytes())), raw=[dict(path=str(path.resolve()),
        SHA256=sha(path.read_bytes())) for path in paths],
        analysis_source=dict(path=str(Path(__file__).resolve()), SHA256=sha(Path(__file__).read_bytes())))
    outputs = {'analysis.json': encode(result), 'timing.csv': _csv_bytes(result),
        'REPORT.md': render(result).encode()}
    _require(not any((args.output / name).exists() for name in outputs),
        'Refusing to overwrite an existing analysis artifact')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, blob in outputs.items():
        with (args.output / name).open('xb') as handle:
            handle.write(blob)
    print(json.dumps(dict(status=result['status'], raw_samples=result['audit']['raw_sample_count'],
        output=str(args.output)), sort_keys=True))


if __name__ == '__main__':
    main()
