"""CPU-only validation of the frozen same-program repeatability experiment.

Validation checks provenance, execution identity and complete observations. It
does not require A/B equivalence, a timing improvement, or a DVFS association.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


REPO = Path(__file__).resolve().parents[3]
CASE = 'N02_k1024_p128_g128'
CUBIN = 'cd5dad8709d09c326f4f9e703fe28f76586f5589ddf8ab1a29e02c15debedd6d'
CONDITIONS = ('fast_gap', 'legacy_gap', 'idle_20ms', 'gpu_preload')
VARIANTS = ('A', 'B')
FIELDS = ('graphics_clock_mhz', 'sm_clock_mhz', 'memory_clock_mhz',
          'power_usage_mw', 'power_limit_mw', 'temperature_c',
          'gpu_utilization_percent', 'memory_utilization_percent',
          'pstate', 'clock_event_reasons')
MANDATORY_CLOCKS = FIELDS[:3]


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_relative(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe relative path: ' + name)
    return Path(name)


def preflight(root):
    """Check frozen CPU artifacts and old-file protection without GPU access."""
    root = Path(root).resolve()
    protocol_path = root / 'preregistration/protocol.json'
    protocol = read(protocol_path)
    require(protocol['stage'] == 'same_program_repeatability', 'Wrong experiment')
    require(protocol['compiler'] == '20d9d98fead78c2f65e1b3ba5a7881d86a836dad', 'Compiler changed')
    require(protocol['clock_control'] == 'unlocked', 'Unexpected clock-control mode')
    require(protocol['GPU_configuration_writes'] is False, 'GPU settings mutation authorized in protocol')
    require(protocol['compiler_calls'] == protocol['native_rebuilds'] == 0, 'Compilation in protocol')
    require(protocol['GPU'] == 'H100!' and protocol['profile'] == 'chenmiaoming', 'GPU/profile changed')
    require(len(protocol['cases']) == len(protocol['programs']) == 1, 'Single case/program required')
    meta, case = protocol['programs'][0], protocol['cases'][0]
    require(case['id'] == CASE and meta['case'] == case and meta['geometry'] == CASE, 'Case mismatch')
    require((case['rows'], case['cols'], case['packed_cols'], case['group_size']) == (1024, 1024, 128, 128), 'Shape changed')
    require(meta['variant'] == 'U' and meta['cubin_SHA256'] == CUBIN, 'Program changed')
    require(meta['fixture']['grid'] == [4, 32, 1] and meta['fixture']['inplace'] == [], 'Fixture changed')
    require(protocol['conditions'] == list(CONDITIONS) and protocol['variants'] == list(VARIANTS), 'Matrix changed')
    timing = protocol['timing']
    require((timing['workers'], timing['visits'], timing['samples']) == (3, 12, 31), 'Sampling matrix changed')
    require(timing['bootstrap_replicates'] == 8192 and timing['practical_threshold_percent'] == 3, 'Analysis protocol changed')
    require(timing['preload_repeats'] == 128 and timing['idle_seconds'] == .02, 'Intervention changed')
    require(timing['telemetry_period_s'] == .01, 'Telemetry period changed')
    execution = protocol['execution']
    for field in ('single_case', 'single_loaded_function', 'single_fixture_during_timing',
                  'single_target_graph', 'A_B_same_function_pointers_params_grid_graph',
                  'A_B_independent_observations'):
        require(execution[field] is True, 'Missing identity contract: ' + field)
    require(execution['Triton_imported'] is False and execution['compile_calls'] == 0, 'Runtime compiler import contract changed')
    require(sorted(execution['target_graph_node_types']) == [0, 7, 7], 'Target graph changed')
    require(protocol['correctness']['patterns'] == ['finite', 'structured', 'edge'], 'Correctness patterns changed')
    require(protocol['correctness']['post_timing_pattern'] == 'finite', 'Post-check changed')

    snapshot = root / protocol['snapshot']
    for name, digest in protocol['source_SHA256'].items():
        relative = safe_relative(name)
        require(file_sha(snapshot / relative) == digest, 'Snapshot mismatch: ' + name)
        require(file_sha(REPO / relative) == digest, 'Current source mismatch: ' + name)
    # Archived probe_run.py intentionally contains the old authorized clock test.
    # Audit only the new executed entrypoints, never reject preserved history.
    for name in ('same_program_worker.py', 'same_program_run.py', 'telemetry.py'):
        text = (snapshot / 'experiments/layout_coordination/clock_control' / name).read_text()
        for token in ('nvmlDeviceSet', 'nvmlDeviceReset', '--lock-gpu-clocks',
                      '--reset-gpu-clocks', '--lock-memory-clocks', 'nvidia-smi'):
            require(token not in text, f'GPU configuration command in executed source {name}: {token}')

    protected_path = root / safe_relative(protocol['protected_inventory_path'])
    require(file_sha(protected_path) == protocol['protected_inventory_SHA256'], 'Protected inventory SHA mismatch')
    protected = read(protected_path)
    require(len(protected) == protocol['protected_file_count'] > 0, 'Protected inventory count mismatch')
    for name, digest in protected.items():
        require(file_sha(REPO / safe_relative(name)) == digest, 'Old protected file changed: ' + name)
    historical_path = root / safe_relative(protocol['historical_audit_path'])
    require(file_sha(historical_path) == protocol['historical_audit_SHA256'], 'Historical audit changed')
    historical = read(historical_path)
    require(historical['status'] == 'OK' and historical['target_case'] == CASE, 'Historical audit failed')
    for item in historical['primary_same_worker']:
        for field in ('raw', 'protocol'):
            reference = item[field]
            require(file_sha(REPO / safe_relative(reference['path'])) == reference['SHA256'], 'Historical raw/protocol changed')
    reference = protocol['reference_driver_binding']
    reference_path = REPO / safe_relative(reference['source']['path'])
    require(file_sha(reference_path) == reference['source']['SHA256'], 'Reference CUDA-driver binding source changed')
    require(read(reference_path)['bindings'][meta['tag']] == reference['binding'], 'Reference CUDA-driver binding mismatch')

    payload_path = root / 'payload.zip'
    require(file_sha(payload_path) == protocol['payload_SHA256'], 'Payload SHA mismatch')
    with zipfile.ZipFile(payload_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate payload archive member')
        for name in names:
            safe_relative(name)
        require(read_archive_json(archive, 'result.json')['programs'] == [meta], 'Payload metadata mismatch')
        prefix = meta['artifact_directory'] + '/'
        for name, digest in meta['artifact_SHA256'].items():
            require(sha(archive.read(prefix + name)) == digest, 'Payload artifact mismatch: ' + name)
        require(sha(archive.read(prefix + 'kernel.cubin')) == CUBIN, 'Payload CUBIN mismatch')
    freeze = read(root / 'preregistration/freeze.json')
    require(freeze['status'] == 'OK' and freeze['CPU_only'] is True and freeze['GPU_dispatches'] == 0, 'Invalid freeze record')
    require(freeze['protocol_SHA256'] == sha(encode(protocol)), 'Frozen protocol SHA mismatch')
    require(freeze['payload_SHA256'] == protocol['payload_SHA256'], 'Frozen payload SHA mismatch')
    require(freeze['protected_inventory_SHA256'] == protocol['protected_inventory_SHA256'], 'Frozen inventory SHA mismatch')
    return protocol, dict(status='OK', protocol_SHA256=sha(encode(protocol)),
        payload_SHA256=protocol['payload_SHA256'], source_files=len(protocol['source_SHA256']),
        protected_files=len(protected), single_case=CASE, single_CUBIN_SHA256=CUBIN,
        clock_control='unlocked', GPU_configuration_writes=False)


def read_archive_json(archive, name):
    return json.loads(archive.read(name))


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def telemetry_audit(raw):
    rows = raw['telemetry']
    require(bool(rows), 'No telemetry')
    previous = -1
    counts = {field: Counter() for field in FIELDS}
    good = Counter()
    contexts = Counter()
    for index, row in enumerate(rows):
        require(row['sequence'] == index, 'Telemetry sequence mismatch')
        require(row['monotonic_ns'] > previous, 'Nonmonotonic telemetry')
        previous = row['monotonic_ns']
        require(row['sample_end_monotonic_ns'] >= previous, 'Invalid telemetry sample interval')
        require(row['query_duration_ns'] == row['sample_end_monotonic_ns'] - previous, 'Telemetry duration mismatch')
        require('sampler_error' not in row, 'Sampler Python-side failure')
        context = row['context']
        contexts[str((context.get('phase'), context.get('condition')))] += 1
        for field in FIELDS:
            reading = row[field]
            require(isinstance(reading, dict) and 'rc' in reading and 'value' in reading, 'Telemetry rc/value missing: ' + field)
            rc, value = reading['rc'], reading['value']
            require(rc is None or isinstance(rc, int), 'Noninteger NVML rc')
            counts[field][str(rc)] += 1
            if rc == 0:
                require(numeric(value), 'Nonfinite successful NVML value: ' + field)
                good[field] += 1
                if field in MANDATORY_CLOCKS:
                    require(value > 0, 'Invalid clock reading')
                if field == 'clock_event_reasons':
                    require(isinstance(value, int) and 0 <= value < (1 << 64), 'Invalid uint64 event mask')
            else:
                require(value is None and bool(reading.get('error')), 'Unavailable NVML reading fabricated: ' + field)
    for field in MANDATORY_CLOCKS:
        require(good[field] > 0, 'No successful mandatory clock samples: ' + field)
    return dict(rows=len(rows), successful_counts=dict(good),
        return_code_counts={key: dict(value) for key, value in counts.items()},
        context_counts=dict(contexts),
        note='Power/utilization have hardware averaging windows. Missing readings are reported, not replaced or interpolated.')


def validate_worker(raw, root, protocol, invocation):
    require(raw['status'] == 'OK' and raw['invocation'] == invocation, 'Failed/wrong worker')
    require(raw['protocol_SHA256'] == sha(encode(protocol)), 'Worker protocol mismatch')
    require(raw['payload_SHA256'] == protocol['payload_SHA256'], 'Worker payload mismatch')
    require(raw['clock_control'] == 'unlocked' and raw['profiler_attached'] is False, 'Clock/profiler mode changed')
    require(raw['compile_calls'] == 0 and raw['Triton_imported'] is False, 'GPU worker imported compiler')
    require(raw['driver_return_codes'] == [0] and not raw.get('cleanup_errors'), 'CUDA/cleanup failure')
    environment = raw['environment']
    require(environment['SM_count'] == 132 and environment['cc'] == [9, 0], 'Wrong GPU architecture')
    require('H100' in environment['name'] and 'H200' not in environment['name'], 'Wrong GPU product')
    readonly = environment['readonly']
    require(readonly['read_only'] is True and readonly['cap_sys_admin_bit'] == 21, 'Missing read-only environment')
    require(readonly['nvml']['requested_uuid'] == environment['gpu_uuid'], 'NVML/device UUID mismatch')
    meta = protocol['programs'][0]
    identity = raw['identity']
    require(identity['case'] == CASE and identity['cubin_SHA256'] == CUBIN, 'Runtime identity changed')
    for field in ('same_loaded_function_for_all_observations', 'same_target_graph_for_all_observations'):
        require(identity[field] is True, 'Runtime identity proof missing: ' + field)
    require(identity['independent_observation_labels'] == list(VARIANTS), 'A/B observation identity changed')
    require(identity['grid'] == meta['fixture']['grid'] and raw['fixture'] == meta['fixture'], 'Timing fixture changed')
    for field in ('function_handle', 'target_graph_handle', 'context_handle', 'stream_handle', 'parameter_vector_address'):
        require(isinstance(identity[field], int) and identity[field] > 0, 'Invalid identity handle: ' + field)
    require(sorted(identity['target_graph_node_types']) == [0, 7, 7], 'Target graph node mismatch')
    require(identity['preload_repeats'] == 128 and identity['preload_graph_node_types'] == [0] * 128, 'Preload graph changed')
    # DriverFixture owns one allocation per storage group and one ABI scratch
    # allocation, even when the archived kernel has zero global scratch bytes.
    require(identity['allocation_count'] == len(meta['fixture']['initial_storage_SHA256']) + 1, 'Unexpected timing allocations')
    require(identity['arguments_unchanged_after_timing'] is True, 'Runtime arguments changed during timing')
    values = identity['runtime_argument_values']
    # The original dequantization ABI passes packed columns (128), while the
    # logical output has 1024 columns. They must not be interchanged here.
    require(len(values) == 9 and values[3] == 128 and values[5:7] == [128, 1024], 'Runtime scalar/shape changed')
    require(all(isinstance(values[i], int) and values[i] > 0 for i in (0, 1, 2, 4)), 'Invalid timing pointers')
    binding = raw['bindings']['same_program']
    require(len(raw['bindings']) == 1 and binding['tag'] == meta['tag'], 'Multiple/wrong function bindings')
    require(binding['cubin_SHA256'] == binding['loaded_SHA256'] == CUBIN, 'Loaded bytes changed')
    require(binding['attributes']['n_regs'] == meta['resources']['REG'], 'Register allocation changed')
    require(binding['attributes']['local_bytes'] == meta['resources']['LOCAL'] == 0, 'Local allocation changed')
    reference = protocol['reference_driver_binding']['binding']
    require(binding['attributes'] == reference['attributes'], 'CUDA driver resource readbacks changed')
    require(binding['ABI'] == reference['ABI'], 'CUDA driver ABI changed')
    require(binding['attributes']['occupancy_capacity_q'] > 0, 'Invalid occupancy capacity')
    checks = {(item['phase'], item['pattern']): item for item in raw['checks']}
    require(len(checks) == len(raw['checks']) == 4, 'Expected exactly four correctness checks')
    require(set(checks) == {('before', p) for p in ('finite', 'structured', 'edge')} | {('after', 'finite')}, 'Correctness coverage changed')
    for check in checks.values():
        require(check['pass'] is True and all(check['guards']), 'Correctness/guard failed')
        require(bool(check['details']) and all(item.get('pass_', True) for item in check['details']), 'Correctness detail failed')
    export = root / 'timing' / f'invocation_{invocation:03}' / 'export'
    for name, digest in meta['artifact_SHA256'].items():
        require(file_sha(export / 'program_files' / meta['artifact_directory'] / name) == digest, 'Exported compiler artifact changed: ' + name)
    expected = {(condition, variant) for condition in CONDITIONS for variant in VARIANTS}
    cells = {(cell['condition'], cell['variant']): cell for cell in raw['cells']}
    require(set(cells) == expected and len(raw['cells']) == 8, 'Incomplete/duplicate eight-cell matrix')
    total = 0
    for (condition, variant), cell in cells.items():
        require(cell['case'] == CASE and len(cell['visits']) == 12, 'Wrong case/visit count')
        require(sorted(v['visit'] for v in cell['visits']) == list(range(12)), 'Missing/duplicate visits')
        for visit in cell['visits']:
            require(sorted(visit['node_types']) == [0, 7, 7], 'Per-visit graph changed')
            for field in ('samples_us', 'host_gaps_ns', 'guard_duration_ns', 'launch_ns', 'completion_ns'):
                require(len(visit[field]) == 31 and all(numeric(x) and x >= 0 for x in visit[field]), 'Invalid sample/vector: ' + field)
            require(all(x > 0 for x in visit['samples_us']), 'Nonpositive timing')
            require(all(visit['start_ns'] <= launch <= complete <= visit['end_ns']
                for launch, complete in zip(visit['launch_ns'], visit['completion_ns'])), 'Launch/completion outside visit')
            require(all(a < b for a, b in zip(visit['launch_ns'], visit['launch_ns'][1:])), 'Repeated/nonmonotonic launches')
            if condition == 'idle_20ms':
                require(min(visit['host_gaps_ns']) >= 19_000_000, 'Idle intervention missing')
            total += 31
    orders = {(row['visit'], row['condition']): row for row in raw['orders']}
    require(len(raw['orders']) == len(orders) == 48, 'Condition-block order coverage incomplete')
    for visit in range(12):
        sequence = orders[visit, CONDITIONS[0]]['condition_order']
        require(sorted(sequence) == sorted(CONDITIONS), 'Invalid condition order')
        for condition in CONDITIONS:
            row = orders[visit, condition]
            require(row['condition_order'] == sequence, 'Condition order inconsistent')
            require(len(row['round_orders']) == 31 and all(sorted(x) == list(VARIANTS) for x in row['round_orders']), 'A/B independent observation order incomplete')
    return dict(invocation=invocation, gpu_uuid=environment['gpu_uuid'], cells=8,
        samples=total, correctness_checks=4, binding=binding, telemetry=telemetry_audit(raw))


def validate(root, *, preregistration_only=False):
    root = Path(root).resolve()
    protocol, registration = preflight(root)
    if preregistration_only:
        return dict(status='OK', scope='CPU preregistration only; GPU results not checked', preregistration=registration)
    paths = sorted((root / 'timing').glob('invocation_*/raw.json'))
    require(len(paths) == 3, 'Expected exactly three completed raw workers')
    raws, workers = [], []
    for invocation, path in enumerate(paths):
        require(path.parent.name == f'invocation_{invocation:03}', 'Unexpected invocation path')
        raw = read(path)
        require(encode(raw) == encode(read(path.parent / 'export/result.json')), 'Raw/export JSON mismatch')
        with zipfile.ZipFile(path.parent / 'raw_export.zip') as archive:
            require(encode(raw) == encode(read_archive_json(archive, 'result.json')), 'Archive/raw result mismatch')
        dispatch = read(path.parent / 'dispatch.json')
        require(dispatch['protocol_SHA256'] == sha(encode(protocol)) and dispatch['payload_SHA256'] == protocol['payload_SHA256'], 'Dispatch SHA mismatch')
        require(dispatch['profile'] == 'chenmiaoming' and bool(dispatch['call_id']), 'Dispatch account/call missing')
        workers.append(validate_worker(raw, root, protocol, invocation))
        raws.append(raw)
    require(len({worker['gpu_uuid'] for worker in workers}) == 3, 'GPU UUIDs are not unique')
    require(sum(worker['samples'] for worker in workers) == 8928, 'Total sample coverage mismatch')
    # Recompute all derived outputs with the frozen analysis code. No result
    # direction or sham-equivalence classification enters a validation gate.
    source = Path(__file__).with_name('analysis.py')
    spec = importlib.util.spec_from_file_location('same_program_frozen_analysis', source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analysis = analyzer.analyze(protocol, raws)
    protocol_path = root / 'preregistration/protocol.json'
    analysis['inputs'] = dict(protocol=dict(path=str(protocol_path.resolve()), SHA256=file_sha(protocol_path)),
        raw=[dict(path=str(path.resolve()), SHA256=file_sha(path)) for path in paths],
        analysis_source=dict(path=str(source.resolve()), SHA256=file_sha(source)))
    outputs = {'analysis.json': analyzer.encode(analysis),
        'timing.csv': analyzer._csv_bytes(analysis), 'REPORT.md': analyzer.render(analysis).encode()}
    for name, expected in outputs.items():
        require((root / 'analysis' / name).read_bytes() == expected, 'Derived output does not reproduce: ' + name)
    return dict(status='OK', scope='CPU provenance, identity, complete observations, telemetry and derived-output validation',
        preregistration=registration, workers=workers, raw_sample_count=8928,
        analysis_recomputed=True, derived_SHA256={name: sha(blob) for name, blob in outputs.items()},
        outcome_requirement='None: adverse/inconclusive timing, clock variation and failed sham equivalence remain valid reported outcomes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.root, preregistration_only=args.preflight)
    output = args.output if args.output is not None else (None if args.preflight else args.root / 'validation.json')
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as handle:
            handle.write(encode(result))
    print(json.dumps(dict(status=result['status'], preflight=args.preflight,
        samples=result.get('raw_sample_count'), output=str(output) if output else None), sort_keys=True))


if __name__ == '__main__':
    main()
