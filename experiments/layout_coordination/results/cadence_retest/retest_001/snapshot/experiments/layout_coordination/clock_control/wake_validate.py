"""CPU validation of the preregistered adaptive same-program wake experiment.

All gates concern provenance, identity, correctness and observation coverage.
No timing direction, recovery magnitude or confidence-interval result is a gate.
"""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location('wake_validation_helpers', HERE / 'same_program_validate.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
require, read, encode, sha = helpers.require, helpers.read, helpers.encode, helpers.sha
file_sha, safe_relative, numeric = helpers.file_sha, helpers.safe_relative, helpers.numeric
CASE = helpers.CASE
CONDITIONS = ('fast_gap', 'idle_20ms', 'idle_then_same_prime', 'idle_then_disjoint_prime')
ROLES = ('target', 'primer')
VARIANTS = ('A', 'B')


def preflight(root):
    root = Path(root).resolve()
    protocol = read(root / 'preregistration/protocol.json')
    require(protocol['stage'] == 'same_program_wake_followup' and protocol['adaptive'] is True, 'Wrong experiment')
    require(protocol['compiler'] == '20d9d98fead78c2f65e1b3ba5a7881d86a836dad', 'Compiler changed')
    require(protocol['compiler_calls'] == protocol['native_rebuilds'] == 0, 'Compiler/rebuild requested')
    require(protocol['clock_control'] == 'unlocked' and protocol['GPU_configuration_writes'] is False, 'GPU settings changed')
    require(protocol['GPU'] == 'H100!' and protocol['profile'] == 'chenmiaoming', 'GPU/profile changed')
    require(protocol['program_roles'] == list(ROLES), 'Wrong roles')
    require(protocol['conditions'] == list(CONDITIONS) and protocol['variants'] == list(VARIANTS), 'Wrong matrix')
    require(len(protocol['cases']) == 1 and len(protocol['programs']) == 2, 'Wrong case/program count')
    case, programs = protocol['cases'][0], protocol['programs']
    require(case['id'] == CASE and case['rows'] == case['cols'] == 1024, 'Wrong target shape')
    require(case['packed_cols'] == case['group_size'] == 128, 'Wrong packed shape')
    require(programs[0]['variant'] == 'U' and programs[0]['cubin_SHA256'] == helpers.CUBIN, 'Target changed')
    require(programs[1]['variant'] == 'GL_packed_compute_vec1', 'Wrong disjoint primer')
    require(programs[1]['cubin_SHA256'] != programs[0]['cubin_SHA256'], 'Primer binary is not distinct')
    require(all(meta['case'] == case and meta['geometry'] == CASE for meta in programs), 'Primer/target case changed')
    fixture = protocol['expected_fixture']
    require(fixture == programs[0]['fixture'] and fixture['grid'] == [4, 32, 1] and fixture['inplace'] == [], 'Fixture changed')
    timing = protocol['timing']
    require((timing['workers'], timing['visits'], timing['samples']) == (3, 12, 31), 'Sampling count changed')
    require(timing['warmups'] == 8 and timing['bootstrap_replicates'] == 8192, 'Warmups/bootstrap changed')
    require(timing['practical_threshold_percent'] == 3 and timing['idle_seconds'] == .02, 'Threshold/idle changed')
    require(timing['telemetry_period_s'] == .01 and timing['prewarm_seconds'] == 2, 'Telemetry/prewarm changed')
    require(protocol['correctness']['patterns'] == ['finite', 'structured', 'edge'] and protocol['correctness']['post_timing_pattern'] == 'finite', 'Correctness coverage changed')
    require(protocol['execution']['target_unchanged'] is True, 'Target identity contract missing')
    require(sorted(protocol['execution']['target_graph_node_types']) == [0, 7, 7], 'Target graph changed')

    snapshot = root / protocol['snapshot']
    for name, digest in protocol['source_SHA256'].items():
        relative = safe_relative(name)
        require(file_sha(REPO / relative) == digest, 'Current source changed: ' + name)
        require(file_sha(snapshot / relative) == digest, 'Snapshot source changed: ' + name)
    for name in ('wake_worker.py', 'wake_run.py', 'telemetry.py'):
        source = (snapshot / 'experiments/layout_coordination/clock_control' / name).read_text()
        for token in ('nvmlDeviceSet', 'nvmlDeviceReset', '--lock-gpu-clocks', '--reset-gpu-clocks', 'nvidia-smi'):
            require(token not in source, f'GPU configuration command in executed {name}: {token}')
    inventory_path = root / safe_relative(protocol['protected_inventory_path'])
    require(file_sha(inventory_path) == protocol['protected_inventory_SHA256'], 'Protected inventory changed')
    protected = read(inventory_path)
    require(len(protected) == protocol['protected_file_count'] > 0, 'Protected count mismatch')
    for name, digest in protected.items():
        require(file_sha(REPO / safe_relative(name)) == digest, 'Previous evidence/source changed: ' + name)

    previous = protocol['motivating_run']
    previous_protocol_path = REPO / safe_relative(previous['path'])
    require(file_sha(previous_protocol_path) == previous['SHA256'], 'Motivating protocol changed')
    previous_protocol = read(previous_protocol_path)
    require(previous_protocol['stage'] == 'same_program_repeatability', 'Wrong motivating experiment')
    previous_root = previous_protocol_path.parents[1]
    require(read(previous_root / 'validation_final.json')['status'] == 'OK', 'Previous final validation absent')
    inherited_path = previous_root / previous_protocol['protected_inventory_path']
    require(file_sha(inherited_path) == previous_protocol['protected_inventory_SHA256'], 'Inherited protected inventory changed')
    inherited = read(inherited_path)
    require(all(protected.get(name) == digest for name, digest in inherited.items()), 'Inherited protection weakened')
    require(all(protected.get(name) == digest for name, digest in previous_protocol['source_SHA256'].items()), 'First experiment sources not protected')
    previous_files = [path for path in previous_root.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    require(bool(previous_files), 'Previous experiment missing')
    for path in previous_files:
        name = str(path.relative_to(REPO))
        require(protected.get(name) == file_sha(path), 'First experiment file not protected: ' + name)
    references = protocol['reference_driver_bindings']
    require(set(references) == set(ROLES), 'Driver reference role missing')
    for role, meta in zip(ROLES, programs):
        reference = references[role]
        path = REPO / safe_relative(reference['source']['path'])
        require(file_sha(path) == reference['source']['SHA256'], 'Reference driver source changed')
        require(read(path)['bindings'][meta['tag']] == reference['binding'], 'Reference driver binding changed')
        require(reference['binding']['cubin_SHA256'] == reference['binding']['loaded_SHA256'] == meta['cubin_SHA256'], 'Reference binary changed')
    payload_path = root / 'payload.zip'
    require(file_sha(payload_path) == protocol['payload_SHA256'], 'Payload changed')
    with zipfile.ZipFile(payload_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate payload member')
        for name in names:
            safe_relative(name)
        for meta in programs:
            for name, digest in meta['artifact_SHA256'].items():
                require(sha(archive.read(meta['artifact_directory'] + '/' + name)) == digest, 'Payload artifact changed: ' + name)
    freeze = read(root / 'preregistration/freeze.json')
    require(freeze['status'] == 'OK' and freeze['CPU_only'] is True and freeze['GPU_dispatches'] == 0, 'Invalid CPU freeze')
    require(freeze['protocol_SHA256'] == sha(encode(protocol)) and freeze['payload_SHA256'] == protocol['payload_SHA256'], 'Freeze SHA mismatch')
    return protocol, dict(status='OK', protocol_SHA256=sha(encode(protocol)),
        source_files=len(protocol['source_SHA256']), protected_files=len(protected),
        inherited_protected_files=len(inherited), first_experiment_protected_files=len(previous_files),
        payload_SHA256=protocol['payload_SHA256'], clock_control='unlocked')


def _ranges(items):
    """Normalize exported allocation intervals after checking their evidence."""
    result = []
    for item in items:
        require(isinstance(item, dict), 'Storage range must be a dictionary')
        start, size = item['start'], item['bytes']
        require(isinstance(size, int) and size > 0, 'Invalid storage size')
        end = start + size
        require(isinstance(start, int) and isinstance(end, int) and 0 < start < end, 'Invalid storage interval')
        result.append((start, end))
    return result


def validate_worker(raw, root, protocol, invocation):
    require(raw['status'] == 'OK' and raw['invocation'] == invocation, 'Failed/wrong worker')
    require(raw['protocol_SHA256'] == sha(encode(protocol)) and raw['payload_SHA256'] == protocol['payload_SHA256'], 'Worker SHA mismatch')
    require(raw['clock_control'] == 'unlocked' and raw['profiler_attached'] is False, 'Wrong timing mode')
    require(raw['compile_calls'] == 0 and raw['Triton_imported'] is False, 'Compiler execution/import')
    require(raw['driver_return_codes'] == [0] and not raw.get('cleanup_errors'), 'CUDA/cleanup failure')
    env = raw['environment']
    require(env['SM_count'] == 132 and env['cc'] == [9, 0] and 'H100' in env['name'] and 'H200' not in env['name'], 'Wrong GPU')
    require(env['readonly']['read_only'] is True and env['readonly']['nvml']['requested_uuid'] == env['gpu_uuid'], 'Read-only NVML/device binding missing')
    identity = raw['identity']
    require(identity['case'] == CASE and identity['storage_disjoint'] is True, 'Case/disjoint storage proof missing')
    require(identity['same_loaded_target_function_for_all_observations'] is True and identity['same_target_graph_for_all_observations'] is True, 'Target identity changed')
    require(identity['independent_observation_labels'] == list(VARIANTS), 'A/B labels changed')
    require(identity['allocation_count'] == 10, 'Wrong allocation count')
    for key in ('context_handle', 'stream_handle'):
        require(isinstance(identity[key], int) and identity[key] > 0, 'Invalid CUDA identity handle')
    require(set(raw['bindings']) == set(raw['fixture']) == set(ROLES), 'Wrong role bindings/fixtures')
    normalized_ranges = {}
    for role, meta in zip(ROLES, protocol['programs']):
        item, binding = identity[role], raw['bindings'][role]
        require(identity[role + '_CUBIN_SHA256'] == meta['cubin_SHA256'], 'Role CUBIN changed')
        require(raw['fixture'][role] == protocol['expected_fixture'] and item['grid'] == protocol['expected_fixture']['grid'], 'Role fixture/grid changed')
        require(identity['arguments_unchanged_after_timing'][role] is True, 'Runtime arguments changed')
        for key in ('function_handle', 'module_handle', 'parameter_vector_address', 'graph_handle'):
            require(isinstance(item[key], int) and item[key] > 0, 'Invalid role handle: ' + key)
        require(sorted(item['graph_node_types']) == [0, 7, 7], 'Role graph changed')
        values = item['runtime_argument_values']
        require(len(values) == 9 and values[3] == 128 and values[5:7] == [128, 1024], 'Packed runtime scalar changed')
        require(all(isinstance(values[index], int) and values[index] > 0 for index in (0, 1, 2, 4, 7)), 'Invalid role pointers')
        require(values[7] == item['scratch'], 'Scratch pointer changed')
        require(binding['tag'] == meta['tag'] and binding['cubin_SHA256'] == binding['loaded_SHA256'] == meta['cubin_SHA256'], 'Loaded role binary changed')
        reference = protocol['reference_driver_bindings'][role]['binding']
        require(binding['attributes'] == reference['attributes'] and binding['ABI'] == reference['ABI'], 'Driver resource/ABI changed')
        normalized_ranges[role] = _ranges(item['storage_ranges'])
        require(len(normalized_ranges[role]) == 5, 'Expected four storages plus ABI scratch')
        logical = item['logical_pointers']
        expected_logical = {name: values[index] for index, name in enumerate(meta['runtime_argument_names'])
                            if name in protocol['expected_fixture']['views']}
        require(logical == expected_logical, 'Logical pointer/ABI mismatch')
        require(all(any(start <= pointer < end for start, end in normalized_ranges[role])
                    for pointer in [*logical.values(), item['scratch']]), 'Pointer outside recorded allocations')
        for name, digest in meta['artifact_SHA256'].items():
            path = root / 'timing' / f'invocation_{invocation:03}' / 'export/program_files' / meta['artifact_directory'] / name
            require(file_sha(path) == digest, 'Exported role artifact changed')
    for field in ('function_handle', 'module_handle', 'parameter_vector_address', 'graph_handle'):
        require(identity['target'][field] != identity['primer'][field], 'Primer is not independently loaded/allocated: ' + field)
    require(all(a1 <= b0 or b1 <= a0 for a0, a1 in normalized_ranges['target'] for b0, b1 in normalized_ranges['primer']), 'Primer/target storage overlap')
    checks = {(item['role'], item['phase'], item['pattern']): item for item in raw['checks']}
    expected_checks = {(role, 'before', pattern) for role in ROLES for pattern in ('finite', 'structured', 'edge')} | {(role, 'after', 'finite') for role in ROLES}
    require(set(checks) == expected_checks and len(raw['checks']) == 8, 'Eight-check coverage incomplete')
    for check in checks.values():
        require(check['pass'] is True and bool(check['guards']) and all(check['guards']), 'Correctness/guard failed')
        require(bool(check['details']) and all(item.get('pass_', True) for item in check['details']), 'Correctness detail failed')
    cells = {(cell['condition'], cell['variant']): cell for cell in raw['cells']}
    require(len(raw['cells']) == len(cells) == 8 and set(cells) == {(c, v) for c in CONDITIONS for v in VARIANTS}, 'Eight-cell matrix incomplete')
    sample_count = 0
    for (condition, variant), cell in cells.items():
        require(cell['case'] == CASE and sorted(v['visit'] for v in cell['visits']) == list(range(12)), 'Visit coverage changed')
        for visit in cell['visits']:
            require(sorted(visit['node_types']) == [0, 7, 7], 'Per-visit target graph changed')
            for field in ('samples_us', 'host_gaps_ns', 'sleep_duration_ns', 'prime_duration_ns', 'guard_duration_ns', 'launch_ns', 'completion_ns'):
                require(len(visit[field]) == 31 and all(numeric(value) and value >= 0 for value in visit[field]), 'Invalid observation vector: ' + field)
            require(all(value > 0 for value in visit['samples_us']), 'Nonpositive event timing')
            require(all(value == 0 for value in visit['guard_duration_ns']), 'Per-sample guard unexpectedly enabled')
            require(all(visit['start_ns'] <= launch <= complete <= visit['end_ns'] for launch, complete in zip(visit['launch_ns'], visit['completion_ns'])), 'Observation outside visit')
            require(all(a < b for a, b in zip(visit['launch_ns'], visit['launch_ns'][1:])), 'Nonmonotonic target launches')
            if condition != 'fast_gap':
                require(min(visit['sleep_duration_ns']) >= 19_000_000, 'Missing 20-ms idle')
            if condition in ('idle_then_same_prime', 'idle_then_disjoint_prime'):
                require(min(visit['prime_duration_ns']) > 0, 'Missing one-kernel primer')
            # Even empty branches retain the two timestamp calls' elapsed
            # overhead; zero is not required for an arm without sleep/prime.
            sample_count += 31
    orders = {(row['visit'], row['condition']): row for row in raw['orders']}
    require(len(raw['orders']) == len(orders) == 48, 'Condition order coverage incomplete')
    for visit in range(12):
        order = orders[visit, CONDITIONS[0]]['condition_order']
        require(sorted(order) == sorted(CONDITIONS), 'Invalid condition order')
        for condition in CONDITIONS:
            item = orders[visit, condition]
            require(item['condition_order'] == order and len(item['round_orders']) == 31, 'Order mismatch')
            require(all(sorted(row) == list(VARIANTS) for row in item['round_orders']), 'A/B round coverage changed')
    prewarm = identity['prewarm_graph_launches']
    require(all(isinstance(prewarm[role], int) and prewarm[role] > 0 for role in ROLES), 'Initial balanced prewarm missing')
    require(prewarm['target'] == prewarm['primer'], 'Initial prewarm is not balanced')
    require(raw['bindings']['target']['graph_launches'] == prewarm['target'] + 4200, 'Target graph execution count changed')
    require(raw['bindings']['primer']['graph_launches'] == prewarm['primer'] + 840, 'Disjoint primer execution count changed')
    require(all(raw['bindings'][role]['direct_launches'] == 5 for role in ROLES), 'Direct check/capture launch count changed')
    activity = identity['observation_activity_by_condition']
    require(set(activity) == set(CONDITIONS), 'Per-condition execution accounting missing')
    for condition in CONDITIONS:
        require(activity[condition] == dict(observations=840,
            target_graph_launches=1680 if condition == 'idle_then_same_prime' else 840,
            primer_graph_launches=840 if condition == 'idle_then_disjoint_prime' else 0,
            prime_graph_launches=840 if condition in CONDITIONS[2:] else 0), 'Per-condition primer/target execution count changed')
    return dict(invocation=invocation, gpu_uuid=env['gpu_uuid'], raw_samples=sample_count,
        correctness_checks=8, telemetry=helpers.telemetry_audit(raw),
        role_bindings=raw['bindings'], target_identity_preserved=True, storage_disjoint=True)


def validate(root, *, preregistration_only=False):
    root = Path(root).resolve()
    protocol, registration = preflight(root)
    if preregistration_only:
        return dict(status='OK', scope='CPU preregistration only', preregistration=registration)
    paths = sorted((root / 'timing').glob('invocation_*/raw.json'))
    require(len(paths) == 3, 'Expected three raw workers')
    workers, raws = [], []
    for invocation, path in enumerate(paths):
        require(path.parent.name == f'invocation_{invocation:03}', 'Wrong invocation path')
        raw = read(path)
        require(encode(raw) == encode(read(path.parent / 'export/result.json')), 'Raw/export mismatch')
        with zipfile.ZipFile(path.parent / 'raw_export.zip') as archive:
            require(encode(raw) == encode(helpers.read_archive_json(archive, 'result.json')), 'Raw/archive mismatch')
        dispatch = read(path.parent / 'dispatch.json')
        require(dispatch['protocol_SHA256'] == sha(encode(protocol)) and dispatch['payload_SHA256'] == protocol['payload_SHA256'], 'Dispatch SHA mismatch')
        require(dispatch['profile'] == 'chenmiaoming' and bool(dispatch['call_id']), 'Dispatch identity missing')
        workers.append(validate_worker(raw, root, protocol, invocation))
        raws.append(raw)
    require(len({row['gpu_uuid'] for row in workers}) == 3, 'Workers do not have unique GPU UUIDs')
    require(sum(row['raw_samples'] for row in workers) == 8928, 'Sample coverage incomplete')
    source = HERE / 'wake_analysis.py'
    spec = importlib.util.spec_from_file_location('wake_frozen_analysis', source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    result = analyzer.analyze(protocol, raws)
    protocol_path = root / 'preregistration/protocol.json'
    base_source = HERE / 'analysis.py'
    result['inputs'] = dict(protocol=dict(path=str(protocol_path.resolve()), SHA256=file_sha(protocol_path)),
        raw=[dict(path=str(path.resolve()), SHA256=file_sha(path)) for path in paths],
        analysis_source=dict(path=str(source.resolve()), SHA256=file_sha(source)),
        base_analysis_source=dict(path=str(base_source.resolve()), SHA256=file_sha(base_source)))
    outputs = {'analysis.json': analyzer.encode(result), 'timing.csv': analyzer._csv_bytes(result), 'REPORT.md': analyzer.render(result).encode()}
    for name, expected in outputs.items():
        require((root / 'analysis' / name).read_bytes() == expected, 'Derived output does not reproduce: ' + name)
    return dict(status='OK', scope='CPU provenance, target identity, disjoint-primer, correctness, complete observations and analysis validation',
        preregistration=registration, workers=workers, raw_sample_count=8928,
        analysis_recomputed=True, derived_SHA256={name: sha(blob) for name, blob in outputs.items()},
        outcome_requirement='None; no equivalence, recovery, improvement or DVFS-direction test is required to pass.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.root, preregistration_only=args.preflight)
    output = args.output if args.output is not None else (None if args.preflight else args.root / 'validation_final.json')
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as handle:
            handle.write(encode(result))
    print(encode(dict(status='OK', preflight=args.preflight, samples=result.get('raw_sample_count'), output=str(output) if output else None)).decode().strip())


if __name__ == '__main__':
    main()
