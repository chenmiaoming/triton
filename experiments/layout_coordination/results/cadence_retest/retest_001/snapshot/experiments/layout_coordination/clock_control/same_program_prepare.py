"""Freeze a CPU-only, same-shape frozen-program repeatability experiment.

Preparation only reads historical evidence.  It neither imports Triton nor
dispatches GPU work, and refuses to overwrite any experiment directory.
"""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import zipfile

from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c
from experiments.tma_reduction_layout.phase11.nonreduction_survey.elf import elf


SOURCE = Path(__file__).resolve().parent
ROOT = c.ROOT
HISTORY = c.OUT
CASE_ID = 'N02_k1024_p128_g128'
CUBIN_SHA256 = 'cd5dad8709d09c326f4f9e703fe28f76586f5589ddf8ab1a29e02c15debedd6d'
REQUIRED_TOOLS = ('same_program_prepare.py', 'same_program_worker.py',
                  'same_program_run.py', 'same_program_validate.py',
                  'analysis.py', 'telemetry.py')


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def reference(path):
    return dict(path=relative(path), SHA256=digest_file(path))


def source_names():
    """Pin execution tools and all their historical Python/JSON dependencies."""
    for name in REQUIRED_TOOLS:
        assert (SOURCE / name).is_file(), ('Execution source not ready', name)
    names = {relative(path) for path in SOURCE.rglob('*.py')
             if '__pycache__' not in path.parts}
    names.update(relative(path) for path in c.SOURCE.rglob('*')
                 if path.is_file() and path.suffix in ('.py', '.json')
                 and '__pycache__' not in path.parts)
    names.update('experiments/tma_reduction_layout/' + name for name in (
        'phase11/operator_gpu/common.py', 'phase11/operator_gpu/driver.py',
        'phase8/launch.py', 'phase4/exact_cuda.py'))
    assert all((ROOT / name).is_file() for name in names)
    return sorted(names)


def selected_program():
    """Copy the one historical U program, preserving every compiler artifact."""
    protocol_path = HISTORY / 'stage_n4/preregistration/protocol.json'
    payload_path = HISTORY / 'stage_n4/selection/selected_payload.zip'
    protocol = c.read(protocol_path)
    assert protocol['compiler'] == '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'
    assert protocol['native_SHA256'] == c.NATIVE
    candidates = [row for row in protocol['programs']
                  if row['geometry'] == CASE_ID and row['variant'] == 'U']
    assert len(candidates) == 1
    meta = candidates[0]
    assert meta['cubin_SHA256'] == CUBIN_SHA256
    assert meta['case']['rows'] == meta['case']['cols'] == 1024
    assert meta['case']['packed_cols'] == meta['case']['group_size'] == 128
    assert meta['fixture']['grid'] == [4, 32, 1]
    assert meta['fixture']['inplace'] == []
    prefix = meta['artifact_directory'] + '/'
    contents = {}
    with zipfile.ZipFile(payload_path) as archive:
        original = json.loads(archive.read('result.json'))
        records = [row for row in original['programs'] if row['tag'] == meta['tag']]
        assert records == [meta], 'N4 selected payload/protocol metadata mismatch'
        for name in archive.namelist():
            if name.startswith(prefix) and not name.endswith('/'):
                assert '..' not in Path(name).parts and not Path(name).is_absolute()
                contents[name] = archive.read(name)
    assert prefix + 'metadata.json' in contents
    for name, digest in meta['artifact_SHA256'].items():
        assert c.sha(contents[prefix + name]) == digest, name
    blob = contents[prefix + 'kernel.cubin']
    assert c.sha(blob) == CUBIN_SHA256
    driver_path = (HISTORY / 'stage_n4/timing/attempt_001'
                   / 'invocation_001/export/result.json')
    driver_raw = c.read(driver_path)
    assert driver_raw['status'] == 'OK'
    assert driver_raw['protocol_SHA256'] == c.sha(c.encode(protocol))
    driver_binding = driver_raw['bindings'][meta['tag']]
    assert driver_binding['cubin_SHA256'] == driver_binding['loaded_SHA256'] == CUBIN_SHA256
    assert driver_binding['attributes'] == dict(local_bytes=0, n_regs=72,
        occupancy_capacity_q=7, static_smem_bytes=0)
    provenance = dict(source_protocol=reference(protocol_path),
        source_payload=reference(payload_path), source_tag=meta['tag'],
        reference_driver_binding=dict(source=reference(driver_path), binding=driver_binding),
        canonical_artifact_directory=meta['artifact_directory'],
        canonical_CUBIN_SHA256=CUBIN_SHA256,
        allocated_ELF_sections=elf(blob),
        artifact_files_SHA256={name: c.sha(value) for name, value in sorted(contents.items())},
        original_metadata_preserved=True,
        scope='One original U function for one fixed AWQ shape; A/B are observation labels, not programs.')
    contents['result.json'] = c.encode(dict(status='OK', programs=[meta], cases=[meta['case']],
        provenance=provenance, compiler_calls=0, native_rebuilds=0))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(contents.items()):
            archive.writestr(name, value)
    return protocol, meta, buffer.getvalue(), provenance


def historical_same_worker():
    """Recompute the two motivating within-worker visit spreads from raw data."""
    rows = []
    for stage, invocation, first, last in (
            ('stage_n1_baseline_r2', 1, 3, 4),
            ('stage_n1_baseline_r3', 2, 6, 8)):
        raw_path = (HISTORY / stage / 'timing/attempt_001'
                    / f'invocation_{invocation:03}/export/result.json')
        protocol_path = HISTORY / stage / 'preregistration/protocol.json'
        raw, protocol = c.read(raw_path), c.read(protocol_path)
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        tag = CASE_ID + ':U'
        cells = [cell for cell in raw['cells'] if cell['tag'] == tag]
        assert len(cells) == 1 and cells[0]['case'] == CASE_ID
        cell = cells[0]
        assert cell['pattern'] == 'finite'
        binding = raw['bindings'][tag]
        assert binding['cubin_SHA256'] == binding['loaded_SHA256'] == CUBIN_SHA256
        assert binding['direct_launches'] == 1
        visits = {entry['visit']: float(statistics.median(entry['samples_us']))
                  for entry in cell['visits']}
        assert len(visits) == len(cell['visits'])
        rows.append(dict(stage=stage, invocation=invocation, case=CASE_ID, tag=tag,
            raw=reference(raw_path), protocol=reference(protocol_path),
            gpu_uuid=raw['environment']['gpu_uuid'],
            binding=binding, pattern=cell['pattern'],
            visit_medians_us=[dict(visit=visit, median_us=value)
                              for visit, value in sorted(visits.items())],
            selected_earlier_visit=first, selected_later_visit=last,
            selected_earlier_us=visits[first], selected_later_us=visits[last],
            selected_later_change_percent=100 * (visits[last] / visits[first] - 1),
            inferential_scope='Exploratory historical observations within one worker and one loaded function; no clock telemetry was recorded.'))
    return rows


def historical_cross_stage():
    """Audit the independent N2/N4 frozen-program repeats without pooling them."""
    protocols = {stage: c.read(HISTORY / stage / 'preregistration/protocol.json')
                 for stage in ('stage_n2', 'stage_n4')}
    raw_paths = {stage: [HISTORY / stage / 'timing/attempt_001'
                        / f'invocation_{worker:03}/export/result.json'
                        for worker in (1, 2, 3)] for stage in protocols}
    raw = {stage: [c.read(path) for path in paths] for stage, paths in raw_paths.items()}
    by_tag = {stage: {row['tag']: row for row in protocol['programs']}
              for stage, protocol in protocols.items()}
    comparable_fields = ('cubin_SHA256', 'runtime_argument_names',
        'specialized_runtime_values', 'dynamic_smem_bytes', 'num_warps',
        'resources', 'fixture', 'case')

    def estimate(stage, tag):
        medians = []
        for result in raw[stage]:
            assert result['status'] == 'OK'
            assert result['protocol_SHA256'] == c.sha(c.encode(protocols[stage]))
            cells = [cell for cell in result['cells']
                     if tag in [cell['tag'], *cell.get('aliases', [])]]
            assert len(cells) == 1
            medians.extend(statistics.median(entry['samples_us']) for entry in cells[0]['visits'])
        return float(statistics.mean(medians))

    rows = []
    for tag, right in sorted(by_tag['stage_n4'].items()):
        left = by_tag['stage_n2'][tag]
        equal = {key: left.get(key) == right.get(key) for key in comparable_fields}
        assert all(equal.values()), (tag, equal)
        a, b = estimate('stage_n2', tag), estimate('stage_n4', tag)
        rows.append(dict(tag=tag, case=right['geometry'], variant=right['variant'],
            CUBIN_SHA256=right['cubin_SHA256'], equality=equal,
            N2_mean_of_visit_medians_us=a, N4_mean_of_visit_medians_us=b,
            N4_vs_N2_change_percent=100 * (b / a - 1)))
    assert len(rows) == 42 and len({row['case'] for row in rows}) == 14
    inputs = [reference(HISTORY / stage / 'preregistration/protocol.json') for stage in protocols]
    inputs += [reference(path) for paths in raw_paths.values() for path in paths]
    return dict(inputs=inputs, rows=rows, cases=14, programs=42,
        estimator='Arithmetic mean of individual worker/visit medians, computed from original raw samples; stages are never pooled.',
        caveat='N2/N4 used different worker cohorts and different surrounding candidate mixes; this is an observational repeatability audit, not an isolated DVFS experiment.')


def protected_inventory():
    """Protect every previously tracked result and the original dependency code."""
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--',
        'experiments/tma_reduction_layout/results', 'experiments/layout_coordination',
        'experiments/tma_reduction_layout/phase11/nonreduction_survey',
        'experiments/tma_reduction_layout/phase11/operator_gpu/common.py',
        'experiments/tma_reduction_layout/phase11/operator_gpu/driver.py',
        'experiments/tma_reduction_layout/phase8/launch.py',
        'experiments/tma_reduction_layout/phase4/exact_cuda.py'], cwd=ROOT).decode().split('\0')
    names = sorted({name for name in tracked if name})
    assert names and all((ROOT / name).is_file() for name in names)
    return {name: digest_file(ROOT / name) for name in names}


def prepare(output):
    output = Path(output).resolve()
    assert not output.exists(), ('Refusing to overwrite evidence', output)
    assert output.is_relative_to(ROOT), ('Output must be inside repository', output)
    names = source_names()
    old_protocol, meta, payload, provenance = selected_program()
    audit = dict(status='OK', primary_same_worker=historical_same_worker(),
        secondary_cross_stage=historical_cross_stage(),
        target_case=CASE_ID, target_CUBIN_SHA256=CUBIN_SHA256,
        selection='The motivating fixed shape was chosen from historical same-worker U visit variation, not a new candidate timing result.')
    protected = protected_inventory()
    sources = {name: digest_file(ROOT / name) for name in names}
    inventory_blob, audit_blob = c.encode(protected), c.encode(audit)
    protocol = dict(stage='same_program_repeatability', phase='same_program_repeatability',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        profile='chenmiaoming', GPU='H100!', base_image=c.BASE_IMAGE,
        compiler=old_protocol['compiler'], native_SHA256=old_protocol['native_SHA256'],
        backend_SHA256=old_protocol['backend_SHA256'], compiler_calls=0, native_rebuilds=0,
        clock_control='unlocked', GPU_configuration_writes=False,
        cases=[meta['case']], programs=[meta], conditions=['fast_gap', 'legacy_gap', 'idle_20ms', 'gpu_preload'],
        variants=['A', 'B'], source_SHA256=sources, payload_SHA256=c.sha(payload),
        snapshot='snapshot', canonical_provenance=provenance,
        reference_driver_binding=provenance['reference_driver_binding'],
        protected_inventory_path='preregistration/protected_SHA256.json',
        protected_inventory_SHA256=c.sha(inventory_blob), protected_file_count=len(protected),
        historical_audit_path='preregistration/historical_audit.json', historical_audit_SHA256=c.sha(audit_blob),
        timing=dict(workers=3, visits=12, samples=31, warmups=8,
            bootstrap_replicates=8192, seed=20261105, practical_threshold_percent=3,
            prewarm_seconds=2.0, telemetry_period_s=0.01,
            preload_repeats=128, idle_seconds=0.02,
            sample_order='Counterbalanced deterministic condition blocks by visit/worker; independently randomized A/B each round; shared worker/visit bootstrap indices.'),
        correctness=dict(patterns=['finite', 'structured', 'edge'], pre_timing=True,
            post_timing_pattern='finite', full_output=True, guards=True, readonly_inputs=True),
        execution=dict(single_case=True, single_loaded_function=True, single_fixture_during_timing=True,
            single_target_graph=True, target_graph_node_types=[0, 7, 7],
            A_B_same_function_pointers_params_grid_graph=True,
            A_B_independent_observations=True, Triton_imported=False, compile_calls=0,
            prewarm='Initial two-second untimed prewarm once before all visits; eight fixed warmups per condition block.',
            fast_gap='No file read/hash or imposed idle between observations; guard outside the timed sample loop.',
            legacy_gap='Read/hash the frozen CUBIN and loaded buffer before every target graph launch, matching the historical guard.',
            idle_20ms='No per-observation file/hash; sleep 20 ms before launching the identical target graph.',
            gpu_preload='Launch an untimed graph with 128 repetitions of the same kernel before the unchanged single-target graph.'),
        interpretation=dict(primary='Repeatability of the identical program on the identical fixed shape, data pointers, scalars, grid, function and graph.',
            boundary='No size comparison, layout comparison, compiler change, clock write, or claim of fixed-frequency operation.',
            telemetry='Query-time clock observations; NVML power is a one-second average on H100, and utilization windows are product-dependent. These are not per-kernel frequency/power proofs.',
            causal_limit='Guard cadence changes multiple host effects; association with sampled clocks does not uniquely identify DVFS.'))
    # All checks precede creation; every output is write-once.
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        blob = (ROOT / name).read_bytes()
        assert c.sha(blob) == sources[name], ('Source changed during preparation', name)
        c.write_once(output / 'snapshot' / name, blob)
    c.write_once(output / 'payload.zip', payload)
    c.write_once(output / 'preregistration/protected_SHA256.json', inventory_blob)
    c.write_once(output / 'preregistration/historical_audit.json', audit_blob)
    c.write_once(output / 'preregistration/protocol.json', protocol)
    c.write_once(output / 'preregistration/freeze.json', dict(status='OK',
        protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
        protected_inventory_SHA256=c.sha(inventory_blob), source_files=len(sources),
        protected_files=len(protected), CPU_only=True, GPU_dispatches=0))
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = prepare(args.output)
    print('Frozen', protocol['stage'], CASE_ID, 'one program; protected', protocol['protected_file_count'], 'files')


if __name__ == '__main__':
    main()
