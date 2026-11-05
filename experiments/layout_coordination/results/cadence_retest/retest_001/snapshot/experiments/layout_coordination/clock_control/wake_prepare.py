"""Adaptive follow-up: preserve target identity, prime distinct code and data."""
import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import zipfile
from . import same_program_prepare as old
from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c


def prepare(output):
    output = Path(output).resolve()
    assert not output.exists() and output.is_relative_to(c.ROOT)
    for name in ['wake_worker.py','wake_run.py','wake_analysis.py','wake_validate.py']:
        assert (Path(__file__).parent/name).is_file(), name
    previous = c.ROOT/'experiments/layout_coordination/results/clock_control/same_program_001'
    assert c.read(previous/'validation_final.json')['status'] == 'OK'
    first = c.read(previous/'preregistration/protocol.json')
    target = first['programs'][0]
    source = c.OUT/'stage_n4'
    prior = c.read(source/'preregistration/protocol.json')
    primer = next(r for r in prior['programs'] if r['geometry'] == target['geometry']
                  and r['variant'] == 'GL_packed_compute_vec1')
    assert primer['cubin_SHA256'] != target['cubin_SHA256']
    members = {}
    with zipfile.ZipFile(source/'selection/selected_payload.zip') as archive:
        for meta in [target, primer]:
            prefix = meta['artifact_directory']+'/'
            for name in archive.namelist():
                if name.startswith(prefix) and not name.endswith('/'):
                    members[name] = archive.read(name)
            for name, digest in meta['artifact_SHA256'].items():
                assert c.sha(members[prefix+name]) == digest
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, blob in sorted(members.items()):
            archive.writestr(name, blob)
    payload = stream.getvalue()
    names = old.source_names()
    sources = {name: old.digest_file(c.ROOT/name) for name in names}
    protected = c.read(previous/'preregistration/protected_SHA256.json')
    for name, digest in first['source_SHA256'].items():
        assert old.digest_file(c.ROOT/name) == digest
        protected[name] = digest
    for path in previous.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            protected[str(path.relative_to(c.ROOT))] = old.digest_file(path)
    raw_path = source/'timing/attempt_001/invocation_001/export/result.json'
    bindings = c.read(raw_path)['bindings']
    p = dict(stage='same_program_wake_followup', phase='same_program_wake_followup',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        adaptive=True, motivating_run=old.reference(previous/'preregistration/protocol.json'),
        motivation='Prior same-shape diagnostic found a 20ms-idle slowdown without sampled nominal-clock variation. This follow-up tests whether one untimed kernel on disjoint data and distinct code restores target timing.',
        profile='chenmiaoming', GPU='H100!', base_image=c.BASE_IMAGE,
        compiler=first['compiler'], native_SHA256=first['native_SHA256'],
        compiler_calls=0, native_rebuilds=0, clock_control='unlocked', GPU_configuration_writes=False,
        cases=first['cases'], programs=[target,primer], program_roles=['target','primer'],
        expected_fixture=target['fixture'],
        variants=['A','B'], conditions=['fast_gap','idle_20ms','idle_then_same_prime','idle_then_disjoint_prime'],
        source_SHA256=sources, snapshot='snapshot', payload_SHA256=c.sha(payload),
        reference_driver_bindings={role:dict(source=old.reference(raw_path),binding=bindings[meta['tag']])
            for role,meta in zip(['target','primer'],[target,primer])},
        protected_inventory_path='preregistration/protected_SHA256.json',
        protected_inventory_SHA256=c.sha(c.encode(protected)),protected_file_count=len(protected),
        timing=dict(workers=3,visits=12,samples=31,warmups=8,
            bootstrap_replicates=8192,seed=20261106,practical_threshold_percent=3,
            prewarm_seconds=2.0,telemetry_period_s=.01,idle_seconds=.02),
        correctness=dict(patterns=['finite','structured','edge'],post_timing_pattern='finite'),
        execution=dict(target_unchanged=True,target_graph_node_types=[0,7,7],
            disjoint_primer='One untimed full-grid GL kernel uses a different function and separate allocations for every input/output/scratch. Target pointers, scalars, shape, grid and graph remain unchanged.',
            same_primer='One untimed execution of the target graph before the unchanged timed target graph.',
            guards='Both modules guarded before/after blocks; no per-observation file/hash/copy.',
            prewarm='Initial balanced two-second activity from both graphs; eight fixed condition-specific warmups per block.',
            order='Deterministic counterbalanced condition rotations and reversals; seeded random A/B order per observation round.'),
        interpretation='Adaptive mechanism diagnostic, not a new layout comparison or a retrospective causal explanation of historical visits. Nominal sampled clocks do not measure microsecond effective-clock/gating transitions.')
    for name in names:
        blob = (c.ROOT/name).read_bytes()
        assert c.sha(blob) == sources[name]
        c.write_once(output/'snapshot'/name,blob)
    c.write_once(output/'payload.zip',payload)
    c.write_once(output/'preregistration/protected_SHA256.json',protected)
    c.write_once(output/'preregistration/protocol.json',p)
    c.write_once(output/'preregistration/freeze.json',dict(status='OK',CPU_only=True,GPU_dispatches=0,
        protocol_SHA256=c.sha(c.encode(p)),payload_SHA256=c.sha(payload)))
    print('Frozen adaptive wake follow-up; target and disjoint primer:',[r['cubin_SHA256'] for r in p['programs']])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    prepare(parser.parse_args().output)
