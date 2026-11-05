"""Read-only cadence exposure audit of historical TMA/Gluon evidence.

This is an inventory and descriptive drift audit, not a test asserting that
old measurements are wrong. It never imports Triton or dispatches GPU work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/tma_reduction_layout/results/phase11'
STAGES = ('stage_c2d', 'stage_d1', 'stage_m2', 'stage_m3', 'stage_m4a', 'stage_m4b')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), SHA256=sha(path.read_bytes()))


def inventory(stage):
    folder = BASE/stage
    gate_path = folder/'artifact_gate.json'
    gate = read(gate_path)
    assert gate['status'] == 'OK'
    protocols = list((folder/'preregistration').rglob('protocol.json'))
    matching = [p for p in protocols if sha(p.read_bytes()) == gate['protocol_SHA256']]
    assert matching, (stage, 'No protocol matches artifact gate')
    protocol_path = min(matching, key=lambda p: (len(p.parts), str(p)))
    protocol = read(protocol_path)
    for name, digest in protocol['worker_source_SHA256'].items():
        assert sha((ROOT/name).read_bytes()) == digest, (stage, name)
    payload_sha = sha((folder/'compiled_payload.zip').read_bytes())
    metadata = gate['metadata']
    raw_root = folder/'timing/revisions/3' if stage == 'stage_m4a' else folder/'timing'
    paths = sorted(raw_root.glob('invocation_*/raw.json'))
    assert len(paths) == 3, (stage, paths)
    drift, workers = [], []
    layout_contexts = set()
    for path in paths:
        raw = read(path)
        assert raw['status'] == 'OK' and raw['protocol_SHA256'] == gate['protocol_SHA256']
        assert raw['payload_SHA256'] == payload_sha
        assert raw['driver_return_codes'] == [0] and not raw['Triton_imported']
        for tag, binding in raw['bindings'].items():
            assert binding.get('CUBIN_SHA256', binding.get('cubin_SHA256')) == metadata[tag]['cubin_SHA256']
        raw_samples = 0
        for cell in raw['cells']:
            meta = metadata[cell['tag']]
            layout_contexts.add((cell['tag'], cell['q'], cell['grid'], cell['K']))
            medians = [float(statistics.median(v['samples_us'])) for v in cell['visits']]
            assert medians and min(medians) > 0
            raw_samples += sum(len(v['samples_us']) for v in cell['visits'])
            drift.append(dict(invocation=raw['invocation'], cell_id=cell['cell_id'],
                tag=cell['tag'], q=cell['q'], grid=cell['grid'], K=cell['K'],
                family=meta['family'], original_once=bool(meta.get('original_once')),
                natural_original_once=bool(meta.get('original_once')) and cell['q']=='natural',
                visit_medians_us=medians,
                within_worker_visit_range_percent=100*(max(medians)/min(medians)-1)))
        workers.append(dict(invocation=raw['invocation'], source=reference(path),
            gpu_uuid=raw['environment']['gpu_uuid'], physical_cells=len(raw['cells']),
            target_samples=raw_samples,
            host_gap_timestamps_recorded=any('host_gaps_ns' in v for cell in raw['cells'] for v in cell['visits']),
            telemetry_recorded=bool(raw.get('telemetry'))))
    assert len({w['gpu_uuid'] for w in workers}) == 3
    natural = [d for d in drift if d['natural_original_once']]
    families = {}
    for meta in metadata.values():
        families[meta['family']] = families.get(meta['family'], 0)+1
    geometry = sorted({(m['M'],m['N'],m['num_warps']) for m in metadata.values() if 'M' in m})
    return dict(stage=stage, protocol=reference(protocol_path), gate=reference(gate_path),
        frozen_payload=reference(folder/'compiled_payload.zip'),
        logical_programs=len(metadata), program_families=families,
        geometry=geometry, unique_recorded_physical_conditions=len(layout_contexts),
        sampling=protocol['timing'], workers=workers,
        worker_source_files_match_frozen_protocol=True,
        within_worker_drift=dict(worker_conditions=len(drift),
            above_3_percent=sum(d['within_worker_visit_range_percent']>3 for d in drift),
            above_10_percent=sum(d['within_worker_visit_range_percent']>10 for d in drift),
            largest_12=sorted(drift,key=lambda d:d['within_worker_visit_range_percent'],reverse=True)[:12],
            natural_original_once_conditions=len(natural),
            natural_original_once_above_3_percent=sum(d['within_worker_visit_range_percent']>3 for d in natural)),
        control_identity='Explicit Gluon layout/consumer/control programs; no source-aligned Triton U/G0 pair exists in this payload.',
        susceptibility='Potential guard/cadence exposure; observed range is descriptive and not a causal diagnosis.',
        correct_retest_name='Seen-case cadence remeasurement, never a new blind D1 evaluation.')


def audit():
    source_paths = [
        'experiments/tma_reduction_layout/phase11/c1/timing.py',
        'experiments/tma_reduction_layout/phase11/c15/timing.py',
        'experiments/tma_reduction_layout/phase11/c16/runtime.py',
        'experiments/tma_reduction_layout/phase11/c16/submission.py',
        'experiments/tma_reduction_layout/phase11/c2/timing.py',
        'experiments/tma_reduction_layout/phase11/c2/driver.py',
        'experiments/tma_reduction_layout/phase11/c2c/runtime.py',
        'experiments/tma_reduction_layout/phase11/c2d/runtime.py',
        'experiments/tma_reduction_layout/phase11/d1/runtime.py',
        'experiments/tma_reduction_layout/phase11/d1/driver.py',
        'experiments/tma_reduction_layout/phase11/m2/runtime.py',
        'experiments/tma_reduction_layout/phase11/m3/runtime.py',
        'experiments/tma_reduction_layout/phase11/m4/runtime.py',
        'experiments/tma_reduction_layout/phase11/artifact_audit.py',
        'experiments/tma_reduction_layout/phase10/run_timing.py',
        'experiments/tma_reduction_layout/phase7/run_timing.py',
        'experiments/tma_reduction_layout/phase8/run_timing.py',
        'experiments/tma_reduction_layout/phase9/run_timing.py',
    ]
    result = dict(status='OK', scope='Read-only legacy timing-method inventory and within-worker visit drift.',
        source_evidence=[reference(ROOT/name) for name in source_paths],
        stages=[inventory(stage) for stage in STAGES],
        findings=[
            'C1/C1.5/C1.6 use per-sample file+loaded-byte guards followed by direct CUDA event launches; C1.6 additionally has an explicit warm/cold cache manipulation.',
            'C2a/b/c/d,D1,M2/M3/M4 use one-kernel graphs with two external event nodes, per-sample guards, and contiguous samples in shuffled cell blocks.',
            'These legacy payloads are extracted under /tmp, unlike historical production/nonreduction Volume extraction. The new /tmp guard negative control prevents treating all per-sample guards as proven harmful.',
            'C2+ has 2–3 condition-specific guarded warmups before each cell block; initial correctness work is outside formal samples, and some checks follow visit zero. No recorded host-gap timeline or NVML telemetry permits retrospective exclusion of transient interference.',
            'Longer repeated K kernels may dilute a fixed startup component, but steady slopes are not automatically immune if cadence effects depend on K or geometry. Use a predefined K/grid sensitivity panel before deciding to rerun an entire old probe matrix.',
            'M3 has actual within-worker >10% visit spreads and is a priority quantitative retest. Most D1/M4 natural FULL cells are stable below 3% by this descriptive range criterion; stability does not prove absence of a common bias.',
            'Keep frozen models, parameters, D1 pre-predictions, historical raw measurements and reports unchanged. A cadence remeasurement is a new seen-case execution-context audit, not a fresh blind test.',
            'Static IR/PTX/SASS execution plans, legality, actual CUBIN resource allocations, correctness, and discrete dynamic instruction/request counts do not need automatic recollection solely because host cadence changed. NCU latency, bandwidth and stall percentages retain their original profiled context and cannot substitute for the new unprofiled timing.',
            'The Phase11 initial U/Gluon reproduction bridge explicitly did not certify machine-code/source-aligned G0 equivalence; its labels must not be rebranded as the corrected production U/G0 baseline.',
            'Earlier Phase7/8/9/10 runners also use direct event timing with per-launch guards and initial warmup suites. Phase8 records boundary nvidia-smi telemetry, which is not a per-observation host-gap/frequency trace. This script inventories their source pattern only; it does not claim that all earlier or cross-architecture raw runs have been individually reaudited.',
            'Compile-time shared-memory OOM/legality results and static spill/resource findings do not become invalid because a timing-cadence effect was found. Runtime latency and speedup claims remain scoped to their original environment until explicitly remeasured.',
        ],
        recommended_scope=dict(
            mandatory_current_operator_claims='Fresh same-run U/G0 plus all candidates in the latest source-aligned operator/nonreduction matrices; obsolete wrong-Gluon programs are historical only.',
            priority_legacy_controls='M3 native/fixed vec2/vec4/vec8 natural and q=1/4 timing contrasts with observed visit drift; original TMA geometry default/candidate FULL controls; small-K/grid C1/C2 probe sentinels.',
            preserve_without_automatic_repeat='Frozen structural/NCU count evidence, CPU-only model fits, obsolete pilot timings and old failed preparations.',
            conditional_expansion='Pre-register a uniform practical equivalence/difference threshold across the sensitivity panel. If cadence materially changes a current claim or its ordering, rerun that whole semantic comparison family with the new common protocol; do not select only winning or drifting cells.'))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as handle:handle.write(encode(result))
    print(json.dumps(dict(status=result['status'],stages=[
        dict(stage=s['stage'],logical_programs=s['logical_programs'],
             worker_conditions=s['within_worker_drift']['worker_conditions'],
             above_3_percent=s['within_worker_drift']['above_3_percent'],
             above_10_percent=s['within_worker_drift']['above_10_percent'])
        for s in result['stages']],output=str(args.output) if args.output else None),sort_keys=True))
