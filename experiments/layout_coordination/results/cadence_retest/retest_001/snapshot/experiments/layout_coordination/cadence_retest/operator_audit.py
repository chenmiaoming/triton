"""Read-only inventory of production-operator timing exposed to host cadence.

This audit does not mark a timing difference as caused by idle, compile code,
or run GPUs.  It selects the complete latest source-aligned matrix for replay
rather than selecting only historical winners or large observed drifts.
"""
from collections import Counter
from pathlib import Path
import statistics

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.operator_gpu import g8_analyze


STAGE = c.OUT / 'stage_g8'


def ref(path):
    path = Path(path)
    return dict(path=str(path.relative_to(c.ROOT)), SHA256=c.sha(path.read_bytes()))


def audit():
    protocol_path = STAGE/'preregistration/protocol.json'
    gate_path = STAGE/'compile_gate.json'
    payload_path = STAGE/'compiled_payload.zip'
    final_path = STAGE/'final_validation.json'
    protocol, gate, final = map(c.read, (protocol_path, gate_path, final_path))
    assert gate['status'] == final['status'] == 'OK'
    assert gate['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert gate['payload_SHA256'] == c.sha(payload_path.read_bytes())
    records = {row['tag']: row for row in gate['records']}
    assert len(records) == len(gate['records']) == 168
    expected = g8_analyze.expected_conditions(protocol)
    assert len(expected) == 517
    physical = []
    drift = []
    sources = list(map(ref, (protocol_path, gate_path, payload_path, final_path)))
    uuids = []
    total_samples = 0
    baseline_keys = set()
    for worker in (1, 2, 3):
        raw_path = STAGE/f'timing/invocation_{worker}/raw.json'
        raw = c.read(raw_path)
        sources.append(ref(raw_path))
        assert raw['status'] == 'OK' and raw['invocation'] == worker
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        assert raw['payload_SHA256'] == gate['payload_SHA256']
        assert raw['inference']['environment']['gpu_uuid'] == raw['training']['environment']['gpu_uuid']
        uuids.append(raw['inference']['environment']['gpu_uuid'])
        for domain in ('inference', 'training'):
            part = raw[domain]
            assert part['status'] == 'OK' and part['driver_return_codes'] == [0]
            assert not part['Triton_imported'] and part.get('compile_calls', 0) == 0
            for cell in part['cells']:
                medians = [float(statistics.median(v['samples_us'])) for v in cell['visits']]
                assert len(medians) == 5 and all(x > 0 for x in medians)
                total_samples += sum(len(v['samples_us']) for v in cell['visits'])
                identity = dict(domain=domain, group=cell.get('group', cell.get('geometry')),
                    geometry=cell.get('geometry'), batch=cell.get('batch'), length=cell.get('length'),
                    mode=cell['mode'], M=cell.get('M'), tag=cell['tag'])
                drift.append(dict(**identity, worker=worker, visit_medians_us=medians,
                    min_us=min(medians), max_us=max(medians), mean_us=statistics.mean(medians),
                    spread_percent=100*(max(medians)/min(medians)-1)))
                if worker == 1:
                    tags = [cell['tag'], *cell.get('aliases', [])]
                    physical.append(dict(**identity, aliases=cell.get('aliases', []),
                        physical_CUBIN_SHA256=records[cell['tag']]['cubin_SHA256'],
                        node_types=cell['visits'][0]['node_types']))
                    for tag in tags:
                        suffix = tag.split(':')[-1]
                        if suffix in ('upstream', 'gluon_anchor'):
                            role = 'U' if suffix == 'upstream' else 'G0'
                            baseline_keys.add((domain, identity['group'], identity['batch'],
                                identity['length'], identity['mode'], identity['M'],
                                tag.rsplit(':', 1)[0], role))
    assert len(set(uuids)) == 3 and len(physical) == 439
    assert total_samples == 204135 and len(drift) == 1317
    assert len(baseline_keys) == 156
    by_domain = {}
    for domain in ('inference', 'training'):
        rows = [r for r in drift if r['domain'] == domain]
        by_domain[domain] = dict(worker_cells=len(rows),
            spread_above_3_percent=sum(r['spread_percent'] > 3 for r in rows),
            spread_above_10_percent=sum(r['spread_percent'] > 10 for r in rows),
            maximum_spread_percent=max(r['spread_percent'] for r in rows))
    return dict(status='OK', audit_type='protocol susceptibility and descriptive repeatability',
        inputs=sources, compiler=protocol['compiler'], native_SHA256=protocol['native_SHA256'],
        historical_protocol=ref(protocol_path), accepted_compile_gate=ref(gate_path),
        frozen_payload=ref(payload_path), payload_bytes=payload_path.stat().st_size,
        inference_groups=18, training_geometries=3, base_geometries=9,
        logical_programs=168, global_full_CUBIN_hashes=len({r['cubin_SHA256'] for r in gate['records']}),
        record_roles=dict(Counter(r['role'] for r in gate['records'])),
        logical_conditions=517, physical_conditions_per_worker=439,
        panel_conditions=dict(Counter(r['panel'] for r in expected.values())),
        same_run_U_G0_workload_pairs=78, historical_raw_samples=total_samples,
        historical_GPU_UUIDs=uuids, historical_timing=protocol['timing'],
        physical_inventory=physical, descriptive_drift_by_domain=by_domain,
        all_worker_cell_drift=drift,
        protocol_exposures=[
            dict(source='experiments/tma_reduction_layout/phase11/operator_gpu/production_runtime.py',
                operation='PipelineGraph.run',
                observation='Every sample guards every stage by reading its CUBIN from /tmp and hashing file plus loaded bytes before launching.'),
            dict(source='experiments/tma_reduction_layout/phase11/operator_gpu/g3b_runtime.py',
                operation='TrainingGraph.run',
                observation='Every training sample similarly hashes each stage before graph launch; pipeline includes three zero nodes inside target timing.'),
            dict(source='experiments/tma_reduction_layout/phase11/operator_gpu/g4_runtime.py',
                operation='run',
                observation='Randomized variant blocks, not per-round interleaving: each block reconstructs a graph, runs eight warmups, then 31 samples. Visit-zero output checks occur between variants.'),
            dict(source='experiments/tma_reduction_layout/phase11/operator_gpu/g3b_runtime.py',
                operation='run',
                observation='Training likewise uses randomized variant blocks with graph construction, resets and first-visit oracle checks outside timing. No per-sample host-gap timestamps or power/clock telemetry were recorded.')],
        stage_decisions=[
            dict(stages=['g1'], action='retain_correctness_no_timing_replay',
                reason='G1 ran correctness only; a timing-cadence issue does not invalidate output checks.'),
            dict(stages=['g2','g3a','g3b','g4a','g4b'], action='retain_obsolete_source_history_do_not_replay_wrong_Gluon',
                reason='Incorrect production Gluon source graph already superseded by source-aligned G8. Replay its corrected replacement, not the obsolete program.'),
            dict(stages=['g5','g6'], action='retain_CPU_compile_equivalence',
                reason='CPU-only executable/ABI/resource evidence; no new timing to correct.'),
            dict(stages=['g7'], action='retain_actual_JIT_and_correctness_revalidate_baselines_with_current_replay',
                reason='Actual tensor binding and machine-code equivalence are not timing claims. Fresh same-run controls will be gathered for all 78 current G8 workload pairs; historical G7-only default-call absolute timings remain historical.'),
            dict(stages=['g8'], action='replay_complete_latest_matrix',
                reason='All 517 logical conditions have data-history/cadence exposure; remeasure full corrected matrix and 78 U/G0 workload pairs with fixed warmup/cadence, preserving all raw/compiler artifacts.')],
        NCU_boundary='Preserve existing exact-CUBIN software instruction and static resource evidence. Cache/stall/throughput counters belong to their original replay protocol; do not silently relabel them as new-cadence counter measurements.',
        interpretation='Visit spread is descriptive, not proof of idle/DVFS causality or a selection filter. None of 1317 G8 worker/cells has >10% visit spread; complete-matrix replay is a protocol-based decision, not a search for favorable replacement results.')
