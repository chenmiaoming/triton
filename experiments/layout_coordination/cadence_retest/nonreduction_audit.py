"""Read-only cadence-impact audit of the frozen N1--N5 operator survey.

No GPU, compiler, historical-file mutation or automatic experiment dispatch.
The output is an inventory and an exposure assessment, not a claim that the
old candidate gains were caused by idle state or should be numerically fixed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c
from experiments.tma_reduction_layout.phase11.nonreduction_survey.elf import elf


ROOT = c.ROOT
HISTORY = c.OUT
TIMED_STAGES = ('stage_n1_baseline_r2', 'stage_n1_baseline_r3',
                'stage_n2', 'stage_n4', 'stage_n5_timing')
CURRENT_STAGES = ('stage_n2', 'stage_n5_timing')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), SHA256=file_sha(path))


def physical_identity(meta):
    """Use the original identity contract; preserve independent U/G0 controls."""
    return (meta['cubin_SHA256'], tuple(meta['runtime_argument_names']),
        c.encode(meta.get('specialized_runtime_values', {})),
        meta['num_warps'], meta['dynamic_smem_bytes'],
        meta.get('global_scratch_size', 0),
        'U' if meta['variant'] == 'U' else 'Gluon')


def physical_groups(programs):
    groups = {}
    for meta in programs:
        key = (meta['geometry'], physical_identity(meta))
        if key not in groups:
            groups[key] = dict(case=meta['geometry'],
                representative_tag=meta['tag'], aliases=[],
                cubin_SHA256=meta['cubin_SHA256'],
                control_bucket='U' if meta['variant'] == 'U' else 'Gluon')
        else:
            groups[key]['aliases'].append(meta['tag'])
    return list(groups.values())


def inventory(stage):
    path = HISTORY / stage / 'preregistration/protocol.json'
    protocol = c.read(path)
    payload = Path(protocol['payload_path'])
    assert file_sha(payload) == protocol['payload_SHA256'], payload
    programs = protocol['programs']
    groups = physical_groups(programs)
    eligibility_path = HISTORY/stage/'check/attempt_001/invocation_000/export/result.json'
    eligibility = c.read(eligibility_path)
    assert eligibility['status'] == 'OK'
    assert eligibility['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert eligibility['compile_calls'] == 0
    assert eligibility['Triton_imported'] is False
    by_tag = defaultdict(set)
    for check in eligibility['checks']:
        assert check['pass'] and all(check['guards'])
        for tag in [check['tag'], *check.get('aliases', [])]:
            by_tag[tag].add(check['pattern'])
    assert all(by_tag[meta['tag']] == {'finite', 'structured', 'edge'} for meta in programs)
    cases = []
    with zipfile.ZipFile(payload) as archive:
        for meta in programs:
            for name, digest in meta['artifact_SHA256'].items():
                assert c.sha(archive.read(meta['artifact_directory'] + '/' + name)) == digest
        for case in protocol['cases']:
            metas = [m for m in programs if m['geometry'] == case['id']]
            u = next(m for m in metas if m['variant'] == 'U')
            g = next(m for m in metas if m['variant'] == 'G0')
            blobs = [archive.read(m['artifact_directory'] + '/kernel.cubin') for m in (u, g)]
            assert elf(blobs[0]) == elf(blobs[1]), case['id']
            for key in ('runtime_argument_names', 'specialized_runtime_values',
                        'num_warps', 'dynamic_smem_bytes', 'global_scratch_size', 'resources'):
                assert u.get(key) == g.get(key), (case['id'], key)
            fixture_meta = u if 'fixture' in u else g
            fixture = fixture_meta['fixture']
            cases.append(dict(case=case, logical_programs=len(metas),
                physical_programs=sum(x['case'] == case['id'] for x in groups),
                U_tag=u['tag'], G0_tag=g['tag'],
                baseline_allocated_ELF_ABI_resources_equal=True,
                fixture=fixture, fixture_source_tag=fixture_meta['tag'],
                in_place=bool(fixture['inplace']),
                program_tags=[m['tag'] for m in metas]))
    return dict(stage=stage, protocol=ref(path), payload=ref(payload),
        compiler=protocol['compiler'], native_SHA256=protocol['native_SHA256'],
        backend_SHA256=protocol['backend_SHA256'],
        logical_programs=len(programs), physical_programs=len(groups),
        cases=cases, physical_groups=groups,
        eligibility=dict(source=ref(eligibility_path), status='OK',
            patterns=['finite','structured','edge'],
            physical_correctness_checks=len(eligibility['checks']),
            all_logical_program_patterns_covered=True),
        program_source='Load unchanged records from the referenced protocol[programs].',
        artifact_source='Copy each original artifact_directory from referenced payload, preserving all artifact_SHA256 bytes.')


def historical_drift(stage):
    """Describe all visits without selecting/dropping a worker or a candidate."""
    protocol_path = HISTORY / stage / 'preregistration/protocol.json'
    protocol = c.read(protocol_path)
    paths = sorted((HISTORY / stage / 'timing/attempt_001').glob('invocation_*/export/result.json'))
    assert len(paths) == protocol['timing']['workers'] == 3
    rows, paired = [], []
    by_case = defaultdict(list)
    for path in paths:
        raw = c.read(path)
        assert raw['status'] == 'OK'
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        logical = {}
        for cell in raw['cells']:
            medians = {v['visit']: float(statistics.median(v['samples_us'])) for v in cell['visits']}
            assert len(medians) == protocol['timing']['visits']
            assert all(len(v['samples_us']) == protocol['timing']['samples'] for v in cell['visits'])
            binding = raw['bindings'][cell['tag']]
            assert binding['cubin_SHA256'] == binding['loaded_SHA256']
            row = dict(case=cell['case'], tag=cell['tag'], aliases=cell.get('aliases', []),
                invocation=raw['invocation'], gpu_uuid=raw['environment']['gpu_uuid'],
                cubin_SHA256=binding['cubin_SHA256'],
                mean_visit_median_us=float(statistics.mean(medians.values())),
                minimum_visit_median_us=min(medians.values()),
                maximum_visit_median_us=max(medians.values()),
                visit_span_percent=100 * (max(medians.values()) / min(medians.values()) - 1),
                visit_medians_us=[dict(visit=v, median_us=x) for v, x in sorted(medians.items())])
            rows.append(row)
            by_case[cell['case']].append(row)
            for tag in [cell['tag'], *cell.get('aliases', [])]:
                assert tag not in logical
                logical[tag] = (cell['case'], medians)
        for case in protocol['cases']:
            cid = case['id']
            u, g = logical[cid + ':U'][1], logical[cid + ':G0'][1]
            assert set(u) == set(g)
            # This relative spread is diagnostic; it is not a new confidence interval.
            gains = [100 * (u[v] - g[v]) / u[v] for v in sorted(u)]
            uu, gg = [u[v] for v in sorted(u)], [g[v] for v in sorted(u)]
            common_correlation = (float(statistics.correlation(uu, gg))
                if statistics.pstdev(uu) and statistics.pstdev(gg) else None)
            paired.append(dict(case=cid, invocation=raw['invocation'],
                U_visit_span_percent=100 * (max(uu) / min(uu) - 1),
                G0_visit_span_percent=100 * (max(gg) / min(gg) - 1),
                U_G0_visit_median_correlation=common_correlation,
                G0_vs_U_visit_gain_min_percent=min(gains),
                G0_vs_U_visit_gain_max_percent=max(gains),
                G0_vs_U_visit_gain_span_pp=max(gains) - min(gains)))
    summary = []
    for case, entries in by_case.items():
        worst = max(entries, key=lambda x: x['visit_span_percent'])
        summary.append(dict(case=case, maximum_observed_visit_span_percent=worst['visit_span_percent'],
            motivating_tag=worst['tag'], motivating_invocation=worst['invocation'],
            minimum_visit_median_us=worst['minimum_visit_median_us'],
            maximum_visit_median_us=worst['maximum_visit_median_us']))
    return dict(stage=stage, protocol=ref(protocol_path), raw_inputs=[ref(p) for p in paths],
        timing=protocol['timing'], rows=rows, case_summary=summary,
        paired_G0_U_visit_diagnostics=paired,
        host_gap_timestamps_recorded=False, frequency_telemetry_recorded=False,
        causal_boundary='Observed visit spans are descriptive only. They neither identify idle-state causality nor show that every paired layout gain is biased.')


def source_evidence():
    patterns = {
        'phase11/nonreduction_survey/runtime.py': [
            'buffers.reset(only_inplace=True)', 'observations[cell[\'tag\']].append(graphs[cell[\'tag\']].run())',
            "if visit==0:", "check=buffers.check()"],
        'phase11/nonreduction_survey/measure_run.py': ["root=Path('/evidence/nonreduction/runtime')"],
        'phase11/operator_gpu/driver.py': ["sha(self.path.read_bytes())", 'self.kernel.guard()',
                                         "self.driver.call('cuGraphLaunch'"],
        'phase11/nonreduction_survey/tune_worker.py': [
            'for _ in range(3):e[\'graph\'].replay()', "for _ in range(9):", "e['graph'].replay();torch.cuda.synchronize()"]}
    result=[]
    for suffix, needles in patterns.items():
        path=ROOT/'experiments/tma_reduction_layout'/suffix
        lines=path.read_text().splitlines()
        matches={needle:[i+1 for i,line in enumerate(lines) if needle in line] for needle in needles}
        assert all(matches.values()), (path, matches)
        result.append(dict(**ref(path), matches=matches))
    return result


def build_audit():
    datasets=[inventory(stage) for stage in CURRENT_STAGES]
    n2=c.read(HISTORY/'stage_n2/preregistration/protocol.json')
    n4=c.read(HISTORY/'stage_n4/preregistration/protocol.json')
    indexed={p['tag']:p for p in n2['programs']}
    assert all(indexed[p['tag']]==p for p in n4['programs'])
    assert sum(len(x['cases']) for x in datasets)==16
    assert sum(x['logical_programs'] for x in datasets)==144
    assert sum(x['physical_programs'] for x in datasets)==134
    return dict(status='OK', audit_kind='CPU read-only historical cadence exposure and frozen retest inventory',
        datasets=datasets, historical=[historical_drift(s) for s in TIMED_STAGES],
        source_evidence=source_evidence(),
        exposure='All N1r2/r3, N2, N4 and N5 formal driver timings use per-observation file+loaded-buffer SHA guards on Modal Volume and have no recorded host-gap timestamps or fixed inter-sample cadence. The uniform complete current matrix is selected by protocol exposure, not observed winners or outliers.',
        replacement=dict(cases=16, logical_programs=144, physical_programs=134,
            unchanged_full_candidate_matrix=True, fresh_U_G0_per_case=True,
            immutable_original_artifacts=True, compiler_calls_required=0,
            N4_frozen_choices='All 42 records are exactly present in N2. Score those preregistered choices on the fresh complete matrix without selecting new winners; no independent duplicate N4 dispatch is necessary for coverage.',
            N1_baseline_timing='Superseded by fresh same-run U/G0 for every case; do not rerun obsolete r2/r3 as separate datasets.',
            in_place_requirement='MRoPE Q/K must be restored before every target/prime that consumes them. A repeated same-program primer without restoration changes semantics. Keep copies outside target events, standardize ordering, and record their cadence.',
            configuration_scope='Retain the frozen BT/W/stages for N05. Do not silently retune baseline configuration while replacing timing evidence.'),
        unaffected_or_separate=dict(
            compiler_binding_correctness='Source/AST, allocated ELF/ABI/resource equality, correctness/guards and original compiler artifacts remain valid frozen evidence; timing disturbance does not require recompilation.',
            N5_tail_stride='Correctness-only N01_tail_stride_m131_n7 has no formal timing claim; no cadence-driven timing rerun is required.',
            NCU='Keep original N3/N5 raw profiles as evidence of their original cache/warmup/profile conditions. Structural code/traffic evidence is not invalidated merely by event-time drift. Do not relabel cache/DRAM/stall percentages as matching new timing; collect matched profiles if needed to explain a materially changed timing conclusion.',
            U_only_autotune='N05 U-only 24-config menu per geometry used immediate 3 warmups +9 Torch-graph samples without per-sample file hashing. It has no fixed cadence telemetry and tiny winner margins, but is separate from the Volume-guard path. Frozen-layout retiming can retain BT32/W8/stages2. Claiming the optimal source-menu configuration after correction would require a separate check of all 48 available frozen U records.',
            intentional_idle_diagnostics='clock_control/same_program_001 and wake_001 deliberately manipulate gaps and are not layout-claim replacements; preserve them as diagnostic evidence.'),
        claim_boundary='Potential exposure is not proof of historical bias. New samples must not be pooled with old samples or used to subtract an assumed idle penalty from old gains.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=build_audit()
    if args.output:
        c.write_once(args.output,result)
    print(json.dumps(dict(status=result['status'], cases=result['replacement']['cases'],
        logical_programs=result['replacement']['logical_programs'],
        physical_programs=result['replacement']['physical_programs'],
        historical_stages=len(result['historical']), output=str(args.output) if args.output else None)))


if __name__=='__main__':main()
