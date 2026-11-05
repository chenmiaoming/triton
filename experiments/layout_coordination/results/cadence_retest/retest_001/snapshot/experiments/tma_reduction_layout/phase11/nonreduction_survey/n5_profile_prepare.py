"""Freeze six footprint counter profiles without opening formal timing files."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import zipfile

from . import common as c
from .profile_prepare import _abi, _member


PAYLOAD_SHA256 = '7fcc58f9e4c163a97b69abbaf0d0ebb8791be9bf33b0024e8c38c938447a3805'
TARGETS = [
    ('N5_N01_scale_footprint', 'N01_m4096_n8192', 'GL_coordinated_fully_coalesced'),
    ('N5_N02_awq_footprint', 'N02_k8192_p1024_g128', 'GL_packed_compute_vec1'),
]


def select(payload):
    payload = Path(payload)
    assert c.sha(payload.read_bytes()) == PAYLOAD_SHA256
    with zipfile.ZipFile(payload) as archive:
        accepted = json.loads(archive.read('result.json'))
        assert accepted['status'] == 'OK' and len(accepted['programs']) == 6
        by_variant = {(program['geometry'],program['variant']): program
                      for program in accepted['programs']}
        groups = []
        for identity, case_id, candidate in TARGETS:
            case = next(case for case in accepted['cases'] if case['id']==case_id)
            variants = ['U', 'G0', candidate]
            records, evidence = [], []
            for variant in variants:
                meta = copy.deepcopy(by_variant[case_id,variant])
                for name,digest in meta['artifact_SHA256'].items():
                    assert c.sha(archive.read(_member(meta,name))) == digest
                assert c.sha(archive.read(_member(meta,'kernel.cubin'))) == meta['cubin_SHA256']
                ptx = archive.read(_member(meta,'kernel.ptx')).decode()
                records.append(dict(metadata=meta, aliases=[], logical_variants=[variant]))
                evidence.append(dict(variant=variant, tag=meta['tag'],
                    cubin_SHA256=meta['cubin_SHA256'], ABI=_abi(ptx),
                    resources=meta['resources'], choice=meta.get('choice')))
            baseline = next(record for record in accepted['records'] if record['case']==case_id)
            assert baseline['status']=='OK' and all(baseline[key] for key in
                ['allocated_sections_equal','ABI_and_launch_equal','resources_equal'])
            groups.append(dict(id=identity, case=case, records=records,
                modes=['warm_application'], logical_variants=variants,
                static_rationale='Already frozen N5 footprint pair: inspect cache/DRAM traffic, request grouping, communication and resource capacity; no new layout selection or retuning.',
                static_evidence=evidence))
    assert len(groups)==2 and sum(len(group['records']) for group in groups)==6
    return groups, accepted['source_binding_protocol_SHA256']


def run(payload, stage='stage_n5_profiles', snapshot=None):
    payload = Path(payload).resolve()
    root = c.OUT/stage
    assert not (root/'preregistration').exists(), root
    groups, binding_sha = select(payload)
    metrics_path = c.old.OUT/'stage_g8/profile_protocol.json'
    metrics = c.read(metrics_path)['profile_metrics']
    assert len(metrics)==len(set(metrics))==26
    names = [str(path.relative_to(c.ROOT)) for path in c.SOURCE.rglob('*')
             if path.is_file() and '__pycache__' not in path.parts]
    names += [str((c.old.SOURCE/name).relative_to(c.ROOT))
              for name in ['common.py','driver.py','layout_parsing.py']]
    names += ['experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    snapshot = Path(snapshot or '/tmp/nonreduction-n5-profile-worker-snapshot')
    hashes = c.old.snapshot(names, snapshot)
    previous = c.read(c.old.OUT/'stage_g8/preregistration/protocol.json')
    protocol = dict(stage=stage, phase='N5_frozen_footprint_transfer_profiles',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        profile='chenmiaoming', snapshot=str(snapshot), source_SHA256=hashes,
        compiler=previous['compiler'], native_SHA256=c.NATIVE,
        backend_SHA256=previous['backend_SHA256'], payload_path=str(payload),
        payload_SHA256=PAYLOAD_SHA256, profile_groups=groups, metrics=metrics,
        metric_provenance=dict(path=str(metrics_path.relative_to(c.ROOT)),
            SHA256=c.sha(metrics_path.read_bytes()), description='Exact prior 26-metric H100 list; availability checked on every actual worker.'),
        source_binding_protocol_SHA256=binding_sha, alias_proofs=[],
        maximum_concurrent_GPU_groups=1, expected_reports=6,
        formal_timing=False, InstructionStats_collected=False,
        selection_uses_candidate_timing=False,
        selection_policy='Two N5 footprint cases and their previously frozen single candidates only. No formal N5 timing file is opened or used. Preserve U/G0 as separate reports.',
        ncu_settings=dict(warm_application=dict(replay='application',cache='none',grid='original full grid')),
        correctness=dict(warmups=8, one_binary_per_report=True,
            unchanged_runtime_sizes_and_specialization=True, full_oracle_and_canaries=True,
            inplace_restore='Before every warmup and target, outside profiler range'),
        concurrency_scope='Host orchestration is strictly sequential spawn/get; profile_run.collect is unchanged and its module-level ceiling field is not an observation of actual concurrency.',
        interpretation='Hardware cache/DRAM/request diagnostics for the frozen larger footprints. Do not infer residency from allocation size, retune layouts, convert profiler time into formal timing, or fabricate missing software InstructionStats.')
    c.write_once(root/'preregistration/protocol.json',protocol)
    c.write_once(root/'preregistration/source_snapshot.zip',c.archive(snapshot))
    c.write_once(root/'compiled_payload.zip',payload.read_bytes())
    print('N5 profiles frozen:',len(groups),'groups,6 reports,warm_application only,max concurrency1')
    return protocol


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload',type=Path,default=c.OUT/'stage_n5/revision_002/binding/attempt_001/accepted_timing_payload.zip')
    parser.add_argument('--stage',default='stage_n5_profiles')
    parser.add_argument('--snapshot',type=Path)
    args = parser.parse_args()
    run(args.payload,args.stage,args.snapshot)
