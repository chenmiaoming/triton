"""Prepare the explicitly post-timing N3b KDA-gate mechanism diagnostic.

This does not alter N2/N4 timings, candidate choices or original N3 files. It
adds exact feature_vec4 counter evidence and a D64 practical-null control.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import zipfile

from . import common as c
from .profile_prepare import _abi, _identity, _member


CASES = ['N05_t2048_h8_d128', 'N05_t2048_h8_d64']
VARIANTS = ['U', 'G0', 'GL_feature_vec2', 'GL_feature_vec4']


def select(payload):
    groups, proofs = [], []
    with zipfile.ZipFile(payload) as archive:
        compiled = json.loads(archive.read('result.json'))
        assert compiled['status']=='OK' and compiled['GPU_allocated'] is False
        assert compiled['GPU_kernel_launches']==0
        indexed={(row['geometry'],row['variant']):row for row in compiled['programs']}
        for case_id in CASES:
            case=copy.deepcopy(indexed[(case_id,'U')]['case'])
            records, evidence, seen = [], [], {}
            for variant in VARIANTS:
                meta=copy.deepcopy(indexed[(case_id,variant)])
                cubin=archive.read(_member(meta,'kernel.cubin'))
                ptx=archive.read(_member(meta,'kernel.ptx'))
                ir=archive.read(_member(meta,'kernel.ttgir'))
                assert c.sha(cubin)==meta['cubin_SHA256']
                assert c.sha(ptx)==meta['artifact_SHA256']['kernel.ptx']
                assert c.sha(ir)==meta['artifact_SHA256']['kernel.ttgir']
                abi=_abi(ptx.decode())
                kind='U' if variant=='U' else 'Gluon'
                key=(kind,_identity(meta,abi))
                if key in seen:
                    representative=seen[key]
                    representative['aliases'].append(meta['tag'])
                    representative['logical_variants'].append(variant)
                    proofs.append(dict(group='N05_kda_gate_d'+str(case['D']),tag=meta['tag'],
                        representative=representative['metadata']['tag'],kind=kind,
                        full_CUBIN_equal=True,ABI_equal=True,runtime_argument_names_equal=True,
                        scalar_specialization_equal=True,launch_resources_equal=True,
                        cubin_SHA256=meta['cubin_SHA256'],ABI=abi,logical_metadata=meta))
                else:
                    record=dict(metadata=meta,aliases=[],logical_variants=[variant])
                    records.append(record);seen[key]=record
                lines=ir.decode().splitlines()
                evidence.append(dict(tag=meta['tag'],variant=variant,
                    cubin_SHA256=meta['cubin_SHA256'],ttgir_SHA256=c.sha(ir),PTX_SHA256=c.sha(ptx),
                    resources=meta.get('resources',{}),dynamic_shared_bytes=meta['dynamic_smem_bytes'],
                    encoding_definitions=[line.strip() for line in lines if re.match(r'\s*#\w+\s*=\s*#ttg\.',line)],
                    conversion_operations=[line.strip().split(' loc(')[0] for line in lines if 'convert_layout' in line]))
            groups.append(dict(id='N05_kda_gate_d'+str(case['D']),case=case,records=records,
                modes=['software','warm_application'],logical_variants=VARIANTS,
                static_evidence=evidence,
                static_rationale='Post-timing diagnostic: compare exact feature_vec4 with U/G0 and the preselected feature_vec2 contrast. Feature_vec4 aligns input/compute with output ownership. D64 is a practical-null control for the same transformation; no timing is rerun.'))
    return groups,proofs


def _trigger(path, variants):
    path=Path(path);value=c.read(path)
    assert value['status']=='OK'
    rows=[row for row in value['rows'] if row['case'] in CASES and row['variant'] in variants]
    assert len(rows)==len(CASES)*len(variants)
    return dict(path=str(path.relative_to(c.ROOT)),SHA256=c.sha(path.read_bytes()),
        protocol_SHA256=value['protocol_SHA256'],payload_SHA256=value['payload_SHA256'],
        rows=[{key:row[key] for key in ['case','variant','time_us','gain_vs_U_percent','gain_vs_U_CI95',
            'gain_vs_G0_percent','gain_vs_G0_CI95']} for row in rows])


def run(parent_protocol, n2_analysis, n4_analysis, stage='stage_n3b', snapshot=None):
    parent_path=Path(parent_protocol).resolve();parent=c.read(parent_path)
    root=c.OUT/stage
    assert not (root/'preregistration').exists(),root
    payload=Path(parent['payload_path']);assert c.sha(payload.read_bytes())==parent['payload_SHA256']
    groups,proofs=select(payload)
    n2=_trigger(n2_analysis,['GL_feature_vec2','GL_feature_vec4'])
    n4=_trigger(n4_analysis,['GL_feature_vec4'])
    assert n2['payload_SHA256']==parent['payload_SHA256']
    threshold=3.0
    observed=[]
    for row in n4['rows']:
        passes=all(row[key][0]>threshold for key in ['gain_vs_U_CI95','gain_vs_G0_CI95'])
        observed.append(dict(case=row['case'],strict_improvement_against_both_controls=passes,
            status='confirmed' if passes else 'not_confirmed_at_strict_both_reference_3percent_threshold',
            exact_CI95_U=row['gain_vs_U_CI95'],exact_CI95_G0=row['gain_vs_G0_CI95']))
    names=[str(path.relative_to(c.ROOT)) for path in c.SOURCE.rglob('*')
        if path.is_file() and '__pycache__' not in path.parts]
    names += [str((c.old.SOURCE/name).relative_to(c.ROOT)) for name in ['common.py','driver.py','layout_parsing.py']]
    names += ['experiments/tma_reduction_layout/phase8/launch.py','experiments/tma_reduction_layout/phase4/exact_cuda.py']
    snapshot=Path(snapshot) if snapshot else Path('/tmp/nonreduction-n3b-worker-snapshot')
    assert not snapshot.exists(),snapshot
    hashes=c.old.snapshot(names,snapshot)
    protocol=dict(stage=stage,phase='N3b_post_timing_gate_diagnostic',profile='chenmiaoming',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        snapshot=str(snapshot),source_SHA256=hashes,compiler=parent['compiler'],
        native_SHA256=c.NATIVE,backend_SHA256=parent['backend_SHA256'],
        payload_path=str(payload),payload_SHA256=parent['payload_SHA256'],
        profile_groups=groups,alias_proofs=proofs,metrics=parent['metrics'],
        metric_provenance=parent['metric_provenance'],maximum_concurrent_GPU_groups=3,
        formal_timing=False,evidence_role='post_timing_mechanism_diagnostic',confirmatory=False,
        selection_uses_candidate_timing=True,formal_timing_or_selector_changes=False,
        original_profile_protocol=dict(path=str(parent_path.relative_to(c.ROOT)),
            SHA256=c.sha(parent_path.read_bytes()),protocol_SHA256=c.sha(c.encode(parent))),
        selection_evidence=dict(N2_exploratory=n2,N4_independent_fixed_choice=n4),
        N4_outcome_boundaries=observed,ncu_settings=parent['ncu_settings'],correctness=parent['correctness'],
        questions=[
            'Does the exact feature_vec4 binary remove shared-memory layout exchange and barriers relative to U/G0? Nonidentity input-to-compute conversion is absent in its static IR, while launch shared memory is zero.',
            'Do measured n_regs and occupancy capacity change as suggested by compilation: D64 REG30->25 versus D128 REG48->36? Actual q comes only from Loaded.binding; capacity is not achieved occupancy.',
            'How do shared wavefronts, global sectors/requests and instruction counts differ from feature_vec2? No count is converted into a latency prediction.',
            'Does the same communication/resource transformation occur at D64 despite no confirmed >3% gain? Preserve this null and the provisional D128 timing boundary.'],
        interpretation='Explicitly post-timing counter diagnostic, not an extension of the original static-selected N3 denominator, not a new timing confirmation, and not a selector change. D128 did not pass the strict both-reference threshold in N4; do not round its G0 CI lower bound up to 3%.',
        report_requirement='The original profile_analyze numerical parser is usable, but its static-selected N3 report preamble must not be used unchanged for this post-timing supplement.')
    c.write_once(root/'preregistration/protocol.json',protocol)
    c.write_once(root/'preregistration/source_snapshot.zip',c.archive(snapshot))
    c.write_once(root/'compiled_payload.zip',payload.read_bytes())
    boundary='''# N3b — 后验 KDA gate 机制诊断\n\n本补充由 N2 的 feature_vec4 结果及 N4 的固定选择复测触发，属于明确标记的后验计数器诊断，不能并入原 N3 静态预选分母。\n\nD128 在 N4 相对 U 的区间超过 +3%，相对 G0 的区间下界为 2.988585172510577%，未通过“两个基线均严格超过 +3%”的确认标准；保持 provisional。D64 作为同语义、同变换的实用无收益对照保留。不会增加或改写正式计时、重新选择候选、改动任何旧原始样本。\n\n两个几何分别采集 U/G0/feature_vec2/feature_vec4，使用 software 和 warm_application 模式，共 16 个逻辑报告。计数不能加权为纳秒；q 来自真实加载二进制的 binding。\n'''
    c.write_once(root/'preregistration/BOUNDARIES.md',boundary.encode())
    print('N3b frozen:',sum(len(g['logical_variants'])*len(g['modes']) for g in groups),
        'logical /',sum(len(g['records'])*len(g['modes']) for g in groups),
        'physical reports; explicitly post-timing, no GPU launched')
    return protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-protocol',type=Path,default=c.OUT/'stage_n3/preregistration/protocol.json')
    parser.add_argument('--n2-analysis',type=Path,default=c.OUT/'stage_n2/analysis/analysis.json')
    parser.add_argument('--n4-analysis',type=Path,default=c.OUT/'stage_n4/analysis/analysis.json')
    parser.add_argument('--stage',default='stage_n3b')
    parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    run(args.parent_protocol,args.n2_analysis,args.n4_analysis,args.stage,args.snapshot)


if __name__=='__main__':main()
