"""Final CPU-only M4 audit and write-once freeze."""
import argparse
import io
import re
import zipfile
from . import common as c
from .runtime import cells_for
from .offline_import import one as offline_check
from experiments.tma_reduction_layout.phase11.operator_gpu.layout_parsing import encodings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    c.protect_prior()
    p=c.protocol();c.verify_sources(p)
    archives=[(c.OUT/'preregistration/protocol.json',c.OUT/'preregistration/source_snapshot.zip')]
    archives += [(path,path.parent/'source_snapshot.zip') for path in (c.OUT/'preregistration/revisions').glob('*/protocol.json')]
    for path,blob in archives:
        value=c.read(path)
        with zipfile.ZipFile(blob) as z:
            for name,digest in value['worker_source_SHA256'].items():assert c.sha(z.read(name))==digest,name
    cg=c.read(c.OUT/'compile_gate.json');ag=c.read(c.OUT/'artifact_gate.json')
    history=c.OUT/'audit_revisions/1'
    if history.exists():
        old=c.read(history/'freeze.json')
        invalid=c.read(history/'invalidation.json')
        assert c.sha((history/'frozen_state.zip').read_bytes())==invalid['prior_frozen_state_SHA256']
        with zipfile.ZipFile(history/'frozen_state.zip') as data,zipfile.ZipFile(history/'source_at_freeze.zip') as source:
            for name,digest in old['files'].items():
                if '/phase11/m4/' in name:blob=source.read(name)
                else:blob=data.read((c.ROOT/name).relative_to(c.OUT).as_posix())
                assert c.sha(blob)==digest,name
                if any(part in name for part in ('/timing/','/profiles/','/cpu_preflight/','/eligibility/')):
                    assert (c.ROOT/name).read_bytes()==blob,name
    assert cg['native_rebuilds']==0
    root=c.ROOT/cg['artifact_directory']
    raw=c.read(root/'result.json')
    assert raw['ccache_before']==raw['ccache_after']
    assert raw['native_SHA256']==raw['native_SHA256_after']==p['baseline']['native_SHA256']
    with zipfile.ZipFile(c.OUT/'compiled_payload.zip') as z:
        assert z.testzip() is None
        for tag,meta in cg['metadata'].items():
            base=tag.replace(':','/')
            assert c.sha(z.read(base+'/kernel.cubin'))==meta['cubin_SHA256']
            folder=root/base
            assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
            for file in folder.glob('kernel.*'):
                assert z.read(base+'/'+file.name)==file.read_bytes()
            assert meta['ABI_kind']=='descriptor' and meta['original_once']
            ir=(folder/'kernel.ttgir').read_text()
            if meta['output_policy']!='control':
                lines=[line for line in ir.splitlines() if 'ttg.local_load' in line and f'tensor<1x{meta["M"]}x{meta["N"]}xbf16' in line]
                assert lines,tag
                alias=re.search(r'tensor<1x\d+x\d+xbf16,\s*(#\w+)>',lines[0])[1]
                assert encodings(ir)[alias]['fields']==meta['layout'],tag
            resource=ag['resources'][tag]
            assert resource['CUBIN_SHA256']==resource['loaded_CUBIN_SHA256']==meta['cubin_SHA256']
            assert resource['ABI']==['descriptor128']+['u32']*3+['u64']*6
    correct=c.read(c.ROOT/ag['eligibility_path'])
    assert correct['status']=='OK' and not correct['Triton_imported'] and correct['driver_return_codes']==[0]
    assert len(correct['checks'])==3*len(cg['metadata'])
    assert all(row['pass'] and row['tail_canary_pass'] and row['bit_exact'] for row in correct['checks'])
    expected=cells_for(p,ag)
    expected_ids={row['cell_id'] for row in expected}
    samples=0;uuids=[]
    for worker in (1,2,3):
        raw=c.read(c.timing_root(p)/f'invocation_{worker}/raw.json')
        assert raw['status']=='OK' and raw['formal_timing'] and not raw['profiler_attached']
        assert not raw['Triton_imported'] and raw['driver_return_codes']==[0]
        assert raw['protocol_SHA256']==c.sha(c.encode(p))
        assert raw['payload_SHA256']==c.sha((c.OUT/'compiled_payload.zip').read_bytes())
        assert {row['cell_id'] for row in raw['cells']}==expected_ids
        maximum=max(row['grid']*c.stride(cg['metadata'][row['tag']])*4 for row in raw['cells'])
        assert raw['output_allocation_bytes']>=maximum
        for row in raw['cells']:
            assert row['correctness']['pass'] and len(row['samples_us'])==p['timing']['samples_per_cell']
            assert row['samples_us']==[s for visit in row['visits'] for s in visit['samples_us']]
            assert all(sorted(visit['node_types'])==[0,7,7] for visit in row['visits'])
            assert all(value>0 for value in row['samples_us'])
            samples+=len(row['samples_us'])
        uuids.append(raw['environment']['gpu_uuid'])
        for tag,binding in raw['bindings'].items():
            assert binding['loaded_CUBIN_SHA256']==binding['CUBIN_SHA256']==cg['metadata'][tag]['cubin_SHA256']
    assert len(set(uuids))==3
    counts=c.read(c.OUT/'counter_gate.json')
    seen=set()
    for group in ag['profile_groups']:
        for tag in group:
            identity=c.binary_execution_identity(cg['metadata'][tag])
            assert identity not in seen
            seen.add(identity)
    assert seen=={c.binary_execution_identity(meta) for meta in cg['metadata'].values()}
    assert not counts['NCU_duration_used_for_timing']
    for report in counts['reports']:offline_check(report)
    assert len(counts['reports'])==2*len(ag['profile_groups'])
    assert len(counts['single_pass'])==sum(map(len,ag['profile_groups']))
    for row in counts['hardware']:
        assert all(metric in row['metrics'] for metric in ag['profile_metrics'])
    for index in range(len(ag['profile_groups'])):
        folder=c.OUT/f'profiles/group_{index}/attempt_1/export'
        for mode in ('software','hardware'):
            assert 'PROFILE_BINDING' in c.read(folder/mode/'collect.json')['output']
    offline=c.read(c.OUT/'offline_import_validation.json')
    assert len(offline['reports'])==len(counts['reports']) and offline['GPU_allocations']==0
    analysis=c.read(c.OUT/'analysis.json')
    assert analysis['event_samples']==samples and not analysis['historical_offset_used']
    diag=c.read(c.OUT/'diagnostics.json')
    assert len(diag['source_code_audits'])==len(cg['metadata'])
    assert (c.OUT/'REPORT.md').stat().st_size>1000
    result={'status':'OK','stage':c.STAGE,'CPU_only_validator':True,'programs':len(cg['metadata']),
            'new_compiles':cg['new_compiles'],'reused':cg['reused'],'CPU_rejections':len(cg['rejected']),
            'correctness_checks':len(correct['checks']),'formal_event_samples':samples,
            'physical_conditions_per_worker':len(expected),'NCU_reports':len(counts['reports']),
            'local_NCU_reimports':len(counts['reports']),'GPU_UUIDs':uuids,
            'source_PC_opcode_audits':len(diag['source_code_audits']),'native_rebuilds':0,
            'prior_evidence_unchanged':True,'selector_updated':False}
    if args.freeze:
        c.write_once(c.OUT/'validation.json',result)
        files=[path for path in c.OUT.rglob('*') if path.is_file() and path!=c.OUT/'freeze.json']
        files += [path for path in c.SOURCE.glob('*') if path.is_file()]
        c.write_once(c.OUT/'freeze.json',{'stage':c.STAGE,'files':{path.relative_to(c.ROOT).as_posix():c.sha(path.read_bytes()) for path in files}})
    else:
        assert c.read(c.OUT/'validation.json')==result
    frozen=c.read(c.OUT/'freeze.json')
    for name,digest in frozen['files'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    print(c.STAGE,'VALIDATOR OK:',result)


if __name__=='__main__':main()
