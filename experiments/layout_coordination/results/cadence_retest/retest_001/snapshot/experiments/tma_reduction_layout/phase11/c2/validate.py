"""Offline final validation of frozen sources, raw bytes, binaries and curves."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile
from experiments.tma_reduction_layout.phase11.c2 import common as c
from experiments.tma_reduction_layout.phase11.c2.analyze import counter_gate,derived,optimized_input_layout_audit
from experiments.tma_reduction_layout.phase11.c2.report import analyze

DERIVED={'compile_gate.json','artifact_gate.json','counter_gate.json','analysis.json','validation.json','raw_manifest.json','REPORT.md','figure_manifest.json'}


def raw_files(batch):
    root=c.out(batch)
    return {p.relative_to(c.ROOT).as_posix():c.sha(p.read_bytes()) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).parts[0]!='figures' and not (p.parent==root and p.name in DERIVED)}


def validate(batch,freeze=False,committed=False):
    c.protect_prior(batch);protocol=c.load_protocol(batch)
    root=c.out(batch)
    source_root=root/'preregistration'
    if protocol['revision']>1:source_root=source_root/'revisions'/str(protocol['revision'])
    with zipfile.ZipFile(source_root/'source_snapshot.zip') as z:
        for name,digest in protocol['worker_source_SHA256'].items():assert c.sha(z.read(name))==digest,name
    if freeze:c.write_once(root/'raw_manifest.json',{'files':raw_files(batch)})
    manifest=c.read(root/'raw_manifest.json')['files'];assert raw_files(batch)==manifest
    archives=entries=0
    for path in root.rglob('raw_export.zip'):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for member in z.infolist():
                if member.is_dir():continue
                exported=path.parent/'export'/member.filename
                assert exported.is_file() and z.read(member)==exported.read_bytes(),exported
                entries+=1
        archives+=1
    with zipfile.ZipFile(root/'frozen_payload.zip') as z:
        gate=c.read(root/'artifact_gate.json')
        for tag,meta in gate['metadata'].items():
            cubin=z.read(tag.replace(':','/')+'/kernel.cubin')
            assert c.sha(cubin)==meta['cubin_SHA256'],tag
            binding=gate['resources'][tag]
            assert binding['CUBIN_SHA256']==binding['loaded_CUBIN_SHA256']==meta['cubin_SHA256']
            assert binding['attributes']['preferred_shared_carveout']==100
            for q in ('1','2','4'):
                r=binding['reservations'][q]
                if r.get('feasible'):assert r['capacity']==int(q) and r['dynamic_bytes']>=meta['dynamic_smem_bytes']
    old=c.read(root/'counter_gate.json');counter_gate(batch);assert old==c.read(root/'counter_gate.json')
    old=c.read(root/'analysis.json');value=analyze(batch);assert old==json.loads(c.encode(value))
    assert not value['empirical_offset_26_6_used']
    figure=c.read(root/'figure_manifest.json')
    assert figure['analysis_SHA256']==c.sha((root/'analysis.json').read_bytes())
    for name,digest in figure['files'].items():assert c.sha((root/'figures'/name).read_bytes())==digest,name
    if batch=='a':
        original=c.read(c.base.OUT/'stage_c15/counter_gate.json')
        assert original['status']=='OK' and original['report_count']==128
        reused=[m for m in gate['metadata'].values() if 'reused_tag' in m];assert len(reused)==16
        for meta in reused:
            assert c.sha((c.ROOT/meta['reused_directory']/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
        # Verify original physical reports still exist; prior manifest protects all bytes.
        oldreports=list((c.base.OUT/'stage_c1/profiles').glob('attempt_*/export/**/report.ncu-rep'))
        oldreports += list((c.base.OUT/'stage_c15/profiles').glob('attempt_*/export/**/report.ncu-rep'))
        assert len(oldreports)>=128,len(oldreports)
    else:
        assert c.read(c.out('a')/'validation.json')['status']=='OK'
        assert optimized_input_layout_audit(batch)==c.read(root/'audits/optimized_input_layouts.json')
        assert all(set(r['predictor_features'])==set(protocol['analysis']['selector_features']) for r in value['services'])
        scope=c.read(root/'audits/aggregate_counter_scope.json')
        with zipfile.ZipFile(root/'audits/aggregate_invariance_initial_gate.zip') as z:
            original=z.read('counter_gate.json');assert c.sha(original)==scope['initial_gate_SHA256']
            initial=json.loads(original)
        current=c.read(root/'counter_gate.json')
        assert all(r['pass'] for r in current['q_dynamic_opcode_checks'])
        assert all(r['pass'] for r in current['q_achieved_occupancy_checks'])
        assert all(all(r['WF_minus_reported_BC_invariant'].values()) for r in current['q_wavefront_bank_conflict_invariance'])
        old={(r['tag'],r['q']):(r['actual'],r['natural']) for r in initial['q_wavefront_bank_conflict_invariance']}
        new={(r['tag'],r['q']):(r['actual'],r['natural']) for r in current['q_wavefront_bank_conflict_invariance']}
        assert old==new and initial['status']=='FAILED'
    result={'status':'OK','stage':protocol['stage'],'raw_files':len(manifest),'raw_archives':archives,
            'archive_entries_matched':entries,'metadata_entries':len(gate['metadata']),
            'new_event_samples':value['timing_samples_new'],'cells_per_worker':value['cells_per_worker'],
            'source_hashes_match':True,'prior_hashes_match':True,'selector_feature_schema':value['selector_feature_schema'],
            '26_6_offset_used':False,'no_C3_or_PR':True}
    derived(root/'validation.json',result)
    if committed:
        prefix=root.relative_to(c.ROOT).as_posix()
        process=subprocess.Popen(['git','archive','--format=tar','HEAD','--',prefix],cwd=c.ROOT,stdout=subprocess.PIPE)
        seen=set()
        with tarfile.open(fileobj=process.stdout,mode='r|') as stream:
            for member in stream:
                if member.name not in manifest:continue
                assert c.sha(stream.extractfile(member).read())==manifest[member.name],member.name
                seen.add(member.name)
        assert process.wait()==0 and seen==set(manifest)
        print('Committed raw bytes PASS',len(seen))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['a','b'],required=True);p.add_argument('--freeze',action='store_true');p.add_argument('--committed',action='store_true')
    args=p.parse_args();validate(args.batch,args.freeze,args.committed)
