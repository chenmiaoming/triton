"""Freeze complete replacement matrices; prior production evidence stays read-only."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import subprocess
from . import common as c
from .g6_analyze import references

OUT = c.OUT/'stage_g8'
SNAPSHOT = Path('/tmp/operator-gpu-g8-worker-snapshot')


def main():
    base = c.read(c.OUT/'stage_g6/attempt_003/preregistration/protocol.json')
    old2 = c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    old3 = c.read(c.OUT/'stage_g3b/preregistration/protocol_v2.json')
    old4a = c.read(c.OUT/'stage_g4a/preregistration/protocol_v2.json')
    old4b = c.read(c.OUT/'stage_g4b/preregistration/protocol.json')
    groups = []
    def variant(name, pv='default', qk='default', **extra):
        return dict(name=name,pv=pv,qk=qk,**extra)
    def standard(case):
        return [variant('upstream',upstream=True),variant('gluon_anchor')]+[
            variant(x['name'],x['name']) for x in case['layouts']]
    for case in old2['geometries']:
        works = [dict(batch=b,length=8192) for b in (1,8,64)] if case['family']=='C06' else [
            dict(batch=b,length=n) for b in (1,8) for n in (256,2048)]
        groups.append(dict(id='g2_'+case['id'],panel='g2',case=case,form_id=case['id'],
                           variants=standard(case),workloads=works))
    for g in old4a['groups']:
        groups.append(dict(id='g4a_'+g['id'],panel='g4a',case=g['case'],form_id=g['case']['id'],
                           variants=g['variants'],workloads=g['workloads']))
    for g in old4b['groups']:
        case=g['case'];gid=g['id']
        if case['family']=='C02': form=case['id']+'_b8_l2048'
        elif g['binding_batch']==1: form=gid
        else: form=case['id']
        variants=[]
        for prefix,form_id in [('actual',form),('B4',case['id'])]:
            for v in g['variants']:
                variants.append({**v,'name':prefix+':'+v['name'],'form_id':form_id})
        groups.append(dict(id='g4b_'+gid,panel='g4b',case=case,form_id=form,
                           variants=variants,workloads=g['workloads']))
    training=[]
    for case in old3['cases']:
        training.append({**case,'variants':standard(case)+[
            variant(f'vec{v}:n_first:native',f'vec{v}:n_first',native=True) for v in (1,8)]})
    refs=references();refroot=Path('/tmp/operator-g8-reference-payload');assert not refroot.exists()
    reference_records={}
    def reference(key,meta,folder):
        value=copy.deepcopy(meta);value['artifact_directory']='references/'+key.replace(':','_')
        for name,digest in meta['artifact_SHA256'].items():
            blob=(folder/name).read_bytes();assert c.sha(blob)==digest
            c.write_once(refroot/value['artifact_directory']/name,blob)
        reference_records[key]=value
    for key,(root,meta) in refs.items(): reference(key+':U',meta,root/meta['artifact_directory'])
    gate2=c.read(c.OUT/'stage_g2/compile_gate.json')
    for row in gate2['records']:
        if row['stage2']:reference(row['tag'],row,c.ROOT/gate2['artifact_directory']/row['artifact_directory'])
    r3=c.OUT/'stage_g3b/cpu/attempt_002/export'
    for row in c.read(r3/'result.json')['records']:
        if row['role']=='dx':reference(row['tag'],row,r3/row['artifact_directory'])
    names=list(base['worker_source_SHA256'])+[str(f.relative_to(c.ROOT)) for f in c.SOURCE.glob('*.py')]
    hashes=c.snapshot(names,SNAPSHOT)
    cli=c.read(c.OUT/'stage_g3a/preregistration/ncu_cli_check.json')
    profile_selectors=[]
    # Union of the original G2/G3a/G4a NCU questions, linked to this new graph.
    questions={
        'C02_n64_d128':['upstream','gluon_anchor','vec1:n_first','vec2:within','vec4:n_first','vec8:within'],
        'C02_n64_d64':['upstream','gluon_anchor','vec1:n_first','vec2:within','vec4:n_first','vec8:within'],
        'C06_s16_d128':['upstream','gluon_anchor','vec1:n_first','vec2:within','vec8:within'],
        'C06_s32_d256':['upstream','gluon_anchor','vec1:n_first','vec2:within','vec4:within','vec8:within']}
    for g in groups:
        if g['panel']=='g2':
            names=questions.get(g['case']['id'],['upstream','gluon_anchor','vec1:n_first','vec2:within','vec8:within'])
            profile_selectors.append(dict(group=g['id'],names=names,modes=['software','warm_application','cold_isolated']))
        elif g['panel']=='g4a':
            profile_selectors.append(dict(group=g['id'],names=[v['name'] for v in g['variants']],modes=['software','warm_application']))
    for case in training:
        profile_selectors.append(dict(group=case['id'],names=[v['name'] for v in case['variants']],modes=['software','warm_application']))
    p=dict(stage='g8',frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        compiler=base['compiler'],native_SHA256=c.NATIVE,backend_SHA256=base['backend_SHA256'],
        profile='chenmiaoming',requested_gpu='H100!:1',compiler_intervention=False,
        inference_groups=groups,training_cases=training,items={i['case']['id']:i for i in base['cases']},
        reference_records=reference_records,reference_payload_SHA256=c.sha(c.archive(refroot)),
        worker_source_SHA256=hashes,correctness=old2['correctness'],training_correctness=old3['correctness'],
        timing={**old4a['timing'],'seed':20261102},
        profile_selectors=profile_selectors,profile_metrics=cli['selected_metrics'],profile_warmups=8,
        formal_timing='Fresh paired U/G0/GL times. No historical U timings mixed with new candidates. No result-dependent repeats.',
        scope='Replace G2/G3a/G3b/G4a/G4b production-layout contrasts only; independent TMA/A10/survey unchanged.',
        protected_SHA256={str(f.relative_to(c.ROOT)):c.sha(f.read_bytes())
            for stage in ['stage_g1','stage_g2','stage_g3a','stage_g3b','stage_g4a','stage_g4b','stage_g5','stage_g6','stage_g7']
            for f in sorted((c.OUT/stage).rglob('*')) if f.is_file()})
    c.write_once(OUT/'preregistration/reference_payload.zip',c.archive(refroot))
    c.write_once(OUT/'preregistration/source_snapshot.zip',c.archive(SNAPSHOT))
    c.write_once(OUT/'preregistration/protocol.json',p)
    logical=sum(len(g['variants'])*len(g['workloads'])*(2 if g['case']['family']=='C02' else 1) for g in groups)
    logical+=sum(len(g['variants'])*(1+len(g['pipeline_M'])) for g in training)
    print('G8 frozen:',len(groups),'inference groups;',len(training),'training geometries;',logical,'logical conditions')


if __name__=='__main__':main()
