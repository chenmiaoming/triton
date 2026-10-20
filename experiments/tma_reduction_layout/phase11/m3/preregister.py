"""Bind M2-derived new hypotheses before any M3 compilation or measurement."""
from datetime import datetime,timezone
import io
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.m3 import common as c
from experiments.tma_reduction_layout.phase11.m3.static import plan
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def main():
    dest=c.OUT/'preregistration'
    assert not dest.exists()
    m2=c.read(c.BASE/'results/phase11/stage_m2/compile_gate.json')
    old=c.read(c.BASE/'results/phase11/stage_c2d/compile_gate.json')
    m2diag=c.read(c.BASE/'results/phase11/stage_m2/diagnostics.json')
    m2code={r['tag']:r for r in m2diag['code']}
    pairs=[(m,n,w,v) for m,n,w in [(32,64,8),(32,128,4)] for v in [2,4,8]]
    cases=[];plans=[];predictions=[]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        def reuse(gate,tag,newtag=None):
            meta=gate['metadata'][tag]
            folder=c.ROOT/gate['artifact_directory']/tag.replace(':','/')
            name=newtag or tag
            row={**meta,'tag':name,'reuse_source':str(folder.relative_to(c.ROOT)),
                 'output_policy':meta.get('output_policy','legacy_fixed'),
                 'layout_key':meta.get('layout_key',meta.get('layout_id'))}
            for path in sorted(folder.glob('kernel.*')):
                archive.write(path,name.replace(':','/')+'/'+path.name)
            assert c.sha((folder/'kernel.cubin').read_bytes())==row['cubin_SHA256']
            cases.append(row)
        for m,n,w,v in pairs:
            for placement in ['n_first','within']:
                case=next(x for x in candidates(m,n,w) if x['placement']==placement and x['register_layout']['size_per_thread'][2]==v)
                key=case['id'];plans.append(plan(case))
                for policy in ['fixed','native']:
                    tag=key+':'+policy+':ONE_FULL'
                    if v!=4:reuse(m2,tag)
                    else:
                        template=next(row for row in m2['metadata'].values() if row['M']==m and row['N']==n and row['num_warps']==w and row['output_policy']==policy)
                        row={k:value for k,value in template.items() if k not in
                            ('cubin_SHA256','function_name','dynamic_smem_bytes','global_scratch_size','tensordesc_meta','compile_cache_key','reuse_source','old_tag')}
                        cases.append({**row,'tag':tag,'layout_key':key,'layout_id':key,'vec':v,
                            'placement':placement,'layout':case['register_layout']})
                if v==4:
                    tag=key+':ONE_FULL'
                    reuse(old,tag,key+':legacy_fixed:ONE_FULL')
                    cases[-1]['layout_key']=key
                if placement=='n_first' and v in (2,8):
                    anchor=m2['metadata'][key+':native:ONE_FULL']
                    fixed=m2code[key+':fixed:ONE_FULL']
                    native=m2code[key+':native:ONE_FULL']
                    meta={k:value for k,value in anchor.items() if k not in
                        ('cubin_SHA256','function_name','dynamic_smem_bytes','global_scratch_size','tensordesc_meta','compile_cache_key','reuse_source','old_tag')}
                    tag=key+':broadcast:ONE_FULL'
                    cases.append({**meta,'tag':tag,'output_policy':'broadcast','profile_only':True})
                    expected=fixed['whole_counts']['shared_load']-fixed['dynamic_by_source'].get('MO',{}).get('shared_load',0)
                    predictions.append({'tag':tag,'anchor_native':anchor['tag'],
                        'native_whole_LDS':native['whole_counts']['shared_load'],
                        'predicted_broadcast_whole_LDS':expected,
                        'predicted_ML_loads_unchanged':True,
                        'predicted_MAX_SHFL_unchanged':True,
                        'timing_comparison_permitted':False})
        for m,n,w in [(32,64,8),(32,128,4)]:
            reuse(m2,f'M{m}_N{n}_w{w}:control:ONE_TRANSFER')
    assert len(cases)==34 and sum('reuse_source' in r for r in cases)==22
    assert sum(r.get('profile_only',False) for r in cases)==4
    names=subprocess.check_output(['git','ls-files','-z','experiments/tma_reduction_layout'],cwd=c.ROOT).decode().split('\0')
    manifest={name:c.sha((c.ROOT/name).read_bytes()) for name in names if name and '/phase11/m3/' not in name and '/stage_m3/' not in name}
    c.write_once(dest/'prior_manifest.json',{'files':manifest})
    c.write_once(dest/'reuse.zip',buffer.getvalue())
    c.write_once(dest/'static_predictions.json',{'layouts':plans,'broadcast':predictions,
        'predicted_before_M3_compile':True,'training_or_timing_used':'M2 motivates hypotheses; no M3 observations'})
    target,hashes=c.snapshot()
    parent=c.read(c.BASE/'results/phase11/stage_m2/preregistration/protocol.json')
    value={**{k:parent[k] for k in ('baseline','target','timing','profiles','correctness')},
        'stage':'M3','revision':1,'frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'authorization':'User approved additional GPU hypothesis generation and verification after M2.',
        'pinned_upstream':c.PIN,'profile':'chenmiaoming','cases':cases,
        'pair_geometries_vectors':pairs,'native_rebuilds':0,'selector_refit':False,'new_blind_claim':False,
        'primary_endpoint':parent['primary_endpoint'],
        'profile_only_consumer':'max(column)+logical_row, BMxBN outputs/CTA; grid=1 counts only; excluded from FULL timing',
        'worker_source_SHA256':hashes,'plan_SHA256':c.sha((c.SOURCE/'PLAN.md').read_bytes()),
        'prior_manifest_SHA256':c.sha((dest/'prior_manifest.json').read_bytes()),
        'predictions_SHA256':c.sha((dest/'static_predictions.json').read_bytes()),
        'retry_policy':parent['retry_policy'],
        'stop':'Finish M3/report/validator/commit/push and stop; no further automatic stage, D2 or PR.'}
    c.write_once(dest/'source_snapshot.zip',c.archive(target))
    c.write_once(dest/'protocol.json',value)
    c.verify_sources(value);c.protect_prior()
    print('M3 preregistration frozen: 34 programs, 12 new/22 reused; four diagnostic consumers excluded from timing',flush=True)


if __name__=='__main__':main()
