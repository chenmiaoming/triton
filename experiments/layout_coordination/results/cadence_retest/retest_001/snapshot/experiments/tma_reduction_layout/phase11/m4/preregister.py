"""Bind M4a/M4b programs and metrics before their compilation or GPU runs."""
from datetime import datetime, timezone
import io
import subprocess
import zipfile
from . import common as c
from .static import GEOMETRIES
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def main():
    dest=c.OUT/'preregistration'
    assert not dest.exists()
    gates={name:c.read(c.BASE/f'results/phase11/stage_{name}/compile_gate.json') for name in ('m2','m3')}
    parent=c.read(c.BASE/'results/phase11/stage_m3/preregistration/protocol.json')
    cases=[]
    stream=io.BytesIO()
    compiled_fields=('cubin_SHA256','function_name','dynamic_smem_bytes','global_scratch_size',
                     'tensordesc_meta','compile_cache_key','reuse_source','old_tag')
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        def reuse(gate,tag,newtag,**updates):
            meta=gate['metadata'][tag]
            folder=c.ROOT/gate['artifact_directory']/tag.replace(':','/')
            for path in sorted(folder.glob('kernel.*')):
                archive.write(path,newtag.replace(':','/')+'/'+path.name)
            assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
            return {**meta,'tag':newtag,'reuse_source':folder.relative_to(c.ROOT).as_posix(),
                    'profile_only':False,**updates}
        def fresh(case,policy,swizzle=128,compile_source='m2_full_output'):
            return {'tag':case['id']+f':sw{swizzle}:'+policy+':ONE_FULL',
                'M':case['BM'],'N':case['BN'],'num_warps':case['W'],'dtype':'bfloat16',
                'shared_layout':{**case['shared_layout'],'swizzle_byte_width':swizzle},
                'layout':case['register_layout'],'layout_key':case['id'],'layout_id':case['id'],
                'vec':case['register_layout']['size_per_thread'][2],'placement':case['placement'],
                'output_policy':policy,'family':'ONE_FULL','kind':'composition','D':1,
                'global_rows':24576,'path':'host','ABI_kind':'descriptor','one_pass':True,
                'original_once':True,'compile_source':compile_source,'profile_only':False}
        if c.STAGE=='m4a':
            for m,n,w in GEOMETRIES:
                for case in candidates(m,n,w):
                    if case['register_layout']['size_per_thread'][2]!=2:continue
                    oldtag=case['id']+':native:ONE_FULL'
                    for sw in (128,64,32,0):
                        row=fresh(case,'native',sw)
                        if sw==128:
                            row=reuse(gates['m2'],oldtag,row['tag'],compile_source='m2_full_output')
                        cases.append(row)
                tag=f'M{m}_N{n}_w{w}:control:ONE_TRANSFER'
                control=gates['m2']['metadata'][tag]
                for sw in (128,64,32,0):
                    newtag=f'M{m}_N{n}_w{w}:control:sw{sw}:ONE_TRANSFER'
                    if sw==128:row=reuse(gates['m2'],tag,newtag)
                    else:
                        row={k:v for k,v in control.items() if k not in compiled_fields}
                        row.update(tag=newtag,shared_layout={**control['shared_layout'],'swizzle_byte_width':sw},
                                   compile_source='c16_transfer' if control['function_name']=='transfer' else 'c2c_transfer')
                    cases.append(row)
            assert len(cases)==36
        else:
            assert c.read(c.BASE/'results/phase11/stage_m4a/validation.json')['status']=='OK'
            for m,n,w in GEOMETRIES[:2]:
                for case in candidates(m,n,w):
                    for policy in ('native','broadcast','centered'):
                        oldtag=case['id']+':'+policy+':ONE_FULL'
                        found=next((gate for gate in (gates['m3'],gates['m2']) if oldtag in gate['metadata']),None)
                        row=fresh(case,policy,compile_source='dense_output' if policy=='centered' else 'm3_full_output' if policy=='broadcast' else 'm2_full_output')
                        if found:
                            row=reuse(found,oldtag,row['tag'])
                        cases.append(row)
                tag=f'M{m}_N{n}_w{w}:control:ONE_TRANSFER'
                cases.append(reuse(gates['m3'],tag,tag))
                default=next(case for case in candidates(m,n,w) if case['id']==upstream(m,n,w))
                row=fresh(default,'copy',compile_source='dense_output')
                row.update(tag=f'M{m}_N{n}_w{w}:copy_control:ONE_COPY',family='ONE_COPY')
                cases.append(row)
            assert len(cases)==49
    for row in cases:
        row.pop('resource_only',None)
        row['profile_only']=False
    c.write_once(dest/'reuse.zip',stream.getvalue())
    target,hashes=c.snapshot()
    c.write_once(dest/'source_snapshot.zip',c.archive(target))
    stall=['smsp__warp_issue_stalled_'+reason+'_per_warp_active.pct' for reason in
           ('barrier','short_scoreboard','long_scoreboard','mio_throttle','math_pipe_throttle','not_selected')]
    stall+=['smsp__issue_active.avg.pct_of_peak_sustained_active',
            'smsp__warps_eligible.avg.per_cycle_active']
    timing={**parent['timing'],'order_seed':20261021,
            'q_conditions':['natural','1'] if c.STAGE=='m4a' else parent['timing']['q_conditions']}
    profiles={**parent['profiles'],'group_limit':9 if c.STAGE=='m4a' else 10,
              'q_conditions':['natural','1','2','4'] if c.STAGE=='m4a' else ['natural'],
              'optional_metrics':stall,'mandatory_metrics':parent['profiles']['metrics']}
    p={**{key:parent[key] for key in ('baseline','target','correctness','retry_policy')},
       'stage':c.STAGE.upper(),'pinned_upstream':c.PIN,'profile':'chenmiaoming',
       'frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
       'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
       'cases':cases,'timing':timing,'profiles':profiles,'native_rebuilds':0,
       'pair_geometries_vectors':[(m,n,w,2) for m,n,w in (GEOMETRIES if c.STAGE=='m4a' else GEOMETRIES[:2])],
       'worker_source_SHA256':hashes,'selector_refit':False,'new_blind_claim':False,
       'operation_plan_SHA256':c.sha((c.BASE/'results/phase11/stage_m4_preparation/operation_plans_v2.json').read_bytes()),
       'authorization':'User approved CPU preparation, M4a and M4b in the supplied recommendation screenshot.',
       'primary_endpoint':'loop-free per-SM tile service slope across 128/256/384 MiB BF16 input; raw 128 MiB us auxiliary',
       'old_q_curves_reused':['stage_m2/analysis.json','stage_m3/analysis.json'],
       'H4_status':'hypothesis, not established throughput-versus-latency decomposition',
       'full_grid_dense_output':True,'outcome_based_repeats':False,
       'CPU_rejections':'retain every rejected compile; only legal/correct binaries enter timing',
       'stop':'Complete authorized M4a/M4b, validators and stage commits/push; stop for review.'}
    c.write_once(dest/'protocol.json',p)
    c.verify_sources(p)
    print(c.STAGE,'preregistered',len(cases),'programs;',sum('reuse_source' in row for row in cases),'reused; no GPU dispatched')


if __name__=='__main__':main()
