"""Freeze the complete development matrix and revised D1 split before compile."""
import argparse
import shutil
import subprocess
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates

GEOMETRIES = ((8,64,8),(8,128,4),(8,256,4),(16,64,4),(16,128,4),
              (32,64,8),(32,128,4),(64,64,8),(16,64,8))
SHARED = {'swizzle_byte_width':128,'element_bitwidth':16,'rank':3,'transposed':False}


def initial():
    old = c.read(c.BASE/'results/phase11/stage_c2c/compile_gate.json')
    history = c.read(c.BASE/'results/phase11/stage_c32/D1_predictions.json')
    kept = [row for row in history['cases'] if row['geometry'] != [16,64,8]]
    assert len(kept) == 10 and all(row['history_category']=='strict_zero_mention' for row in kept)
    split = {'moved_to_development':[16,64,8], 'reason':'Explicit user authorization; code-only history, repair R4 full-geometry calibration hole.',
        'historical_source':'stage_c32/D1_predictions.json',
        'historical_SHA256':c.sha((c.BASE/'results/phase11/stage_c32/D1_predictions.json').read_bytes()),
        'reserved_geometries':[row['geometry'] for row in kept],
        'cases':[{key:row[key] for key in ('geometry','stratum','history_category')} for row in kept],
        'D1_compiled':False,'D1_timed':False}
    dest = c.OUT/'preregistration'
    c.write_once(dest/'D1_split.json',split)
    cases = []
    for m,n,w in GEOMETRIES:
        for layout in candidates(m,n,w):
            meta = {'M':m,'N':n,'num_warps':w,'D':1,'kind':'composition','dtype':'bfloat16',
                'path':'host','input_kind':'bf16_matrix','layout':layout['register_layout'],
                'layout_id':layout['id'],'placement':layout['placement'],
                'vec':layout['register_layout']['size_per_thread'][2],'shared_layout':SHARED}
            for family in (('ONE_FULL','MC') if (m,n,w)==(16,64,8) else ('ONE_FULL',)):
                previous = next((item for item in old['metadata'].values()
                    if item.get('layout_id')==layout['id'] and item.get('original_once')
                    and item['family'] in ('ONE_FULL','ORIGINAL_FULL')),None) if family=='ONE_FULL' else None
                item = {**(previous or {}),**meta,'family':family,'tag':layout['id']+':'+family,
                    'original_once':family=='ONE_FULL','one_pass':family=='ONE_FULL',
                    'ABI_kind':'descriptor' if family=='ONE_FULL' else 'descriptor_K','global_rows':24576}
                if previous:
                    item['reuse_source']=str((c.ROOT/old['artifact_directory']/previous['tag'].replace(':','/')).relative_to(c.ROOT))
                cases.append(item)
        previous = next((item for item in old['metadata'].values() if
            (item['M'],item['N'],item['num_warps'])==(m,n,w) and
            item['family'] in ('ONE_TRANSFER','ORIGINAL_TRANSFER')),None)
        item = {**(previous or {}),'M':m,'N':n,'num_warps':w,'D':1,'kind':'composition',
            'dtype':'bfloat16','path':'host','ABI_kind':'descriptor','original_once':True,'one_pass':True,
            'input_kind':'bf16_matrix','shared_layout':SHARED,'family':'ONE_TRANSFER',
            'tag':f'M{m}_N{n}_w{w}:control:ONE_TRANSFER','global_rows':24576}
        if previous:
            item['reuse_source']=str((c.ROOT/old['artifact_directory']/previous['tag'].replace(':','/')).relative_to(c.ROOT))
        cases.append(item)
    assert len(cases)==76 and sum(bool(row.get('reuse_source')) for row in cases)==45
    reuse = dest/'reuse'
    for meta in cases:
        if not meta.get('reuse_source'):
            continue
        target = reuse/meta['tag'].replace(':','/')
        target.mkdir(parents=True)
        for path in (c.ROOT/meta['reuse_source']).glob('kernel.*'):
            shutil.copyfile(path,target/path.name)
        assert c.sha((target/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
    c.write_once(dest/'reuse.zip',c.archive(reuse))
    c.write_once(dest/'prior_manifest.json',{'files':c.prior_manifest()})
    return {'stage':'C2d','date':'2026-10-18','revision':1,
        'authorization':'User approved C2d and C3.3, separate commits and push; stop before ten protected D1 GPU compiles/timing.',
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'pinned_upstream':c.PIN,'baseline':c.read(c.BASE/'results/phase11/stage_c2a/preregistration/protocol.json')['baseline'],
        'profile':'chenmiaoming','target':{'GPU':'H100!:1','cc':[9,0],'SM_count':132},
        'calibration_geometries':GEOMETRIES,'reserved_D1':split['reserved_geometries'],'cases':cases,
        'MC_K':[1,16,64,256],'fit_K':[16,64,256],
        'timing':{'workers':3,'visits':3,'samples_per_visit':4,'samples_per_cell':12,'warmups':2,
            'q_conditions':['natural','1','2','4','8'],'full_grid_factors':[1,2,3],
            'MC_grid_SM_multipliers':[32,64,96],'order_seed':20261018,
            'outcome_based_repeats':False,'single_kernel_graph':True},
        'correctness':{'patterns':['quarter','random','boundary']},
        'profiles':{'group_limit':12,'MC_software_K':[16,64],'q_conditions':['natural','4'],
            'metrics':c.read(c.BASE/'results/phase11/stage_c2c/preregistration/protocol.json')['profiles']['metrics'],
            'duration_used_for_timing':False},
        'global_rows':24576,'native_rebuilds':0,'D1_compiled':False,'D1_timed':False,
        'historical_26_6_used':False,'selector_runtime_grid_feature':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision',type=int,default=1)
    parser.add_argument('--reason',default='Initial authorized C2d calibration')
    args=parser.parse_args()
    value=initial() if args.revision==1 else c.protocol()
    if args.revision>1:
        value={**value,'revision':args.revision,'previous_protocol_SHA256':c.sha(c.encode(value)),
               'technical_reason':args.reason}
    _,hashes=c.snapshot()
    value['worker_source_SHA256']=hashes
    value['plan_SHA256']=c.sha((c.SOURCE/'PLAN.md').read_bytes())
    value['C33_method_SHA256']=c.sha((c.SOURCE.parent/'c33/PLAN.md').read_bytes())
    c.verify_sources(value)
    dest=c.OUT/'preregistration'
    if args.revision>1:
        dest=dest/'revisions'/str(args.revision)
    c.write_once(dest/'protocol.json',value)
    c.write_once(dest/'source_snapshot.zip',c.archive(c.snapshot()[0]))
    print('C2d frozen',len(value['cases']),'programs;',len(value['reserved_D1']),'protected D1; revision',args.revision)


if __name__=='__main__':
    main()
