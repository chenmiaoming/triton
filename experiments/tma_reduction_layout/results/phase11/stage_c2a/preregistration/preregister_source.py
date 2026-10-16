"""Freeze authorized C2 batches before compilation or GPU execution."""
import argparse
from pathlib import Path
import shutil
import subprocess
from experiments.tma_reduction_layout.phase11.c2 import common as c


def cases_a():
    layouts=c.read(c.base.SOURCE/'followup_review_matrix.json')['unique_logical_layouts']
    old=c.read(c.base.OUT/'stage_c15/artifact_gate.json')['metadata']
    cases=[]
    for layout in layouts:
        lid=layout['layout_id']
        features={'M':layout['BM'],'N':layout['BN'],'num_warps':layout['num_warps'],
                  'vec':layout['vec'],'placement':layout['placement'],
                  'layout':layout['ownership']['gluon_blocked_layout'],'layout_id':lid}
        for family,d in [('ML',1),('ML',2),('MR',1),('MR',2),('MO',1),('MC',1)]:
            meta={**features,'tag':f'{lid}:{family}:D{d}','family':family,'D':d,
                  'dtype':'bfloat16','path':'host','ABI_kind':'pointer' if family=='MO' else 'descriptor_K',
                  'input_kind':'result_f32' if family=='MO' else 'bf16_matrix',
                  'shared_layout':{'swizzle_byte_width':128,'element_bitwidth':16,'rank':3,'transposed':False}}
            if layout['placement']=='n_first' and family in ('ML','MR') and d==1:
                tag=f'M32_N{meta["N"]}_w{meta["num_warps"]}:vec{meta["vec"]}:{family}'
                meta.update({**old[tag],**meta,'reused_tag':tag,
                             'reused_directory':f'experiments/tma_reduction_layout/results/phase11/stage_c15/artifacts/attempt_1/export/{tag.replace(":","/")}'})
            cases.append(meta)
    assert len(cases)==90 and sum('reused_tag' in m for m in cases)==16
    return cases


def cases_b():
    cases=[]
    for family,values in [('P',[1,2,4,8,16,32,64]),('U',[1,2,4,8,16,32]),('W',[1,2,4,8]),('E',[2,8,32])]:
        for value in values:
            m,n,w=value,128,4
            s,t,a=[1,value,1],[1,1,32],[1,1,4]
            if family=='U': s,t=[1,1,1],[1,value,32//value]
            if family=='W': n,w,s,t,a=256,8,[1,1,1],[1,1,32],[1,value,8//value]
            for d in (1,2):
                cases.append({'tag':f'probe_{family}{value}:D{d}','family':family,'parameter':value,'D':d,
                    'M':m,'N':n,'num_warps':w,'dtype':'bfloat16','path':'direct','ABI_kind':'pointer_K',
                    'input_kind':'bf16_matrix','layout':{'size_per_thread':s,'threads_per_warp':t,'warps_per_cta':a,'order':[2,1,0]}})
    assert len(cases)==40
    return cases


def freeze_prior():
    # Freeze committed evidence and stage-A inputs, never unrelated untracked work.
    names=subprocess.check_output(['git','ls-files','experiments/tma_reduction_layout/results','experiments/tma_reduction_layout/phase11'],cwd=c.ROOT,text=True).splitlines()
    return {name:c.sha((c.ROOT/name).read_bytes()) for name in names if not name.startswith('experiments/tma_reduction_layout/phase11/c2/')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['a','b'],required=True)
    p.add_argument('--revision',type=int,default=1);p.add_argument('--reason',default='Initial authorized C2')
    args=p.parse_args();batch=args.batch
    snap,hashes=c.snapshot(batch)
    if args.revision>1:
        old=c.load_protocol(batch)
        protocol={**old,'revision':args.revision,'technical_revision_reason':args.reason,
                  'previous_protocol_SHA256':c.sha(c.encode(old)),'worker_source_SHA256':hashes}
        dest=c.out(batch)/'preregistration/revisions'/str(args.revision)
    else:
        prior=c.prior.load_protocol()
        cases=cases_a() if batch=='a' else cases_b()
        protocol={'stage':'C2'+batch,'date':'2026-10-16','revision':1,
            'authorization':'User formally approved C2a full matrix and C2b parameter/residency probes; commit per batch, no C3 or PR',
            'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
            'baseline':prior['baseline'],'profile':'chenmiaoming','target':{'GPU':'H100!:1','cc':[9,0],'SM_count':132},
            'cases':cases,'logical_programs':len(cases),'new_programs':74 if batch=='a' else 40,
            'K':[1,16,64,256],'fit_K':[16,64,256],'profile_K':[16,256],
            'correctness':{'patterns':['quarter','random','boundary'],'complete_output_grid':2},
            'timing':{'workers':3,'samples_per_cell':20,'visits':4,'samples_per_visit':5,'warmups':3,
                      'grid_SM_multipliers':[32,64,96],'latency_grid':1,'graphs':True,'single_kernel_per_sample':True,
                      'order_seed':20261016,'q':['natural'] if batch=='a' else ['natural','1','2','4'],
                      'q_controls_previous_matrix':batch=='b','outcome_based_repeats':False},
            'profiles':{'hardware_repeats':3,'software_repeats':1,'grid':1,'metrics':[
                'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
                'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
                'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
                'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum','smsp__inst_executed.sum',
                'sm__warps_active.avg.pct_of_peak_sustained_active'],
                'q_occupancy_grid_SM_multiplier':64,'q_hardware_repeats':1,
                'q_target_instructions_must_match_natural':True,'duration_not_used_for_timing':True,
                'replay':'kernel','cache_control':'none','clock_control':'none','group_program_limit':10},
            'reuse_original_reports':128 if batch=='a' else 0,
            'analysis':{'steady_fit':'T(K)=a+bK on K16/64/256; K1 separate',
                'normalization':'b(G)=a_grid+(G/SM)*D*s_tile; us per tile per SM; grid is measurement provenance only',
                'HBM':'Exact C1.6 TMA controls at original full grids, no cropped-cache claim of HBM',
                'MO':'Input load is fixture work; conversion deletion makes standalone MO diagnostic',
                'MC':'Distinct observable output each K; two fixed shared slots; K1 single TMA transfer',
                'empirical_offset_26_6_used':False,'selector_features':['M','N','num_warps','dtype','layout'],
                'q':'unused launch shared reservation, exact CUDA capacity; achieved occupancy from NCU separately',
                'parameter_fixture':'BF16 private updates and loop work remain part of measured combinations; no claim of naked opcode latency'},
            'worker_source_SHA256':hashes,'native_rebuild':False,'stop_after':'C2 calibration report, no C3/D/PR'}
        dest=c.out(batch)/'preregistration'
        c.write_once(dest/'prior_manifest.json',{'files':freeze_prior()})
        reuse=Path('/tmp/tma-phase11-c2'+batch+'-reuse');reuse.mkdir(exist_ok=False)
        origins={}
        if batch=='a':
            for case in cases:
                if 'reused_directory' not in case:continue
                folder=c.ROOT/case['reused_directory'];target=reuse/case['tag'].replace(':','/')
                target.mkdir(parents=True)
                origins[case['tag']]={'directory':case['reused_directory'],'files':{}}
                for path in folder.glob('kernel.*'):
                    shutil.copyfile(path,target/path.name)
                    origins[case['tag']]['files'][path.name]=c.sha(path.read_bytes())
            oldcounter=c.read(c.base.OUT/'stage_c15/counter_gate.json')
            c.write_once(dest/'reused_counter_bindings.json',{'source':'experiments/tma_reduction_layout/results/phase11/stage_c15/counter_gate.json',
                'SHA256':c.sha(c.encode(oldcounter)),'report_count':128,
                'programs':{m['tag']:m['reused_tag'] for m in cases if 'reused_tag' in m}})
        else:
            assert c.read(c.out('a')/'validation.json')['status']=='OK'
        # Independently checked transfer controls are references, not new programs.
        anchors=c.read(c.base.OUT/'stage_c16/artifact_gate.json')['metadata']
        protocol['anchors']={}
        for tag,meta in anchors.items():
            if meta['family']!='TMA':continue
            source=c.base.OUT/'stage_c16/artifacts/attempt_3/export'/tag.replace(':','/')
            target=reuse/tag.replace(':','/');target.mkdir(parents=True)
            for path in source.glob('kernel.*'):shutil.copyfile(path,target/path.name)
            protocol['anchors'][tag]={**meta,'D':1,'ABI_kind':'descriptor','input_kind':'bf16_matrix'}
            origins[tag]={'directory':source.relative_to(c.ROOT).as_posix(),'files':{p.name:c.sha(p.read_bytes()) for p in source.glob('kernel.*')}}
        c.write_once(dest/'reuse_bindings.json',origins)
        c.write_once(dest/'reuse.zip',c.archive(reuse))
    c.write_once(dest/'protocol.json',protocol)
    c.write_once(dest/'source_snapshot.zip',c.archive(snap))
    c.write_once(dest/'preregister_source.py',Path(__file__).read_bytes())
    print(protocol['stage'],'frozen',protocol['logical_programs'],'logical',protocol['new_programs'],'new')


if __name__=='__main__':main()
