"""Freeze hypotheses, source snapshot, reused CUBINs and timing endpoints."""
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.m2 import common as c
from experiments.tma_reduction_layout.phase11.m2.static import output_plan
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def main():
    dest=c.OUT/'preregistration'
    assert not dest.exists(),'Write-once preregistration'
    pairs=[(32,64,8,2),(32,128,4,2),(32,64,8,8),(32,128,4,8),(16,256,8,2)]
    cases=[]
    plans=[]
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as archive:
        for m,n,w in sorted({row[:3] for row in pairs}):
            stage='stage_d1' if (m,n,w)==(16,256,8) else 'stage_c2d'
            gate=c.read(c.BASE/'results/phase11'/stage/'compile_gate.json')
            rows=[row for row in pairs if row[:3]==(m,n,w)]
            for _,_,_,vec in rows:
                for placement in ('n_first','within'):
                    case=next(row for row in candidates(m,n,w) if row['placement']==placement and row['register_layout']['size_per_thread'][2]==vec)
                    plans.append(output_plan(case))
                    key=case['id']
                    oldtag=key+':ONE_FULL'
                    oldmeta=gate['metadata'][oldtag]
                    folder=c.ROOT/gate['artifact_directory']/oldtag.replace(':','/')
                    for policy in ('legacy_fixed','fixed','native'):
                        tag=key+':'+policy+':ONE_FULL'
                        if policy=='legacy_fixed':
                            meta={**oldmeta,'tag':tag,'layout_key':key,'output_policy':policy,
                                  'reuse_source':str(folder.relative_to(c.ROOT)),'old_tag':oldtag}
                            for path in sorted(folder.glob('kernel.*')):
                                archive.write(path,tag.replace(':','/')+'/'+path.name)
                            assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
                        else:
                            meta={'M':m,'N':n,'num_warps':w,'D':1,'dtype':'bfloat16',
                                'kind':'composition','path':'host','input_kind':'bf16_matrix',
                                'layout':case['register_layout'],'shared_layout':case['shared_layout'],
                                'layout_key':key,'layout_id':key,'placement':placement,'vec':vec,
                                'family':'ONE_FULL','original_once':True,'one_pass':True,
                                'ABI_kind':'descriptor','global_rows':24576,'tag':tag,'output_policy':policy}
                        cases.append(meta)
            tag=f'M{m}_N{n}_w{w}:control:ONE_TRANSFER'
            oldmeta=gate['metadata'][tag]
            folder=c.ROOT/gate['artifact_directory']/tag.replace(':','/')
            meta={**oldmeta,'reuse_source':str(folder.relative_to(c.ROOT)),
                  'output_policy':'control','layout_key':f'M{m}_N{n}_w{w}:control'}
            for path in sorted(folder.glob('kernel.*')):
                archive.write(path,tag.replace(':','/')+'/'+path.name)
            cases.append(meta)
    assert len(cases)==33 and sum('reuse_source' in row for row in cases)==13
    for row in cases:
        assert row.get('global_rows')==24576 and row['original_once'] and row['ABI_kind']=='descriptor'
    names=subprocess.check_output(['git','ls-files','-z','experiments/tma_reduction_layout'],cwd=c.ROOT).decode().split('\0')
    manifest={name:c.sha((c.ROOT/name).read_bytes()) for name in names if name and
              '/phase11/m2/' not in name and '/stage_m2/' not in name}
    c.write_once(dest/'prior_manifest.json',{'files':manifest})
    c.write_once(dest/'reuse.zip',payload.getvalue())
    c.write_once(dest/'operation_plans.json',{'plans':plans,'timing_data_used':False})
    pinned={}
    for name in ('lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp',
                 'lib/Analysis/Utility.cpp','lib/Tools/LinearLayout.cpp',
                 'lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp'):
        blob=c.git_blob(name)
        pinned[name]=c.sha(blob)
        c.write_once(dest/'compiler_reference'/name,blob)
    target,hashes=c.snapshot()
    old=c.read(c.BASE/'results/phase11/stage_d1/preregistration/protocol.json')
    timing={**old['timing'],'order_seed':20261020}
    profiles={**old['profiles'],'group_limit':12,'q_conditions':['natural','1','4'],
        'metrics':old['profiles']['metrics']+['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum']}
    value={'stage':'M2','revision':1,'frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'authorization':'User approved GPU experiments and additional hypothesis-driven experiments; preserve evidence and stage commits/push; no PR.',
        'pinned_upstream':c.PIN,'baseline':old['baseline'],'profile':'chenmiaoming',
        'target':{'GPU':'H100!','CC':[9,0],'SM_count':132},'cases':cases,
        'pair_geometries_vectors':pairs,'primary_pairs':pairs[:2],'extension_pairs':pairs[2:],
        'timing':timing,'profiles':profiles,'correctness':old['correctness'],
        'primary_endpoint':'FULL time-versus-grid/SM OLS slope at 128/256/384 MiB, ns/tile/SM',
        'primary_contrast':'(within_fixed-NF_fixed)-(within_native-NF_native), same worker and exact q',
        'source_bridge':'Original fixed CUBIN versus matched-source fixed; all source/code/resource differences retained.',
        'causal_scope':'Output-policy interaction, not automatically an isolated MO latency. Inspect upstream and global-store changes.',
        'q_meaning':'driver occupancy capacity, not instantaneous residency',
        'native_rebuilds':0,'selector_refit':False,'D1_new_blind_claim':False,
        'retry_policy':{'max_infrastructure_attempts':3,'performance_based_retries':False,'budget_error':'stop; user switches/funds profile'},
        'worker_source_SHA256':hashes,'compiler_reference_SHA256':pinned,
        'plan_SHA256':c.sha((c.SOURCE/'PLAN.md').read_bytes()),
        'prior_manifest_SHA256':c.sha((dest/'prior_manifest.json').read_bytes()),
        'stop':'Finish M2/report/validator/commit/push; no automatic D2, model promotion or PR.'}
    c.write_once(dest/'source_snapshot.zip',c.archive(target))
    c.write_once(dest/'protocol.json',value)
    c.verify_sources(value)
    c.protect_prior()
    print('M2 protocol frozen: 33 programs; 20 CPU compiles, 13 reused CUBINs; no GPU yet',flush=True)


if __name__=='__main__':main()
