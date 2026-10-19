"""CPU-only, write-once policy/endpoint binding before any D1 compilation."""
import subprocess
from datetime import datetime, timezone
from experiments.tma_reduction_layout.phase11.d1 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream, plan

MODELS = {
    'C34_resource_only':('stage_c34/resource_only_D1_predictions.json','primary_outer_CV_selected'),
    'C34_full_auto':('stage_c34/D1_predictions.json','automated_inner_CV_comparator'),
    'C33':('stage_c33/D1_predictions.json','historical_comparator'),
}


def main():
    dest=c.OUT/'preregistration'
    assert not dest.exists(), 'Preregistration is write-once'
    old=c.prior.protocol()
    models={};policies={};split=None
    for label,(name,role) in MODELS.items():
        path=c.BASE/'results/phase11'/name;blob=path.read_bytes();value=c.read(path)
        assert value['D1_compiled'] is False and value['D1_timed'] is False
        rows=value['cases']
        current=[{k:r[k] for k in ('geometry','stratum','history_category')} for r in rows]
        if split is None:split=current
        assert current==split
        assert all(r['history_category']=='strict_zero_mention' for r in rows)
        c.write_once(dest/'bindings'/label/'predictions.json',blob)
        models[label]={'role':role,'original_path':str(path.relative_to(c.ROOT)),
            'SHA256':c.sha(blob),'bound_copy':f'bindings/{label}/predictions.json',
            'frozen_switches':sum(r['prediction']['changed'] for r in rows)}
        for suffix,key in (('conservative','selected'),('nominal','nominal_choice')):
            policies[label+'_'+suffix]=[{'geometry':r['geometry'],'layout':r['prediction'][key]} for r in rows]
    assert len(split)==10 and sum(r['stratum']=='empirically_unseen_core' for r in split)==5
    cases=[]
    for row in split:
        m,n,w=row['geometry'];pool=candidates(m,n,w);default=upstream(m,n,w)
        for name in ('upstream_default','fixed_vec1','fixed_vec2','fixed_vec4','fanout_only'):
            if name=='upstream_default':choice=default
            elif name.startswith('fixed_vec'):
                possible=[r for r in pool if r['placement']=='n_first' and r['register_layout']['size_per_thread'][2]==int(name[-1])]
                choice=possible[0]['id'] if possible else default
            else:
                choice=min(pool,key=lambda r:(plan(r)['u']*plan(r)['w'],
                    -r['register_layout']['size_per_thread'][2],r['placement']!='n_first'))['id']
            policies.setdefault(name,[]).append({'geometry':row['geometry'],'layout':choice})
        for layout in pool:
            cases.append({'M':m,'N':n,'num_warps':w,'D':1,'kind':'composition',
                'dtype':'bfloat16','path':'host','input_kind':'bf16_matrix',
                'layout':layout['register_layout'],'layout_id':layout['id'],
                'placement':layout['placement'],'vec':layout['register_layout']['size_per_thread'][2],
                'shared_layout':layout['shared_layout'],'family':'ONE_FULL','tag':layout['id']+':ONE_FULL',
                'original_once':True,'one_pass':True,'ABI_kind':'descriptor','global_rows':24576})
        cases.append({'M':m,'N':n,'num_warps':w,'D':1,'kind':'composition','dtype':'bfloat16',
            'path':'host','input_kind':'bf16_matrix','shared_layout':pool[0]['shared_layout'],
            'family':'ONE_TRANSFER','tag':f'M{m}_N{n}_w{w}:control:ONE_TRANSFER',
            'original_once':True,'one_pass':True,'ABI_kind':'descriptor','global_rows':24576})
    assert len(cases)==69 and sum(r['family']=='ONE_FULL' for r in cases)==59
    assert len(policies)==11 and models['C34_resource_only']['frozen_switches']==3
    assert models['C34_full_auto']['frozen_switches']==2
    for choices in policies.values():
        for r in choices:assert r['layout'] in [v['id'] for v in candidates(*r['geometry'])]
    names=subprocess.check_output(['git','ls-files','-z','experiments/tma_reduction_layout'],cwd=c.ROOT).decode().split('\0')
    manifest={name:c.sha((c.ROOT/name).read_bytes()) for name in names if name and
              '/phase11/d1/' not in name and '/stage_d1/' not in name}
    c.write_once(dest/'prior_manifest.json',{'files':manifest})
    c.write_once(dest/'reuse.zip',c.archive(dest/'empty_reuse'))
    target,hashes=c.snapshot()
    protocol={'stage':'D1','revision':1,'frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'authorization':'User explicitly approved CPU protocol binding then all ten D1 H100 shapes, validator, commit and push; stop for review.',
        'pinned_upstream':c.PIN,'baseline':old['baseline'],'profile':'chenmiaoming',
        'target':old['target'],'D1_cases':split,'cases':cases,'models':models,'policies':policies,
        'strata':{'core':[r['geometry'] for r in split if r['stratum']=='empirically_unseen_core'],
                  'boundary':[r['geometry'] for r in split if r['stratum']=='legacy_boundary'],
                  'all':[r['geometry'] for r in split]},
        'endpoint':{'primary':'natural-q loop-free FULL 128/256/384 MiB time-versus-grid/SM OLS slope, ns/tile/SM',
            'auxiliary':'128 MiB raw event time, us, reported separately',
            'practical_threshold_percent':3,'case_aggregate':'median of three worker-paired gain/regret ratios',
            'stratum_aggregate':'equal-weight arithmetic case mean','descriptive_worker_CI':'mean +/- 4.303 * sample_stdev / sqrt(3)',
            'post_test_model_promotion':False,'outcome_based_repeats':False},
        'development_only_primary_selection':{'FULL_relative_RMSE_percent':[7.45,7.71],
            'paired_ratio_RMSE_pp':[8.27,9.05],'upper_tail_coverage':['49/52','42/52'],
            'conservative_mean_gain_percent':[20.20,18.27],
            'primary_switched_gain_le_3_percent_count':0,
            'scope':'Nine outer CV geometries used to select primary; not independent confirmation.'},
        'timing':old['timing'],'correctness':old['correctness'],'profiles':old['profiles'],
        'global_rows':24576,'MC_K':[1],'fit_K':[],'native_rebuilds':0,
        'worker_source_SHA256':hashes,'plan_SHA256':c.sha((c.SOURCE/'PLAN.md').read_bytes()),
        'prior_manifest_SHA256':c.sha((dest/'prior_manifest.json').read_bytes()),
        'C2d_protocol_SHA256':c.sha(c.encode(old)),'optional_low_q_diagnostic':'not performed',
        'historical_26_6_used':False,'selector_runtime_grid_feature':False,
        'retry_policy':{'max_infrastructure_attempts':3,'performance_based_retries':False,
            'budget_error':'stop and report; do not switch profiles'},
        'failure_policy':'Keep all artifacts and spills. Executable spills timed. Unavailable candidates and frozen selections are reported as failures without substitution.',
        'coverage':'Frozen precompile predictions only; no conditioning on measured n_regs/q; all eligible FULL layouts and nondefault pairs, stratified.',
        'D1_compiled_at_freeze':False,'D1_timed_at_freeze':False}
    c.verify_sources(protocol);c.protect_prior()
    c.write_once(dest/'source_snapshot.zip',c.archive(target))
    c.write_once(dest/'protocol.json',protocol)
    print('D1 CPU preregistration frozen: 69 programs, 11 policies, core5/boundary5; SHA256',c.sha(c.encode(protocol)))


if __name__=='__main__':main()
