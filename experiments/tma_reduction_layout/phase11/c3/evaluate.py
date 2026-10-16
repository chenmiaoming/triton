"""Development comparisons and frozen D1 predictions; no GPU execution."""
import statistics as st
from experiments.tma_reduction_layout.phase11.c3 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, plan, upstream
from experiments.tma_reduction_layout.phase11.c3.model import select, predict

MODES=('full','Load-blind','Comm-blind','Resource/Envelope-blind')


def evaluate(theta):
    analysis=c.read(c.BASE/'results/phase11/stage_c2b/analysis.json')
    c16=c.read(c.BASE/'results/phase11/stage_c16/analysis.json')
    body={r['tag']:r['median_ns_per_tile_SM'] for r in analysis['aggregate'] if r['q']=='natural'}
    one={};one_workers={}
    for r in analysis['setup_and_K1_services']:
        if r['q']=='natural' and r['K1_ns_per_CTA_SM'] is not None:
            one.setdefault(r['tag'],[]).append(r['K1_ns_per_CTA_SM'])
            one_workers[r['tag'],r['invocation']]=r['K1_ns_per_CTA_SM']
    decisions=[];candidate_rows=[];counterfactuals=[]
    for geometry in ((32,64,8),(32,128,4)):
        cases=candidates(*geometry)
        predictions={mode:select(*geometry,theta,mode) for mode in MODES}
        default=upstream(*geometry)
        measured={case['id']:{'MC_body_ns':body[case['id']+':MC:D1'],
                             'MC_K1_ns':st.median(one[case['id']+':MC:D1'])} for case in cases}
        for case in cases:
            p=predict(case,theta)
            measured_row=measured[case['id']]
            candidate_rows.append({'case':case,'prediction':p,'measured_development':measured_row,
                'body_enclosed':p['onchip_steady_ns'][0]<=measured_row['MC_body_ns']<=p['onchip_steady_ns'][1],
                'K1_enclosed':p['K1_ns'][0]<=measured_row['MC_K1_ns']<=p['K1_ns'][1],
                'calibration_coverage_only':True})
        choices={'upstream_default':default}
        for vec in (1,2,4,8):
            choices['fixed_vec'+str(vec)]=next(x['id'] for x in cases if
                x['placement']=='n_first' and x['register_layout']['size_per_thread'][2]==vec)
        # Fanout-only sees contributor count, not lane/warp distinction, bank
        # traffic or instruction schedule. Stable candidate-order tie-break.
        choices['fanout_only']=min(cases,key=lambda x:plan(x)['u']*plan(x)['w'])['id']
        choices.update({mode:predictions[mode]['choice'] for mode in MODES})
        choices.update({mode+'_point_diagnostic':predictions[mode]['point_choice'] for mode in MODES})
        full={}
        for r in c16['cells']:
            if r['cache']!='warm' or ':FULL' not in r['tag']:continue
            if not r['tag'].startswith(f'M32_N{geometry[1]}_w{geometry[2]}:'):continue
            if r['grid'] != (32768 if geometry[1]==64 else 16384):continue
            lid=r['tag'].replace(':FULL',':n_first')
            full[lid]=r['median_us']
        oracle_body=min(r['MC_body_ns'] for r in measured.values())
        oracle_one=min(r['MC_K1_ns'] for r in measured.values())
        for name,choice in choices.items():
            actual=measured[choice]
            worker_regret=[100*(one_workers[choice+':MC:D1',i]/min(
                one_workers[x['id']+':MC:D1',i] for x in cases)-1) for i in (1,2,3)]
            decisions.append({'geometry':list(geometry),'method':name,'choice':choice,
                'MC_body_regret_percent':100*(actual['MC_body_ns']/oracle_body-1),
                'MC_K1_regret_percent':100*(actual['MC_K1_ns']/oracle_one-1),
                'MC_K1_regret_worker_range_percent':[min(worker_regret),max(worker_regret)],
                'worker_range_is_confidence_interval':False,
                'MC_K1_improvement_vs_default_percent':100*(1-actual['MC_K1_ns']/measured[default]['MC_K1_ns']),
                'original_N_first_FULL_us':full.get(choice),
                'original_N_first_regret_percent':100*(full[choice]/min(full.values())-1) if choice in full and full else None,
                'FULL_scope':'Only eight N-first binaries exist; within not evaluated as complete-kernel oracle.',
                'MC_scope':'MC body/K1 fixture development comparison; not original full-kernel speedup.'})
        # Explicit same-vec ambiguity left by total fanout.
        for v in (2,4,8):
            same=[x for x in cases if x['register_layout']['size_per_thread'][2]==v]
            if len(same)==2:
                counterfactuals.append({'geometry':list(geometry),'vec':v,
                    'fanout_tie':all(plan(x)['u']*plan(x)['w']==plan(same[0])['u']*plan(same[0])['w'] for x in same),
                    'layouts':[{'id':x['id'],'u':plan(x)['u'],'w':plan(x)['w'],
                        'ML_bank_conflicts':plan(x)['ML']['bank_conflicts'],
                        'shuffle':plan(x)['MR']['shuffle_warp_instructions'],
                        **measured[x['id']]} for x in same]})
    result={'decisions':decisions,'candidate_predictions':candidate_rows,
        'fanout_same_vec_diagnostics':counterfactuals,
        'independent_decision_geometries':2,'logical_development_layouts':15,
        'robust_switch_count':sum(r['method']=='full' and r['choice']!=upstream(*r['geometry']) for r in decisions),
        'body_calibration_interval_coverage':sum(r['body_enclosed'] for r in candidate_rows),
        'K1_calibration_interval_coverage':sum(r['K1_enclosed'] for r in candidate_rows),
        'not_held_out_validation':True,
        'full_kernel_data_used_for_parameter_fitting':False}
    c.write(c.OUT/'development.json',result)
    protocol=c.read(c.OUT/'preregistration/protocol.json')
    blind=[select(r['BM'],r['BN'],r['W'],theta) for r in protocol['blind_cases']]
    c.write(c.OUT/'D1_predictions.json',{'stage':'D1 prediction freeze only',
        'GPU_timed':False,'compiled':False,'cases':blind,
        'theta_SHA256':c.digest((c.OUT/'theta_H100.json').read_bytes()),
        'protocol_SHA256':c.digest((c.OUT/'preregistration/protocol.json').read_bytes()),
        'static_plan_revision_SHA256':c.digest((c.OUT/'preregistration/static_plan_revision_1.json').read_bytes()),
        'C16_evaluation_only_SHA256':c.digest((c.BASE/'results/phase11/stage_c16/analysis.json').read_bytes()),
        'interpretation':'Default choices are abstentions, not a prediction that vec8 is optimal. Point choices and all competitor intervals are frozen separately.'})
    return result


if __name__=='__main__':
    value=evaluate(c.read(c.OUT/'theta_H100.json'))
    print('robust switches',value['robust_switch_count'],'/2; body/K1 enclosed',
          value['body_calibration_interval_coverage'],value['K1_calibration_interval_coverage'],'/15')
    for r in value['decisions']:
        if r['method'] in ('upstream_default','fanout_only','full','Load-blind','Comm-blind','Resource/Envelope-blind'):
            print(r['geometry'],r['method'],r['choice'],'MC K1 regret',round(r['MC_K1_regret_percent'],2),'FULL',r['original_N_first_FULL_us'])
