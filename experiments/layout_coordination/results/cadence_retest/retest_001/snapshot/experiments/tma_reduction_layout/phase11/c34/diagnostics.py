"""Before/resource-only/full decompositions, including unsuccessful cases."""
import statistics as st
from experiments.tma_reduction_layout.phase11.c34 import common as c


def candidate_summary(decision, layout):
    item=next(r for r in decision['candidates'] if r['layout']==layout)
    default=next(r for r in decision['candidates'] if r['layout']==decision['upstream_default'])
    scenes=item['resource_scenarios']
    if not scenes:
        return {'layout':layout,'nominal_gain_percent':0.,'worst_gain_percent':0.,
                'q_pairs':[],'is_default':True}
    wi=max(range(len(scenes)),key=lambda i:scenes[i]['predicted_delta_relative']+scenes[i]['uncertainty_upper_relative'])
    old_max=max(max(8.,item['resource']['n_regs_point']+s['common_register_residual']+
                    s['paired_register_residual']) for s in scenes)
    return {'layout':layout,'nominal_gain_percent':item['nominal_gain_percent'],
        'fixed_q_bound_percent':item['nominal_gain_percent']-100*item['uncertainty']['upper_relative'],
        'nominal_performance_upper_pp':100*item['uncertainty']['upper_relative'],
        'q_only_worst_gain_percent':-100*max(s['predicted_delta_relative'] for s in scenes),
        'worst_gain_percent':item['worst_gain_percent'],
        'worst_q_pair':[scenes[wi]['candidate_q'],scenes[wi]['default_q']],
        'worst_performance_upper_pp':100*scenes[wi]['uncertainty_upper_relative'],
        'q_pairs':sorted({(s['candidate_q'],s['default_q']) for s in scenes}),
        'n_regs_point':item['resource']['n_regs_point'],
        'default_n_regs_point':default['resource']['n_regs_point'],
        'resource_n_regs_interval':item['resource']['n_regs_interval'],
        'largest_registers_among_recorded_scenario_witnesses':old_max,
        'new_interval_is_continuous_not_only_witnesses':'margin_common' in item['resource'],
        'margin_common':item['resource'].get('margin_common',8.),
        'margin_paired':item['resource'].get('margin_paired',8.),
        'resource_residual_geometries':item['resource_residual_geometries'],
        'nominal_performance_pool':item['uncertainty'],
        'worst_performance_pool':item['worst_uncertainty'],
        'reasons':item['reasons'],'robust_switch':item['robust_switch']}


def build(write=True):
    dev=c.read(c.OUT/'development.json');old=c.read(c.prior.OUT/'development.json')
    blind=c.read(c.OUT/'D1_predictions.json')['cases']
    rb=c.read(c.OUT/'resource_only_D1_predictions.json')['cases']
    ob=c.read(c.prior.OUT/'D1_predictions.json')['cases']
    targets=[('development',[16,64,8]),('development',[8,64,8]),
             ('D1',[8,64,4]),('D1',[8,128,8]),('D1',[8,512,8])]
    focused=[]
    for split,geo in targets:
        if split=='development':
            name=f'{geo[0]}x{geo[1]}w{geo[2]}'
            before=next(r['decision'] for r in old['outer_decisions'] if r['held']==name)
            resource=next(r['decision'] for r in dev['resource_only_outer'] if r['held']==name)
            after=next(r['decision'] for r in dev['outer'] if r['held']==name)
        else:
            before=next(r['prediction'] for r in ob if r['geometry']==geo)
            resource=next(r['prediction'] for r in rb if r['geometry']==geo)
            after=next(r['prediction'] for r in blind if r['geometry']==geo)
        old_nominal=before['nominal_choice']
        focused.append({'split':split,'geometry':geo,'anchor_layout':old_nominal,
            'before':candidate_summary(before,old_nominal),
            'resource_only':candidate_summary(resource,old_nominal),
            'full_same_layout':candidate_summary(after,old_nominal),
            'full_new_nominal':candidate_summary(after,after['nominal_choice']),
            'before_selected':before['selected'],'resource_only_selected':resource['selected'],
            'full_selected':after['selected'],'D1_truth_available':False if split=='D1' else None})
    cv=c.read(c.OUT/'cross_validation.json');profiles=[]
    for fold in cv['full:auto']['folds']:
        scores=fold['inner_scores']
        best={}
        for key,allowed in (('scalar',('full_structural','partial_overlap')),
                            ('q_dependent',('full_structural','partial_overlap_q'))):
            values=[r for r in scores if r['family'] in allowed]
            best[key]=min(values,key=lambda r:r['inner_relative_loss'])
        profiles.append({'held':fold['held'],'best_inner':best,
            'selected':fold['selected_recipe']})
    coverage=[]
    for item in dev['outer']:
        pairs=[r for r in cv['full:auto']['paired_outer_residuals'] if r['geometry']==item['held']]
        values={}
        for row in pairs: values.setdefault(row['layout'],[]).append(row['relative_residual'])
        for case in item['decision']['candidates']:
            if case['uncertainty'] is None: continue
            residual=st.median(values[case['layout']]);upper=case['uncertainty']['upper_relative']
            coverage.append({'held':item['held'],'layout':case['layout'],
                'observed_relative_residual':residual,'nominal_q_upper_relative':upper,
                'covered':residual<=upper,
                'calibration_geometries':case['uncertainty']['training_geometries']})
    result={'focus_cases':focused,'eta_inner_profiles':profiles,
        'performance_upper_coverage':coverage,
        'performance_upper_covered':sum(r['covered'] for r in coverage),
        'performance_upper_total':len(coverage),
        'performance_rule_unchanged_from_C33':True,
        'no_case_outcomes_used_to_select_resource_margins':True}
    legacy=c.read(c.prior.OUT/'cross_validation.json')
    reproduction=[]
    for key,record in cv.items():
        if record['group']!='auto':continue
        old=legacy[record['mode']]
        differences=[]
        for fold in record['folds']:
            previous=next(f for f in old['folds'] if f['held']==fold['held'])
            index={(s['family'],s['eta'],s['ridge']):s for s in previous['inner_scores']}
            for s in fold['inner_scores']:
                if s['family']=='partial_overlap_q':continue
                p=index[s['family'],s['eta'],s['ridge']]
                differences.append(abs(p['inner_relative_loss']-s['inner_relative_loss']))
        reproduction.append({'mode':record['mode'],'scalar_scores_compared':len(differences),
            'max_absolute_inner_loss_difference':max(differences)})
    result['C33_scalar_score_reproduction']=reproduction
    if write:c.derived(c.OUT/'diagnostics.json',result)
    return result


if __name__=='__main__':build()
