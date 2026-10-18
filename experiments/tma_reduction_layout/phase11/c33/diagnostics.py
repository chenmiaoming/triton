"""FULL within-warp counters, transfer q response and false-zero audit."""
from experiments.tma_reduction_layout.phase11.c33 import common as c, fit, resources
from experiments.tma_reduction_layout.phase11.c3.static import candidates
from experiments.tma_reduction_layout.phase11.c32.features import static


def run():
    theta=c.read(c.OUT/'theta_H100.json');fit.BUNDLE={'local_tables':theta['fixed_R_tables']}
    matrix=c.read(c.prior.OUT/'matrix.json');cv=c.read(c.OUT/'cross_validation.json')
    dev=c.read(c.OUT/'development.json')
    audit=[]
    for row in matrix['matrix']:
        if row['geometry'] not in ([32,64,8],[32,128,4],[16,64,8]): continue
        case=next(r for r in candidates(*row['geometry']) if r['id']==row['layout']);g=static(case)
        audit.append({**row,'static_ML_ideal_WF':g['ML']['ideal_wavefronts'],
            'static_ML_conflict_WF':g['ML']['bank_conflicts'],
            'static_ML_actual_WF':g['ML']['wavefronts'],
            'static_MR_shared_WF':g['MR']['shared_load_wavefronts'],
            'static_MR_shuffle':g['MR']['shuffle_warp_instructions'],
            'static_MR_barrier':g['MR']['barrier_warp_instructions'],
            'WHOLE_minus_static_ML_MR_WF':row['shared_load_wavefronts_per_CTA']-
                g['ML']['wavefronts']-g['MR']['shared_load_wavefronts'],
            'remaining_scope':'Whole FULL includes output conversion/setup/wait traffic. Residual WF is not independently timed MO latency.'})
    false_zero=[]
    for r in dev['development_evaluation']:
        if r['geometry'] in ([64,64,8],[8,64,8]): false_zero.append({'scope':'all-development fit',**r})
    for r in dev['outer_evaluation']:
        if r['geometry'] in ([64,64,8],[8,64,8]): false_zero.append({'scope':'held geometry',**r})
    # eta is a fitted response-family parameter, not proof of a physically
    # isolated overlap fraction. Profile scores show sensitivity/identifiability.
    profile=theta['inner_scores']
    physical=c.read(c.prior.OUT/'artifact_gate.json')
    resource_check=[]
    for fold in theta['outer_folds']:
        mm=fold['resource_point_model']
        for tag,meta in physical['metadata'].items():
            g=f'{meta["M"]}x{meta["N"]}w{meta["num_warps"]}'
            if meta['family']!='ONE_FULL' or g!=fold['held']: continue
            case=next(r for r in candidates(meta['M'],meta['N'],meta['num_warps']) if r['id']==meta['layout_id'])
            truth=physical['resources'][tag]
            resource_check.append({'geometry':g,'layout':case['id'],
                'observed_n_regs':truth['attributes']['num_regs'],
                'predicted_n_regs':resources.point(mm,case),
                'observed_q':truth['reservations']['natural']['capacity'],
                'predicted_q':resources.q_for(case,resources.point(mm,case))})
    value={'FULL_counter_audit':audit,'false_zero_cases':false_zero,
        'overlap_profile':profile,'physical_B_identified':False,'physical_eta_identified':False,
        'q_definition':'CUDA occupancy capacity under static/dynamic shared reservation, not a trace asserting exactly q CTAs remain active at every instant.',
        'bank_metric_scope':'NCU whole-kernel dynamic bank-conflict count; small extra counts from non-ML operations are retained. Static ML 64-conflict prediction does not imply whole FULL total exactly 64.',
        'control_q_responses':matrix['controls'],'held_geometry_resource_checks':resource_check,
        'held_geometry_resource_q_exact':sum(r['observed_q']==r['predicted_q'] for r in resource_check)}
    c.derived(c.OUT/'mechanism_diagnostics.json',value)
    print('C3.3 FULL within/false-zero diagnostics recorded')
    return value


if __name__=='__main__': run()
