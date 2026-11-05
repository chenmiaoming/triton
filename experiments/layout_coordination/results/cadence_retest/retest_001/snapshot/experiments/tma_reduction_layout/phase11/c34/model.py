"""C3.3 performance bound unchanged; calibrated correlated resource rectangle."""
from experiments.tma_reduction_layout.phase11.c34 import common as c, fit, resources
from experiments.tma_reduction_layout.phase11.c33.model import structure, upper_bound
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream
from experiments.tma_reduction_layout.phase11.c32.features import static


def predict(geometry,theta,bank=None,resource_model=None):
    mm=theta['model'];rm=resource_model or theta['resource_model']
    bank=theta['paired_outer_residuals'] if bank is None else bank
    pool=candidates(*geometry);default_id=upstream(*geometry)
    default=next(r for r in pool if r['id']==default_id)
    estimates={r['id']:resources.estimate(r,rm) for r in pool}
    points={r['id']:fit.predict(mm,r,estimates[r['id']]['q_point']) for r in pool}
    dp=points[default_id];results=[]
    for case in pool:
        cp=points[case['id']];rr=estimates[case['id']];s=structure(case,default,cp,dp)
        nominal=100*(1-cp['tile_ns']/dp['tile_ns'])
        if case['id']==default_id:
            results.append({'layout':case['id'],'resource':rr,'prediction':cp,
                'nominal_gain_percent':0.,'worst_gain_percent':0.,'robust_switch':False,
                'uncertainty':None,'structure':s,'resource_scenarios':[]})
            continue
        upper=upper_bound(bank,s,cp['leverage']['inflation']**2)
        raw_scenarios,cal=resources.pair(case,default,rm)
        assert raw_scenarios
        scenarios=[];bounded=[];uncertainties=[];structures=[]
        for raw in raw_scenarios:
            dq,cq=raw['default_q'],raw['candidate_q']
            d=fit.predict(mm,default,dq);v=fit.predict(mm,case,cq)
            ratio=v['tile_ns']/d['tile_ns']-1
            ss=structure(case,default,v,d)
            uu=upper_bound(bank,ss,v['leverage']['inflation']**2)
            bounded.append(ratio+uu['upper_relative'])
            uncertainties.append(uu);structures.append(ss)
            scenarios.append({**raw,'predicted_delta_relative':ratio,
                'uncertainty_upper_relative':uu['upper_relative'],'structure':ss})
        wi=max(range(len(bounded)),key=lambda i:bounded[i])
        worst=-100*bounded[wi]
        reasons=[]
        if not rr['shared_legal'] or not estimates[default_id]['shared_legal']: reasons.append('shared_illegal')
        common,_,_=resources.envelope(rm,case)
        default_high=resources.point(rm,default)+common[1]
        if rr['spill_uncertainty'] or default_high>255: reasons.append('spill_uncertainty')
        if not cal['calibration_target_met']: reasons.append('resource_calibration_target_not_met')
        if rr['new_warp_count'] or estimates[default_id]['new_warp_count']: reasons.append('unseen_warp_count')
        support=theta['performance_support'];r=static(case)['elements_per_thread']
        if r<min(support['R']) or r>max(support['R']): reasons.append('FULL_R_outside_support')
        if case['W'] not in support['W']: reasons.append('FULL_W_outside_support')
        if worst<=c.protocol()['practical_switch_percent']: reasons.append('no_robust_FULL_gain')
        results.append({'layout':case['id'],'resource':rr,'prediction':cp,
            'nominal_gain_percent':nominal,'worst_gain_percent':worst,'structure':s,
            'fixed_q_bound_percent':nominal-100*upper['upper_relative'],
            'q_only_worst_gain_percent':-100*max(s['predicted_delta_relative'] for s in scenarios),
            'uncertainty':upper,'resource_scenarios':scenarios,
            'worst_scenario_index':wi,'worst_uncertainty':uncertainties[wi],
            'worst_structure':structures[wi],
            'resource_residual_geometries':cal['base_envelope']['pool']['geometries'],
            'paired_resource_residual_geometries':cal['base_envelope']['paired_geometries'],
            'resource_calibration_key':resources.key_for(case),
            'robust_switch':not reasons,'reasons':reasons})
    nominal=min(results,key=lambda r:(r['prediction']['tile_ns'],r['layout']!=default_id,r['layout']))
    robust=[r for r in results if r['robust_switch']]
    selected=max(robust,key=lambda r:(r['worst_gain_percent'],r['nominal_gain_percent'])) if robust else next(r for r in results if r['layout']==default_id)
    return {'geometry':list(geometry),'upstream_default':default_id,
        'nominal_choice':nominal['layout'],'nominal_gain_percent':nominal['nominal_gain_percent'],
        'selected':selected['layout'],'selected_worst_gain_percent':selected['worst_gain_percent'],
        'changed':selected['layout']!=default_id,'candidates':results,
        'runtime_grid_feature':False,'D1_compiled':False,'D1_timed':False}
