"""Static steady-state selection with joint paired residual scenarios."""
from itertools import product
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c31.features import cached_plan
from experiments.tma_reduction_layout.phase11.c31.features import features
from experiments.tma_reduction_layout.phase11.c31.calibrate import predict as point
from experiments.tma_reduction_layout.phase11.c3.static import candidates,upstream,compact
import json


def plan(case):return cached_plan(json.dumps(case,sort_keys=True))


def q_values(case,mode):
    if mode=='Resource/Envelope-blind':return [16]
    lo,hi=plan(case)['resources']['q_interval']
    return list(range(max(1,lo),max(1,hi)+1))


def transfer(case,theta,mode):
    if mode in ('Resource/Envelope-blind','Envelope-only-blind'):
        return [0.,0.],0.,'disabled ablation'
    measured=[r for r in theta['transfer']['measurements'] if
              (r['BM'],r['BN'])==(case['BM'],case['BN'])]
    if measured:
        values=[r['service_ns'] for r in measured]
        controls=[r['service_ns'] for r in measured if r['cohort']=='C1.6']
        return [min(values),max(values)],st.median(controls or values),'independent measured controls'
    byte_count=2*case['BM']*case['BN']+4*case['BN']
    rates=theta['transfer']['ns_per_tile_byte_interval']
    return [byte_count*r for r in rates],byte_count*sum(rates)/2,'byte-rate extrapolation; unmeasured geometry'


def band(qc,qd):
    q=min(qc,qd)
    return 'q1' if q<2 else 'q2' if q<3 else 'q_ge_3'


def joint_errors(theta,mode,group):
    if mode=='Envelope-only-blind':mode='full'
    rows=theta['models'][mode]['paired_error_bands'][group]['records']
    values=np.array([[r['default_residual_ns'],r['delta_residual_ns']] for r in rows])
    # Include the nominal model and reflect joint errors about zero. A missing
    # communication model cannot gain an optimistically one-sided correction
    # merely because its development residuals all have the same sign.
    return np.vstack((np.zeros((1,2)),values,-values)),rows


def compare(candidate,default,theta,mode='full'):
    fitted_mode='full' if mode=='Envelope-only-blind' else mode
    model=theta['models'][fitted_mode]['model'];tables=theta['primitive_shape_tables']
    qs_c=q_values(candidate,mode);qs_d=q_values(default,mode)
    sc={q:point(model,candidate,q,tables) for q in qs_c}
    sd={q:point(model,default,q,tables) for q in qs_d}
    tma,tma_point,scope=transfer(default,theta,mode)
    delta_s=[float('inf'),-float('inf')];delta_t=delta_s.copy();ratio=delta_s.copy()
    extrema={};count=0
    for qc,qd in product(qs_c,qs_d):
        errors,source=joint_errors(theta,mode,band(qc,qd))
        # Shared default error is applied to both. Delta error remains paired.
        d=sd[qd]+errors[:,0]
        c=sc[qc]+errors[:,0]+errors[:,1]
        raw_delta=c-d
        delta_s=[min(delta_s[0],float(raw_delta.min())),max(delta_s[1],float(raw_delta.max()))]
        for roof in sorted(set(tma)):
            td=np.maximum(roof,np.maximum(d,0.));tc=np.maximum(roof,np.maximum(c,0.))
            relative=(tc-td)/np.maximum(td,1e-12)
            change=tc-td;count+=len(errors)
            if float(relative.max())>ratio[1]:
                j=int(np.argmax(relative))
                extrema['worst']={'candidate_q':qc,'default_q':qd,'TMA_ns':roof,
                    'joint_default_error_ns':float(errors[j,0]),'joint_delta_error_ns':float(errors[j,1]),
                    'default_MC_ns_before_clamp':float(d[j]),'candidate_MC_ns_before_clamp':float(c[j]),
                    'default_tile_ns':float(td[j]),'candidate_tile_ns':float(tc[j]),
                    'relative_change':float(relative[j]),'error_band':band(qc,qd)}
            ratio=[min(ratio[0],float(relative.min())),max(ratio[1],float(relative.max()))]
            delta_t=[min(delta_t[0],float(change.min())),max(delta_t[1],float(change.max()))]
    nominal_d=sd[max(qs_d)];nominal_c=sc[max(qs_c)]
    td=max(tma_point,nominal_d);tc=max(tma_point,nominal_c)
    resource=plan(candidate)['resources']
    illegal=not resource['shared_legal'] or resource['definite_register_infeasible'] or not resource['static_shape_legal']
    uncertain_spill=resource['spill_possible']
    safe=not illegal and not uncertain_spill
    return {'candidate':candidate['id'],'default':default['id'],
        'delta_MC_ns_interval':delta_s,'delta_tile_ns_interval':delta_t,
        'relative_change_interval':ratio,'worst_scenario_improvement_percent':-100*ratio[1],
        'nominal_improvement_percent':100*(1-tc/td),'nominal_candidate_MC_ns':nominal_c,
        'nominal_default_MC_ns':nominal_d,'nominal_candidate_tile_ns':tc,'nominal_default_tile_ns':td,
        'TMA_ns_interval':tma,'TMA_point_ns':tma_point,'transfer_scope':scope,
        'candidate_q_interval':[min(qs_c),max(qs_c)],'default_q_interval':[min(qs_d),max(qs_d)],
        'scenarios_evaluated':count,'worst_scenario':extrema['worst'],
        'resource_gate_pass':safe,'static_spill_uncertainty':uncertain_spill,
        'robust_switch_eligible':safe and ratio[1]<-0.03,
        'covariance':'Same joint residual record and common TMA draw for candidate/default. Reflection preserves joint dependence; q ranges are conservatively crossed.'}


def predict(case,theta,mode='full'):
    fitted_mode='full' if mode=='Envelope-only-blind' else mode
    model=theta['models'][fitted_mode]['model'];tables=theta['primitive_shape_tables']
    qs=q_values(case,mode);values=[point(model,case,q,tables) for q in qs]
    tma,tp,scope=transfer(case,theta,mode)
    nominal=values[-1]
    f=features(case,max(qs),model['recipe'],theta['primitive_shape_tables'],model['mode'])
    contributions={name:coef*f[name] for name,coef in zip(model['feature_names'],model['coefficients'])}
    # Marginal diagnostics are never independently subtracted for selection.
    point_errors=theta['models'][fitted_mode]['cross_validation']['point_errors']
    errors=[r['error_ns'] for r in point_errors if
            (r['q']=='natural' or r['q']=='4') or min(qs)<3]
    radius=max(map(abs,errors));mc=[max(0.,min(values)-radius),max(values)+radius]
    return {'id':case['id'],'case':case,'static':compact(plan(case)),
        'q_evaluated':qs,'MC_point_ns':nominal,'MC_q_curve_ns':dict(zip(map(str,qs),values)),
        'nominal_static_features':f,'feature_contributions_ns':contributions,
        'MC_marginal_diagnostic_ns':mc,'TMA_ns_interval':tma,'TMA_point_ns':tp,
        'tile_point_ns':max(tp,nominal),
        'rho_point':nominal/tp if tp else None,
        'rho_marginal_diagnostic_interval':[mc[0]/tma[1],mc[1]/tma[0]] if tma[0] else None,
        'transfer_scope':scope,'marginal_interval_used_for_decisions':False,
        'extrapolation_flags':[flag for condition,flag in [
            (case['BM']!=32,'new_BM_combination'),(case['BN'] not in (64,128),'new_BN_ML_pattern'),
            (case['W'] not in (4,8),'new_warp_count'),(plan(case)['p']>64,'P_size_extrapolation'),
            (plan(case)['resources']['spill_possible'],'possible_spill')] if condition]}


def select(m,n,w,theta,mode='full'):
    cases=candidates(m,n,w);default_id=upstream(m,n,w)
    default=next(x for x in cases if x['id']==default_id)
    predictions=[predict(x,theta,mode) for x in cases]
    comparisons=[compare(x,default,theta,mode) for x in cases if x['id']!=default_id]
    robust=[r for r in comparisons if r['robust_switch_eligible']]
    if robust:
        # A common roof can make several fast layouts equivalent. Preserve
        # the upstream placement among safe near-optimal candidates, then
        # minimize directly predicted MC work; no measured winner tie-break.
        minimum=min(r['nominal_candidate_tile_ns'] for r in robust)
        near=[r for r in robust if r['nominal_candidate_tile_ns']<=1.03*minimum]
        same=[r for r in near if r['candidate'].endswith(':'+default['placement'])]
        chosen=min(same or near,key=lambda r:(r['nominal_candidate_MC_ns'],r['candidate']))
        choice=chosen['candidate'];reason='paired_steady_roofline_improvement_over_3_percent'
    else:
        choice=default_id
        all_flat=all(abs(r['relative_change_interval'][0])+abs(r['relative_change_interval'][1])<1e-10 for r in comparisons)
        unresolved_resources=plan(default)['resources']['spill_possible'] or all(
            not r['resource_gate_pass'] for r in comparisons)
        reason='resource_uncertainty_default' if unresolved_resources else (
            'transfer_limited_comparable_set' if all_flat else 'no_robust_steady_improvement_or_resource_gate')
    nominal_min=min(predictions,key=lambda r:r['tile_point_ns'])
    default_point=next(r for r in predictions if r['id']==default_id)['tile_point_ns']
    nominal_choice=nominal_min['id'] if nominal_min['tile_point_ns']<0.97*default_point else default_id
    return {'geometry':[m,n,w],'mode':mode,'default':default_id,'choice':choice,
        'switched':choice!=default_id,'reason':reason,'comparisons':comparisons,
        'robust_candidates':[r['candidate'] for r in robust],
        'nominal_choice':nominal_choice,
        'nominal_comparable_set':[p['id'] for p in predictions if p['tile_point_ns']<=1.03*nominal_min['tile_point_ns']],
        'default_resource_uncertainty':plan(default)['resources']['spill_possible'],
        'predictions':predictions,'primary_metric':'steady MC/TMA bottleneck envelope',
        'K1_MO_grid1_hard_veto':False}


if __name__=='__main__':
    import argparse
    from experiments.tma_reduction_layout.phase11.c31 import common as c
    parser=argparse.ArgumentParser();parser.add_argument('--BM',type=int,required=True)
    parser.add_argument('--BN',type=int,required=True);parser.add_argument('--W',type=int,required=True)
    args=parser.parse_args();print(json.dumps(select(args.BM,args.BN,args.W,c.read(c.V1/'theta_H100.json')),indent=2))
