"""Static predictions and scale/structure-conditioned paired uncertainty."""
import math
import statistics as st
import re
from experiments.tma_reduction_layout.phase11.c33 import common as c, fit, resources
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream
from experiments.tma_reduction_layout.phase11.c32.features import static


def structure(candidate,default,cp,dp):
    if cp['branch']=='direct_response' or dp['branch']=='direct_response':
        branch='direct_response'
    elif cp['branch']==dp['branch']=='platform': branch='both_platform'
    elif cp['branch']==dp['branch']=='onchip': branch='both_onchip'
    else: branch='crossing'
    return {'branch':branch,'family':candidate['placement'],
        'vec_group':'low_vec' if candidate['register_layout']['size_per_thread'][2]<=2 else 'high_vec',
        'q_change':cp['q']!=dp['q'],
        'R':candidate['BM']*candidate['BN']//(32*candidate['W'])}


def residual_structure(row):
    branch='direct_response' if 'direct_response' in (row['candidate_branch'],row['default_branch']) else (
        'both_platform' if row['candidate_branch']==row['default_branch']=='platform' else
        'both_onchip' if row['candidate_branch']==row['default_branch']=='onchip' else 'crossing')
    vec=int(row['layout'].split(':vec')[1].split(':')[0])
    return {'branch':branch,'family':row['candidate_family'],
        'vec_group':'low_vec' if vec<=2 else 'high_vec',
        'q_change':row['candidate_q']!=row['default_q'],'R':row['R']}


def upper_bound(bank,s,leverage):
    minimum=c.protocol()['minimum_distinct_geometries_per_stratum']
    levels=[('branch','family','vec_group','q_change'),('family','vec_group','q_change'),
            ('family','vec_group'),('family',),()]
    chosen=None
    for keys in levels:
        rows=[r for r in bank if all(residual_structure(r)[key]==s[key] for key in keys)]
        # Also seek nearby scale (a factor of two in R); record backoff when
        # this has fewer than three independent geometries.
        near=[r for r in rows if .5<=r['R']/s['R']<=2]
        if len({r['geometry'] for r in near})>=minimum:
            chosen=(keys,near,True);break
        if len({r['geometry'] for r in rows})>=minimum:
            chosen=(keys,rows,False);break
    assert chosen is not None
    keys,rows,local_scale=chosen
    by_geometry={}
    for row in rows:
        by_geometry.setdefault(row['geometry'],[]).append(row['relative_residual'])
    # Within-geometry candidates/workers do not create new independent samples.
    maxima=[max(v) for v in by_geometry.values()]
    positive_rms=math.sqrt(st.mean(max(0.,v)**2 for v in maxima))
    margin=max(c.protocol()['uncertainty_relative_margin_floor'],positive_rms)
    inflation=min(2.5,max(1.,math.sqrt(leverage)))
    r_support=sorted({r['R'] for r in rows})
    missing_R=s['R'] not in r_support
    outside_R=not min(r_support)<=s['R']<=max(r_support)
    if missing_R: inflation*=1.25
    if outside_R: inflation*=1.5
    observed_upper=max(0.,max(maxima))
    return {'upper_relative':observed_upper+margin*inflation,
        'signed_residual_range':[min(r['relative_residual'] for r in rows),max(r['relative_residual'] for r in rows)],
        'positive_tail_RMS':positive_rms,'additional_margin_relative':margin*inflation,
        'conditioning_keys':list(keys),'nearby_R_scale':local_scale,
        'distinct_geometries':len(by_geometry),'training_geometries':sorted(by_geometry),
        'residual_records':len(rows),'R_support':r_support,'R_missing':missing_R,
        'R_extrapolation':outside_R,'negative_residuals_reflected':False,
        'scope':'Training-only cross-fitted empirical upper tail with positive margin; not a distribution-free confidence guarantee.'}


def predict(geometry,theta,bank=None,resource_model=None):
    mm=theta['model'];rm=resource_model or theta['resource_model']
    bank=theta['paired_outer_residuals'] if bank is None else bank
    pool=candidates(*geometry);default_id=upstream(*geometry)
    default=next(r for r in pool if r['id']==default_id)
    estimates={r['id']:resources.estimate(r,rm) for r in pool}
    points={r['id']:fit.predict(mm,r,estimates[r['id']]['q_point']) for r in pool}
    dp=points[default_id];results=[]
    # A shared register error is varied together, then a paired allocation
    # error adjusts only the candidate. No independent time-base extrema.
    for case in pool:
        cp=points[case['id']];rr=estimates[case['id']];s=structure(case,default,cp,dp)
        nominal=100*(1-cp['tile_ns']/dp['tile_ns'])
        if case['id']==default_id:
            results.append({'layout':case['id'],'resource':rr,'prediction':cp,
                'nominal_gain_percent':0.,'worst_gain_percent':0.,'robust_switch':False,
                'uncertainty':None,'structure':s,'resource_scenarios':[]})
            continue
        upper=upper_bound(bank,s,cp['leverage']['inflation']**2)
        def nearby(row):
            m,n,w=map(int,re.fullmatch(r'(\d+)x(\d+)w(\d+)',row['geometry']).groups())
            return .5<=m*n/(32*w)/s['R']<=2
        errors=[r for r in rm['held_geometry_errors'] if nearby(r)]
        if len({r['geometry'] for r in errors})<3: errors=rm['held_geometry_errors']
        common=[min(r['residual'] for r in errors)-8,0.,max(r['residual'] for r in errors)+8]
        paired=[r for r in rm['paired_register_errors'] if nearby(r) and
            r['layout'].endswith(':'+case['placement'])]
        if len({r['geometry'] for r in paired})<3:
            paired=[r for r in rm['paired_register_errors'] if nearby(r)]
        if len({r['geometry'] for r in paired})<3: paired=rm['paired_register_errors']
        delta=[min((r['delta_residual'] for r in paired),default=0.)-8,0.,
               max((r['delta_residual'] for r in paired),default=0.)+8]
        ratios=[];scenarios=[];bounded=[];scenario_uncertainty=[];scenario_structure=[]
        for shift in common:
            for diff in delta:
                dq=resources.q_for(default,max(8.,resources.point(rm,default)+shift))
                cq=resources.q_for(case,max(8.,resources.point(rm,case)+shift+diff))
                d=fit.predict(mm,default,dq);v=fit.predict(mm,case,cq)
                ratios.append(v['tile_ns']/d['tile_ns']-1)
                ss=structure(case,default,v,d)
                uu=upper_bound(bank,ss,v['leverage']['inflation']**2)
                bounded.append(ratios[-1]+uu['upper_relative'])
                scenario_uncertainty.append(uu);scenario_structure.append(ss)
                scenarios.append({'common_register_residual':shift,'paired_register_residual':diff,
                    'candidate_q':cq,'default_q':dq,'predicted_delta_relative':ratios[-1],
                    'uncertainty_upper_relative':uu['upper_relative'],'structure':ss})
        worst_index=max(range(len(bounded)),key=lambda i:bounded[i])
        worst= -100*bounded[worst_index]
        reasons=[]
        if not rr['shared_legal'] or not estimates[default_id]['shared_legal']: reasons.append('shared_illegal')
        if rr['spill_uncertainty'] or estimates[default_id]['spill_uncertainty']: reasons.append('spill_uncertainty')
        if rr['new_warp_count'] or estimates[default_id]['new_warp_count']: reasons.append('unseen_warp_count')
        support=theta['performance_support']
        r=static(case)['elements_per_thread']
        if r<min(support['R']) or r>max(support['R']): reasons.append('FULL_R_outside_support')
        if case['W'] not in support['W']: reasons.append('FULL_W_outside_support')
        if worst<=c.protocol()['practical_switch_percent']: reasons.append('no_robust_FULL_gain')
        results.append({'layout':case['id'],'resource':rr,'prediction':cp,
            'nominal_gain_percent':nominal,'worst_gain_percent':worst,'structure':s,
            'uncertainty':upper,'resource_scenarios':scenarios,
            'worst_uncertainty':scenario_uncertainty[worst_index],
            'worst_structure':scenario_structure[worst_index],
            'resource_residual_geometries':sorted({r['geometry'] for r in errors}),
            'paired_resource_residual_geometries':sorted({r['geometry'] for r in paired}),
            'robust_switch':not reasons,'reasons':reasons})
    nominal=min(results,key=lambda r:(r['prediction']['tile_ns'],r['layout']!=default_id,r['layout']))
    robust=[r for r in results if r['robust_switch']]
    selected=max(robust,key=lambda r:(r['worst_gain_percent'],r['nominal_gain_percent'])) if robust else next(r for r in results if r['layout']==default_id)
    return {'geometry':list(geometry),'upstream_default':default_id,
        'nominal_choice':nominal['layout'],'nominal_gain_percent':nominal['nominal_gain_percent'],
        'selected':selected['layout'],'selected_worst_gain_percent':selected['worst_gain_percent'],
        'changed':selected['layout']!=default_id,'candidates':results,
        'runtime_grid_feature':False,'D1_compiled':False,'D1_timed':False}
