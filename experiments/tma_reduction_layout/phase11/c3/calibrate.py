"""Fit shared parameters using C2 only, retaining every diagnostic."""
from math import ceil
import statistics as st
from experiments.tma_reduction_layout.phase11.c3 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, plan
from experiments.tma_reduction_layout.phase11.c3.model import (
    fit, probe_features, load_features, graph_schedule, PROBE_NAMES, LOAD_NAMES)


def calibrate_linear():
    analysis=c.read(c.BASE/'results/phase11/stage_c2b/analysis.json')
    gate=c.read(c.BASE/'results/phase11/stage_c2b/artifact_gate.json')
    dev={x['id']:x for geometry in ((32,64,8),(32,128,4)) for x in candidates(*geometry)}
    theta={'version':'C3-v1','target':'H100-sm90','units':'ns per tile on one SM',
           'load':{},'probe':{},'source':'C2a/b frozen effective-service curves',
           'time_intervals_are':'Empirical structural error/sensitivity envelopes, not statistical confidence intervals or hardware guarantees.'}
    for d in (1,2):
        prows=[];lrows=[]
        for row in analysis['services']:
            if row['D']!=d:continue
            tag=row['tag'];meta=gate['metadata'][tag]
            reservation=gate['resources'][tag]['reservations'][row['q']]
            q=reservation['capacity']
            common={'tag':tag,'q':row['q'],'capacity':q,
                    'ns':row['service_ns_per_tile_SM'],'worker':row['invocation']}
            if meta['family'] in ('P','U','W','E'):
                family=meta['family'];v=int(tag.split(':')[0].split('_')[1][1:])
                p=v if family=='P' else 1
                u=v if family=='U' else 1
                w=v if family=='W' else 1
                elements=meta['M']*meta['N']//(32*meta['num_warps'])
                columns=elements//p
                features=probe_features(p,u,w,columns,meta['num_warps'],elements,q,family)
                prows.append({**common,'group':f'{family}{v}','features':features})
            if meta['family']=='ML':
                p=plan(dev[meta['layout_id']])
                lrows.append({**common,'group':f'N{meta["N"]}_e{meta["vec"]}',
                              'features':load_features(p,meta['num_warps'],q)})
        theta['probe'][str(d)]=fit(prows,PROBE_NAMES)
        theta['load'][str(d)]=fit(lrows,LOAD_NAMES)
    # Independent TMA controls: byte normalization, no runtime grid feature.
    rates=[]
    for row in analysis['HBM_anchors']:
        rates.append(row['HBM_tile_SM_us']*1000/(2*32*row['N']+4*row['N']))
    theta['transfer']={'ns_per_tile_byte_interval':[min(rates),max(rates)],
        'calibration':analysis['HBM_anchors'],
        'extrapolation':'Only BM32,N64/128 full-grid controls measured. Other geometries are flagged; byte scaling is a frozen prediction hypothesis, not an independently verified baseline.'}
    # MC bounds constrain composition. Preserve actual residuals separately;
    # a global ratio cannot encode shape/layout-specific offsets.
    bounds=[];points=[];chains=[];k1=[];diagnostics=[]
    index={(r['tag'],r['q'],r['invocation']):r for r in analysis['services']}
    one={(r['tag'],r['q'],r['invocation']):r for r in analysis['setup_and_K1_services']}
    for row in analysis['services']:
        if row['family']!='MC':continue
        tag=row['tag'];meta=gate['metadata'][tag];p=plan(dev[meta['layout_id']])
        q=gate['resources'][tag]['reservations'][row['q']]['capacity']
        schedules=[graph_schedule(p,meta['num_warps'],q,theta,streams=d) for d in (1,2)]
        lower=min(s['pipeline_lower_ns'] for s in schedules)
        upper=max(s['serial_upper_ns'] for s in schedules)
        point=st.median(s['point_ns'] for s in schedules)
        measured=row['service_ns_per_tile_SM']
        bounds.append(measured/upper);bounds.append(measured/lower)
        points.append(measured/point)
        setup=one[(tag,row['q'],row['invocation'])]
        if setup['K1_ns_per_CTA_SM'] is not None:
            k1.append(setup['K1_ns_per_CTA_SM']/measured)
        if row['grid1_effective_chain_us'] is not None:
            chains.append(1000*row['grid1_effective_chain_us']/measured)
        diagnostics.append({'tag':tag,'q':row['q'],'worker':row['invocation'],
            'observed_body_ns':measured,'dependency_pipeline_ns':lower,
            'dependency_serial_ns':upper,'observed_over_pipeline':measured/lower,
            'observed_over_serial':measured/upper,
            'K1_service_ns':setup['K1_ns_per_CTA_SM']})
    # Residual/worker uncertainty extends bounds multiplicatively. C2's largest
    # steady-K fit residual is retained as explicit sensitivity, not an offset.
    residual=analysis['fit_quality']['max_steady_K_residual_percent']/100
    theta['composition']={'steady_ratio':[min(bounds)*(1-residual),max(bounds)*(1+residual)],
        'steady_point_ratio':st.median(points),
        'K1_to_body_ratio':[min(k1)*(1-residual),max(k1)*(1+residual)],
        'K1_to_body_point':st.median(k1),
        'grid1_chain_to_body_ratio':[min(chains)*(1-residual),max(chains)*(1+residual)],
        'diagnostics':diagnostics,
        'MO_bound_ns':[min(r['service_ns_per_tile_SM'] for r in analysis['services'] if r['family']=='MO'),
                       max(r['service_ns_per_tile_SM'] for r in analysis['services'] if r['family']=='MO')],
        'MO_usage':'Diagnostic output/setup upper constraint only; not added to repeated service. K1/body hull already includes output/setup uncertainty.',
        'interpretation':'Global empirical pipeline/serialization and fixture discrepancy hull; does not identify pure overlap or a universal latency. Requires held-out validation.',
        'residual_fraction':residual}
    # Cross-validation is reported independently, not allowed to masquerade as
    # an unseen-shape validation. Missing independent identification stays wide.
    c.write(c.OUT/'attempts/recomputed_linear_v0.json',theta)
    return theta


def calibrate():
    from experiments.tma_reduction_layout.phase11.c3.surfaces import table, leave_size_out
    analysis=c.read(c.BASE/'results/phase11/stage_c2b/analysis.json')
    gate=c.read(c.BASE/'results/phase11/stage_c2b/artifact_gate.json')
    dev={x['id']:x for geometry in ((32,64,8),(32,128,4)) for x in candidates(*geometry)}
    probes=[];loads=[]
    for row in analysis['services']:
        tag=row['tag'];meta=gate['metadata'][tag]
        q=gate['resources'][tag]['reservations'][row['q']]['capacity']
        common={'D':row['D'],'capacity':q,'ns':row['service_ns_per_tile_SM']}
        if meta['family'] in ('P','U','W','E'):
            value=int(tag.split(':')[0].split('_')[1][1:])
            probes.append({**common,'family':meta['family'],'parameter':value})
        elif meta['family']=='ML':
            p=plan(dev[meta['layout_id']])
            if p['ML']['bank_conflicts']==0:
                loads.append({**common,'family':'ML','parameter':meta['vec'],
                              'ns':common['ns']/p['ML']['ideal_wavefronts']})
    primitive=table(probes);cv=leave_size_out(primitive)
    theta={'version':'C3-v1-surfaces','target':'H100-sm90',
        'primitive_surfaces':primitive,'load_surfaces':table(loads),
        'primitive_leave_size_out':cv,
        'primitive_leave_size_out_max_error':max(r['relative_error'] for r in cv),
        'units':'ns per tile on one SM; load surface is ns per ideal wavefront',
        'curve_interpolation':'Piecewise linear in inverse residency; log-log interpolation in primitive size. Outside support: explicit sensitivity/extrapolation.',
        'primitive_identification':'Effective fixture-inclusive P/U/W/E per-column increments. Shared p/u/w/e tables, no layout/case-specific weights. P/U/W neutral-probe subtraction is not claimed to isolate naked operations.',
        'failed_linear_attempt':'attempts/linear_v0.json; rejected for probe and held-size errors',
        'interval_type':'Empirical model/sensitivity envelopes, not confidence intervals or guaranteed hardware bounds.'}
    # One shared address-conflict coefficient interval; paired same-vec dev
    # geometry contrasts are not claimed a pure latency identification.
    idx={(r['tag'],r['q'],r['invocation']):r for r in analysis['services']}
    conflict=[]
    for row in analysis['services']:
        if row['family']!='ML':continue
        meta=gate['metadata'][row['tag']];p=plan(dev[meta['layout_id']])
        if not p['ML']['bank_conflicts']:continue
        other=row['tag'].replace(':within:',':n_first:')
        match=idx.get((other,row['q'],row['invocation']))
        if match:
            conflict.append({'tag':row['tag'],'q':row['q'],'D':row['D'],
                'ns_per_conflict':(row['service_ns_per_tile_SM']-match['service_ns_per_tile_SM'])/p['ML']['bank_conflicts']})
    theta['bank_conflict_effective_ns_interval']=[0.,
                                                max(r['ns_per_conflict'] for r in conflict)]
    theta['bank_conflict_contrasts']=conflict
    theta['bank_conflict_coefficient_is_pure_latency']=False
    rates=[r['HBM_tile_SM_us']*1000/(2*32*r['N']+4*r['N']) for r in analysis['HBM_anchors']]
    theta['transfer']={'ns_per_tile_byte_interval':[min(rates),max(rates)],
        'calibration':analysis['HBM_anchors'],
        'support':'Independent full-grid controls only BM32,N64/128. Byte scaling elsewhere is unvalidated; conservative selection abstains.'}
    one={(r['tag'],r['q'],r['invocation']):r for r in analysis['setup_and_K1_services']}
    lower_ratios=[];upper_ratios=[];points=[];k1=[];chains=[];diagnostics=[]
    mo_rates=[]
    for row in analysis['services']:
        if row['family']=='MO':
            mo_rates.append(row['service_ns_per_tile_SM']/(4*row['N']))
        if row['family']!='MC':continue
        meta=gate['metadata'][row['tag']];p=plan(dev[meta['layout_id']])
        q=gate['resources'][row['tag']]['reservations'][row['q']]['capacity']
        schedules=[graph_schedule(p,meta['num_warps'],q,theta,streams=d) for d in (1,2)]
        lower=min(s['pipeline_lower_ns'] for s in schedules)
        upper=max(s['serial_upper_ns'] for s in schedules)
        point=st.median(s['point_ns'] for s in schedules)
        measured=row['service_ns_per_tile_SM']
        lower_ratios.append(measured/upper);upper_ratios.append(measured/lower)
        points.append(measured/point)
        setup=one[(row['tag'],row['q'],row['invocation'])]
        mo=idx[(row['tag'].replace(':MC:',':MO:'),row['q'],row['invocation'])]['service_ns_per_tile_SM']
        if setup['K1_ns_per_CTA_SM'] is not None:
            k1.append(setup['K1_ns_per_CTA_SM']/max(measured,mo))
        if row['grid1_effective_chain_us'] is not None:
            chains.append(1000*row['grid1_effective_chain_us']/measured)
        diagnostics.append({'tag':row['tag'],'q':row['q'],'worker':row['invocation'],
            'observed_body_ns':measured,'pipeline_lower_ns':lower,
            'serial_upper_ns':upper,'MO_diagnostic_ns':mo,
            'K1_service_ns':setup['K1_ns_per_CTA_SM'],
            'MC_minus_ML_MR_MO_ns':measured-sum(idx[(row['tag'].replace(':MC:',':'+f+':'),row['q'],row['invocation'])]['service_ns_per_tile_SM'] for f in ('ML','MR','MO'))})
    residual=analysis['fit_quality']['max_steady_K_residual_percent']/100
    theta['composition']={'steady_ratio':[min(lower_ratios)*(1-residual),max(upper_ratios)*(1+residual)],
        'steady_point_ratio':st.median(points),
        'K1_combination_ratio':[min(k1)*(1-residual),max(k1)*(1+residual)],
        'K1_combination_point':st.median(k1),
        'MO_ns_per_output_byte_interval':[min(mo_rates),max(mo_rates)],
        'MO_usage':'Constrains max(body,output/setup proxy) K1 envelope, never summed linearly. Input load/setup remain confounded.',
        'grid1_chain_to_body_ratio':[min(chains)*(1-residual),max(chains)*(1+residual)],
        'diagnostics':diagnostics,'residual_fraction':residual,
        'interpretation':'Global dimensionless MC combination/fixture bounds. No per-layout residual constant, no physical pure-overlap identification.',
        'calibration_is_validation':False}
    c.write(c.OUT/'theta_H100.json',theta)
    return theta


if __name__=='__main__':
    theta=calibrate()
    print('primitive response entries',len(theta['primitive_surfaces']),
          'load entries',len(theta['load_surfaces']),
          'leave-size-out max error',theta['primitive_leave_size_out_max_error'])
    print('composition', {k:v for k,v in theta['composition'].items() if k!='diagnostics'})
