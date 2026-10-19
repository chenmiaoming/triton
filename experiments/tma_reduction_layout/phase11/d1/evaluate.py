"""Blind scoring: consume frozen decisions, never refit using D1 outcomes."""
from collections import defaultdict
import math
import statistics as st
from experiments.tma_reduction_layout.phase11.d1 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import upstream


def geo(values):return f'{values[0]}x{values[1]}w{values[2]}'


def rmse(values):return math.sqrt(st.mean(v*v for v in values)) if values else None


def interval(values):
    assert len(values)==3
    mean=st.mean(values);half=4.303*st.stdev(values)/math.sqrt(3)
    return {'median':st.median(values),'worker_values':values,'mean':mean,
            'descriptive_t95':[mean-half,mean+half]}


def truth():
    p=c.protocol();gate=c.read(c.OUT/'artifact_gate.json');analysis=c.read(c.OUT/'analysis.json')
    workers={i:c.read(c.OUT/'timing'/f'invocation_{i}/raw.json') for i in (1,2,3)}
    services={}
    for r in analysis['oneshot']:
        if r['q_condition']=='natural':services[r['tag'],r['invocation']]=r
    cases=[];by_geo={}
    for definition in p['D1_cases']:
        g=definition['geometry'];rows=[]
        metadata={tag:r for tag,r in gate['metadata'].items() if [r['M'],r['N'],r['num_warps']]==g}
        controls=[]
        for tag,meta in metadata.items():
            sample=[services[tag,i] for i in (1,2,3)]
            measured=[workers[i]['bindings'][tag] for i in (1,2,3)]
            slopes=[r['tile_slope_ns'] for r in sample]
            assert all(v>0 for v in slopes),(tag,slopes)
            row={'tag':tag,'layout':meta.get('layout_id'),'geometry':g,
                'slope_ns':interval(slopes),'raw128_us':interval([r['full_us'] for r in sample]),
                'worker_resources':[{'invocation':i,'n_regs':r['attributes']['num_regs'],
                    'local_bytes':r['attributes']['local_bytes'],
                    'q':r['reservations']['natural']['capacity'],
                    'dynamic_shared_bytes':r['reservations']['natural']['dynamic_bytes']}
                    for i,r in zip((1,2,3),measured)],
                'grid_fits':[r['grid_fit'] for r in sample],
                'cubin_SHA256':meta['cubin_SHA256'],
                'register_layout':meta.get('layout'),
                'static_audit':c.read(c.OUT/'compile_gate.json')['static_audit'][tag]}
            (rows if meta['family']=='ONE_FULL' else controls).append(row)
        assert len(controls)==1
        default=upstream(*g);index={r['layout']:r for r in rows};d=index[default]
        oracle_by_worker=[min(rows,key=lambda r:r['slope_ns']['worker_values'][i])['layout'] for i in range(3)]
        oracle_values=[index[o]['slope_ns']['worker_values'][i] for i,o in enumerate(oracle_by_worker)]
        for row in rows:
            row['gain_percent']=interval([100*(1-v/base) for v,base in zip(row['slope_ns']['worker_values'],d['slope_ns']['worker_values'])])
            row['oracle_regret_percent']=interval([100*(v/o-1) for v,o in zip(row['slope_ns']['worker_values'],oracle_values)])
            row['aux_raw128_gain_percent']=interval([100*(1-v/base) for v,base in zip(row['raw128_us']['worker_values'],d['raw128_us']['worker_values'])])
        item={**definition,'default':default,'candidates':rows,'control':controls[0],
            'oracle_by_worker':oracle_by_worker,'oracle_slope_ns_by_worker':oracle_values,
            'oracle_median_point_layout':min(rows,key=lambda r:r['slope_ns']['median'])['layout'],
            'oracle_gain_percent':interval([100*(1-v/base) for v,base in zip(oracle_values,d['slope_ns']['worker_values'])])}
        cases.append(item);by_geo[tuple(g)]=item
    value={'status':'OK','cases':cases,'FULL_candidates':sum(len(r['candidates']) for r in cases),
        'controls':len(cases),'all_programs_measured':len(services)==69*3,
        'oracle_scope':'Same-worker minimum measured eligible slope; finite sample oracle, can favor noise. Point-layout column uses minimum median slope.',
        'primary_endpoint':p['endpoint'],'failures':[]}
    assert value['FULL_candidates']==59 and value['all_programs_measured']
    c.derived(c.OUT/'truth.json',value)
    return by_geo


def score(p,observed):
    tables={}
    for stratum,geometries in p['strata'].items():
        table=[]
        for policy,choices in p['policies'].items():
            items=[]
            for r in choices:
                if r['geometry'] not in geometries:continue
                t=observed[tuple(r['geometry'])]
                candidate=next(v for v in t['candidates'] if v['layout']==r['layout'])
                gain=candidate['gain_percent']['median'];regret=candidate['oracle_regret_percent']['median']
                changed=r['layout']!=t['default']
                items.append({'geometry':r['geometry'],'layout':r['layout'],'default':t['default'],
                    'changed':changed,'gain_percent':gain,'oracle_regret_percent':regret,
                    'gain_interval':candidate['gain_percent'],
                    'aux_raw128_gain_percent':candidate['aux_raw128_gain_percent'],
                    'adverse_over_3_percent':gain < -3.,
                    'switched_gain_le_3_percent':changed and gain<=3.})
            table.append({'policy':policy,'cases':len(items),'failures':0,'items':items,
                'switches':sum(r['changed'] for r in items),
                'mean_gain_percent':st.mean(r['gain_percent'] for r in items),
                'mean_oracle_regret_percent':st.mean(r['oracle_regret_percent'] for r in items),
                'adverse_over_3_percent':sum(r['adverse_over_3_percent'] for r in items),
                'switched_gain_le_3_percent':sum(r['switched_gain_le_3_percent'] for r in items)})
        assert len(table)==11
        tables[stratum]=table
    c.derived(c.OUT/'scorecards.json',{'status':'OK','tables':tables,
        'roles':p['models'],'all_policies_shown_in_every_stratum':True,
        'no_D1_model_selection_or_refitting':True,'endpoint':p['endpoint']})
    return tables


def coverage(p,observed):
    results={}
    for label,definition in p['models'].items():
        bound=c.OUT/'preregistration'/definition['bound_copy']
        assert c.sha(bound.read_bytes())==definition['SHA256']
        frozen=c.read(bound);records=[]
        for case in frozen['cases']:
            t=observed[tuple(case['geometry'])];d=case['prediction']['upstream_default']
            pts={r['layout']:r for r in case['prediction']['candidates']}
            truths={r['layout']:r for r in t['candidates']};dp=pts[d]['prediction']['tile_ns']
            for layout,pt in pts.items():
                tr=truths[layout];dr=truths[d];rs=pt['resource'];pred=pt['prediction']['tile_ns']
                point=tr['slope_ns']['median'];base=dr['slope_ns']['median']
                obs_delta=st.median(v-z for v,z in zip(tr['slope_ns']['worker_values'],dr['slope_ns']['worker_values']))
                obs_ratio=st.median(v/z-1 for v,z in zip(tr['slope_ns']['worker_values'],dr['slope_ns']['worker_values']))
                predicted_delta=pred-dp;predicted_ratio=pred/dp-1
                residual=obs_ratio-predicted_ratio
                rmeas=tr['worker_resources'][0];dmeas=dr['worker_resources'][0]
                assert all(v['n_regs']==rmeas['n_regs'] and v['q']==rmeas['q'] for v in tr['worker_resources'])
                qlo,qhi=rs['q_interval'];joint=sorted({(v['candidate_q'],v['default_q']) for v in pt['resource_scenarios']})
                record={'geometry':case['geometry'],'stratum':case['stratum'],'layout':layout,
                    'default':d,'nondefault':layout!=d,'predicted_FULL_ns':pred,'observed_FULL_ns':point,
                    'absolute_error_ns':pred-point,'relative_error':pred/point-1,
                    'predicted_paired_delta_ns':predicted_delta,'observed_paired_delta_ns':obs_delta,
                    'paired_residual_ns':obs_delta-predicted_delta,
                    'predicted_paired_ratio':predicted_ratio,'observed_paired_ratio':obs_ratio,
                    'paired_relative_residual':residual,
                    'observed_gain_percent':-100*obs_ratio,
                    'nominal_q_upper_relative':pt['uncertainty']['upper_relative'] if pt['uncertainty'] else None,
                    'upper_tail_covered':residual<=pt['uncertainty']['upper_relative'] if pt['uncertainty'] else None,
                    'propagated_lower_gain_percent':pt['worst_gain_percent'],
                    'propagated_lower_gain_covered':-100*obs_ratio>=pt['worst_gain_percent'],
                    'measured_n_regs':rmeas['n_regs'],'measured_q':rmeas['q'],
                    'measured_default_q':dmeas['q'],'local_bytes':rmeas['local_bytes'],
                    'n_regs_upper':rs['n_regs_interval'][1],
                    'n_regs_upper_covered':rmeas['n_regs']<=rs['n_regs_interval'][1],
                    'integer_ceil_upper_covered':rmeas['n_regs']<=math.ceil(rs['n_regs_interval'][1]),
                    'q_interval':[qlo,qhi],'q_interval_covered':qlo<=rmeas['q']<=qhi,
                    'q_values':rs.get('q_values'),
                    'q_discrete_covered':rmeas['q'] in rs['q_values'] if 'q_values' in rs else None,
                    'joint_q_pairs':joint,'joint_q_covered':(rmeas['q'],dmeas['q']) in joint if layout!=d else None,
                    'resource_outside_R':rs['R_outside_support'],'resource_new_W':rs['new_warp_count'],
                    'FULL_outside_support':'FULL_R_outside_support' in pt.get('reasons',[]),
                    'selection_reasons':pt.get('reasons',[]),
                    'worker_relative_errors':[pred/v-1 for v in tr['slope_ns']['worker_values']]}
                records.append(record)
        summary={}
        for stratum,geometries in p['strata'].items():
            rows=[r for r in records if r['geometry'] in geometries];pairs=[r for r in rows if r['nondefault']]
            def count(key,pool):
                valid=[r for r in pool if r[key] is not None]
                return {'covered':sum(r[key] for r in valid),'total':len(valid)}
            summary[stratum]={'FULL_candidates':len(rows),'nondefault_pairs':len(pairs),
                'FULL_relative_RMSE_percent':100*rmse([r['relative_error'] for r in rows]),
                'FULL_absolute_RMSE_ns':rmse([r['absolute_error_ns'] for r in rows]),
                'paired_RMSE_ns':rmse([r['paired_residual_ns'] for r in pairs]),
                'paired_ratio_RMSE_pp':100*rmse([r['paired_relative_residual'] for r in pairs]),
                'n_regs_upper':count('n_regs_upper_covered',rows),
                'n_regs_integer_ceil_upper':count('integer_ceil_upper_covered',rows),
                'q_interval':count('q_interval_covered',rows),
                'q_discrete':count('q_discrete_covered',rows),
                'joint_q':count('joint_q_covered',pairs),
                'performance_upper':count('upper_tail_covered',pairs),
                'propagated_gain_lower':count('propagated_lower_gain_covered',pairs),
                'R_outside_count':sum(r['resource_outside_R'] for r in rows),
                'new_W_count':sum(r['resource_new_W'] for r in rows)}
        results[label]={'role':definition['role'],'summary':summary,'records':records,
            'frozen_prediction_SHA256':definition['SHA256']}
    value={'status':'OK','models':results,'no_refit':True,'no_measured_q_conditioning':True,
        'units':'ns/tile/SM; paired ratios relative to same-worker upstream default',
        'scope':'Empirical unseen-case coverage; 10 geometry groups, not 59 independent geometry tests. Intervals are not distribution-free guarantees.'}
    c.derived(c.OUT/'coverage.json',value)
    return value


def run():
    p=c.protocol();c.protect_prior();observed=truth();score(p,observed);coverage(p,observed)
    print('D1 scored: 59 FULL truths, 10 controls, 11 frozen policies x 3 strata; no refit')


if __name__=='__main__':run()
