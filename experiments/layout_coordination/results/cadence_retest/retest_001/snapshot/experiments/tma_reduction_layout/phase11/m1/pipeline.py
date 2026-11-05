"""Frozen-model substitutions and same-worker FULL contrasts; no refitting."""
from collections import defaultdict
from functools import lru_cache
import math
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.m1 import common as c
from experiments.tma_reduction_layout.phase11.c33 import features
from experiments.tma_reduction_layout.phase11.c32 import regression as reg
from experiments.tma_reduction_layout.phase11.c3.static import candidates,upstream

VARIANTS=('static','measured_q','measured_control','measured_q_and_control')


def load():
    full=[];controls=[];gates={};compiles={};counters={}
    for cohort,stage in (('development','stage_c2d'),('D1','stage_d1')):
        analysis=c.read(c.source(stage,'analysis.json'));gate=c.read(c.source(stage,'artifact_gate.json'))
        gates[cohort]=gate;compiles[cohort]=c.read(c.source(stage,'compile_gate.json'))
        counters[cohort]=c.read(c.source(stage,'counter_gate.json'))
        for r in analysis['oneshot']:
            meta=gate['metadata'][r['tag']];geometry=r['geometry']
            item={**r,'cohort':cohort,'geometry_vector':geometry,'geometry':c.geo(geometry),
                'worker':r['invocation'],'target_ns':r['tile_slope_ns']}
            if meta['family']=='ONE_FULL':
                case=next(x for x in candidates(*geometry) if x['id']==meta['layout_id'])
                full.append({**item,'case':case,'layout':case['id']})
            else:controls.append(item)
    def unique(rows):
        seen={}
        for r in sorted(rows,key=lambda x:(x['q_condition']!='natural',x['q_condition'])):
            key=(r['cohort'],r.get('layout',r['geometry']),r['q'],r['worker'],r['GPU_UUID'])
            if key in seen:
                assert math.isclose(seen[key]['target_ns'],r['target_ns'],abs_tol=1e-7)
                seen[key]['condition_aliases'].append(r['q_condition'])
            else:seen[key]={**r,'condition_aliases':[]}
        return list(seen.values())
    return {'full':unique(full),'controls':unique(controls),'gates':gates,'compiles':compiles,'counters':counters}


def contexts():
    legacy=c.read(c.source('stage_c33','theta_H100.json'))
    auto=c.read(c.source('stage_c34','theta_H100.json'))
    dev=c.read(c.source('stage_c34','development.json'))
    olddev=c.read(c.source('stage_c33','development.json'))
    result={}
    definitions=(('C34_resource_only',legacy,'resource_only_outer','stage_c34/resource_only_D1_predictions.json'),
                 ('C34_full_auto',auto,'outer','stage_c34/D1_predictions.json'),
                 ('C33',legacy,'outer_decisions','stage_c33/D1_predictions.json'))
    for label,theta,field,path in definitions:
        folds={r['held']:r for r in theta['outer_folds']}
        rows=(olddev if label=='C33' else dev)[field]
        for row in rows:
            d=row['decision'];g=c.geo(d['geometry'])
            result[label,'development',g]={'model':folds[g]['model'],'decision':d,'tables':theta['fixed_R_tables']}
        for row in c.read(c.BASE/'results/phase11'/path)['cases']:
            d=row['prediction'];g=c.geo(row['geometry'])
            result[label,'D1',g]={'model':theta['model'],'decision':d,'tables':theta['fixed_R_tables']}
    return result


def response(model,case,q,tables,control=None):
    mc=max(0.,reg.predict(model['MC'],features.operation(case,q,tables,model['mode'])))
    hf=features.control([case['BM'],case['BN'],case['W']],q)
    hm=model['control'];mask=np.asarray(hm['bulk_mask'],bool)
    x=np.array([hf[n] for n in hm['feature_names']]);beta=np.asarray(hm['coefficients'])
    hhat=max(1.,float(x[mask]@beta[mask]),float(x[~mask]@beta[~mask]));h=hhat if control is None else control
    fs=features.operation(case,q,tables,model['mode'],mc=mc,
        transfer=h if model['kind']=='full_structural' and model['mode']!='Resource/Envelope-blind' else None)
    a=max(0.,reg.predict(model['full'],fs))
    eta=min(1.,max(0.,model['eta']+model.get('eta_q_slope',0.)*math.log2(q/8.)))
    partial=model['kind'] in ('partial_overlap','partial_overlap_q')
    t=h+max(0.,a-eta*h) if partial else max(1.,a)
    return {'tile_ns':t,'control_ns':h,'model_control_ns':hhat,'A_FULL_ns':a,
            'MC_covariate_ns':mc,'q':q,'eta':eta,'partial_overlap':partial,
            'exposed_response_ns':max(0.,a-eta*h) if partial else None}


def metrics(records,variant):
    groups=defaultdict(list)
    for r in records:
        v=r['predictions'][variant]
        if v is not None:groups[r['cohort'],r['geometry'],r['layout'],r['actual_q']].append(r)
    errors=[];absolute=[];pairs=[]
    aggregate={}
    for key,rows in groups.items():
        y=st.median(r['observed_FULL_ns'] for r in rows)
        v=st.median(r['predictions'][variant]['tile_ns'] for r in rows)
        errors.append(v/y-1);absolute.append(v-y);aggregate[key]=rows
    for key,rows in aggregate.items():
        d=upstream(*rows[0]['geometry_vector'])
        if rows[0]['layout']==d:continue
        diffs=[];rel=[]
        for r in rows:
            default=next((z for z in records if z['cohort']==r['cohort'] and z['layout']==d and
                z['worker']==r['worker'] and z['GPU_UUID']==r['GPU_UUID'] and z['q_condition']==r['q_condition'] and
                z['predictions'][variant] is not None),None)
            if default is None:continue
            actual=r['observed_FULL_ns']-default['observed_FULL_ns']
            prediction=r['predictions'][variant]['tile_ns']-default['predictions'][variant]['tile_ns']
            diffs.append(actual-prediction)
            rel.append(r['observed_FULL_ns']/default['observed_FULL_ns']-
                       r['predictions'][variant]['tile_ns']/default['predictions'][variant]['tile_ns'])
        if diffs:pairs.append((st.median(diffs),st.median(rel)))
    rms=lambda vs:math.sqrt(st.mean(x*x for x in vs)) if vs else None
    return {'conditions':len(groups),'paired_conditions':len(pairs),'geometry_groups':len({r['geometry'] for r in records}),
        'FULL_relative_RMSE_percent':100*rms(errors) if errors else None,'FULL_RMSE_ns':rms(absolute),
        'paired_RMSE_ns':rms([r[0] for r in pairs]),
        'paired_ratio_RMSE_pp':100*rms([r[1] for r in pairs]) if pairs else None}


def run():
    bundle=load();ctx=contexts();records=[];gate_records=[];natural_reproduced=0
    controls={(r['cohort'],r['geometry'],r['worker'],r['GPU_UUID'],r['q']):r for r in bundle['controls']}
    for (label,cohort,g),pack in ctx.items():
        decision=pack['decision'];points={r['layout']:r for r in decision['candidates']}
        rows=[r for r in bundle['full'] if r['cohort']==cohort and r['geometry']==g]
        for row in rows:
            point=points[row['layout']];natural=row['q_condition']=='natural'
            qpoint=point['prediction']['q'];qp=qpoint if natural else min(qpoint,int(row['q_condition']))
            qa=row['q'];model=pack['model'];case=row['case'];tables=pack['tables']
            key=(cohort,g,row['worker'],row['GPU_UUID'])
            at_pred=controls.get((*key,qp));at_actual=controls.get((*key,qa))
            values={'static':response(model,case,qp,tables),
                'measured_q':response(model,case,qa,tables),
                'measured_control':response(model,case,qp,tables,at_pred['target_ns']) if at_pred else None,
                'measured_q_and_control':response(model,case,qa,tables,at_actual['target_ns']) if at_actual else None}
            if natural:
                assert math.isclose(values['static']['tile_ns'],point['prediction']['tile_ns'],rel_tol=1e-10,abs_tol=1e-7),(label,g,row['layout'])
                natural_reproduced+=1
            interaction=(values['measured_q_and_control']['tile_ns']-values['measured_q']['tile_ns']-
                values['measured_control']['tile_ns']+values['static']['tile_ns']) if all(v is not None for v in values.values()) else None
            records.append({'model':label,'cohort':cohort,'geometry':g,'geometry_vector':row['geometry_vector'],
                'layout':row['layout'],'tag':row['tag'],'worker':row['worker'],'GPU_UUID':row['GPU_UUID'],
                'q_condition':row['q_condition'],'condition_aliases':row['condition_aliases'],
                'predicted_q':qp,'actual_q':qa,'observed_FULL_ns':row['target_ns'],
                'observed_control_at_actual_q_ns':at_actual['target_ns'] if at_actual else None,
                'predictions':values,'substitution_interaction_ns':interaction,
                'missing_control_at_predicted_q':at_pred is None})
        for candidate in decision['candidates']:
            default=candidate['layout']==decision['upstream_default'];scenes=candidate['resource_scenarios']
            gate_records.append({'model':label,'cohort':cohort,'geometry':g,'geometry_vector':decision['geometry'],
                'layout':candidate['layout'],'default':default,'selected':candidate['layout']==decision['selected'],
                'nominal_choice':candidate['layout']==decision['nominal_choice'],
                'nominal_gain_percent':candidate['nominal_gain_percent'],
                'performance_upper_pp':100*candidate['uncertainty']['upper_relative'] if candidate['uncertainty'] else None,
                'nominal_q_gain_lower_percent':candidate['nominal_gain_percent']-100*candidate['uncertainty']['upper_relative'] if candidate['uncertainty'] else 0.,
                'q_only_worst_gain_percent':-100*max(s['predicted_delta_relative'] for s in scenes) if scenes else 0.,
                'final_gain_lower_percent':candidate['worst_gain_percent'],
                'q_pairs':sorted({(s['candidate_q'],s['default_q']) for s in scenes}),
                'reasons':candidate.get('reasons',[]),'resource':candidate['resource']})
    summary=[]
    for label in ('C34_resource_only','C34_full_auto','C33'):
        for cohort in ('development','D1'):
            for endpoint in ('natural','all_physical_q'):
                keep=[r for r in records if r['model']==label and r['cohort']==cohort and
                      (endpoint!='natural' or r['q_condition']=='natural')]
                common=[r for r in keep if all(v is not None for v in r['predictions'].values())]
                summary.append({'model':label,'cohort':cohort,'endpoint':endpoint,
                    'all_available':{v:metrics(keep,v) for v in VARIANTS},
                    'four_way_common_support':{v:metrics(common,v) for v in VARIANTS},
                    'worker_rows':len(keep),'four_way_worker_rows':len(common),
                    'q_prediction_mismatch_rows':sum(r['predicted_q']!=r['actual_q'] for r in keep),
                    'control_unavailable_at_predicted_q_rows':sum(r['missing_control_at_predicted_q'] for r in keep)})
    result={'status':'OK','records':records,'summary':summary,'gate_records':gate_records,
        'natural_frozen_predictions_reproduced':natural_reproduced,
        'no_refit':True,'D1_rescored':False,'control_interpolation':False,
        'scope':'Post-hoc computational substitutions. Diagnostic q is occupancy capacity, not instantaneous resident CTAs. No replacement blind-test score.'}
    c.derived(c.OUT/'pipeline.json',result)
    print('M1 pipeline complete',len(records),'rows;',natural_reproduced,'frozen natural predictions reproduced',flush=True)
    return result


if __name__=='__main__':run()
