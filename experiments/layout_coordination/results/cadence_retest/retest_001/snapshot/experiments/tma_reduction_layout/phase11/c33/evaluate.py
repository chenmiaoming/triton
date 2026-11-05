"""Score every policy on the complete development matrix, with held folds."""
import statistics as st
from experiments.tma_reduction_layout.phase11.c33 import common as c, data, fit, resources
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream
from experiments.tma_reduction_layout.phase11.c32.features import static
from experiments.tma_reduction_layout.phase11.c33.calibrate import measure


def score_policy(name,choices,observed):
    items=[]
    for geometry,selected in choices.items():
        rows=observed[geometry];case=rows[0]['case'];d=upstream(case['BM'],case['BN'],case['W'])
        values={r['layout']:r['target_ns'] for r in rows}
        oracle=min(values,key=values.get)
        items.append({'geometry':geometry,'selected':selected,'default':d,
            'observed_gain_percent':100*(1-values[selected]/values[d]),
            'oracle_regret_percent':100*(values[selected]/values[oracle]-1),
            'changed':selected!=d,'adverse_over_3_percent':values[selected]>values[d]*1.03})
    return {'policy':name,'items':items,'switches':sum(r['changed'] for r in items),
        'adverse_over_3_percent':sum(r['adverse_over_3_percent'] for r in items),
        'mean_gain_percent':st.mean(r['observed_gain_percent'] for r in items),
        'mean_oracle_regret_percent':st.mean(r['oracle_regret_percent'] for r in items)}


def run():
    bundle=data.load();_,rr=resources.census();bundle['resource_rows']=rr;fit.initialize(bundle)
    theta=c.read(c.OUT/'theta_H100.json');cv=c.read(c.OUT/'cross_validation.json')
    dev=c.read(c.OUT/'development.json');observed={}
    for row in data.medians([r for r in bundle['full'] if r['q_condition']=='natural']):
        observed.setdefault(row['geometry'],[]).append(row)
    choices={};baselines=[]
    for name in ('upstream_default','fixed_vec1','fixed_vec2','fixed_vec4','fixed_vec8','fanout_only','zero_base_current_calibration'):
        choice={}
        for geo,rows in observed.items():
            case=rows[0]['case'];m,n,w=case['BM'],case['BN'],case['W'];pool=candidates(m,n,w);d=upstream(m,n,w)
            if name=='upstream_default': selected=d
            elif name.startswith('fixed_vec'):
                v=int(name[-1]);possible=[r for r in pool if r['placement']=='n_first' and r['register_layout']['size_per_thread'][2]==v]
                selected=possible[0]['id'] if possible else d
            elif name=='fanout_only':
                selected=min(pool,key=lambda r:(static(r)['u']*static(r)['w'],
                    -r['register_layout']['size_per_thread'][2],r['placement']!='n_first'))['id']
            else:
                mm=theta['model'];rs=theta['resource_model'];pred={}
                for r in pool:
                    q=resources.estimate(r,rs)['q_point'];p=fit.predict(mm,r,q)
                    pred[r['id']]=max(p['control_ns'],p['MC_covariate_ns'])
                selected=min(pred,key=lambda key:(pred[key],key!=d,key))
            choice[geo]=selected
        choices[name]=choice;baselines.append(score_policy(name,choice,observed))
    for label,key in (('C33_final_nominal','development_decisions'),('C33_final_conservative','development_decisions'),
                      ('C33_outer_nominal','outer_decisions'),('C33_outer_conservative','outer_decisions')):
        rows=dev[key];rows=[r['decision'] for r in rows] if key=='outer_decisions' else rows
        choices[label]={f'{r["geometry"][0]}x{r["geometry"][1]}w{r["geometry"][2]}':
            r['nominal_choice'] if label.endswith('nominal') else r['selected'] for r in rows}
        baselines.append(score_policy(label,choices[label],observed))
    ablations=[]
    for mode,record in cv.items():
        choice={}
        for fold in record['folds']:
            rows=observed[fold['held']];case=rows[0]['case'];pool=candidates(case['BM'],case['BN'],case['W'])
            d=upstream(case['BM'],case['BN'],case['W']);rp=fold['resource_point_model'];mm=fold['model']
            points={r['id']:fit.predict(mm,r,resources.q_for(r,resources.point(rp,r)))['tile_ns'] for r in pool}
            choice[fold['held']]=min(points,key=lambda key:(points[key],key!=d,key))
        ablations.append({**score_policy(mode,choice,observed),**record['metrics']})
    zero_rows=[];zero_predictions=[];zero_choices={}
    for fold in theta['outer_folds']:
        rows=[r for r in bundle['full'] if r['geometry']==fold['held'] and r['q_condition']=='natural']
        points=[]
        for row in rows:
            case=row['case'];q=resources.q_for(case,resources.point(fold['resource_point_model'],case))
            p=fit.predict(fold['model'],case,q)
            points.append({**p,'tile_ns':max(p['control_ns'],p['MC_covariate_ns']),'branch':'diagnostic_zero_base'})
        zero_rows+=rows;zero_predictions+=points
        # Same predicted value for each worker; aliases do not add policies.
        pred={r['layout']:p['tile_ns'] for r,p in zip(rows,points)}
        case=rows[0]['case'];default=upstream(case['BM'],case['BN'],case['W'])
        zero_choices[fold['held']]=min(pred,key=lambda k:(pred[k],k!=default,k))
    zero=measure(zero_rows,zero_predictions)
    baselines.append(score_policy('zero_base_outer',zero_choices,observed))
    value={'baselines':baselines,'ablations':ablations,
        'zero_base_outer_metrics':{k:v for k,v in zero.items() if k!='paired'},
        'all_policy_choices_measured':True,'fixed_vec_illegal_falls_back_to_default':True,
        'fanout_tie_rule':'min u*w, larger vec, N-first; no time or hardware counter feature',
        'zero_base_scope':'Diagnostic max(current fitted operational control,current fitted MC); no FULL-target fitting of envelope.',
        'final_model_scores_are_in_sample':True,'outer_scores_are_unseen_geometry':True}
    c.derived(c.OUT/'baselines.json',value)
    print('C3.3 baselines scored on all 61 measured candidates')
    return value


if __name__=='__main__': run()
