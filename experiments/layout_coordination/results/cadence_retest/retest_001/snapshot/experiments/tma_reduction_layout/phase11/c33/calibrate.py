"""FULL-target nested LOGO calibration, including training-only decision bands."""
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c33 import common as c, data, fit, resources, model
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def support(rows):
    return {'R':sorted({r['case']['BM']*r['case']['BN']//(32*r['case']['W']) for r in rows}),
        'W':sorted({r['case']['W'] for r in rows}),
        'BM':sorted({r['case']['BM'] for r in rows}),
        'BN':sorted({r['case']['BN'] for r in rows})}


def prediction_rows(mm,rows,rm=None,conditional=False):
    out=[]
    for r in rows:
        if conditional: q=r['q']
        elif rm is None: q=fit.static_q(r['case'],tuple(mm['excluded_geometries']),r['q_condition'])
        else:
            q=resources.estimate(r['case'],rm)['q_point']
            if r['q_condition']!='natural': q=min(q,int(r['q_condition']))
        out.append(fit.predict(mm,r['case'],q))
    return out


def measure(rows,pred):
    paired=data.paired(rows,pred)
    errors=[p['tile_ns']-r['target_ns'] for r,p in zip(rows,pred)]
    relative=[e/r['target_ns'] for r,e in zip(rows,errors)]
    return {'observations':len(rows),'FULL_RMSE_ns':float(np.sqrt(np.mean(np.square(errors)))),
        'FULL_relative_RMSE_percent':100*float(np.sqrt(np.mean(np.square(relative)))),
        'FULL_median_absolute_percent':100*st.median(abs(v) for v in relative),
        'FULL_max_absolute_percent':100*max(abs(v) for v in relative),
        'paired_RMSE_ns':float(np.sqrt(np.mean([p['delta_residual_ns']**2 for p in paired]))),
        'paired_relative_RMSE_percent':100*float(np.sqrt(np.mean([p['relative_residual']**2 for p in paired]))),
        'paired':paired}


def score(decision,rows):
    obs={}
    for r in rows:
        obs.setdefault(r['layout'],[]).append(r['target_ns'])
    point={key:st.median(values) for key,values in obs.items()}
    d=point[decision['upstream_default']]
    oracle=min(point,key=point.get)
    return {'geometry':decision['geometry'],'all_candidates_measured':len(point)==len(decision['candidates']),
        'default_ns':d,'oracle':oracle,'oracle_gain_percent':100*(1-point[oracle]/d),
        'nominal':decision['nominal_choice'],'nominal_predicted_gain_percent':decision['nominal_gain_percent'],
        'nominal_observed_gain_percent':100*(1-point[decision['nominal_choice']]/d),
        'selected':decision['selected'],'selected_observed_gain_percent':100*(1-point[decision['selected']]/d),
        'changed':decision['changed'],'adverse_over_3_percent':point[decision['selected']]>d*1.03,
        'oracle_regret_percent':100*(point[decision['selected']]/point[oracle]-1)}


def run():
    bundle=data.load();resource_rows,unique=resources.census();bundle['resource_rows']=unique
    fit.initialize(bundle)
    full_geos=sorted({r['geometry'] for r in bundle['full']})
    resource_model=resources.calibrate(unique,'full_single_pass')
    c.derived(c.OUT/'resource_calibration.json',{'rows':resource_rows,'unique':unique,
        'full_single_pass_model':resource_model})
    outer,ablation=[],{}
    for mode in c.protocol()['modes']:
        folds=[];pairs=[];all_rows=[];all_pred=[];conditional_pairs=[]
        for held in full_geos:
            selected=fit.choose((held,),mode)
            mm=selected['model']
            rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
            rp=fit.resource_point_model((held,))
            pred=prediction_rows(mm,rows)
            diagnostics=measure(rows,pred)
            conditional=measure(rows,prediction_rows(mm,rows,conditional=True))
            fold={'held':held,'selected_recipe':selected['selected_recipe'],'inner_scores':selected['inner_scores'],
                'model':mm,'resource_point_model':rp,
                'metrics':{k:v for k,v in diagnostics.items() if k!='paired'},
                'conditional_metrics':{k:v for k,v in conditional.items() if k!='paired'}}
            pairs+=diagnostics['paired'];conditional_pairs+=conditional['paired']
            all_rows+=rows;all_pred+=pred;folds.append(fold)
            print('C3.3',mode,'outer',held,'recipe',mm['kind'],mm['eta'],mm['ridge'],
                'FULL relRMSE',round(diagnostics['FULL_relative_RMSE_percent'],3),flush=True)
        pooled=measure(all_rows,all_pred)
        ablation[mode]={'metrics':{k:v for k,v in pooled.items() if k!='paired'},
            'paired_outer_residuals':pairs,'conditional_paired_residuals':conditional_pairs,
            'folds':folds}
        if mode=='full': outer=folds
        c.derived(c.OUT/'calibration_checkpoint.json',{'completed_modes':list(ablation),
            'modes':ablation,'D1_compiled':False,'D1_timed':False})
    selected=fit.choose((),'full')
    theta={'stage':'C3.3','model':selected['model'],'selected_recipe':selected['selected_recipe'],
        'inner_scores':selected['inner_scores'],'outer_folds':outer,
        'paired_outer_residuals':ablation['full']['paired_outer_residuals'],
        'fixed_R_tables':bundle['local_tables'],'resource_model':resource_model,
        'performance_support':support(bundle['full']),
        'development_geometries':full_geos,'development_FULL_candidates':61,
        'C2d_analysis_SHA256':bundle['analysis_SHA256'],
        'physical_overlap_identified':False,'FULL_primary_target':True,
        'historical_26_6_used':False,'selector_runtime_grid_feature':False,
        'D1_compiled':False,'D1_timed':False}
    c.derived(c.OUT/'theta_H100.json',theta)
    c.derived(c.OUT/'cross_validation.json',ablation)
    decisions(bundle)


def decisions(bundle=None):
    if bundle is None:
        bundle=data.load();_,unique=resources.census();bundle['resource_rows']=unique
        fit.initialize(bundle)
    else:
        unique=bundle['resource_rows']
    theta=c.read(c.OUT/'theta_H100.json')
    outer=theta['outer_folds']
    full_geos=theta['development_geometries']
    decisions=[];evaluation=[];outer_decisions=[];outer_evaluation=[];coverage=[]
    for fold in outer:
        held=fold['held']
        rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
        case=rows[0]['case'];geo=[case['BM'],case['BN'],case['W']]
        final=model.predict(geo,theta);decisions.append(final);evaluation.append(score(final,rows))
        bank=[]
        # Conditional-on-selected-recipe calibration uses only the eight outer
        # training geometries. The ninth evaluation geometry is never included.
        for calib in full_geos:
            if calib==held: continue
            excluded=tuple(sorted((held,calib)))
            recipe=fold['selected_recipe']
            mm=fit.train(excluded,'full',recipe['family'],recipe['eta'],recipe['ridge'])
            rr=[r for r in bundle['full'] if r['geometry']==calib and r['q_condition']=='natural']
            bank+=data.paired(rr,prediction_rows(mm,rr))
        rm=resources.calibrate(unique,'full_single_pass',held)
        local={**theta,'model':fold['model'],'resource_model':rm,
            'performance_support':support([r for r in bundle['full'] if r['geometry']!=held])}
        decision=model.predict(geo,local,bank=bank)
        outer_decisions.append({'held':held,'decision':decision,'calibration_bank':bank,
            'resource_model':rm,'calibration_conditional_on_inner_selected_recipe':True})
        outer_evaluation.append(score(decision,rows))
        actual=data.paired(rows,prediction_rows(fold['model'],rows))
        observed={r['layout']:[] for r in actual}
        for r in actual: observed[r['layout']].append(r['relative_residual'])
        for candidate in decision['candidates']:
            if candidate['uncertainty'] is None: continue
            res=st.median(observed[candidate['layout']])
            coverage.append({'geometry':held,'layout':candidate['layout'],
                'observed_relative_residual':res,'upper':candidate['uncertainty']['upper_relative'],
                'covered':res<=candidate['uncertainty']['upper_relative'],
                'stratum':candidate['structure'],
                'calibration_geometries':candidate['uncertainty']['training_geometries']})
        print('C3.3 outer selector',held,'changed',decision['changed'],flush=True)
    result={'development_decisions':decisions,'development_evaluation':evaluation,
        'outer_decisions':outer_decisions,'outer_evaluation':outer_evaluation,
        'outer_upper_tail_coverage':coverage,'outer_covered':sum(r['covered'] for r in coverage),
        'outer_total':len(coverage),'development_switches':sum(r['changed'] for r in evaluation),
        'outer_switches':sum(r['changed'] for r in outer_evaluation),
        'outer_adverse_selections_over_3_percent':sum(r['adverse_over_3_percent'] for r in outer_evaluation),
        'uncertainty_is_not_a_formal_confidence_bound':True}
    c.derived(c.OUT/'development.json',result)
    print('C3.3 calibration complete; development/outer switches',result['development_switches'],result['outer_switches'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--decisions-only',action='store_true')
    if parser.parse_args().decisions_only: decisions()
    else: run()
