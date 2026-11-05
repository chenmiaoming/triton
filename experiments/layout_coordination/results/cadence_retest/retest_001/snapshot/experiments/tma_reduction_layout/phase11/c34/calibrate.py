"""Nested overlap-family comparison. Resource calibration is an independent input."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import statistics as st
from experiments.tma_reduction_layout.phase11.c34 import common as c, fit, model
from experiments.tma_reduction_layout.phase11.c33 import data, resources as old_resources
from experiments.tma_reduction_layout.phase11.c33.calibrate import measure, score, support
from experiments.tma_reduction_layout.phase11.c33.evaluate import score_policy
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def initialize():
    bundle=data.load();_,unique=old_resources.census();bundle['resource_rows']=unique
    fit.initialize(bundle)
    return bundle


def prediction_rows(mm,rows):
    key=tuple(mm['excluded_geometries'])
    return [fit.predict(mm,r['case'],fit.static_q(r['case'],key,r['q_condition'])) for r in rows]


def without_pairs(metrics):
    return {k:v for k,v in metrics.items() if k!='paired'}


def cv_mode(mode):
    bundle=fit.BUNDLE
    geos=sorted({r['geometry'] for r in bundle['full']})
    folds=[];all_rows=[];all_pred=[];pairs=[]
    for held in geos:
        selected=fit.choose((held,),mode,'auto')
        rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
        pp=prediction_rows(selected['model'],rows);measured=measure(rows,pp)
        folds.append({'held':held,**selected,'metrics':without_pairs(measured),
            'resource_point_model':fit.resource_point_model((held,))})
        all_rows+=rows;all_pred+=pp;pairs+=measured['paired']
        print('C3.4',mode,'auto',held,selected['selected_recipe']['family'],
              'FULL RMSE%',round(measured['FULL_relative_RMSE_percent'],3),flush=True)
        c.derived(c.OUT/'cv_checkpoints'/f'{mode}.json',{'mode':mode,'folds':folds,
            'protocol_SHA256':c.sha((c.OUT/'protocol.json').read_bytes())})
    record={'folds':folds,'metrics':without_pairs(measure(all_rows,all_pred)),
        'paired_outer_residuals':pairs,'mode':mode,'group':'auto'}
    if mode!='full':return {mode+':auto':record}
    result={'full:auto':record}
    # The auto inner menu is the union of these two restricted searches.
    # Reuse every evaluated inner score, refit only each restricted winner.
    for group in ('scalar','q_dependent'):
        ff=[];rr=[];pp=[];pairs=[]
        allowed={'full_structural','partial_overlap' if group=='scalar' else 'partial_overlap_q'}
        for fold in folds:
            scores=[s for s in fold['inner_scores'] if s['family'] in allowed]
            recipe=min(scores,key=fit.recipe_key);held=fold['held']
            mm=fit.train((held,),'full',recipe['family'],recipe['eta'],recipe['ridge'],recipe['eta_q_slope'])
            rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
            pred=prediction_rows(mm,rows);measured=measure(rows,pred)
            ff.append({'held':held,'model':mm,'selected_recipe':recipe,'inner_scores':scores,
                'selection_group':group,'metrics':without_pairs(measured),
                'resource_point_model':fold['resource_point_model']})
            rr+=rows;pp+=pred;pairs+=measured['paired']
        result['full:'+group]={'folds':ff,'metrics':without_pairs(measure(rr,pp)),
            'paired_outer_residuals':pairs,'mode':'full','group':group}
    return result


def final_fit():return fit.choose((),'full','auto')


def cross_validation(bundle):
    records={};final=None
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        jobs={pool.submit(cv_mode,mode):mode for mode in c.protocol()['modes']}
        jobs[pool.submit(final_fit)]='final'
        for job in as_completed(jobs):
            result=job.result()
            if jobs[job]=='final':final=result
            else:records.update(result)
            c.derived(c.OUT/'performance_checkpoint.json',records)
    assert final is not None
    c.derived(c.OUT/'cross_validation.json',records)
    c.derived(c.OUT/'final_time_model.json',final)
    return records,final


def training_bank(bundle,fold):
    bank=[];held=fold['held'];recipe=fold['selected_recipe']
    for geo in sorted({r['geometry'] for r in bundle['full']} - {held}):
        excluded=tuple(sorted((held,geo)))
        mm=fit.train(excluded,'full',recipe['family'],recipe['eta'],recipe['ridge'],recipe.get('eta_q_slope',0.))
        rows=[r for r in bundle['full'] if r['geometry']==geo and r['q_condition']=='natural']
        bank+=data.paired(rows,prediction_rows(mm,rows))
    return bank


def decisions(bundle,cv,final):
    resource=c.read(c.OUT/'resource_calibration.json')
    theta={'stage':'C3.4','model':final['model'],'selected_recipe':final['selected_recipe'],
        'inner_scores':final['inner_scores'],'outer_folds':cv['full:auto']['folds'],
        'paired_outer_residuals':cv['full:auto']['paired_outer_residuals'],
        'fixed_R_tables':bundle['local_tables'],'resource_model':resource['final_model'],
        'performance_support':support(bundle['full']),
        'development_geometries':sorted({r['geometry'] for r in bundle['full']}),
        'development_FULL_candidates':61,'C2d_analysis_SHA256':bundle['analysis_SHA256'],
        'resource_inputs_SHA256':c.sha((c.OUT/'resource_inputs.json').read_bytes()),
        'FULL_primary_target':True,'physical_overlap_identified':False,
        'historical_26_6_used':False,'selector_runtime_grid_feature':False,
        'D1_compiled':False,'D1_timed':False,'GPU_allocations':0}
    c.derived(c.OUT/'theta_H100.json',theta)
    oldtheta=c.read(c.prior.OUT/'theta_H100.json')
    olddev=c.read(c.prior.OUT/'development.json')
    out=[];out_resource=[];final_decisions=[]
    for fold in cv['full:auto']['folds']:
        held=fold['held'];rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
        case=rows[0]['case'];geo=[case['BM'],case['BN'],case['W']]
        rm=next(f['resource_model'] for f in resource['folds'] if f['held']==held)
        bank=training_bank(bundle,fold)
        local={**theta,'model':fold['model'],'resource_model':rm,
               'performance_support':support([r for r in bundle['full'] if r['geometry']!=held])}
        prediction=model.predict(geo,local,bank=bank)
        out.append({'held':held,'decision':prediction,'evaluation':score(prediction,rows),
            'calibration_bank':bank,'resource_model_key':held,
            'calibration_conditional_on_inner_selected_recipe':True})
        previous=next(o for o in olddev['outer_decisions'] if o['held']==held)
        previous_fold=next(f for f in oldtheta['outer_folds'] if f['held']==held)
        resource_local={**oldtheta,'model':previous_fold['model'],'resource_model':rm,
            'performance_support':local['performance_support']}
        rp=model.predict(geo,resource_local,bank=previous['calibration_bank'])
        out_resource.append({'held':held,'decision':rp,'evaluation':score(rp,rows),
            'performance_model_and_bank':'unchanged C3.3 outer fit and training-only bank'})
        fp=model.predict(geo,theta);final_decisions.append({'decision':fp,'evaluation':score(fp,rows)})
        print('C3.4 decisions',held,'resource-only',rp['changed'],'full',prediction['changed'],flush=True)
    result={'outer':out,'resource_only_outer':out_resource,'full_data_fit':final_decisions,
            'final_fit_is_in_sample':True,'outer_independent_geometries':9}
    c.derived(c.OUT/'development.json',result)
    # Preserve the old historical split; only predictions are updated.
    split=c.read(c.prior.OUT/'D1_predictions.json')
    newblind=[];resourceblind=[]
    for item in split['cases']:
        meta={k:v for k,v in item.items() if k!='prediction'}
        newblind.append({**meta,'prediction':model.predict(item['geometry'],theta)})
        resourceblind.append({**meta,'prediction':model.predict(item['geometry'],
            {**oldtheta,'resource_model':resource['final_model']})})
    c.derived(c.OUT/'D1_predictions.json',{'stage':'D1','model_version':'C3.4',
        'count':10,'cases':newblind,'theta_SHA256':c.sha((c.OUT/'theta_H100.json').read_bytes()),
        'historical_split_source_SHA256':c.sha((c.prior.OUT/'history_audit.json').read_bytes()),
        'D1_compiled':False,'D1_timed':False})
    c.derived(c.OUT/'resource_only_D1_predictions.json',{'count':10,'cases':resourceblind,
        'time_model_and_performance_bank':'unchanged C3.3','D1_compiled':False,'D1_timed':False})
    evaluate(bundle,cv,result)


def evaluate(bundle,cv,dev):
    observed={}
    for row in data.medians([r for r in bundle['full'] if r['q_condition']=='natural']):
        observed.setdefault(row['geometry'],[]).append(row)
    previous=c.read(c.prior.OUT/'baselines.json')['baselines']
    policies=[r for r in previous if r['policy'] in ('upstream_default','fixed_vec1',
              'fixed_vec2','fixed_vec4','fixed_vec8','fanout_only','C33_outer_conservative','C33_outer_nominal')]
    for key,prefix in (('outer','C34_outer'),('resource_only_outer','C34_resource_only'),('full_data_fit','C34_in_sample')):
        for suffix,field in (('nominal','nominal_choice'),('conservative','selected')):
            choices={}
            for item in dev[key]:
                dec=item['decision'];geo=dec['geometry'];name=f'{geo[0]}x{geo[1]}w{geo[2]}'
                choices[name]=dec[field]
            policies.append(score_policy(prefix+'_'+suffix,choices,observed))
    ablations=[]
    for name,record in cv.items():
        choices={}
        for fold in record['folds']:
            rows=observed[fold['held']];case=rows[0]['case'];pool=candidates(case['BM'],case['BN'],case['W'])
            default=upstream(case['BM'],case['BN'],case['W'])
            pred={x['id']:fit.predict(fold['model'],x,fit.static_q(x,(fold['held'],)))['tile_ns'] for x in pool}
            choices[fold['held']]=min(pred,key=lambda k:(pred[k],k!=default,k))
        ablations.append({**score_policy(name,choices,observed),**record['metrics']})
    c.derived(c.OUT/'baselines.json',{'policies':policies,'ablations':ablations,
        'metric':'128/256/384 MiB FULL ns/tile/SM service slopes',
        'fixed_footprint_raw_time_is_different_estimand':True,
        'all_selected_candidates_measured':True})


def run():
    bundle=initialize();cv,final=cross_validation(bundle)
    decisions(bundle,cv,final)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--decisions-only',action='store_true')
    if parser.parse_args().decisions_only:
        decisions(initialize(),c.read(c.OUT/'cross_validation.json'),c.read(c.OUT/'final_time_model.json'))
    else: run()
