"""CPU validator: immutable inputs, nested exclusions, resource and policy arithmetic."""
import argparse
import ast
import math
import statistics as st
import subprocess
from collections import Counter
import numpy as np
from experiments.tma_reduction_layout.phase11.c34 import common as c, resources, fit, model
from experiments.tma_reduction_layout.phase11.c34 import diagnostics
from experiments.tma_reduction_layout.phase11.c34.calibrate import (
    initialize, prediction_rows, without_pairs, training_bank)
from experiments.tma_reduction_layout.phase11.c33 import data, resources as old_resources
from experiments.tma_reduction_layout.phase11.c33.calibrate import measure, score, support
from experiments.tma_reduction_layout.phase11.c33.evaluate import score_policy
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def same(a,b):
    assert c.encode(a)==c.encode(b)


def close(a,b):
    assert abs(a-b)<=1e-7*max(1.,abs(a),abs(b)),(a,b)


def metrics_close(a,b):
    for k,v in a.items():
        if isinstance(v,(int,float)):close(float(v),float(b[k]))


def recompute_resources(calibration):
    _,unique=old_resources.census();resources.initialize(unique)
    inputs=c.read(c.OUT/'resource_inputs.json')
    same(inputs['rows'],list(resources.ROWS));assert inputs['timing_fields']==[]
    assert len(inputs['rows'])==74
    assert all(set(r)=={'geometry','case','cubin_SHA256','n_regs','natural_q',
        'local_bytes','static_shared_bytes','dynamic_shared_bytes'} for r in inputs['rows'])
    models=[];checks=[];pair_checks=[];rectangle_probes=0
    olddev=c.read(c.prior.OUT/'development.json')
    protected={tuple(g) for g in c.protocol()['protected_D1']}
    for row in inputs['rows']:
        ca=row['case'];assert (ca['BM'],ca['BN'],ca['W']) not in protected
    for fold in calibration['folds']:
        held=fold['held'];rm=fold['resource_model']
        obs=[r for r in resources.ROWS if r['geometry']==held]
        geo=obs[0]['case'];geo=[geo['BM'],geo['BN'],geo['W']]
        pool=candidates(*geo)
        same(rm,resources.calibrate((held,),pool));models.append(rm)
        assert held not in rm['training_geometries'] and held in rm['excluded_geometries']
        previous=next(r for r in olddev['outer_decisions'] if r['held']==held)
        default=next(r for r in pool if r['id']==upstream(*geo))
        truth={r['case']['id']:r for r in obs}
        for ca in pool:
            observed=truth[ca['id']]
            new=resources.estimate(ca,rm);prev=old_resources.estimate(ca,previous['resource_model'])
            record=next(r for r in calibration['checks'] if r['layout']==ca['id'])
            same(record['before'],prev);same(record['after'],new)
            assert record['n_regs_observed']==observed['n_regs'] and record['q_observed']==observed['natural_q']
            assert record['after_regs_upper_covered']==(observed['n_regs']<=new['n_regs_interval'][1])
            assert record['before_regs_upper_covered']==(observed['n_regs']<=prev['n_regs_interval'][1])
            assert record['after_q_interval_covered']==(observed['natural_q'] in new['q_values'])
            assert record['before_q_interval_covered']==(prev['q_interval'][0]<=observed['natural_q']<=prev['q_interval'][1])
            checks.append(record)
            if ca['id']==default['id']:continue
            scenes,_=resources.pair(ca,default,rm)
            pp=next(r for r in calibration['paired_checks'] if r['layout']==ca['id'])
            qnew=sorted({(s['candidate_q'],s['default_q']) for s in scenes})
            same(qnew,pp['after_q_pairs'])
            common,delta,_=resources.envelope(rm,ca)
            pc,pd=resources.point(rm,ca),resources.point(rm,default)
            for shift in (common[0],common[1],0.,sum(common)/2):
                for diff in (delta[0],delta[1],0.,sum(delta)/2):
                    for suc in (False,True):
                        for sud in (False,True):
                            direct=(old_resources.q_for(ca,pc+shift+diff,suc),
                                    old_resources.q_for(default,pd+shift,sud))
                            assert direct in qnew,(ca['id'],direct,qnew)
                            rectangle_probes+=1
            oldcase=next(r for r in previous['decision']['candidates'] if r['layout']==ca['id'])
            qold=sorted({(s['candidate_q'],s['default_q']) for s in oldcase['resource_scenarios']})
            same(qold,pp['before_q_pairs'])
            actual=(observed['natural_q'],truth[default['id']]['natural_q'])
            assert pp['after_covered']==(actual in qnew)
            assert pp['before_covered']==(actual in qold)
            pair_checks.append(pp)
    final=calibration['final_model']
    query_geos=c.prior.prior.protocol()['calibration_geometries']+c.protocol()['protected_D1']
    queries=[case for geo in query_geos for case in candidates(*geo)]
    same(final,resources.calibrate((),queries));models.append(final)
    calibration_records=0
    for rm in models:
        for key,cal in rm['calibrations'].items():
            assert not cal['uses_timing'] and cal['calibration_pool']['distinct_geometries']>=3
            assert cal['calibration_total']==len(cal['observations'])
            for obs in cal['observations']:
                assert obs['geometry'] not in obs['base_envelope']['pool']['geometries']
                assert obs['geometry'] in obs['model_excluded_geometries']
                assert all(h in obs['model_excluded_geometries'] for h in rm['excluded_geometries'])
                assert all(h not in obs['base_envelope']['pool']['geometries'] for h in rm['excluded_geometries'])
            # Rebuilding above rechecks every margin candidate in frozen order;
            # explicitly ensure there was no smaller passing pair in its trace.
            assert all(t['failed_indices'] for t in cal['trials'][:-1])
            assert cal['calibration_target_met']==(cal['calibration_covered']==cal['calibration_total'])
            if cal['target_R']<=4:
                assert max(cal['base_envelope']['pool']['R_support'])<=8
            calibration_records+=1
    totals={k:sum(r[k] for r in checks) for k in ('before_regs_upper_covered',
        'after_regs_upper_covered','before_q_interval_covered','after_q_interval_covered')}
    totals.update(layouts=len(checks),pairs=len(pair_checks),independent_held_geometries=9,
        before_joint_q_covered=sum(r['before_covered'] for r in pair_checks),
        after_joint_q_covered=sum(r['after_covered'] for r in pair_checks))
    same(totals,calibration['totals']);assert len(checks)==61 and len(pair_checks)==52
    assert rectangle_probes==3328
    return calibration_records


def validate_cv(bundle,cv,final):
    folds_rebuilt=0;winner_losses=0
    for key,record in cv.items():
        assert len(record['folds'])==9
        all_rows=[];all_pred=[];all_pairs=[]
        for fold in record['folds']:
            held=fold['held'];mm=fold['model'];rr=fold['selected_recipe']
            assert held not in mm['training_geometries']+mm['control_training_geometries']+mm['MC']['training_geometries']
            assert held in mm['excluded_geometries']
            assert held not in fold['resource_point_model']['training_geometries']
            same(fold['resource_point_model'],fit.resource_point_model((held,)))
            recipes=fit.recipes(record['mode'],record['group'])
            assert {(r['family'],r['eta'],r['ridge'],r['eta_q_slope']) for r in fold['inner_scores']}==set(recipes)
            same(min(fold['inner_scores'],key=fit.recipe_key),rr)
            for row in fold['inner_scores']:
                assert set(row['inner_held_geometries'])==set(bundle_geo(bundle))-{held}
                assert held not in row['inner_losses_by_geometry']
                close(row['inner_relative_loss'],st.mean(row['inner_losses_by_geometry'].values()))
            rebuilt=fit.train((held,),record['mode'],rr['family'],rr['eta'],rr['ridge'],rr['eta_q_slope'])
            for component in ('full','MC','control'):
                np.testing.assert_allclose(rebuilt[component]['coefficients'],mm[component]['coefficients'],rtol=1e-7,atol=1e-8)
            rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
            pp=prediction_rows(mm,rows);m=measure(rows,pp)
            metrics_close(without_pairs(m),fold['metrics'])
            all_rows+=rows;all_pred+=pp;all_pairs+=m['paired'];folds_rebuilt+=1
            # Recompute the selected recipe's inner validation score; no second
            # exhaustive grid search is required by this arithmetic validator.
            for inner in rr['inner_held_geometries']:
                excluded=tuple(sorted((held,inner)))
                im=fit.train(excluded,record['mode'],rr['family'],rr['eta'],rr['ridge'],rr['eta_q_slope'])
                ir=data.medians([r for r in bundle['full_fit'] if r['geometry']==inner])
                loss=fit.objective(ir,np.array([p['tile_ns'] for p in prediction_rows(im,ir)]))
                close(loss,rr['inner_losses_by_geometry'][inner]);winner_losses+=1
        metrics_close(without_pairs(measure(all_rows,all_pred)),record['metrics'])
        same(all_pairs,record['paired_outer_residuals'])
    rr=final['selected_recipe'];same(min(final['inner_scores'],key=fit.recipe_key),rr)
    rebuilt=fit.train((),'full',rr['family'],rr['eta'],rr['ridge'],rr['eta_q_slope'])
    np.testing.assert_allclose(rebuilt['full']['coefficients'],final['model']['full']['coefficients'],rtol=1e-7,atol=1e-8)
    for inner in rr['inner_held_geometries']:
        im=fit.train((inner,),'full',rr['family'],rr['eta'],rr['ridge'],rr['eta_q_slope'])
        rows=data.medians([r for r in bundle['full_fit'] if r['geometry']==inner])
        loss=fit.objective(rows,np.array([p['tile_ns'] for p in prediction_rows(im,rows)]))
        close(loss,rr['inner_losses_by_geometry'][inner]);winner_losses+=1
    return folds_rebuilt,winner_losses


def bundle_geo(bundle):return {r['geometry'] for r in bundle['full']}


def run(pre_freeze=False,committed=False):
    c.protect_prior();p=c.protocol()
    assert p['plan_SHA256']==c.sha((c.SOURCE/'PLAN.md').read_bytes())
    assert p['C33_theta_SHA256']==c.sha((c.prior.OUT/'theta_H100.json').read_bytes())
    assert not p['resource_calibration_uses_timing'] and not p['GPU_allocations']
    resource=c.read(c.OUT/'resource_calibration.json');calibration_records=recompute_resources(resource)
    bundle=initialize();cv=c.read(c.OUT/'cross_validation.json')
    final=c.read(c.OUT/'final_time_model.json');fold_count,inner_count=validate_cv(bundle,cv,final)
    theta=c.read(c.OUT/'theta_H100.json');dev=c.read(c.OUT/'development.json')
    same(theta['model'],final['model']);same(theta['resource_model'],resource['final_model'])
    same(theta['paired_outer_residuals'],cv['full:auto']['paired_outer_residuals'])
    assert not theta['physical_overlap_identified'] and theta['FULL_primary_target']
    protected={tuple(g) for g in p['protected_D1']}
    assert len(protected)==10
    for row in bundle['full']+bundle['steady']:
        ca=row['case'];assert (ca['BM'],ca['BN'],ca['W']) not in protected
    for row in bundle['control']:assert tuple(row['geometry_vector']) not in protected
    oldtheta=c.read(c.prior.OUT/'theta_H100.json');olddev=c.read(c.prior.OUT/'development.json')
    scored=0
    for item in dev['outer']:
        held=item['held'];fold=next(f for f in cv['full:auto']['folds'] if f['held']==held)
        rm=next(f['resource_model'] for f in resource['folds'] if f['held']==held)
        bank=training_bank(bundle,fold);same(bank,item['calibration_bank'])
        assert all(r['geometry']!=held for r in bank)
        local={**theta,'model':fold['model'],'resource_model':rm,
            'performance_support':support([r for r in bundle['full'] if r['geometry']!=held])}
        expected=model.predict(item['decision']['geometry'],local,bank=bank)
        same(expected,item['decision'])
        rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
        same(score(expected,rows),item['evaluation']);scored+=1
        ro=next(r for r in dev['resource_only_outer'] if r['held']==held)
        of=next(f for f in oldtheta['outer_folds'] if f['held']==held)
        ob=next(f for f in olddev['outer_decisions'] if f['held']==held)['calibration_bank']
        rp=model.predict(expected['geometry'],{**oldtheta,'model':of['model'],'resource_model':rm,
            'performance_support':local['performance_support']},bank=ob)
        same(rp,ro['decision']);same(score(rp,rows),ro['evaluation']);scored+=1
        for cand in expected['candidates']:
            if cand['uncertainty'] is None:continue
            assert held not in cand['uncertainty']['training_geometries']
            assert not cand['uncertainty']['negative_residuals_reflected']
            assert cand['uncertainty']['additional_margin_relative']>=.01
            worst=-100*max(s['predicted_delta_relative']+s['uncertainty_upper_relative'] for s in cand['resource_scenarios'])
            close(worst,cand['worst_gain_percent'])
            if cand['robust_switch']:assert worst>3 and not cand['reasons']
    for item in dev['full_data_fit']:
        dec=item['decision'];same(dec,model.predict(dec['geometry'],theta))
        geo=dec['geometry'];name=f'{geo[0]}x{geo[1]}w{geo[2]}'
        rows=[r for r in bundle['full'] if r['geometry']==name and r['q_condition']=='natural']
        same(score(dec,rows),item['evaluation']);scored+=1
    observed={}
    for row in data.medians([r for r in bundle['full'] if r['q_condition']=='natural']):
        observed.setdefault(row['geometry'],[]).append(row)
    baselines=c.read(c.OUT/'baselines.json')
    for record in baselines['policies']+baselines['ablations']:
        choices={r['geometry']:r['selected'] for r in record['items']}
        measured=score_policy(record['policy'],choices,observed)
        for key,value in measured.items():same(value,record[key])
    blind=c.read(c.OUT/'D1_predictions.json');resourceblind=c.read(c.OUT/'resource_only_D1_predictions.json')
    assert blind['count']==10==len(blind['cases'])
    assert {tuple(r['geometry']) for r in blind['cases']}==protected
    assert Counter(r['stratum'] for r in blind['cases'])=={'empirically_unseen_core':5,'legacy_boundary':5}
    assert blind['theta_SHA256']==c.sha((c.OUT/'theta_H100.json').read_bytes())
    for r in blind['cases']:
        assert r['history_category']=='strict_zero_mention'
        same(r['prediction'],model.predict(r['geometry'],theta))
    for r in resourceblind['cases']:
        same(r['prediction'],model.predict(r['geometry'],{**oldtheta,'resource_model':resource['final_model']}))
    saved_diag=c.read(c.OUT/'diagnostics.json');same(saved_diag,diagnostics.build(write=False))
    assert all(r['max_absolute_inner_loss_difference']<1e-10 for r in saved_diag['C33_scalar_score_reproduction'])
    early_path=c.OUT/'execution/resource_only_focus_preflight.json'
    early=c.read(early_path)
    correction=c.read(c.OUT/'execution/resource_only_focus_correction.json')
    assert correction['prior_snapshot_SHA256']==c.sha(early_path.read_bytes())
    for r in early:
        focus=next(f for f in saved_diag['focus_cases'] if f['geometry']==r['geometry'])
        # Numeric preflight decomposition was unaffected by the omitted support
        # veto. Its early selection metadata is preserved as an audited error.
        for key in ('nominal_gain_percent','fixed_q_bound_percent','worst_gain_percent','q_pairs'):
            same(r['resource_only'][key],focus['resource_only'][key])
        fixed=next(f for f in correction['corrected_cases'] if f['geometry']==r['geometry'])
        same(fixed['resource_only'],focus['resource_only'])
        assert fixed['selected']==focus['resource_only_selected']
    for root in (c.SOURCE,c.OUT):
        assert not any(root.rglob('*.cubin')) and not any(root.rglob('*.ptx')) and not any(root.rglob('*.ncu-rep'))
    for name in ('fit.py','resources.py','model.py','calibrate.py','resource_audit.py','diagnostics.py'):
        tree=ast.parse((c.SOURCE/name).read_text())
        imported=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(x and x.split('.')[0] in ('triton','torch','modal','ctypes','subprocess') for x in imported),name
    if not pre_freeze:
        frozen=c.read(c.OUT/'freeze.json')
        for name,digest in frozen['files'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    if committed:
        assert not subprocess.check_output(['git','diff','HEAD','--',str(c.SOURCE.relative_to(c.ROOT)),str(c.OUT.relative_to(c.ROOT))],cwd=c.ROOT)
    result={'stage':'C3.4','status':'OK','protected_prior_files':len(c.read(c.OUT/'prior_manifest.json')['files']),
        'resource_calibrations_rebuilt':calibration_records,'resource_timing_fields':0,
        'outer_resource_upper_coverage':resource['totals']['after_regs_upper_covered'],
        'outer_resource_layouts':61,'outer_resource_joint_q_covered':resource['totals']['after_joint_q_covered'],
        'outer_resource_pairs':52,'outer_performance_folds_rebuilt':fold_count,
        'inner_winner_losses_recomputed':inner_count,'policies_scored':scored,
        'D1_predictions_recomputed':10,'D1_strict_zero_mention':10,
        'D1_compiled':False,'D1_timed':False,'GPU_allocations':0,'native_rebuilds':0,
        'runtime_grid_selector_feature':False,'historical_26_6_used':False,
        'scope':'Integrity, exclusion, coverage, arithmetic and freeze validation. Empirical coverage and predictive efficacy are reported separately.'}
    if not pre_freeze:c.derived(c.OUT/'validation.json',result)
    print('C3.4 validator OK; resource/time calibrations separate; 61 FULL and 10 static D1 predictions',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pre-freeze',action='store_true');parser.add_argument('--committed',action='store_true')
    a=parser.parse_args();run(a.pre_freeze,a.committed)
