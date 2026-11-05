"""CPU-only independent arithmetic, holdout exclusions and freeze validator."""
import argparse
import ast
from collections import Counter
import math
import statistics as st
import subprocess
import numpy as np
from experiments.tma_reduction_layout.phase11.c33 import common as c, data, fit, resources, model
from experiments.tma_reduction_layout.phase11.c33.calibrate import measure, prediction_rows, score
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def close(a,b):
    assert abs(a-b)<=1e-7*max(1.,abs(a),abs(b)),(a,b)


def metric_close(a,b):
    for key,value in a.items():
        if isinstance(value,(int,float)) and key in b: close(float(value),float(b[key]))


def run(pre_freeze=False,committed=False):
    c.prior.protect_prior()
    cv2=c.read(c.prior.OUT/'validation.json');assert cv2['status']=='OK' and cv2['FULL_candidates']==61
    for name,digest in c.read(c.prior.OUT/'freeze.json')['files'].items():
        assert c.sha((c.ROOT/name).read_bytes())==digest,name
    p=c.protocol();binding=c.prior.protocol()
    assert p['C2d_protocol_SHA256']==c.sha(c.encode(binding))
    assert p['plan_SHA256']==c.sha((c.SOURCE/'PLAN.md').read_bytes())==binding['C33_method_SHA256']
    assert not p['historical_26_6_used'] and not p['global_grid_feature'] and not p['signed_residual_reflection']
    theta=c.read(c.OUT/'theta_H100.json');cv=c.read(c.OUT/'cross_validation.json')
    dev=c.read(c.OUT/'development.json');baselines=c.read(c.OUT/'baselines.json')
    bundle=data.load();allrows,unique=resources.census();bundle['resource_rows']=unique;fit.initialize(bundle)
    calibration=c.read(c.OUT/'resource_calibration.json')
    assert calibration['rows']==allrows and calibration['unique']==unique
    assert theta['resource_model']==calibration['full_single_pass_model']
    assert theta['C2d_analysis_SHA256']==bundle['analysis_SHA256']
    protected={tuple(v) for v in p['protected_D1']}
    assert len(protected)==10 and (16,64,8) not in protected
    actual={r['layout'] for r in bundle['full']}
    expected={case['id'] for geo in binding['calibration_geometries'] for case in candidates(*geo)}
    assert actual==expected and len(actual)==61
    assert len({r['geometry'] for r in bundle['full']})==9
    assert 4 in theta['performance_support']['R']
    for row in bundle['full']+bundle['control']+bundle['steady']:
        if row.get('case'):
            x=row['case'];assert (x['BM'],x['BN'],x['W']) not in protected
    assert theta['FULL_primary_target'] and not theta['physical_overlap_identified']
    assert not theta['historical_26_6_used'] and not theta['selector_runtime_grid_feature']
    assert not theta['D1_compiled'] and not theta['D1_timed']
    held_geometry_metrics={};recount=0
    for mode,record in cv.items():
        assert len(record['folds'])==9
        all_held_rows=[];all_predictions=[];all_pairs=[]
        for fold in record['folds']:
            held=fold['held'];mm=fold['model'];rm=fold['resource_point_model']
            assert held in mm['excluded_geometries'] and held in rm['excluded_geometries']
            assert all(held not in mm[key] for key in ('training_geometries','control_training_geometries'))
            assert held not in mm['MC']['training_geometries'] and held not in rm['training_geometries']
            assert all(held not in r['inner_held_geometries'] for r in fold['inner_scores'])
            winner=min(fold['inner_scores'],key=lambda r:(r['inner_relative_loss'],r['family']!='full_structural',-r['ridge'],r['eta']))
            assert winner==fold['selected_recipe']
            assert all(v>=0 for v in mm['full']['coefficients'])
            assert not any('grid' in n.lower() or 'P_shape' in n or '26.6' in n for n in mm['full']['feature_names'])
            rows=[r for r in bundle['full'] if r['geometry']==held and r['q_condition']=='natural']
            pp=prediction_rows(mm,rows)
            measured=measure(rows,pp);conditional=measure(rows,prediction_rows(mm,rows,conditional=True))
            metric_close(measured,fold['metrics']);metric_close(conditional,fold['conditional_metrics'])
            all_held_rows+=rows;all_predictions+=pp;all_pairs+=measured['paired'];recount+=1
            if mode=='full': held_geometry_metrics[held]=measured
        total=measure(all_held_rows,all_predictions)
        metric_close(total,record['metrics'])
        assert all_pairs==record['paired_outer_residuals']
    assert theta['paired_outer_residuals']==cv['full']['paired_outer_residuals']
    # Re-fit selected recipes only (not a second exhaustive hyperparameter
    # search). This checks saved coefficients against immutable development data.
    for fold in theta['outer_folds']:
        rr=fold['selected_recipe']
        rebuilt=fit.train((fold['held'],),'full',rr['family'],rr['eta'],rr['ridge'])
        for key in ('full','MC'):
            np.testing.assert_allclose(rebuilt[key]['coefficients'],fold['model'][key]['coefficients'],rtol=1e-7,atol=1e-8)
    winner=min(theta['inner_scores'],key=lambda r:(r['inner_relative_loss'],r['family']!='full_structural',-r['ridge'],r['eta']))
    assert theta['selected_recipe']==winner
    rr=winner;rebuilt=fit.train((),'full',rr['family'],rr['eta'],rr['ridge'])
    np.testing.assert_allclose(rebuilt['full']['coefficients'],theta['model']['full']['coefficients'],rtol=1e-7,atol=1e-8)
    eval_count=0
    for decision,evaluation in zip(dev['development_decisions'],dev['development_evaluation']):
        assert decision==model.predict(decision['geometry'],theta)
        geo=decision['geometry'];rows=[r for r in bundle['full'] if r['q_condition']=='natural' and r['geometry']==f'{geo[0]}x{geo[1]}w{geo[2]}']
        assert evaluation==score(decision,rows) and evaluation['all_candidates_measured'];eval_count+=1
    for outer,evaluation in zip(dev['outer_decisions'],dev['outer_evaluation']):
        held=outer['held'];fold=next(f for f in theta['outer_folds'] if f['held']==held)
        bank=outer['calibration_bank'];rm=outer['resource_model']
        assert held not in rm['training_geometries'] and held in rm['excluded_geometries']
        assert all(r['geometry']!=held for r in bank)
        seen={r['geometry'] for r in bank};assert len(seen)==8
        local={**theta,'model':fold['model'],'resource_model':rm,
            'performance_support':{'R':sorted({r['case']['BM']*r['case']['BN']//(32*r['case']['W']) for r in bundle['full'] if r['geometry']!=held}),
                                  'W':sorted({r['case']['W'] for r in bundle['full'] if r['geometry']!=held})}}
        expected_decision=model.predict(outer['decision']['geometry'],local,bank=bank)
        assert outer['decision']==expected_decision
        for candidate in expected_decision['candidates']:
            u=candidate['uncertainty']
            if u:
                assert held not in u['training_geometries'] and u['distinct_geometries']>=3
                assert u['additional_margin_relative']>=.01 and not u['negative_residuals_reflected']
            if candidate['robust_switch']:
                assert candidate['worst_gain_percent']>3 and not candidate['resource']['spill_uncertainty']
        rows=[r for r in bundle['full'] if r['q_condition']=='natural' and r['geometry']==held]
        assert evaluation==score(expected_decision,rows);eval_count+=1
    assert dev['outer_covered']==sum(r['covered'] for r in dev['outer_upper_tail_coverage'])
    assert dev['outer_total']==len(dev['outer_upper_tail_coverage'])==52
    assert dev['outer_switches']==sum(r['changed'] for r in dev['outer_evaluation'])
    assert dev['outer_adverse_selections_over_3_percent']==sum(r['adverse_over_3_percent'] for r in dev['outer_evaluation'])
    # Baseline and ablation outcomes must all be scoreable measured candidates.
    obs={}
    for r in data.medians([r for r in bundle['full'] if r['q_condition']=='natural']):
        obs.setdefault(r['geometry'],{})[r['layout']]=r['target_ns']
    for policy in baselines['baselines']+baselines['ablations']:
        assert len(policy['items'])==9
        for item in policy['items']:
            values=obs[item['geometry']];d=values[item['default']];s=values[item['selected']]
            close(item['observed_gain_percent'],100*(1-s/d))
            close(item['oracle_regret_percent'],100*(s/min(values.values())-1))
    history=c.read(c.OUT/'history_audit.json');split=history['D1_split']
    assert split==c.read(c.prior.OUT/'preregistration/D1_split.json')
    blind=c.read(c.OUT/'D1_predictions.json')
    assert blind['count']==10==len(blind['cases'])
    assert Counter(r['stratum'] for r in blind['cases'])=={'empirically_unseen_core':5,'legacy_boundary':5}
    assert {tuple(r['geometry']) for r in blind['cases']}==protected
    assert blind['theta_SHA256']==c.sha((c.OUT/'theta_H100.json').read_bytes())
    for row in blind['cases']:
        assert row['history_category']=='strict_zero_mention'
        assert row['prediction']==model.predict(row['geometry'],theta)
    assert not list(c.OUT.rglob('*.cubin')) and not list(c.OUT.rglob('*.ptx')) and not list(c.OUT.rglob('*.ncu-rep'))
    for name in ('model.py','features.py','fit.py','resources.py','data.py'):
        tree=ast.parse((c.SOURCE/name).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(x and x.split('.')[0] in ('triton','torch','modal','ctypes','subprocess') for x in imports),(name,imports)
    if not pre_freeze:
        frozen=c.read(c.OUT/'freeze.json')
        for name,digest in frozen['files'].items(): assert c.sha((c.ROOT/name).read_bytes())==digest,name
        previous=subprocess.check_output(['git','show',frozen['C2d_commit']+':'+str((c.prior.OUT/'validation.json').relative_to(c.ROOT))],cwd=c.ROOT)
        assert c.sha(previous)==frozen['C2d_validation_SHA256']
    if committed:
        assert not subprocess.check_output(['git','diff','HEAD','--',str(c.SOURCE.relative_to(c.ROOT)),str(c.OUT.relative_to(c.ROOT))],cwd=c.ROOT)
    result={'stage':'C3.3','status':'OK','outer_modes':5,'outer_folds_recomputed':recount,
        'selected_recipe_fits_recomputed':10,'development_FULL_candidates':61,
        'development_decisions_scored':eval_count,'unmeasured_selected_candidates':0,
        'D1_predictions_recomputed':10,'D1_strict_zero_mention':10,
        'D1_core':5,'D1_boundary':5,'D1_compiled':False,'D1_timed':False,
        'C33_GPU_allocations':0,'native_rebuilds':0,'signed_residuals_reflected':False,
        'runtime_grid_selector_feature':False,'historical_26_6_used':False,
        'outer_switches':dev['outer_switches'],'outer_adverse_over_3_percent':dev['outer_adverse_selections_over_3_percent'],
        'scope':'Integrity/exclusions/numerical recomputation PASS. Nine-geometry development validation is separate from unexecuted D1 efficacy.'}
    if not pre_freeze: c.derived(c.OUT/'validation.json',result)
    print('C3.3 validator OK; 61 measured candidates, 45 outer folds, 10 D1 predictions; GPU untouched')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pre-freeze',action='store_true');parser.add_argument('--committed',action='store_true')
    a=parser.parse_args();run(a.pre_freeze,a.committed)
