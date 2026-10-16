"""Development-only model search and same-fold paired cross-validation."""
from itertools import product
from collections import defaultdict
import numpy as np
from experiments.tma_reduction_layout.phase11.c31 import common as c
from experiments.tma_reduction_layout.phase11.c31.data import development,primitive_tables
from experiments.tma_reduction_layout.phase11.c31.features import features
from experiments.tma_reduction_layout.phase11.c31.fit import nnls
from experiments.tma_reduction_layout.phase11.c3.static import upstream

MODES=('full','Load-blind','Comm-blind','Resource/Envelope-blind')


def fit(rows,recipe,tables,mode):
    fs=[features(r['case'],r['q'],recipe,tables,mode) for r in rows]
    names=sorted(fs[0]);assert all(sorted(f)==names for f in fs)
    a=np.array([[f[n] for n in names] for f in fs]);y=np.array([r['service_ns'] for r in rows])
    beta,diagnostic=nnls(a,y,recipe['ridge'])
    return {'recipe':recipe,'mode':mode,'feature_names':names,
            'coefficients':beta.tolist(),'diagnostic':diagnostic}


def predict(model,case,q,tables):
    f=features(case,q,model['recipe'],tables,model['mode'])
    return float(np.dot(model['coefficients'],[f[n] for n in model['feature_names']]))


def folds(rows,level):
    groups=sorted({r['geometry'] if level=='LOGO' else r['layout'] for r in rows})
    for group in groups:
        train=[r for r in rows if (r['geometry'] if level=='LOGO' else r['layout'])!=group]
        yield group,train


def paired_rows(rows,model,tables,level,held):
    # Both predictions always use this same fitted model. Hold out all q and
    # workers of a layout/geometry; worker residuals remain paired within cohort.
    idx={(r['layout'],r['q_condition'],r['cohort'],r['worker'],r['GPU_UUID']):r for r in rows}
    pairs=[]
    for r in rows:
        case=r['case'];default=upstream(case['BM'],case['BN'],case['W'])
        if r['layout']==default:continue
        included=r['geometry']==held if level=='LOGO' else held in (r['layout'],default)
        if not included:continue
        key=(default,r['q_condition'],r['cohort'],r['worker'],r['GPU_UUID'])
        assert key in idx,key
        d=idx[key]
        cp=predict(model,case,r['q'],tables);dp=predict(model,d['case'],d['q'],tables)
        observed=r['service_ns']-d['service_ns'];estimated=cp-dp
        pairs.append({'level':level,'held':held,'layout':r['layout'],'default':default,
            'geometry':r['geometry'],'q_condition':r['q_condition'],
            'candidate_q':r['q'],'default_q':d['q'],
            'cohort':r['cohort'],'worker':r['worker'],'GPU_UUID':r['GPU_UUID'],
            'observed_delta_ns':observed,'predicted_delta_ns':estimated,
            'delta_residual_ns':observed-estimated,
            'default_residual_ns':d['service_ns']-dp,
            'observed_default_ns':d['service_ns'],'predicted_default_ns':dp,
            'observed_candidate_ns':r['service_ns'],'predicted_candidate_ns':cp})
    return pairs


def cross_validate(cells,workers,recipe,tables,mode,levels=('LOGO','LOLO')):
    residuals=[];models=[];point_errors=[]
    for level in levels:
        for held,train in folds(cells,level):
            model=fit(train,recipe,tables,mode)
            residuals.extend(paired_rows(workers,model,tables,level,held))
            models.append({'level':level,'held':held,'model':model})
            for r in workers:
                if (r['geometry'] if level=='LOGO' else r['layout'])!=held:continue
                estimated=predict(model,r['case'],r['q'],tables)
                point_errors.append({'level':level,'layout':r['layout'],'q':r['q_condition'],
                    'observed_ns':r['service_ns'],'predicted_ns':estimated,
                    'error_ns':r['service_ns']-estimated})
    errors=[r['delta_residual_ns'] for r in residuals]
    return {'paired_residuals':residuals,'fold_models':models,'point_errors':point_errors,
        'paired_rmse_ns':float(np.sqrt(np.mean(np.square(errors)))),
        'paired_max_abs_error_ns':max(map(abs,errors))}


def residual_bands(cv):
    grouped=defaultdict(list)
    for r in cv['paired_residuals']:
        # Never pool a low-q communication/latency response into a bulk pair.
        q=min(r['candidate_q'],r['default_q'])
        band='q1' if q<2 else 'q2' if q<3 else 'q_ge_3'
        grouped[band].append(r)
    return {band:{'records':rs,
        'delta_residual_ns_interval':[min(r['delta_residual_ns'] for r in rs),max(r['delta_residual_ns'] for r in rs)],
        'default_residual_ns_interval':[min(r['default_residual_ns'] for r in rs),max(r['default_residual_ns'] for r in rs)],
        'interpretation':'Retain joint (default residual, difference residual) records, not a Cartesian product of marginal extremes.'}
            for band,rs in grouped.items()}


def run():
    cases,workers,cells=development();tables=primitive_tables()
    protocol=c.read(c.OUT/'preregistration/protocol.json')
    options=protocol['model_recipes'];search=[]
    for basis,q_response,ridge in product(options['features'],options['q_response'],options['ridge']):
        recipe={'features':basis,'q_response':q_response,'ridge':ridge}
        cv=cross_validate(cells,workers,recipe,tables,'full',levels=('LOGO',))
        model=fit(cells,recipe,tables,'full')
        search.append({'recipe':recipe,'LOGO_paired_rmse_ns':cv['paired_rmse_ns'],
            'LOGO_paired_max_abs_error_ns':cv['paired_max_abs_error_ns'],
            'feature_count':len(model['feature_names']),'full_fit':model})
    best=min(search,key=lambda r:(r['LOGO_paired_rmse_ns'],r['feature_count'],-r['recipe']['ridge']))
    recipe=best['recipe'];results={}
    for mode in MODES:
        model=fit(cells,recipe,tables,mode)
        cv=cross_validate(cells,workers,recipe,tables,mode)
        results[mode]={'model':model,'cross_validation':cv,'paired_error_bands':residual_bands(cv)}
    # Common transfer-rate uncertainty, with cohort identity retained. No FULL
    # timings or K1/MO/grid1 values enter these parameters.
    transfer=[]
    a=c.read(c.BASE/'results/phase11/stage_c2b/analysis.json')
    for r in a['HBM_anchors']:
        transfer.append({'BM':32,'BN':r['N'],'W':8 if r['N']==64 else 4,
            'cohort':'C2a','worker':r['invocation'],
            'service_ns':1000*r['HBM_tile_SM_us'],'event_us':r['median_us'],
            'calibration_grid':r['grid'],'SMs':132})
    c16=c.read(c.BASE/'results/phase11/stage_c16/analysis.json')
    for r in c16['cells']:
        if ':control:TMA' not in r['tag'] or r['cache']!='warm':continue
        n=64 if 'N64_' in r['tag'] else 128
        full=32768 if n==64 else 16384
        if r['grid']!=full:continue
        for worker,value in enumerate(r['worker_medians_us'],1):
            transfer.append({'BM':32,'BN':n,'W':8 if n==64 else 4,'cohort':'C1.6',
                'worker':worker,'service_ns':value*1000*132/full,
                'event_us':value,'calibration_grid':full,'SMs':132})
    for r in transfer:r['ns_per_tile_byte']=r['service_ns']/(2*r['BM']*r['BN']+4*r['BN'])
    theta={'version':'c3_v1','stage':'C3.1','target':'H100-sm90',
        'recipe':recipe,'primitive_shape_tables':tables,'models':results,
        'transfer':{'measurements':transfer,
            'ns_per_tile_byte_interval':[min(r['ns_per_tile_byte'] for r in transfer),max(r['ns_per_tile_byte'] for r in transfer)],
            'normalization':'event_us * 1000 * SMs / calibration_grid; grid is provenance, not a predictor feature',
            'extrapolation':'Unmeasured geometry uses byte-rate envelope, a D1-testable assumption, not an independent transfer measurement.'},
        'roofline':{'base_ns':0.,'formula':'T_tile = max(s_TMA, S_MC)',
            'interpretation':'Steady bottleneck approximation; single-pass setup remains auxiliary, not an unidentified offset.'},
        'training':{'target':'MC D1 steady service_ns','cell_medians':len(cells),'worker_estimates':len(workers),
            'layouts':len(cases),'geometry_groups':2,'FULL_K1_MO_grid1_in_fit':False},
        'interval_type':'Empirical paired cross-validation error envelope; not a statistical guarantee or held-out validation.'}
    c.write(c.V1/'model_search.json',{'selection_criterion':options['selection_loss'],'candidates':search,'selected':recipe})
    c.write(c.V1/'theta_H100.json',theta)
    print('selected',recipe)
    for mode,r in results.items():
        print(mode,'fit RMSE',round(r['model']['diagnostic']['training_rmse_ns'],3),
            'paired CV RMSE',round(r['cross_validation']['paired_rmse_ns'],3),
            'bands',{k:[round(x,3) for x in v['delta_residual_ns_interval']] for k,v in r['paired_error_bands'].items()})
    return theta


if __name__=='__main__':run()
