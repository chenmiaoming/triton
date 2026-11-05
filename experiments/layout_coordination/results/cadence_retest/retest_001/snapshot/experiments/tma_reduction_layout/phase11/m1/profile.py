"""Development-only scalar-eta profiles and actual design dependencies."""
import numpy as np
from experiments.tma_reduction_layout.phase11.m1 import common as c
from experiments.tma_reduction_layout.phase11.c34 import fit
from experiments.tma_reduction_layout.phase11.c33 import data,resources


def svd_design(a,names):
    scale=np.linalg.norm(a,axis=0);scale[scale==0]=1.
    z=a/scale
    _,s,v=np.linalg.svd(z,full_matrices=False)
    tol=max(z.shape)*np.finfo(float).eps*s[0]
    rank=int(np.sum(s>tol));null=[]
    for vector in v[rank:]:
        vector=vector/np.max(np.abs(vector))
        null.append({'normalized_column_relation':{n:float(x) for n,x in zip(names,vector) if abs(x)>1e-6},
            'relative_residual_norm':float(np.linalg.norm(z@vector)/max(np.linalg.norm(vector),1e-20))})
    correlations=z.T@z
    pairs=[{'a':names[i],'b':names[j],'cosine':float(correlations[i,j])}
           for i in range(len(names)) for j in range(i+1,len(names)) if abs(correlations[i,j])>.98]
    return {'rows':len(a),'columns':len(names),'rank':rank,'singular_values':s.tolist(),
        'numerical_tolerance':float(tol),'column_norms':scale.tolist(),'null_relations':null,
        'near_parallel_columns_cosine_over_098':pairs,
        'positive_spectrum_condition_number':float(s[0]/s[rank-1])}


def run():
    p=c.protocol();c.protect_prior()
    frozen=c.read(c.source('stage_c34','theta_H100.json'))
    bundle=data.load();_,rr=resources.census();bundle['resource_rows']=rr;fit.initialize(bundle)
    assert len({r['geometry'] for r in bundle['full']})==9
    forbidden={tuple(r['geometry']) for r in c.read(c.source('stage_d1','preregistration/protocol.json'))['D1_cases']}
    assert all(tuple(r['geometry_vector']) not in forbidden for r in bundle['full'])
    recipes=[s for s in frozen['inner_scores'] if s['family']=='partial_overlap']
    assert len(recipes)==25
    curves=[];models=[]
    for eta in p['eta_grid']:
        candidates=[r for r in recipes if r['eta']==eta]
        selected=min(candidates,key=fit.recipe_key)
        curves.append({'eta':eta,'best_existing_inner_CV_ridge':selected['ridge'],
            'best_existing_inner_CV_loss':selected['inner_relative_loss'],
            'fixed_ridge_inner_CV_loss':next(r['inner_relative_loss'] for r in candidates if r['ridge']==p['fixed_ridge']),
            'all_existing_inner_CV_scores':candidates})
        for label,ridge in (('fixed_ridge',p['fixed_ridge']),('CV_ridge',selected['ridge'])):
            mm=fit.train((),'full','partial_overlap',eta,ridge,0.)
            base=fit.basis((),'full');rows=base['rows'];names=mm['full']['feature_names']
            f=[fit.components(base,r['case'],r['q'],'partial_overlap')[0] for r in rows]
            a=np.array([[x[n] for n in names] for x in f]);beta=np.array(mm['full']['coefficients'])
            preds=[fit.predict(mm,r['case'],r['q']) for r in rows]
            active=np.array([x['branch']=='onchip' for x in preds])
            contribution=a*beta[None,:]
            nonzero=np.flatnonzero(beta>1e-9)
            mccoef=dict(zip(mm['MC']['feature_names'],mm['MC']['coefficients']))
            mcraw=np.array([sum(mccoef.get(n,0.)*x.get(n,0.) for n in mccoef) for x in f])
            collapsed={n:float(beta[j]+beta[names.index('calibrated_MC_response')]*mccoef.get(n,0.))
                       for j,n in enumerate(names) if n!='calibrated_MC_response'}
            acollapsed=np.array([sum(collapsed[n]*x[n] for n in collapsed) for x in f])
            redundancy={'MC_raw_min_ns':float(mcraw.min()),'MC_clipped_rows':int(np.sum(mcraw<0)),
                'MC_column_max_reconstruction_error_ns':float(np.max(np.abs(mcraw-a[:,names.index('calibrated_MC_response')]))),
                'collapsed_coefficients':collapsed,
                'A_FULL_max_collapse_error_ns':float(np.max(np.abs(acollapsed-a@beta))),
                'scope':'Exact over the audited design when MC nonnegative clipping is inactive. Collapsed coefficients are an algebraic diagnostic, not a new fitted selector.'}
            record={'profile':label,'eta':eta,'ridge':ridge,'model':mm,
                'FULL_feature_design':svd_design(a,names),
                'exposed_branch_design':svd_design(a[active],names) if np.any(active) else None,
                'active_column_design':svd_design(a[:,nonzero],[names[i] for i in nonzero]),
                'coefficients':dict(zip(names,beta.tolist())),
                'RMS_model_contribution_ns':dict(zip(names,np.sqrt(np.mean(contribution**2,axis=0)).tolist())),
                'MC_redundancy':redundancy,
                'training_joint_loss':fit.objective(rows,np.array([r['tile_ns'] for r in preds])),
                'rows':[{'geometry':r['geometry'],'layout':r['layout'],'q':r['q'],
                    'observed_FULL_ns':r['target_ns'],**pred} for r,pred in zip(rows,preds)]}
            models.append(record)
            c.derived(c.OUT/'checkpoints'/f'profile_{label}_{eta}.json',record)
            print('M1 eta profile',label,eta,'ridge',ridge,'design rank',record['FULL_feature_design']['rank'],flush=True)
    outer=[]
    for fold in frozen['outer_folds']:
        scores=[r for r in fold['inner_scores'] if r['family']=='partial_overlap']
        outer.append({'held':fold['held'],'scalar_eta_profiles':[
            {'eta':e,'CV_loss':min(r['inner_relative_loss'] for r in scores if r['eta']==e)} for e in p['eta_grid']],
            'original_selected_recipe':fold['selected_recipe']})
    # Prediction spread and coefficient spread answer different questions.
    comparison=[]
    for label in ('fixed_ridge','CV_ridge'):
        variants=[r for r in models if r['profile']==label]
        for i,row in enumerate(variants[0]['rows']):
            values=[v['rows'][i] for v in variants]
            comparison.append({'profile':label,'geometry':row['geometry'],'layout':row['layout'],'q':row['q'],
                'FULL_range_ns':[min(x['tile_ns'] for x in values),max(x['tile_ns'] for x in values)],
                'A_FULL_range_ns':[min(x['FULL_onchip_response_ns'] for x in values),max(x['FULL_onchip_response_ns'] for x in values)],
                'FULL_range_relative_to_observed':(max(x['tile_ns'] for x in values)-min(x['tile_ns'] for x in values))/row['observed_FULL_ns']})
    value={'status':'OK','development_geometries':sorted({r['geometry'] for r in bundle['full']}),
        'D1_used_for_fit':False,'CV_curve':curves,'fits':models,'outer_training_profiles':outer,
        'prediction_spread':comparison,'weighted_nnls_diagnostic_units':'Dimensionless weighted/paired objective residual, despite historical field name training_rmse_ns.',
        'interpretation':'Rank deficiency is not proof that all nonnegative active coefficients are nonunique. CV profiles are not likelihood confidence intervals. Contributions describe the model, not separately measured hardware latency.'}
    c.derived(c.OUT/'identifiability.json',value)
    print('M1 identifiability complete: ten development-only fits; no D1 fitting',flush=True)
    return value


if __name__=='__main__':run()
