"""Nested whole-geometry fits to FULL and paired FULL differences."""
from collections import Counter, defaultdict
from functools import lru_cache
import math
import numpy as np
from experiments.tma_reduction_layout.phase11.c34 import common as c
from experiments.tma_reduction_layout.phase11.c33 import data, features
from experiments.tma_reduction_layout.phase11.c33 import resources
from experiments.tma_reduction_layout.phase11.c31.fit import nnls
from experiments.tma_reduction_layout.phase11.c32 import regression as reg
from experiments.tma_reduction_layout.phase11.c3.static import upstream

BUNDLE=None


def initialize(bundle):
    global BUNDLE
    BUNDLE=bundle
    basis.cache_clear();train.cache_clear();choose.cache_clear();resource_point_model.cache_clear()


@lru_cache(maxsize=None)
def resource_point_model(excluded):
    rows=[r for r in BUNDLE['resource_rows'] if r['family']=='full_single_pass'
        and r['geometry'] not in excluded and not r['local_bytes']]
    model=resources.fit_model(rows)
    model['training_geometries']=sorted({r['geometry'] for r in rows})
    model['excluded_geometries']=list(excluded)
    return model


def static_q(case,excluded,condition='natural'):
    model=resource_point_model(excluded)
    q=resources.q_for(case,resources.point(model,case))
    return q if condition=='natural' else min(q,int(condition))


def weights(rows,scale=None):
    sizes=Counter((r['geometry'],r['q_condition']=='natural') for r in rows)
    w=np.array([1/math.sqrt(sizes[r['geometry'],r['q_condition']=='natural']) for r in rows])
    if scale is not None: w=w/np.maximum(1.,scale)
    return w


def control_fit(rows):
    names=sorted(features.control(rows[0]['geometry_vector'],rows[0]['q']))
    a=np.array([[features.control(r['geometry_vector'],r['q'])[n] for n in names] for r in rows])
    y=np.array([r['target_ns'] for r in rows]);ww=weights(rows,y)
    # Bandwidth and finite-residency response are separate max branches.
    bulk=np.array([n in ('bytes_KiB','output_KiB') for n in names]);lat=~bulk
    x=a*ww[:,None];target=y*ww
    scale=np.maximum(np.linalg.norm(x,axis=0),1e-10);z=x/scale
    ca,_=nnls(z[:,bulk],target,.001);cb,_=nnls(z[:,lat],target,.001)
    best=None
    for sa,sb in ((1.,.5),(.5,1.),(1.,1.)):
        beta=np.zeros(len(names));beta[bulk]=ca*sa;beta[lat]=cb*sb
        step=1/(2*np.linalg.norm(z,2)**2+.002)
        for _ in range(250):
            pa=z[:,bulk]@beta[bulk];pb=z[:,lat]@beta[lat]
            active=pa>=pb;error=np.maximum(pa,pb)-target
            grad=np.zeros(len(names))
            grad[bulk]=2*z[:,bulk].T@(error*active)
            grad[lat]=2*z[:,lat].T@(error*(~active))
            beta=np.maximum(0,beta-step*(grad+.002*beta))
        loss=float(np.mean((np.maximum(z[:,bulk]@beta[bulk],z[:,lat]@beta[lat])-target)**2))
        if best is None or loss<best[0]: best=(loss,beta.copy())
    return {'feature_names':names,'coefficients':(best[1]/scale).tolist(),
        'bulk_mask':bulk.tolist(),'kind':'operational_control_max_response',
        'physical_transfer_latency_identified':False}


def control_predict(model,geom,q):
    f=features.control(geom,q);x=np.array([f[n] for n in model['feature_names']])
    beta=np.array(model['coefficients']);bulk=np.array(model['bulk_mask'])
    return max(1.,float(x[bulk]@beta[bulk]),float(x[~bulk]@beta[~bulk]))


def matching_pairs(rows):
    index={(r['layout'],r['q_condition']):i for i,r in enumerate(rows)}
    result=[]
    for i,r in enumerate(rows):
        case=r['case'];default=upstream(case['BM'],case['BN'],case['W'])
        if r['layout']==default: continue
        j=index.get((default,r['q_condition']))
        if j is None and r['q_condition']!='natural':
            matches=[k for k,d in enumerate(rows) if d['layout']==default and d['q']==r['q']]
            if matches: j=matches[0]
        if j is not None: result.append((i,j))
    return result


@lru_cache(maxsize=128)
def basis(excluded,mode):
    excluded=set(excluded)
    full=data.medians([r for r in BUNDLE['full_fit'] if r['geometry'] not in excluded])
    control=data.medians([r for r in BUNDLE['control_fit'] if r['geometry'] not in excluded])
    mc=data.medians([r for r in BUNDLE['steady'] if r['geometry'] not in excluded])
    hm=control_fit(control)
    fs=[features.operation(r['case'],r['q'],BUNDLE['local_tables'],mode) for r in mc]
    wm=weights(mc,np.array([r['target_ns'] for r in mc]))
    names=sorted(fs[0]);a=np.array([[f[n] for n in names] for f in fs])
    beta,diag=nnls(a*wm[:,None],np.array([r['target_ns'] for r in mc])*wm,.001)
    mm={'feature_names':names,'coefficients':beta.tolist(),'diagnostic':diag,
        'training_geometries':sorted({r['geometry'] for r in mc})}
    return {'rows':full,'control':hm,'MC':mm,'mode':mode,
        'excluded_geometries':sorted(excluded),
        'training_geometries':sorted({r['geometry'] for r in full}),
        'control_training_geometries':sorted({r['geometry'] for r in control})}


def components(base,case,q,family):
    mc=max(0.,reg.predict(base['MC'],features.operation(case,q,BUNDLE['local_tables'],base['mode'])))
    h=control_predict(base['control'],[case['BM'],case['BN'],case['W']],q)
    f=features.operation(case,q,BUNDLE['local_tables'],base['mode'],mc=mc,
        transfer=h if family=='full_structural' and base['mode']!='Resource/Envelope-blind' else None)
    return f,h,mc


def row_scales(rows):
    defaults={r['geometry']:r['target_ns'] for r in rows if
        r['q_condition']=='natural' and r['layout']==upstream(r['case']['BM'],r['case']['BN'],r['case']['W'])}
    idx={(r['layout'],r['q_condition']):r for r in rows}
    return np.array([idx.get((upstream(r['case']['BM'],r['case']['BN'],r['case']['W']),r['q_condition']),
        {'target_ns':defaults[r['geometry']]})['target_ns'] for r in rows])


def objective(rows,pred):
    y=np.array([r['target_ns'] for r in rows]);s=row_scales(rows)
    abs_errors=(pred-y)/s
    buckets=defaultdict(list)
    for r,e in zip(rows,abs_errors): buckets[r['geometry'],r['q_condition']=='natural'].append(float(e*e))
    absolute=np.mean([np.mean(e) for e in buckets.values()])
    pair_buckets=defaultdict(list)
    for i,j in matching_pairs(rows):
        err=((pred[i]-pred[j])-(y[i]-y[j]))/y[j]
        pair_buckets[rows[i]['geometry'],rows[i]['q_condition']=='natural'].append(float(err*err))
    paired=np.mean([np.mean(e) for e in pair_buckets.values()]) if pair_buckets else 0.
    return float(.5*absolute+.5*paired)


@lru_cache(maxsize=4096)
def train(excluded,mode,family,eta,ridge,eta_q_slope=0.):
    base=basis(excluded,mode);rows=base['rows']
    if mode=='Resource/Envelope-blind': family='full_structural'
    items=[components(base,r['case'],r['q'],family) for r in rows]
    fs=[t[0] for t in items];names=sorted(fs[0]);a=np.array([[f[n] for n in names] for f in fs])
    h=np.array([t[1] for t in items]);y=np.array([r['target_ns'] for r in rows]);sc=row_scales(rows)
    w=weights(rows,sc);pairs=matching_pairs(rows)
    count=Counter((rows[i]['geometry'],rows[i]['q_condition']=='natural') for i,j in pairs)
    pw=np.array([1/(max(1.,y[j])*math.sqrt(count[rows[i]['geometry'],rows[i]['q_condition']=='natural'])) for i,j in pairs])
    pi=np.array([i for i,j in pairs],int);pj=np.array([j for i,j in pairs],int)
    partial=family in ('partial_overlap','partial_overlap_q')
    eq=np.clip(eta+eta_q_slope*np.log2(np.array([r['q'] for r in rows])/8.),0.,1.)
    offset=(1-eq)*h if partial else np.zeros(len(rows));floor=h if partial else np.zeros(len(rows))
    def design(active):
        aa=a*active[:,None];cc=np.where(active,offset,floor)
        x=np.vstack((aa*w[:,None],(aa[pi]-aa[pj])*pw[:,None]))
        target=np.r_[(y-cc)*w,((y[pi]-y[pj])-(cc[pi]-cc[pj]))*pw]
        return x,target
    beta,diag=nnls(*design(np.ones(len(rows),bool)),ridge)
    def predicted(bb): return np.maximum(floor,offset+a@bb)
    best=objective(rows,predicted(beta))
    # Piecewise-linear active-set refits. Preserve the best joint FULL/pair loss;
    # platform observations remain explicit in every objective evaluation.
    if partial:
        for _ in range(12):
            active=(offset+a@beta>=floor)|(y>floor)
            proposal,_=nnls(*design(active),ridge)
            trials=[((1-t)*beta+t*proposal) for t in (1.,.5,.25,.125)]
            values=[objective(rows,predicted(t)) for t in trials]
            k=int(np.argmin(values))
            if values[k]>=best-1e-10: break
            beta=trials[k];best=values[k]
    scale=np.maximum(np.linalg.norm(a*w[:,None],axis=0),1e-10);z=a/scale
    inv=np.linalg.pinv((a*w[:,None]/scale).T@(a*w[:,None]/scale)+max(ridge,.001)*np.eye(len(names)))
    lev=np.einsum('ij,jk,ik->i',z,inv,z)
    result={key:value for key,value in base.items() if key!='rows'}
    result.update(kind=family,eta=float(eta),eta_q_slope=float(eta_q_slope),ridge=float(ridge),
        full={'feature_names':names,'coefficients':beta.tolist(),'diagnostic':diag,
              'scale':scale.tolist(),'inverse_gram':inv.tolist(),
              'feature_support':{n:[float(a[:,i].min()),float(a[:,i].max())] for i,n in enumerate(names)},
              'median_training_leverage':float(np.median(lev))},
        joint_training_relative_loss=best,physical_overlap_identified=False,
        FULL_primary_target=True,global_grid_feature=False,historical_26_6_used=False)
    return result


def predict(model,case,q):
    f,h,mc=components(model,case,q,model['kind'])
    a=max(0.,reg.predict(model['full'],f))
    eq=min(1.,max(0.,model['eta']+model.get('eta_q_slope',0.)*math.log2(q/8.)))
    if model['kind'] in ('partial_overlap','partial_overlap_q'):
        t=h+max(0.,a-eq*h)
        branch='platform' if a<=eq*h else 'onchip'
    else:
        t=max(1.,a);branch='direct_response'
    return {'tile_ns':t,'control_ns':h,'FULL_onchip_response_ns':a,
        'MC_covariate_ns':mc,'q':q,'eta_at_q':eq,'branch':branch,
        'leverage':reg.leverage(model['full'],f)}


def recipes(mode,group='auto'):
    if mode=='Resource/Envelope-blind':
        return [('full_structural',0.,r,0.) for r in c.protocol()['ridge_candidates']]
    result=[('full_structural',0.,r,0.) for r in c.protocol()['ridge_candidates']]
    if group in ('auto','scalar'):
        result += [('partial_overlap',e,r,0.) for e in c.protocol()['overlap_eta_candidates']
                   for r in c.protocol()['ridge_candidates']]
    if group in ('auto','q_dependent'):
        result += [('partial_overlap_q',e,r,b) for e in c.protocol()['overlap_eta_candidates']
                   for r in c.protocol()['ridge_candidates'] for b in c.protocol()['eta_q_slopes']]
    return result


def recipe_key(r):
    complexity={'full_structural':0,'partial_overlap':1,'partial_overlap_q':2}
    return (r['inner_relative_loss'],complexity[r['family']],-r['ridge'],r['eta'],
            abs(r.get('eta_q_slope',0.)),r.get('eta_q_slope',0.))


@lru_cache(maxsize=None)
def choose(excluded,mode='full',group='auto'):
    eligible=sorted({r['geometry'] for r in BUNDLE['full']} - set(excluded))
    results=[]
    for family,eta,ridge,slope in recipes(mode,group):
        errors=[]
        for held in eligible:
            key=tuple(sorted(set(excluded)|{held}))
            m=train(key,mode,family,eta,ridge,slope)
            rows=data.medians([r for r in BUNDLE['full_fit'] if r['geometry']==held])
            pred=np.array([predict(m,r['case'],static_q(r['case'],key,r['q_condition']))['tile_ns'] for r in rows])
            errors.append(objective(rows,pred))
        results.append({'family':family,'eta':eta,'ridge':ridge,'eta_q_slope':slope,
            'inner_relative_loss':float(np.mean(errors)),'inner_held_geometries':eligible,
            'inner_losses_by_geometry':dict(zip(eligible,errors))})
    winner=min(results,key=recipe_key)
    model=train(tuple(excluded),mode,winner['family'],winner['eta'],winner['ridge'],winner['eta_q_slope'])
    return {'model':model,'selected_recipe':winner,'inner_scores':results,'selection_group':group}
