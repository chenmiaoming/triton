"""Shared effective-service fit and dependency/resource interval scheduling.

No kernel compilation, CUDA or runtime grid argument. Calibration features
include fixture work; the composition envelope explicitly retains that
identification uncertainty rather than naming coefficients naked latencies.
"""
from math import ceil, log2
import numpy as np
from experiments.tma_reduction_layout.phase11.c3.static import plan, compact, candidates, upstream

PROBE_NAMES=('loop_warps','private_update_values','local_max_equivalents',
             'intra_shuffle','inter_shuffle','comm_wavefronts','comm_barriers',
             'E_consumer_values','private_chain_over_q','local_chain_over_q',
             'shuffle_chain_over_q','sync_chain_over_q')
LOAD_NAMES=('load_wavefronts','load_warp_instructions','checksum_values',
            'loop_warps','load_chain_over_q')


def probe_features(p,u,w,columns,warps,elements,q,family='MR'):
    rounds=0 if w==1 else (1 if w<=u else 2)
    local=0 if family=='E' else warps*columns*(p-1)
    return [warps,warps*elements,local,
        0 if family=='E' else warps*columns*log2(u),
        0 if family=='E' else warps*columns*log2(w),
        0 if family=='E' else rounds*warps*columns,
        0 if family=='E' else 2*rounds*warps*ceil(columns/4),
        warps*elements if family=='E' else 0,
        elements/q,0 if family=='E' else columns*log2(p)/q,
        0 if family=='E' else columns*log2(u*w)/q,
        0 if family=='E' else rounds*ceil(columns/4)/q]


def load_features(p,warps,q):
    ml=p['ML']
    return [ml['wavefronts'],ml['warp_load_instructions'],
            warps*p['elements_per_thread'],warps,
            ml['warp_load_instructions']/warps/q]


def nnls(a,b):
    """Deterministic nonnegative least squares by cyclic coordinate descent."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    scale=np.linalg.norm(a,axis=0);scale[scale==0]=1
    x=a/scale;coef=np.zeros(x.shape[1]);res=b.copy()
    for iteration in range(4000):
        before=coef.copy()
        for j in range(len(coef)):
            col=x[:,j];norm=float(col@col)
            if norm==0:continue
            new=max(0.,coef[j]+float(col@res)/norm)
            res-=col*(new-coef[j]);coef[j]=new
        if np.max(np.abs(coef-before))<1e-8:break
    return coef/scale,iteration+1


def fit(rows,names):
    a=np.array([r['features'] for r in rows]);b=np.array([r['ns'] for r in rows])
    coef,iterations=nnls(a,b)
    predicted=a@coef
    cv=[]
    groups=sorted({r['group'] for r in rows})
    for group in groups:
        train=[i for i,r in enumerate(rows) if r['group']!=group]
        test=[i for i,r in enumerate(rows) if r['group']==group]
        k,_=nnls(a[train],b[train])
        for i in test:
            estimate=max(float(a[i]@k),1e-9)
            cv.append({'tag':rows[i]['tag'],'q':rows[i]['q'],
                'group':group,'observed_ns':float(b[i]),'predicted_ns':estimate,
                'ratio':float(b[i]/estimate)})
    ratios=[r['ratio'] for r in cv]
    return {'names':list(names),'coefficient_ns_per_unit':coef.tolist(),
        'design_rank':int(np.linalg.matrix_rank(a)),
        'columns':len(names),'rows':len(rows),'iterations':iterations,
        'fit_max_relative_error':float(np.max(np.abs(predicted-b)/b)),
        'leave_structure_out_ratio_interval':[min(ratios),max(ratios)],
        'leave_structure_out':cv,
        'coefficient_interpretation':'Effective fixture-inclusive service coefficients; rank deficiency forbids identifying all separate naked instruction latencies.',
        'training_rows':rows}


def dot(fitted,features):
    return float(np.dot(fitted['coefficient_ns_per_unit'],features))


def graph_schedule(p,warps,q,theta,mode='full',streams=1):
    """Macro DAG: load -> local tree -> lane exchanges -> warp exchanges -> out.

    Lower bound permits pipelining independent CTAs across resource pools;
    upper bound serializes the same dependent nodes. q response is calibrated
    effective dependence response, not a cycle-accurate hardware scheduler.
    Both share measured effective fixture costs until MC bounds constrain them.
    """
    if 'primitive_surfaces' in theta:
        from experiments.tma_reduction_layout.phase11.c3.surfaces import primitive_schedule
        return primitive_schedule(p,warps,q,theta,mode,streams)
    lf=load_features(p,warps,q)
    rf=probe_features(p['p'],p['u'],p['w'],p['columns_per_thread'],
                      warps,p['elements_per_thread'],q)
    load=0. if mode=='Load-blind' else dot(theta['load'][str(streams)],lf)
    surface=theta['probe'][str(streams)]
    contributions=dict(zip(PROBE_NAMES,np.array(surface['coefficient_ns_per_unit'])*rf))
    private=sum(contributions[k] for k in ('loop_warps','private_update_values',
                  'local_max_equivalents','private_chain_over_q','local_chain_over_q'))
    lane=sum(contributions[k] for k in ('intra_shuffle','shuffle_chain_over_q'))
    cross=sum(contributions[k] for k in ('inter_shuffle','comm_wavefronts',
                                        'comm_barriers','sync_chain_over_q'))
    if mode=='Comm-blind':lane=cross=0.
    nodes=[{'node':'shared_load_consumer','depends':[],'ns':load},
           {'node':'local_tree_fixture','depends':['shared_load_consumer'],'ns':private},
           {'node':'lane_exchange','depends':['local_tree_fixture'],'ns':lane},
           {'node':'warp_exchange','depends':['lane_exchange'],'ns':cross}]
    serial=load+private+lane+cross
    pipeline=max(load,private,lane,cross)
    return {'pipeline_lower_ns':pipeline,'serial_upper_ns':serial,
            'point_ns':(pipeline+serial)/2,'nodes':nodes}


def predict(case,theta,mode='full'):
    p=plan(case);r=p['resources'];ql,qh=r['q_interval']
    qvals=sorted({max(1,ql),max(1,qh),*[q for q in (1,2,4) if ql<=q<=qh]})
    if mode=='Resource/Envelope-blind':qvals=[16]
    schedules=[graph_schedule(p,case['W'],q,theta,mode,d) for q in qvals for d in (1,2)]
    # Global multiplicative envelope, never a layout-specific additive residual.
    lo=min(s['pipeline_lower_ns'] for s in schedules)*theta['composition']['steady_ratio'][0]
    hi=max(s['serial_upper_ns'] for s in schedules)*theta['composition']['steady_ratio'][1]
    point=np.median([s['point_ns'] for s in schedules])*theta['composition']['steady_point_ratio']
    mo_upper=theta['composition']['MO_ns_per_output_byte_interval'][1]*4*case['BN']
    k1lo=lo*theta['composition']['K1_combination_ratio'][0]
    k1hi=max(hi,mo_upper)*theta['composition']['K1_combination_ratio'][1]
    # MO constrains the global K1/output uncertainty above; it is not added here.
    byte_count=2*case['BM']*case['BN']+4*case['BN']
    tma=[b*byte_count for b in theta['transfer']['ns_per_tile_byte_interval']]
    if mode=='Resource/Envelope-blind':tma=[0.,0.]
    steady=[max(lo,tma[0]),max(hi,tma[1])]
    k1=[max(k1lo,tma[0]),max(k1hi,tma[1])]
    # Single-CTA effective dependency-chain uncertainty is estimated separately
    # from grid1 slopes, retaining fixture/setup identification uncertainty.
    chain=[lo*theta['composition']['grid1_chain_to_body_ratio'][0],
           hi*theta['composition']['grid1_chain_to_body_ratio'][1]]
    unsupported=[]
    if not r['static_shape_legal'] or not r['shared_legal']:
        unsupported.append('static_shared_illegal')
    if r['definite_register_infeasible']:unsupported.append('register_lower_bound_illegal')
    if r['spill_possible']:unsupported.append('static_spill_uncertainty')
    if p['p']>64:unsupported.append('local_tree_outside_probe_support')
    if p['elements_per_thread']>32:unsupported.append('consumer_outside_probe_support')
    if case['BM']!=32 or case['BN'] not in (64,128):
        unsupported.append('independent_transfer_control_unmeasured_geometry')
    if case['BN'] not in (64,128):unsupported.append('ML_shape_extrapolation')
    if case['W'] not in (4,8):unsupported.append('warp_count_extrapolation')
    return {'id':case['id'],'static':compact(p),'q_evaluated':qvals,
        'onchip_steady_ns':[lo,hi],'steady_ns':steady,'K1_ns':k1,
        'single_CTA_effective_chain_ns':chain,'TMA_ns':tma,
        'rho_interval':[lo/tma[1],hi/tma[0]] if tma[0]>0 else None,
        'point_ns':max(float(point),sum(tma)/2),
        'K1_point_ns':max(max(float(point),sum(theta['composition']['MO_ns_per_output_byte_interval'])/2*4*case['BN'])*theta['composition']['K1_combination_point'],sum(tma)/2),
        'unsupported_reasons':unsupported,
        'scheduling_assumptions':'Macro dependency DAG; ideal inter-CTA pipeline versus serialized resource service, empirical global composition hull. Not a calibrated cycle simulator.'}


def select(m,n,w,theta,mode='full'):
    cases=candidates(m,n,w);pred=[predict(x,theta,mode) for x in cases]
    default=upstream(m,n,w)
    base=next(p for p in pred if p['id']==default)
    robust=[]
    for p in pred:
        if p['id']==default or p['unsupported_reasons']:continue
        if all(p[key][1] < 0.97*base[key][0]
               for key in ('steady_ns','K1_ns','single_CTA_effective_chain_ns')):
            robust.append(p)
    choice=min(robust,key=lambda p:p['K1_ns'][1])['id'] if robust else default
    comparable=[p['id'] for p in pred if p['K1_ns'][0]<=
                1.03*min(x['K1_ns'][1] for x in pred)]
    return {'geometry':[m,n,w],'mode':mode,'default':default,'choice':choice,
        'switched':choice!=default,'robust_candidates':[p['id'] for p in robust],
        'comparable_set':comparable,'point_choice':min(pred,key=lambda p:p['K1_point_ns'])['id'],
        'reason':'interval dominance >3% in steady/K1/chain' if robust else
                 'abstain: intervals overlap or static resource/transfer extrapolation unsupported',
        'predictions':pred}


if __name__=='__main__':
    import argparse,json
    from experiments.tma_reduction_layout.phase11.c3 import common as c
    parser=argparse.ArgumentParser(description='CPU-only frozen C3 selector')
    parser.add_argument('--BM',type=int,required=True)
    parser.add_argument('--BN',type=int,required=True)
    parser.add_argument('--W',type=int,required=True)
    parser.add_argument('--theta',default=str(c.OUT/'theta_H100.json'))
    args=parser.parse_args()
    print(json.dumps(select(args.BM,args.BN,args.W,c.read(args.theta)),indent=2))
