"""Shared primitive response surfaces, not per-candidate timing lookup."""
from collections import defaultdict
from math import log2, exp, log
import statistics as st
import numpy as np


def table(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r['family'],r['parameter'],r['D'],r['capacity']].append(r['ns'])
    return [{'family':f,'parameter':p,'D':d,'q':q,
             'low':min(v),'point':st.median(v),'high':max(v),'observations':len(v)}
            for (f,p,d,q),v in sorted(groups.items())]


def interpolate(rows,q):
    rows=sorted(rows,key=lambda r:1/r['q'])
    xs=[1/r['q'] for r in rows]
    # Outside the measured occupancy range: sensitivity interval includes
    # constant throughput and ideal inverse-occupancy extrapolation.
    if q>max(r['q'] for r in rows):
        r=rows[0]
        return [r['low']*r['q']/q,r['point'],r['high']]
    if q<min(r['q'] for r in rows):
        r=rows[-1]
        return [r['low'],r['point']*r['q']/q,r['high']*r['q']/q]
    return [float(np.interp(1/q,xs,[r[k] for r in rows])) for k in ('low','point','high')]


def query(surface,family,param,q,d):
    rows=[r for r in surface if r['family']==family and r['D']==d]
    sizes=sorted({r['parameter'] for r in rows})
    if param in sizes:
        return interpolate([r for r in rows if r['parameter']==param],q)
    # Log-size interpolation; out-of-support linear value scaling is flagged
    # by the caller. It is a preregistered hypothesis, not a new measurement.
    if param<min(sizes) or param>max(sizes):
        near=min(sizes,key=lambda x:abs(log2(x/param)))
        v=interpolate([r for r in rows if r['parameter']==near],q)
        return [x*param/near for x in v]
    lo=max(x for x in sizes if x<param);hi=min(x for x in sizes if x>param)
    a=interpolate([r for r in rows if r['parameter']==lo],q)
    b=interpolate([r for r in rows if r['parameter']==hi],q)
    t=log2(param/lo)/log2(hi/lo)
    return [exp((1-t)*log(max(x,1e-12))+t*log(max(y,1e-12))) for x,y in zip(a,b)]


def primitive_schedule(p,warps,q,theta,mode='full',streams=1):
    surface=theta['primitive_surfaces']
    cols=p['columns_per_thread'];elements=p['elements_per_thread']
    base=query(surface,'P',1,q,streams)
    local=query(surface,'P',p['p'],q,streams)
    lane=query(surface,'U',p['u'],q,streams)
    lane0=query(surface,'U',1,q,streams)
    cross=query(surface,'W',p['w'],q,streams)
    cross0=query(surface,'W',1,q,streams)
    consumer=query(surface,'E',elements,q,streams)
    scale=warps*cols/4
    # Differences are effective per-column template increments. Their
    # intervals retain fixture and packing ambiguity; not naked max latency.
    private=[max(0.,local[0]-base[2])*scale,
             max(0.,local[1]-base[1])*scale,
             max(0.,local[2]-base[0])*scale]
    if p['w']>1 and cols>=2 and p['p']>1:
        # Operation packing does not imply a proportional time saving.
        # P has no matching cross-column packed tree: retain unidentified
        # saving [0,unpacked template] rather than inventing a time weight.
        private[0]=0.
    intra=[max(0.,lane[0]/p['u']-lane0[2])*scale,
           max(0.,lane[1]/p['u']-lane0[1])*scale,
           max(0.,lane[2]/p['u']-lane0[0])*scale]
    inter=[max(0.,cross[0]/p['w']-cross0[2])*warps*cols/8,
           max(0.,cross[1]/p['w']-cross0[1])*warps*cols/8,
           max(0.,cross[2]/p['w']-cross0[0])*warps*cols/8]
    initial=[x*scale for x in base]
    rate=query(theta['load_surfaces'],'ML',p['e_vector'],q,streams)
    bc_rate=theta['bank_conflict_effective_ns_interval']
    load=[rate[0]*p['ML']['ideal_wavefronts']+bc_rate[0]*p['ML']['bank_conflicts'],
          rate[1]*p['ML']['ideal_wavefronts']+sum(bc_rate)/2*p['ML']['bank_conflicts'],
          rate[2]*p['ML']['ideal_wavefronts']+bc_rate[1]*p['ML']['bank_conflicts']]
    if mode=='Load-blind':load=[0.,0.,0.]
    if mode=='Comm-blind':intra=inter=[0.,0.,0.]
    # E bounds full-element fixture consumer work. This optional cost is
    # concurrent with, rather than appended to, the macro compute path.
    fixture_high=consumer[2]*warps/4
    private=[initial[i]+private[i] for i in range(3)]
    pipeline=max(load[0],private[0],intra[0],inter[0])
    serial=load[2]+max(private[2],fixture_high)+intra[2]+inter[2]
    pt=(max(load[1],private[1],intra[1],inter[1])+
        load[1]+private[1]+intra[1]+inter[1])/2
    nodes=[{'node':name,'depends':[] if i==0 else [names[i-1]],'ns_interval':values}
           for names in [('load','local_tree','lane_exchange','warp_exchange')]
           for i,(name,values) in enumerate(zip(names,[load,private,intra,inter]))]
    return {'pipeline_lower_ns':pipeline,'serial_upper_ns':serial,
        'point_ns':pt,'nodes':nodes,'E_fixture_upper_ns':fixture_high}


def leave_size_out(surface):
    result=[]
    for f in ('P','U','W','E'):
        sizes=sorted({r['parameter'] for r in surface if r['family']==f})
        for size in sizes:
            remain=[r for r in surface if not (r['family']==f and r['parameter']==size)]
            for row in surface:
                if row['family']!=f or row['parameter']!=size:continue
                pred=query(remain,f,size,row['q'],row['D'])[1]
                result.append({'family':f,'size':size,'D':row['D'],'q':row['q'],
                    'observed_ns':row['point'],'predicted_ns':pred,
                    'relative_error':abs(pred-row['point'])/row['point'],
                    'extrapolation':size in (min(sizes),max(sizes))})
    return result
