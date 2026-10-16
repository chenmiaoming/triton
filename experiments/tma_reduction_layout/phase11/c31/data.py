"""Read immutable MC service estimates; no auxiliary timings in fit targets."""
from collections import defaultdict
import statistics as st
from experiments.tma_reduction_layout.phase11.c31 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates,plan,compact


def development():
    analysis=c.read(c.BASE/'results/phase11/stage_c2b/analysis.json')
    gate=c.read(c.BASE/'results/phase11/stage_c2b/artifact_gate.json')
    cases={x['id']:x for g in ((32,64,8),(32,128,4)) for x in candidates(*g)}
    rows=[];cells=defaultdict(list)
    for r in analysis['services']:
        if r['family']!='MC' or r['D']!=1:continue
        meta=gate['metadata'][r['tag']];case=cases[meta['layout_id']]
        q=gate['resources'][r['tag']]['reservations'][r['q']]['capacity']
        row={'layout':case['id'],'case':case,'geometry':f'{case["BM"]}x{case["BN"]}w{case["W"]}',
             'q_condition':r['q'],'q':q,'worker':r['invocation'],
             'GPU_UUID':r['GPU_UUID'],'cohort':r['cohort'],
             'service_ns':r['service_ns_per_tile_SM'],
             'grid_fit_residual_ns':1000*r['grid_fit']['max_abs_residual'],
             'fit_provenance':'C2 K16/64/256 slope, three G/SM slopes; D1'}
        rows.append(row);cells[case['id'],r['q']].append(row)
    medians=[]
    for (lid,qcond),group in sorted(cells.items()):
        assert len(group)==3 and len({r['cohort'] for r in group})==1
        medians.append({**group[0],'service_ns':st.median(r['service_ns'] for r in group),
            'worker_range_ns':[min(r['service_ns'] for r in group),max(r['service_ns'] for r in group)],
            'cell_workers':[{k:r[k] for k in ('worker','GPU_UUID','service_ns')} for r in group]})
    assert len(rows)==180 and len(medians)==60
    return cases,rows,medians


def primitive_tables():
    """Isotonic q projection of D1 primitive shapes, never of MC targets."""
    tables=c.read(c.V0/'theta_H100.json')['primitive_surfaces']
    groups=defaultdict(list)
    for r in tables:
        if r['D']==1 and r['family'] in ('P','U','W'):
            groups[r['family'],r['parameter']].append(r)
    result=[]
    for (family,parameter),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['q']);blocks=[]
        for i,r in enumerate(rows):
            blocks.append({'indices':[i],'weight':r['observations'],'value':r['point']})
            while len(blocks)>=2 and blocks[-2]['value']<blocks[-1]['value']:
                b=blocks.pop();a=blocks.pop();weight=a['weight']+b['weight']
                blocks.append({'indices':a['indices']+b['indices'],'weight':weight,
                    'value':(a['value']*a['weight']+b['value']*b['weight'])/weight})
        values={i:b['value'] for b in blocks for i in b['indices']}
        for i,r in enumerate(rows):
            result.append({**r,'original_point':r['point'],'point':values[i],
                           'low':min(r['low'],values[i]),'high':max(r['high'],values[i])})
    return result


def static_cases():
    cases,_,_=development()
    split=c.read(c.OUT/'preregistration/protocol.json')['blind_cases']
    blind=[x for g in split for x in candidates(g['BM'],g['BN'],g['W'])]
    return list(cases.values()),blind
