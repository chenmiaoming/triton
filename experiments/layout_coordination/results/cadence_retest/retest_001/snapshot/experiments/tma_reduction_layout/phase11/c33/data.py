"""Complete same-worker FULL targets; no D1 binary or timing input."""
from collections import defaultdict
import statistics as st
from experiments.tma_reduction_layout.phase11.c33 import common as c
from experiments.tma_reduction_layout.phase11.c32 import data as historical
from experiments.tma_reduction_layout.phase11.c3.static import upstream


def geometry(case):
    return f'{case["BM"]}x{case["BN"]}w{case["W"]}'


def load():
    old = historical.load()
    a = c.read(c.prior.OUT/'analysis.json')
    gate = c.read(c.prior.OUT/'artifact_gate.json')
    full, controls, mc = [], [], []
    for row in a['oneshot']:
        meta = gate['metadata'][row['tag']]
        g = row['geometry']
        item = {**row,'geometry_vector':g,'geometry':f'{g[0]}x{g[1]}w{g[2]}',
            'worker':row['invocation'],'target_ns':row['tile_slope_ns']}
        if meta['family']=='ONE_FULL':
            case = historical.case_for(meta)
            full.append({**item,'case':case,'layout':case['id']})
        else:
            controls.append(item)
    for row in a['services']:
        meta = gate['metadata'][row['tag']]
        case = historical.case_for(meta)
        mc.append({**row,'case':case,'geometry':geometry(case),'layout':case['id'],
            'worker':row['invocation'],'target_ns':row['service_ns']})
    # Deduplicate physical q aliases for fitting, retain requested labels for
    # same-condition pairing and natural decisions.
    def unique(rows):
        out, seen = [], set()
        for row in sorted(rows,key=lambda item:item['q_condition']!='natural'):
            key=(row.get('layout',row['geometry']),row['q'],row['cohort'],row['worker'])
            if key not in seen:
                seen.add(key);out.append(row)
        return out
    return {'full':full,'control':controls,'full_fit':unique(full),'control_fit':unique(controls),
        'steady':old['steady']+unique(mc),'local_tables':historical.local_tables(old),
        'gate':gate,'analysis_SHA256':c.sha((c.prior.OUT/'analysis.json').read_bytes())}


def medians(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row.get('layout',row['geometry']),row['q'],row['cohort']].append(row)
    return [{**items[0],'target_ns':st.median(item['target_ns'] for item in items),
        'workers':len(items),'worker_range_ns':[min(item['target_ns'] for item in items),
        max(item['target_ns'] for item in items)]} for items in groups.values()]


def paired(rows, predictions):
    index={(row['layout'],row['q_condition'],row['worker'],row['GPU_UUID']):(row,p)
        for row,p in zip(rows,predictions)}
    result=[]
    for row,p in zip(rows,predictions):
        case=row['case']; default=upstream(case['BM'],case['BN'],case['W'])
        if case['id']==default: continue
        key=(default,row['q_condition'],row['worker'],row['GPU_UUID'])
        if key not in index: continue  # Infeasible cap for the default.
        d,dp=index[key]
        observed=row['target_ns']-d['target_ns']
        predicted=p['tile_ns']-dp['tile_ns']
        result.append({'geometry':row['geometry'],'layout':case['id'],'default':default,
            'worker':row['worker'],'GPU_UUID':row['GPU_UUID'],'q_condition':row['q_condition'],
            'candidate_q':p['q'],'default_q':dp['q'],
            'observed_delta_ns':observed,'predicted_delta_ns':predicted,
            'delta_residual_ns':observed-predicted,
            'normalized_delta_residual':(observed-predicted)/d['target_ns'],
            'relative_residual':observed/d['target_ns']-predicted/dp['tile_ns'],
            'observed_default_ns':d['target_ns'],'predicted_default_ns':dp['tile_ns'],
            'observed_gain_percent':-100*observed/d['target_ns'],
            'predicted_gain_percent':-100*predicted/dp['tile_ns'],
            'candidate_branch':p['branch'],'default_branch':dp['branch'],
            'candidate_family':case['placement'], 'default_family':'n_first',
            'R':case['BM']*case['BN']//(32*case['W'])})
    return result
