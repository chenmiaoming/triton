"""Resource-only nested LOGO calibration. No time, gain or policy input."""
from functools import lru_cache
from math import log2
import json
from experiments.tma_reduction_layout.phase11.c34 import common as c
from experiments.tma_reduction_layout.phase11.c33 import resources as old
from experiments.tma_reduction_layout.phase11.c3.static import upstream

ROWS = ()


def initialize(rows):
    global ROWS
    # Explicit resource-only interface: a timing label cannot reach calibration.
    keys = ('geometry','case','cubin_SHA256','n_regs','natural_q',
            'local_bytes','static_shared_bytes','dynamic_shared_bytes')
    ROWS = tuple({k:r[k] for k in keys} for r in rows
                 if r['family']=='full_single_pass' and not r['local_bytes'])
    assert len({r['case']['id'] for r in ROWS}) == len(ROWS)
    point_model.cache_clear(); errors.cache_clear(); calibration.cache_clear()
    segments.cache_clear()


def r_of(case):
    return case['BM']*case['BN']//(32*case['W'])


@lru_cache(maxsize=None)
def point_model(excluded):
    keep = [r for r in ROWS if r['geometry'] not in excluded]
    mm = old.fit_model(keep)
    return {**mm, 'training_geometries':sorted({r['geometry'] for r in keep}),
            'excluded_geometries':list(excluded)}


@lru_cache(maxsize=None)
def errors(excluded):
    """Each residual comes from a model excluding its complete geometry."""
    keep = [r for r in ROWS if r['geometry'] not in excluded]
    result = []
    for geo in sorted({r['geometry'] for r in keep}):
        mm = point_model(tuple(sorted(set(excluded)|{geo})))
        held = [r for r in keep if r['geometry']==geo]
        default = next(r for r in held if r['case']['id']==upstream(
            r['case']['BM'],r['case']['BN'],r['case']['W']))
        de = default['n_regs']-old.point(mm,default['case'])
        for row in held:
            pred = old.point(mm,row['case'])
            result.append({'geometry':geo,'R':r_of(row['case']),
                'layout':row['case']['id'],'family':row['case']['placement'],
                'default':default['case']['id'],'residual':row['n_regs']-pred,
                'default_residual':de,'delta_residual':row['n_regs']-pred-de,
                'observed':row['n_regs'],'predicted':pred,
                'default_observed':default['n_regs'],
                'default_predicted':old.point(mm,default['case']),
                'candidate_q_observed':row['natural_q'],
                'default_q_observed':default['natural_q'],
                'model_excluded_geometries':mm['excluded_geometries']})
    return result


def neighborhood(rows, target_R, minimum=3):
    geo_R = {r['geometry']:r['R'] for r in rows}
    radii = sorted({max(1.,abs(log2(v/target_R))) for v in geo_R.values()})
    radius = 1.
    for radius in radii:
        selected = {g for g,v in geo_R.items() if abs(log2(v/target_R))<=radius+1e-10}
        if len(selected)>=min(minimum,len(geo_R)):
            break
    selected = {g for g,v in geo_R.items() if abs(log2(v/target_R))<=radius+1e-10}
    chosen = [r for r in rows if r['geometry'] in selected]
    return chosen, {'target_R':target_R,'log2_radius':radius,
        'expanded':radius>1.,'geometries':sorted(selected),
        'distinct_geometries':len(selected),'R_support':sorted({r['R'] for r in chosen}),
        'minimum_met':len(selected)>=minimum}


def base_envelope(bank, target_R, family):
    near, pool = neighborhood(bank,target_R)
    common = {r['geometry']:r['default_residual'] for r in near}
    paired = [r for r in near if r['layout']!=r['default'] and r['family']==family]
    relaxed = len({r['geometry'] for r in paired})<3
    if relaxed:
        paired = [r for r in near if r['layout']!=r['default']]
    cv = list(common.values())+[0.]
    dv = [r['delta_residual'] for r in paired]+[0.]
    return {'common_interval':[min(cv),max(cv)],'delta_interval':[min(dv),max(dv)],
        'pool':pool,'paired_placement_relaxed':relaxed,
        'paired_geometries':sorted({r['geometry'] for r in paired}),
        'common_sources':sorted(common.items()),
        'paired_sources':[{k:r[k] for k in ('geometry','layout','delta_residual')}
                          for r in paired]}


@lru_cache(maxsize=1024)
def segments(encoded, shared_upper):
    """All exact register-allocation plateaus; low endpoint is open."""
    case = json.loads(encoded)
    result=[]
    for slot in range(1,33):
        q = old.q_for(case,min(255,slot*8),shared_upper)
        lo = -1e100 if slot==1 else (slot-1)*8
        hi = 1e100 if slot==32 else slot*8
        if result and result[-1][0]==q:
            result[-1]=(q,result[-1][1],hi)
        else:
            result.append((q,lo,hi))
    return tuple(result)


def q_scenarios(candidate, default, pc, pd, common, delta, only_pair=None):
    out={}
    ec,ed = json.dumps(candidate,sort_keys=True),json.dumps(default,sort_keys=True)
    for su_c in (False,True):
        for su_d in (False,True):
            for dq,dlo,dhi in segments(ed,su_d):
                if only_pair and dq!=only_pair[1]: continue
                for cq,clo,chi in segments(ec,su_c):
                    if only_pair and cq!=only_pair[0]: continue
                    lo=max(common[0],dlo-pd+1e-8,clo-pc-delta[1]+1e-8)
                    hi=min(common[1],dhi-pd,chi-pc-delta[0])
                    if lo>hi: continue
                    shift=(lo+hi)/2
                    dl=max(delta[0],clo-pc-shift+1e-8)
                    dh=min(delta[1],chi-pc-shift)
                    if dl>dh: continue
                    diff=(dl+dh)/2
                    assert old.q_for(default,pd+shift,su_d)==dq
                    assert old.q_for(candidate,pc+shift+diff,su_c)==cq
                    out.setdefault((cq,dq),{'candidate_q':cq,'default_q':dq,
                        'common_register_residual':shift,'paired_register_residual':diff,
                        'candidate_n_regs':max(8.,pc+shift+diff),
                        'default_n_regs':max(8.,pd+shift),
                        'candidate_shared_upper':su_c,'default_shared_upper':su_d})
                    if only_pair: return list(out.values())
    return [out[k] for k in sorted(out)]


def shifted(base,mc,mp):
    return ([base['common_interval'][0]-mc,base['common_interval'][1]+mc],
            [base['delta_interval'][0]-mp,base['delta_interval'][1]+mp])


def observation_covered(obs, mc, mp):
    common,delta = shifted(obs['base_envelope'],mc,mp)
    if obs['observed']>obs['predicted']+common[1]+delta[1]+1e-8: return False
    if obs['default_observed']>obs['default_predicted']+common[1]+1e-8: return False
    return bool(q_scenarios(obs['candidate_case'],obs['default_case'],
        obs['predicted'],obs['default_predicted'],common,delta,
        only_pair=(obs['candidate_q_observed'],obs['default_q_observed'])))


@lru_cache(maxsize=None)
def calibration(excluded, target_R, family):
    bank = errors(excluded)
    _, pool = neighborhood(bank,target_R)
    observations=[]
    by_id={r['case']['id']:r['case'] for r in ROWS}
    for held in pool['geometries']:
        nested_excluded=tuple(sorted(set(excluded)|{held}))
        remaining_bank=errors(nested_excluded)
        # The held predictions in bank already exclude held and the outer fold.
        for row in bank:
            if row['geometry']!=held or row['layout']==row['default']: continue
            if row['family']!=family: continue
            base=base_envelope(remaining_bank,row['R'],row['family'])
            observations.append({**row,'base_envelope':base,
                'candidate_case':by_id[row['layout']], 'default_case':by_id[row['default']]})
    assert observations
    candidates = sorted(((mc,mp) for mc in c.protocol()['resource_margin_candidates']
                         for mp in c.protocol()['resource_margin_candidates']),
                        key=lambda x:(sum(x),max(x),x[0]))
    trials=[]; selected=None
    for mc,mp in candidates:
        failures=[i for i,o in enumerate(observations) if not observation_covered(o,mc,mp)]
        trials.append({'common_margin':mc,'paired_margin':mp,
                       'covered':len(observations)-len(failures),'failed_indices':failures})
        if not failures:
            selected=trials[-1]; break
    if selected is None: selected=trials[-1]
    return {'common_margin':selected['common_margin'],'paired_margin':selected['paired_margin'],
        'calibration_covered':selected['covered'],'calibration_total':len(observations),
        'calibration_target_met':selected['covered']==len(observations),
        'target_R':target_R,'family':family,'calibration_pool':pool,
        'excluded_geometries':list(excluded),'trials':trials,
        'observations':observations,'base_envelope':base_envelope(bank,target_R,family),
        'uses_timing':False,'selection_rule':'100% training-only upper-register and joint-q coverage; min total margin, min max margin, min common margin',
        'scope':'Empirical nested LOGO coverage; no distribution-free or spill guarantee.'}


def key_for(case):
    return str(r_of(case))+':'+case['placement']


def calibrate(excluded, queries):
    excluded=tuple(sorted(excluded))
    return {'point_model':point_model(excluded),
        'training_geometries':point_model(excluded)['training_geometries'],
        'excluded_geometries':list(excluded),
        'calibrations':{key_for(case):calibration(excluded,r_of(case),case['placement'])
                        for case in queries},
        'resource_input_SHA256':c.sha(c.encode(list(ROWS))),
        'uses_timing':False,'joint_calibration':True}


def point(rm, case):
    return old.point(rm['point_model'],case)


def envelope(rm, case):
    cal=rm['calibrations'][key_for(case)]
    common,delta=shifted(cal['base_envelope'],cal['common_margin'],cal['paired_margin'])
    return common,delta,cal


def pair(candidate, default, rm):
    pc,pd=point(rm,candidate),point(rm,default)
    common,delta,cal=envelope(rm,candidate)
    return q_scenarios(candidate,default,pc,pd,common,delta),cal


def estimate(case, rm):
    center=point(rm,case)
    common,delta,cal=envelope(rm,case)
    is_default=case['id']==upstream(case['BM'],case['BN'],case['W'])
    low=max(8.,center+common[0]+(0. if is_default else delta[0]))
    high=max(low,center+common[1]+(0. if is_default else delta[1]))
    p=old.static(case)
    possible=set()
    for su in (False,True):
        for q,lo,hi in segments(json.dumps(case,sort_keys=True),su):
            if max(low,lo+1e-8)<=min(high,hi): possible.add(q)
    assert possible
    mm=rm['point_model']
    return {'n_regs_point':center,'n_regs_interval':[low,high],
        'q_interval':[min(possible),max(possible)],'q_values':sorted(possible),
        'q_point':old.q_for(case,center),
        'shared_bytes_interval':p['resources']['shared_bytes_interval'],
        'shared_legal':p['resources']['shared_legal'],'spill_uncertainty':high>255,
        'new_warp_count':case['W'] not in mm['W_support'],
        'R_outside_support':not mm['R_support'][0]<=r_of(case)<=mm['R_support'][1],
        'margin_common':cal['common_margin'],'margin_paired':cal['paired_margin'],
        'resource_pool':cal['base_envelope']['pool'],
        'interval_scope':'Jointly calibrated common/default and paired residual rectangle; empirical bound.'}
