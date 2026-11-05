"""Static exact demand plus shared primitive shape/residency features."""
from math import log2
from functools import lru_cache
import json
from experiments.tma_reduction_layout.phase11.c3.static import plan
from experiments.tma_reduction_layout.phase11.c3.surfaces import query


@lru_cache(maxsize=256)
def cached_plan(encoded):return plan(json.loads(encoded))


def features(case,q,recipe,tables,mode='full'):
    p=cached_plan(json.dumps(case,sort_keys=True));w=case['W'];cols=p['columns_per_thread']
    base={'CTA_warps':float(w),
        'ML_wavefronts':float(p['ML']['wavefronts']),
        'ML_warp_instructions':float(p['ML']['warp_load_instructions']),
        'local_max_equivalents':float(p['MR']['local_max_scalar_equivalents']),
        'local_tree_depth_columns':float(w*cols*log2(p['p'])),
        'MR_shuffle':float(p['MR']['shuffle_warp_instructions']),
        'MR_wavefronts':float(p['MR']['shared_load_wavefronts']),
        'MR_barriers':float(p['MR']['barrier_warp_instructions'])}
    if recipe['q_response']=='inverse_q':
        pressures={'inverse_q':1/q}
    else:
        pressures={'q1_pressure':max(2/q-1,0.),'q2_pressure':max(4/q-1,0.),
                   'q4_pressure':1/q}
    for pressure,value in pressures.items():
        for name in ('CTA_warps','ML_warp_instructions','local_tree_depth_columns','MR_shuffle','MR_barriers'):
            base[name+'__'+pressure]=base[name]*value
    if recipe['features']=='counts_plus_primitive_shapes':
        for family,size,scale in [('P',p['p'],w*cols/4*(p['p']-1)/p['p']),
                                  ('U',p['u'],w*cols/(4*p['u'])*(p['u']>1)),
                                  ('W',p['w'],w*cols/(8*p['w'])*(p['w']>1))]:
            base[family+'_shape_q']=query(tables,family,size,q,1)[1]*scale
    if mode=='Load-blind':
        base={k:v for k,v in base.items() if not k.startswith('ML_')}
    if mode=='Comm-blind':
        base={k:v for k,v in base.items() if not k.startswith(('MR_','U_shape','W_shape'))}
    if mode=='Resource/Envelope-blind':
        # Remove residency response while retaining the same structural model.
        base={k:v for k,v in base.items() if '__' not in k and not k.endswith('_shape_q')}
        if recipe['features']=='counts_plus_primitive_shapes':
            base['P_shape_saturated']=query(tables,'P',p['p'],64,1)[1]*w*cols/4*(p['p']-1)/p['p']
            base['U_shape_saturated']=query(tables,'U',p['u'],64,1)[1]*w*cols/(4*p['u'])*(p['u']>1)
            base['W_shape_saturated']=query(tables,'W',p['w'],64,1)[1]*w*cols/(8*p['w'])*(p['w']>1)
            if mode=='Comm-blind':
                base={k:v for k,v in base.items() if not k.startswith(('U_shape','W_shape'))}
    assert all(v>=0 for v in base.values())
    return base
