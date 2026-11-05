"""Static FULL operation demand; q is predicted residency, never grid."""
from math import log2, ceil
from experiments.tma_reduction_layout.phase11.c32.features import static, local_query


def control(geometry,q):
    m,n,w=geometry
    return {'bytes_KiB':2*m*n/1024,'output_KiB':4*n/1024,
        'CTA_per_q':1/q,'warps_per_q':w/q,'segments_per_q':n/64/q}


def operation(case,q,tables,mode='full',mc=None,transfer=None):
    if mode=='Resource/Envelope-blind': q=1
    g=static(case);w=case['W'];cols=g['columns_per_thread'];r=g['elements_per_thread']
    f={'CTA_warps':float(w),
        'ML_ideal_wavefronts':float(g['ML']['ideal_wavefronts']),
        'ML_conflict_wavefronts':float(g['ML']['bank_conflicts']),
        'ML_warp_loads':float(g['ML']['warp_load_instructions']),
        'local_BF16_max_scalar_equivalents':float(g['MR']['local_max_scalar_equivalents']),
        'local_BF16_packed_demand':float(w*ceil(cols*(g['p']-1)/2)),
        'local_widened_partials':float(w*cols),
        'MR_shuffle':float(g['MR']['shuffle_warp_instructions']),
        'MR_warp_fanout':float(w*cols*(g['u']-1)),
        'MR_wavefronts':float(g['MR']['shared_load_wavefronts']),
        'MR_barrier':float(g['MR']['barrier_warp_instructions']),
        'MO_thread_redistribution':float(w*g['output']['requires_thread_redistribution']),
        'MO_warp_redistribution':float(w*g['output']['warp_redistribution_possible']),
        'MO_register_column_reorder':float(w*cols*log2(g['e_vector'])),
        'output_warp_values':float(case['BN']/32)}
    for name in ('CTA_warps','ML_warp_loads','ML_conflict_wavefronts',
                 'local_widened_partials','MR_shuffle','MR_warp_fanout',
                 'MR_barrier','MO_thread_redistribution','MO_register_column_reorder'):
        if mode!='Resource/Envelope-blind': f[name+'__inverse_q']=f[name]/q
    for variant,fixture in (('native','fixture'),('wide','fixture_wide')):
        f['fixed_R_'+variant+'_net']=(local_query(tables,r,g['p'],q*w/4,variant)-
            local_query(tables,r,g['p'],q*w/4,fixture))*w/4
    if mc is not None: f['calibrated_MC_response']=mc
    if transfer is not None: f['calibrated_control_response']=transfer
    if mode=='Load-blind': f={k:v for k,v in f.items() if not k.startswith('ML_')}
    if mode=='Comm-blind': f={k:v for k,v in f.items() if not k.startswith('MR_')}
    if mode=='Fixed-R-blind': f={k:v for k,v in f.items() if not k.startswith('fixed_R_')}
    return f
