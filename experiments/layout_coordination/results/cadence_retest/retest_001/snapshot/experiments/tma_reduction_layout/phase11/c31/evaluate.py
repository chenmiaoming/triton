"""Developed selections, fully reported ablations and unchanged blind split."""
from collections import defaultdict
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c31 import common as c
from experiments.tma_reduction_layout.phase11.c31.data import development
from experiments.tma_reduction_layout.phase11.c31.model import select,predict,compare,plan
from experiments.tma_reduction_layout.phase11.c31.calibrate import predict as point,MODES
from experiments.tma_reduction_layout.phase11.c3.static import candidates,upstream


def metrics(rows):
    return {'pairs':len(rows),'MAE_ns':st.mean(abs(r['observed_delta_ns']-r['predicted_delta_ns']) for r in rows),
            'RMSE_ns':float(np.sqrt(st.mean((r['observed_delta_ns']-r['predicted_delta_ns'])**2 for r in rows)))}


def run(theta):
    cases,workers,cells=development()
    idx={(r['layout'],r['q_condition'],r['cohort'],r['worker'],r['GPU_UUID']):r for r in workers}
    cell_idx={(r['layout'],r['q_condition']):r for r in cells}
    selections=[];baselines=[];contrasts=[];absolute=[]
    old=c.read(c.BASE/'results/phase11/stage_c16/analysis.json')
    full={}
    for r in old['cells']:
        if ':FULL' not in r['tag'] or r['cache']!='warm':continue
        n=64 if 'N64_' in r['tag'] else 128
        grid=32768 if n==64 else 16384
        if r['grid']!=grid:continue
        full[r['tag'].replace(':FULL',':n_first')]={
            'event_us':r['median_us'],'tile_SM_ns':r['median_us']*1000*132/grid,
            'worker_event_us':r['worker_medians_us'],'provenance':'C1.6 original FULL frozen binary'}
    for g in ((32,64,8),(32,128,4)):
        current=candidates(*g);default=upstream(*g)
        for mode in (*MODES,'Envelope-only-blind'):
            s=select(*g,theta,mode);selections.append(s)
        choices={'upstream_default':default,'c3_v1':selections[-5]['choice'],
                 'c3_v0_overconservative':default,
                 'fanout_only':min(current,key=lambda x:plan(x)['u']*plan(x)['w'])['id']}
        for v in (1,2,4,8):
            choices['fixed_vec'+str(v)]=next(x['id'] for x in current if
                x['placement']=='n_first' and x['register_layout']['size_per_thread'][2]==v)
        choices.update({s['mode']:s['choice'] for s in selections[-5:]})
        mc_best=min(cell_idx[x['id'],'natural']['service_ns'] for x in current)
        for method,choice in choices.items():
            actual=cell_idx[choice,'natural']['service_ns']
            value=full.get(choice)
            baselines.append({'geometry':list(g),'method':method,'choice':choice,
                'MC_natural_ns':actual,'MC_regret_percent':100*(actual/mc_best-1),
                'FULL_event_us':value['event_us'] if value else None,
                'FULL_improvement_vs_default_percent':100*(1-value['event_us']/full[default]['event_us']) if value else None,
                'not_new_GPU_timing':True,'not_a_FULL_oracle_for_within':True})
        # Full timing remains evaluation-only; no base is fitted from it.
        for x in current:
            if x['id'] not in full:continue
            p=predict(x,theta);actual=full[x['id']]
            absolute.append({'layout':x['id'],'predicted_bottleneck_ns':p['tile_point_ns'],
                'old_FULL_tile_SM_ns':actual['tile_SM_ns'],
                'absolute_error_percent':100*(p['tile_point_ns']/actual['tile_SM_ns']-1),
                'scope':'Absolute-time diagnostic; base=0 leaves single-pass setup/output mismatch unisolated.'})
    # Pre-specified address-conflict subset: within N128 vec2/4/8 versus the
    # matching N-first vec at the same worker/cohort/q, not selected by winners.
    for mode in MODES:
        model=theta['models'][mode]['model'];cv=theta['models'][mode]['cross_validation']
        fold_map={(f['level'],f['held']):f['model'] for f in cv['fold_models']}
        for r in workers:
            if not plan(r['case'])['ML']['bank_conflicts']:continue
            nf=r['layout'].replace(':within',':n_first')
            other=idx[nf,r['q_condition'],r['cohort'],r['worker'],r['GPU_UUID']]
            for level in ('fit','LOLO','LOGO'):
                held=r['layout'] if level=='LOLO' else r['geometry']
                m=model if level=='fit' else fold_map[level,held]
                difference=point(m,r['case'],r['q'],theta['primitive_shape_tables'])-point(m,other['case'],other['q'],theta['primitive_shape_tables'])
                contrasts.append({'mode':mode,'level':level,'layout':r['layout'],'other':nf,
                    'q_condition':r['q_condition'],'cohort':r['cohort'],'worker':r['worker'],
                    'observed_delta_ns':r['service_ns']-other['service_ns'],
                    'predicted_delta_ns':difference})
    grouped=defaultdict(list)
    for r in contrasts:grouped[r['mode'],r['level']].append(r)
    contrast_summary=[{'mode':m,'level':l,**metrics(rs)} for (m,l),rs in sorted(grouped.items())]
    diagnostics=c.read(c.BASE/'results/phase11/stage_c2b/analysis.json')
    # Auxiliaries are archived for review, never supplied to select/calibrate.
    auxiliary={'K1_and_setup':diagnostics['setup_and_K1_services'],
        'MO_services':[r for r in diagnostics['aggregate'] if ':MO:' in r['tag']],
        'grid1_effective_chains':[{'tag':r['tag'],'q':r['q'],'invocation':r['invocation'],
            'ns':1000*r['grid1_effective_chain_us']} for r in diagnostics['services']
            if r['family']=='MC' and r['grid1_effective_chain_us'] is not None],
        'decision_veto':False,'training_target':False}
    summary={'selections':selections,'baselines':baselines,'bank_conflict_subset':contrasts,
        'bank_conflict_subset_metrics':contrast_summary,'absolute_FULL_diagnostics':absolute,
        'two_development_geometries_not_independent_validation':True,
        'acceptance':{'M32_N64_w8_small_vec_switch':any(s['mode']=='full' and s['geometry']==[32,64,8] and
            s['choice'].split(':')[1] in ('vec1','vec2') and s['switched'] for s in selections),
            'M32_N128_w4_default_kept':any(s['mode']=='full' and s['geometry']==[32,128,4] and not s['switched'] for s in selections)},
        'envelope_only_is_additional_control':'Same full fit and same paired residuals; only remove the shared transfer roof.'}
    c.write(c.V1/'development.json',summary)
    c.write(c.V1/'auxiliary_diagnostics.json',auxiliary)
    c.write(c.V1/'calibration_cells.json',{'cells':cells,'workers':workers})
    protocol=c.read(c.OUT/'preregistration/protocol.json')
    blind=[select(r['BM'],r['BN'],r['W'],theta) for r in protocol['blind_cases']]
    c.write(c.V1/'D1_predictions.json',{'version':'c3_v1','stage':'C3.1 prediction update',
        'compiled':False,'GPU_timed':False,'cases':blind,
        'same_cases_as_c3_v0':True,'theta_SHA256':c.sha((c.V1/'theta_H100.json').read_bytes()),
        'protocol_SHA256':c.sha((c.OUT/'preregistration/protocol.json').read_bytes()),
        'old_predictions_preserved_SHA256':c.sha((c.V0/'D1_predictions.json').read_bytes()),
        'timing_data_used_from_D1':False})
    print('development acceptance',summary['acceptance'])
    for r in blind:print('blind',r['geometry'],r['choice'],r['reason'])
    return summary


if __name__=='__main__':run(c.read(c.V1/'theta_H100.json'))
