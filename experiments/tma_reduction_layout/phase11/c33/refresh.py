"""Recompute decision-only derivatives from fitted models and saved CV banks."""
from experiments.tma_reduction_layout.phase11.c33 import common as c, data, fit, model
from experiments.tma_reduction_layout.phase11.c33.calibrate import score


def run():
    theta=c.read(c.OUT/'theta_H100.json');dev=c.read(c.OUT/'development.json')
    bundle=data.load();fit.BUNDLE={'local_tables':theta['fixed_R_tables']}
    decisions=[];evaluation=[];outer=[];outer_evaluation=[]
    for old in dev['development_decisions']:
        geo=old['geometry'];g=f'{geo[0]}x{geo[1]}w{geo[2]}'
        rows=[r for r in bundle['full'] if r['geometry']==g and r['q_condition']=='natural']
        new=model.predict(geo,theta);decisions.append(new);evaluation.append(score(new,rows))
    for old in dev['outer_decisions']:
        fold=next(f for f in theta['outer_folds'] if f['held']==old['held'])
        train_rows=[r for r in bundle['full'] if r['geometry']!=old['held']]
        local={**theta,'model':fold['model'],'resource_model':old['resource_model'],
            'performance_support':{'R':sorted({r['case']['BM']*r['case']['BN']//(32*r['case']['W']) for r in train_rows}),
                                  'W':sorted({r['case']['W'] for r in train_rows})}}
        new=model.predict(old['decision']['geometry'],local,bank=old['calibration_bank'])
        rows=[r for r in bundle['full'] if r['geometry']==old['held'] and r['q_condition']=='natural']
        outer.append({**old,'decision':new});outer_evaluation.append(score(new,rows))
    result={**dev,'development_decisions':decisions,'development_evaluation':evaluation,
        'outer_decisions':outer,'outer_evaluation':outer_evaluation,
        'development_switches':sum(r['changed'] for r in evaluation),
        'outer_switches':sum(r['changed'] for r in outer_evaluation),
        'outer_adverse_selections_over_3_percent':sum(r['adverse_over_3_percent'] for r in outer_evaluation),
        'uncertainty_conditioned_for_every_resource_scenario':True}
    c.derived(c.OUT/'development.json',result)
    print('C3.3 decisions refreshed; uncertainty conditioned for each correlated resource scenario')


if __name__=='__main__': run()
