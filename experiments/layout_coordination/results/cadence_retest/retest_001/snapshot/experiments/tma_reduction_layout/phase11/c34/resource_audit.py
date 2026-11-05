"""Resource-only calibration and nine held-geometry coverage records."""
from experiments.tma_reduction_layout.phase11.c34 import common as c, resources
from experiments.tma_reduction_layout.phase11.c33 import resources as old
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def cases_for(geo):
    return candidates(*geo)


def run():
    c.protect_prior()
    _,unique=old.census();resources.initialize(unique)
    expected=c.read(c.OUT/'resource_inputs.json')['rows']
    assert expected==list(resources.ROWS)
    geometries=c.prior.prior.protocol()['calibration_geometries']
    olddev=c.read(c.prior.OUT/'development.json')
    oldtheta=c.read(c.prior.OUT/'theta_H100.json')
    folds=[];checks=[];pair_checks=[]
    for geo in geometries:
        held=f'{geo[0]}x{geo[1]}w{geo[2]}'
        pool=cases_for(geo)
        rm=resources.calibrate((held,),pool)
        before=next(r for r in olddev['outer_decisions'] if r['held']==held)
        default=next(r for r in pool if r['id']==upstream(*geo))
        truth={r['case']['id']:r for r in resources.ROWS if r['geometry']==held}
        for case in pool:
            row=truth[case['id']]
            new=resources.estimate(case,rm);prev=old.estimate(case,before['resource_model'])
            checks.append({'held':held,'layout':case['id'],
                'n_regs_observed':row['n_regs'],'q_observed':row['natural_q'],
                'before':prev,'after':new,
                'before_regs_upper_covered':row['n_regs']<=prev['n_regs_interval'][1],
                'after_regs_upper_covered':row['n_regs']<=new['n_regs_interval'][1],
                'before_q_interval_covered':prev['q_interval'][0]<=row['natural_q']<=prev['q_interval'][1],
                'after_q_interval_covered':row['natural_q'] in new['q_values']})
            if case['id']==default['id']: continue
            oldc=next(r for r in before['decision']['candidates'] if r['layout']==case['id'])
            scenes,cal=resources.pair(case,default,rm)
            actual=(row['natural_q'],truth[default['id']]['natural_q'])
            oldq=sorted({(r['candidate_q'],r['default_q']) for r in oldc['resource_scenarios']})
            newq=sorted({(r['candidate_q'],r['default_q']) for r in scenes})
            pair_checks.append({'held':held,'layout':case['id'],'observed_pair':list(actual),
                'before_q_pairs':oldq,'after_q_pairs':newq,
                'before_covered':actual in oldq,'after_covered':actual in newq,
                'margin_common':cal['common_margin'],'margin_paired':cal['paired_margin']})
        folds.append({'held':held,'resource_model':rm})
        print('C3.4 resource fold',held,'calibrated from n_regs/q only',flush=True)
    queries=[case for geo in geometries+c.protocol()['protected_D1'] for case in cases_for(geo)]
    final=resources.calibrate((),queries)
    totals={key:sum(r[key] for r in checks) for key in (
        'before_regs_upper_covered','after_regs_upper_covered',
        'before_q_interval_covered','after_q_interval_covered')}
    totals.update(layouts=len(checks),pairs=len(pair_checks),
        before_joint_q_covered=sum(r['before_covered'] for r in pair_checks),
        after_joint_q_covered=sum(r['after_covered'] for r in pair_checks),
        independent_held_geometries=len(folds))
    value={'stage':'C3.4','folds':folds,'final_model':final,'checks':checks,
        'paired_checks':pair_checks,'totals':totals,'uses_timing':False,
        'resource_inputs_SHA256':c.sha((c.OUT/'resource_inputs.json').read_bytes()),
        'selection_target':'Training-only empirical n_regs-upper and joint-q coverage; no policy outcome target'}
    c.derived(c.OUT/'resource_calibration.json',value)
    print('C3.4 resource coverage',totals,flush=True)


if __name__=='__main__': run()
