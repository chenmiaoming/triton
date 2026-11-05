"""Freeze ten static D1 predictions; never compile or time those cases."""
import subprocess
from experiments.tma_reduction_layout.phase11.c33 import common as c, fit, model


def prepare():
    theta=c.read(c.OUT/'theta_H100.json')
    fit.BUNDLE={'local_tables':theta['fixed_R_tables']}
    split=c.read(c.prior.OUT/'preregistration/D1_split.json')
    historical=c.read(c.BASE/'results/phase11/stage_c32/history_audit.json')
    history={'source_stage':'C3.2 historical audit, before any C2d compilation',
        'historical_audit_SHA256':c.sha((c.BASE/'results/phase11/stage_c32/history_audit.json').read_bytes()),
        'historical_audit':historical,'authorized_moved_to_development':[16,64,8],
        'D1_split':split,'current_predictions_do_not_count_as_empirical_exposure':True}
    predictions=[]
    for row in split['cases']:
        assert row['history_category']=='strict_zero_mention'
        predictions.append({**row,'prediction':model.predict(row['geometry'],theta)})
    c.derived(c.OUT/'history_audit.json',history)
    c.derived(c.OUT/'D1_predictions.json',{'stage':'D1','model_version':'c33_FULL_target',
        'count':10,'theta_SHA256':c.sha((c.OUT/'theta_H100.json').read_bytes()),
        'cases':predictions,'D1_compiled':False,'D1_timed':False})
    print('C3.3 prepared 10 static D1 predictions; GPU untouched')


def seal():
    from experiments.tma_reduction_layout.phase11.c33.validate import run
    run(pre_freeze=True)
    parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    prior_name=str((c.prior.OUT/'validation.json').relative_to(c.ROOT))
    previous=subprocess.check_output(['git','show',parent+':'+prior_name],cwd=c.ROOT)
    assert c.sha(previous)==c.sha((c.prior.OUT/'validation.json').read_bytes())
    files={str(p.relative_to(c.ROOT)):c.sha(p.read_bytes())
        for root in (c.SOURCE,c.OUT) for p in sorted(root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('freeze.json','validation.json')}
    c.write_once(c.OUT/'freeze.json',{'stage':'C3.3','files':files,
        'C2d_commit':parent,'C2d_validation_SHA256':c.sha(previous),
        'GPU_allocations':0,'D1_compiled':False,'D1_timed':False})
    run()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','seal'])
    {'prepare':prepare,'seal':seal}[parser.parse_args().action]()
