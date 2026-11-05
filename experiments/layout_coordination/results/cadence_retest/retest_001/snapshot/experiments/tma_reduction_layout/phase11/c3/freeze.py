"""Freeze C3 code/parameters/predictions in a hash-bound local manifest."""
from experiments.tma_reduction_layout.phase11.c3 import common as c


def run():
    files={}
    for root in (c.SOURCE,c.OUT):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.name in ('freeze.json','validation.json'):continue
            files[p.relative_to(c.ROOT).as_posix()]=c.digest(p.read_bytes())
    value={'model_version':c.read(c.OUT/'theta_H100.json')['version'],
        'files':files,'new_GPU_jobs':0,'new_kernel_compiles':0,
        'D1_GPU_timing_started':False,'CXX_changes':False,
        'frozen_choices':{r['default']:r['choice'] for r in c.read(c.OUT/'D1_predictions.json')['cases']},
        'coverage_is_validation':False}
    c.write(c.OUT/'freeze.json',value)
    print('C3 freeze',len(files),'files')


if __name__=='__main__':
    run()
