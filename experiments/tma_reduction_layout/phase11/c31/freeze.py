"""Hash-bind C3.1 sources, model, predictions and their derivation."""
from experiments.tma_reduction_layout.phase11.c31 import common as c


def run():
    files={}
    for root in (c.SOURCE,c.OUT):
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if path.name in ('freeze.json','validation.json'):continue
            files[str(path.relative_to(c.ROOT))]=c.sha(path.read_bytes())
    blind=c.read(c.V1/'D1_predictions.json')
    c.write(c.V1/'freeze.json',{'version':'c3_v1','files':files,
        'old_C3_write_once':True,'D1_compiled':False,'D1_timed':False,'new_GPU_jobs':0,
        'choices':{r['default']:r['choice'] for r in blind['cases']},
        'not_held_out_validation':True})
    print('C3.1 freeze',len(files),'files')


if __name__=='__main__':run()
