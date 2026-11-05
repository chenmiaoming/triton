"""Seal only C3.4; old evidence remains byte-for-byte unchanged."""
from experiments.tma_reduction_layout.phase11.c34 import common as c
from experiments.tma_reduction_layout.phase11.c34.validate import run


def seal():
    result=run(pre_freeze=True)
    files={str(p.relative_to(c.ROOT)):c.sha(p.read_bytes())
        for root in (c.SOURCE,c.OUT) for p in sorted(root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('freeze.json','validation.json')}
    c.write_once(c.OUT/'freeze.json',{'stage':'C3.4','files':files,
        'parent_C33_commit':c.protocol()['C33_commit'],
        'GPU_allocations':0,'D1_compiled':False,'D1_timed':False})
    for name,digest in files.items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    c.derived(c.OUT/'validation.json',result)
    print('C3.4 source/results sealed after validator PASS',flush=True)


if __name__=='__main__':seal()
