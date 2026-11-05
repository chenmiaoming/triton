"""Seal completed C2d evidence without modifying any raw record."""
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c2d.validate import run


def main():
    validation=run()
    files={str(p.relative_to(c.ROOT)):c.sha(p.read_bytes())
        for root in (c.SOURCE,c.OUT) for p in sorted(root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('freeze.json','validation.json')}
    c.write_once(c.OUT/'freeze.json',{'stage':'C2d','files':files,
        'validation_SHA256':c.sha(c.encode(validation)),
        'D1_protected':10,'D1_compiled':False,'D1_timed':False,'native_rebuilds':0})
    run()


if __name__=='__main__': main()
