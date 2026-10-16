"""Batch checked launches of frozen CUBINs inside one NCU capture region."""
import argparse
import ctypes as C
import json
import sys
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c2 import common as c
from experiments.tma_reduction_layout.phase11.c2.timing import Session


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True)
    args=p.parse_args();job=c.read(args.job);metadata=job['metadata']
    s=Session(Path(args.root),metadata,maxgrid=max(cell['grid'] for cell in job['cells']))
    # Warm every program/K before starting; no source import or compilation.
    for cell in job['cells']:
        kernel=s.kernels[cell['tag']]
        owner,params,_=s.params(cell['tag'],cell['K'])
        kernel.guard();kernel.launch(cell['grid'],params,s.stream,cell['q'])
    s.driver.call('cuStreamSynchronize',[C.c_void_p],s.stream)
    records=[]
    s.driver.call('cuProfilerStart',[])
    for cell in job['cells']:
        kernel=s.kernels[cell['tag']]
        owner,params,serialized=s.params(cell['tag'],cell['K'])
        kernel.guard();kernel.launch(cell['grid'],params,s.stream,cell['q'])
        s.driver.call('cuStreamSynchronize',[C.c_void_p],s.stream)
        checked=s.check(cell['tag'],cell['K'],[0])
        records.append({**cell,'correctness':checked,'parameters':serialized,'binding':kernel.binding()})
    s.driver.call('cuProfilerStop',[])
    imported=any(m=='triton' or m.startswith('triton.') for m in sys.modules)
    assert not imported
    print('PROFILE_BINDING '+json.dumps({'records':records,'environment':s.environment(),
          'Triton_imported':imported,'compile_calls':0,'driver_return_codes':sorted(s.driver.codes)}),flush=True)
    s.close()


if __name__=='__main__':main()
