"""NCU launches of the same SHA-checked CUBIN payload used for timing."""
import argparse
import ctypes as C
import json
import sys
from pathlib import Path
from experiments.tma_reduction_layout.phase11.m3 import common as c
from experiments.tma_reduction_layout.phase11.m3.runtime import Session


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    job=c.read(args.job)
    session=Session(Path(args.root),job['metadata'],maxgrid=max(r['grid'] for r in job['cells']),maxk=1)
    records=[]
    for cell in job['cells']:
        owners,params,_=session.params(cell['tag'],1)
        kernel=session.kernels[cell['tag']]
        kernel.guard()
        kernel.launch(cell['grid'],params,session.stream,cell['q'])
    session.driver.call('cuStreamSynchronize',[C.c_void_p],session.stream)
    session.driver.call('cuProfilerStart',[])
    for cell in job['cells']:
        owners,params,serialized=session.params(cell['tag'],1)
        kernel=session.kernels[cell['tag']]
        kernel.guard()
        kernel.launch(cell['grid'],params,session.stream,cell['q'])
        session.driver.call('cuStreamSynchronize',[C.c_void_p],session.stream)
        check=session.check(cell['tag'],1,[0])
        assert check['pass'],check
        records.append({**cell,'correctness':check,'parameters':serialized,'binding':kernel.binding()})
    session.driver.call('cuProfilerStop',[])
    imported=any(n=='triton' or n.startswith('triton.') for n in sys.modules)
    assert not imported
    print('PROFILE_BINDING '+json.dumps({'records':records,'environment':session.environment(),
        'Triton_imported':imported,'compile_calls':0,'driver_return_codes':sorted(session.driver.codes)}),flush=True)
    session.close()


if __name__=='__main__':main()
