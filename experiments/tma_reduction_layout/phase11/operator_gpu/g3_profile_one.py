"""Compiler-free, cache-primed profiling of the exact G2 production binaries."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from . import common as c
from .driver import Device, Loaded
from .production_buffers import Buffers
from .production_data import make_data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--job',required=True)
    args=p.parse_args()
    job=c.read(args.job)
    case=job['case']
    software=job['mode']=='software'
    data=make_data(case,batch=1 if software else 8,
                   length=8192 if case['family']=='C06' else 2048)
    if software:
        for name,old in data['masks'].items():
            mask=np.zeros_like(old,dtype=bool)
            if case['family']=='C06':mask[0,0,:]=old[0,0,:]
            elif name=='Att_Out':mask[0,0,0,:]=old[0,0,0,:]
            elif name=='Att_Lse':mask[0,0,0]=old[0,0,0]
            data['masks'][name]=mask
    device=Device()
    buffers=Buffers(device,case,data)
    grid=(1,1,1) if software else data['grid']
    records=[]
    # One report/one binary: every application replay primes the same target.
    assert len(job['metadata'])==1
    for meta in job['metadata']:
        kernel=Loaded(device,Path(args.root),meta)
        buffers.reset()
        owners,params=kernel.parameters(buffers.arguments(meta))
        for _ in range(job['warmups']):
            kernel.guard();kernel.launch(grid,params,device.stream)
        device.synchronize()
        device.driver.call('cuProfilerStart',[])
        kernel.guard();kernel.launch(grid,params,device.stream)
        device.synchronize()
        device.driver.call('cuProfilerStop',[])
        check=buffers.check(stage2=False,include_inputs=True)
        assert check['pass'],check
        records.append({'tag':meta['tag'],'grid':list(grid),'correctness':check,'binding':kernel.binding()})
    imported=any(x=='triton' or x.startswith('triton.') for x in sys.modules)
    assert not imported and sorted(device.driver.codes)==[0]
    print('PROFILE_BINDING '+json.dumps({'records':records,'environment':device.environment,
          'Triton_imported':False,'compile_calls':0,'mode':job['mode'],'batch':data['batch'],
          'length':data['length'],'warmups':job['warmups'],'driver_return_codes':[0]},allow_nan=False),flush=True)
    device.close()


if __name__=='__main__':main()
