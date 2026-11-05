"""Profile exactly the exported stage1/merge binaries; full decode timing stays separate."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import numpy as np
from . import common as c
from .driver import Device, Loaded
from .production_data import make_data
from .production_buffers import Buffers


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    job=c.read(args.job)
    case=job['case']
    software=job['mode']=='software'
    batch=1 if software else 8
    length=8192 if case['family']=='C06' else 2048
    data=make_data(case,batch=batch,length=length)
    if software:
        for name,mask in data['masks'].items():
            active=mask.copy()
            # Some complete-output masks are read-only broadcast views. Build a
            # private mask for the single-CTA capture; never mutate that view.
            mask=np.zeros_like(active,dtype=bool)
            data['masks'][name]=mask
            if case['family']=='C06':
                mask[0,0,:]=active[0,0,:]
            elif name=='Att_Out':
                mask[0,0,0,:]=active[0,0,0,:]
            elif name=='Att_Lse':
                mask[0,0,0]=active[0,0,0]
    device=Device()
    kernels=[Loaded(device,Path(args.root),meta) for meta in job['metadata']]
    buffers=Buffers(device,case,data)
    grid=(1,1,1) if software else data['grid']
    for kernel in kernels:
        owners,params=kernel.parameters(buffers.arguments(kernel.meta))
        kernel.guard()
        kernel.launch(grid,params,device.stream)
    device.synchronize()
    records=[]
    device.driver.call('cuProfilerStart',[])
    for kernel in kernels:
        buffers.reset()
        owners,params=kernel.parameters(buffers.arguments(kernel.meta))
        kernel.guard()
        kernel.launch(grid,params,device.stream)
        device.synchronize()
        check=buffers.check(stage2=False,include_inputs=False)
        assert check['pass'],check
        records.append({'tag':kernel.meta['tag'],'grid':list(grid),'correctness':check,'binding':kernel.binding()})
    device.driver.call('cuProfilerStop',[])
    imported=any(name=='triton' or name.startswith('triton.') for name in sys.modules)
    assert not imported and sorted(device.driver.codes)==[0]
    print('PROFILE_BINDING '+json.dumps({'records':records,'environment':device.environment,
          'Triton_imported':imported,'compile_calls':0,'mode':job['mode'],'batch':batch,'length':length,
          'driver_return_codes':sorted(device.driver.codes)},allow_nan=False),flush=True)
    device.close()


if __name__=='__main__':
    main()
