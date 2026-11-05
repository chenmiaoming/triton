"""NCU target for one archived training merge; no compiler imported."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from . import common as c
from .driver import Device,Loaded
from .g3b_runtime import Buffers
from .production_buffers import unpack


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True)
    a=p.parse_args();job=c.read(a.job);case=job['case'];software=job['mode']=='software'
    device=Device();buffers=Buffers(device,case)
    meta=job['metadata'][0];kernel=Loaded(device,Path(a.root),meta)
    owners,params=kernel.parameters(buffers.args(meta));grid=(1,1,1) if software else ((case['N']+127)//128,1,1)
    for _ in range(8):kernel.guard();kernel.launch(grid,params,device.stream)
    device.synchronize();device.driver.call('cuProfilerStart',[])
    kernel.guard();kernel.launch(grid,params,device.stream);device.synchronize();device.driver.call('cuProfilerStop',[])
    if not software:check=buffers.check()
    else:
        items=[]
        for name,row in buffers.rows.items():
            raw=device.download(row['base'],row['size']+512)
            okay=raw[:256]==row['initial'][:256] and raw[-256:]==row['initial'][-256:]
            if not row['writable']:okay=okay and raw==row['initial']
            else:
                actual=unpack(raw[256:-256],row['dtype'],row['shape'])
                okay=okay and bool(np.allclose(actual[:128],buffers.expected[name][:128],rtol=.008,atol=.001))
                okay=okay and raw[256+256:-256]==row['initial'][256+256:-256]
            items.append({'name':name,'pass':okay})
        check={'pass':all(x['pass'] for x in items),'details':items,'active_output_columns':128}
    assert check['pass'],check
    assert not any(x=='triton' or x.startswith('triton.') for x in sys.modules)
    assert sorted(device.driver.codes)==[0]
    print('PROFILE_BINDING '+json.dumps({'records':[{'tag':meta['tag'],'grid':list(grid),
          'correctness':check,'binding':kernel.binding()}],'environment':device.environment,
          'Triton_imported':False,'compile_calls':0,'mode':job['mode'],'warmups':8,'driver_return_codes':[0]},allow_nan=False),flush=True)
    device.close()


if __name__=='__main__':main()
