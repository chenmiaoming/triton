"""Bounded CPU preflight and complete replacement H100 correctness/timing."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal
from . import common as c
from .g8_prepare import OUT,SNAPSHOT

app=modal.App('triton-g8-source-aligned-candidate-remeasurement')
image=(modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
          'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc']))
cache=modal.Volume.from_name('triton-build-cache');evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,cpu=8,memory=32768,timeout=3600,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def cpu(p,reference_payload):
    from .g8_compile import run
    root=Path('/evidence/operator_gpu/g8/cpu')/modal.current_function_call_id()
    run(p,reference_payload,root);cache.commit();evidence.commit();return c.archive(root)


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=5400,
              single_use_containers=True,include_source=False,volumes={'/evidence':evidence})
def measure(p,gate,payload,invocation):
    from .g8_runtime import run
    root=Path('/evidence/operator_gpu/g8/runtime')/modal.current_function_call_id()
    result=run(p,gate,payload,root,invocation);result['call_id']=modal.current_function_call_id()
    c.write_once(root/'returned_result.json',result);evidence.commit();return result


@app.local_entrypoint()
def main(action:str='cpu',attempt:int=1):
    import os
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(OUT/'preregistration/protocol.json');c.verify_sources(p)
    if action=='cpu':
        jobs=[(OUT/'cpu'/f'attempt_{attempt:03}',cpu,(p,(OUT/'preregistration/reference_payload.zip').read_bytes()))]
    else:
        gate=c.read(OUT/'compile_gate.json');payload=(OUT/'compiled_payload.zip').read_bytes()
        assert gate['status']=='OK' and gate['protocol_SHA256']==c.sha(c.encode(p)) and gate['payload_SHA256']==c.sha(payload)
        if action=='check':jobs=[(OUT/'correctness'/f'attempt_{attempt:03}',measure,(p,gate,payload,0))]
        else:
            assert action=='timing' and c.read(OUT/'correctness/attempt_001/raw.json')['status']=='OK'
            jobs=[(OUT/'timing'/f'invocation_{i}',measure,(p,gate,payload,i)) for i in range(1,p['timing']['workers']+1)]
    calls=[]
    for folder,fn,args in jobs:
        assert not folder.exists(),folder
        call=fn.spawn(*args);c.write_once(folder/'dispatch.json',dict(call_id=call.object_id,profile='chenmiaoming',
            action=action,protocol_SHA256=c.sha(c.encode(p))))
        calls.append((folder,call))
    def save(item):
        folder,call=item;value=call.get()
        if isinstance(value,bytes):
            c.write_once(folder/'raw_export.zip',value);c.extract(value,folder/'export');value=c.read(folder/'export/result.json')
        else:c.write_once(folder/'raw.json',value)
        print('G8',action,folder.name,value['status'],flush=True);return value['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:states=list(pool.map(save,calls))
    assert all(s=='OK' for s in states),states
