"""Frozen compiler cache, CPU preflight, real-JIT verification, correctness, timing."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal
from . import common as c
from .g3b_prepare import OUT,SNAPSHOT,PROTOCOL_FILE

app=modal.App('triton-operator-layout-g3b-training')
image=(modal.Image.from_id(c.BASE_IMAGE).env({'PYTHONPATH':'/opt/triton-src',
       'TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache',
       'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
       .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc']))
cache=modal.Volume.from_name('triton-build-cache');evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,cpu=8,memory=16384,timeout=1800,include_source=False,volumes={'/cache':cache,'/evidence':evidence})
def cpu(protocol):
    from .g3b_compile import run
    root=Path('/evidence/operator_gpu/g3b/cpu')/modal.current_function_call_id()
    run(protocol,root);cache.commit();evidence.commit();return c.archive(root)


@app.function(image=image,gpu='H100!:1',cpu=4,memory=16384,timeout=1800,single_use_containers=True,
              include_source=False,volumes={'/cache':cache,'/evidence':evidence})
def binding(protocol,compiled):
    import traceback
    from .g3b_bind import run
    c.verify_sources(protocol)
    try:result=run(protocol,compiled)
    except Exception:result={'status':'FAILED','error':traceback.format_exc()};print(result['error'],flush=True)
    root=Path('/evidence/operator_gpu/g3b/binding')/modal.current_function_call_id()
    c.write_once(root/'result.json',result);cache.commit();evidence.commit();return result


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=5400,single_use_containers=True,
              include_source=False,volumes={'/evidence':evidence})
def execute(protocol,compiled,payload,invocation):
    from .g3b_runtime import run
    root=Path('/evidence/operator_gpu/g3b/runtime')/modal.current_function_call_id()
    result=run(protocol,compiled,payload,root,invocation);evidence.commit();return result


@app.local_entrypoint()
def main(action:str='cpu',attempt:int=1):
    import os,io,zipfile
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(OUT/'preregistration'/PROTOCOL_FILE);c.verify_sources(p)
    if action=='cpu':jobs=[(OUT/'cpu'/f'attempt_{attempt:03}',cpu,(p,))]
    else:
        compiled=c.read(OUT/'cpu/attempt_002/export/result.json');assert compiled['status']=='OK'
        if action=='bind':jobs=[(OUT/'binding'/f'attempt_{attempt:03}',binding,(p,compiled))]
        else:
            assert c.read(OUT/'binding/attempt_001/raw.json')['status']=='OK'
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
                root=OUT/'cpu/attempt_002/export'
                for path in sorted((root/'artifacts').rglob('*')):
                    if path.is_file():z.write(path,path.relative_to(root))
            payload=stream.getvalue()
            if not (OUT/'compiled_payload.zip').exists():c.write_once(OUT/'compiled_payload.zip',payload)
            else:assert (OUT/'compiled_payload.zip').read_bytes()==payload
            if action=='check':jobs=[(OUT/'correctness'/f'attempt_{attempt:03}',execute,(p,compiled,payload,0))]
            else:
                assert action=='timing'
                assert c.read(OUT/'correctness/attempt_001/raw.json')['status']=='OK'
                jobs=[(OUT/'timing'/f'invocation_{i}',execute,(p,compiled,payload,i)) for i in range(1,p['timing']['workers']+1)]
    calls=[]
    for folder,fn,args in jobs:
        assert not folder.exists(),folder
        call=fn.spawn(*args);c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':'chenmiaoming'});calls.append((folder,call))
    def save(item):
        folder,call=item;result=call.get()
        if isinstance(result,bytes):
            c.write_once(folder/'raw_export.zip',result);c.extract(result,folder/'export');result=c.read(folder/'export/result.json')
        else:c.write_once(folder/'raw.json',result)
        print('G3b return',folder,result['status'],flush=True);return result['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:states=list(pool.map(save,calls))
    assert all(x=='OK' for x in states),states
