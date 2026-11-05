"""CPU-only pass audit using the existing native compiler image and cache volume."""
from pathlib import Path
import modal
from . import common as c
from .g5_prepare import OUT,SNAPSHOT,protocol_path

app=modal.App('triton-operator-g5-baseline-compiler-audit')
image=(modal.Image.from_id(c.BASE_IMAGE)
       .env({'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache',
             'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
       .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                       ignore=['**/__pycache__/**','**/*.pyc']))
cache=modal.Volume.from_name('triton-build-cache');evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,cpu=8,memory=32768,timeout=3600,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def audit(p,pilot):
    from experiments.tma_reduction_layout.phase11.operator_gpu.g5_worker import run
    root=Path('/evidence/operator_gpu/g5')/modal.current_function_call_id()
    run(p,root,pilot);cache.commit();evidence.commit();return c.archive(root)


@app.local_entrypoint()
def main(action:str='pilot',attempt:int=1):
    import os
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming' and action in ('pilot','matrix')
    p=c.read(protocol_path());c.verify_sources(p)
    folder=OUT/action/f'attempt_{attempt:03}';assert not folder.exists()
    call=audit.spawn(p,action=='pilot')
    c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'protocol_SHA256':c.sha(c.encode(p)),
                                       'profile':'chenmiaoming','GPU_allocated':False})
    blob=call.get();c.write_once(folder/'raw_export.zip',blob);c.extract(blob,folder/'export')
    result=c.read(folder/'export/result.json');print('G5',action,result['status'],len(result['programs']),'CPU programs')
    assert result['status']=='OK'
