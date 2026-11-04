"""Bounded Modal dispatch for individually frozen experiment revisions."""
from pathlib import Path
import os
import modal
from . import common as c

REVISION=int(os.environ.get('NONREDUCE_REVISION','1'))
ROOT=c.OUT/'stage_n1'/f'revision_{REVISION:03}'
snapshot=Path('/tmp')/f'nonreduction-n1-revision-{REVISION:03}'
app=modal.App('triton-nonreduction-source-aligned-layouts')
image=(modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
          'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(snapshot/'experiments','/opt/triton-src/experiments',copy=True,
                   ignore=['**/__pycache__/**','**/*.pyc']))
cache=modal.Volume.from_name('triton-build-cache');evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=5400,
              single_use_containers=True,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def bind(p):
    from .bind_worker import run
    import traceback
    root=Path('/evidence/nonreduction/n1_bind')/modal.current_function_call_id()
    try:run(p,root)
    except Exception:
        c.write_once(root/'uncaught_failure.json',dict(status='FAILED',error=traceback.format_exc()))
        if not (root/'result.json').exists():
            c.write_once(root/'result.json',dict(status='FAILED',error=traceback.format_exc(),
                protocol_SHA256=c.sha(c.encode(p))))
    cache.commit();evidence.commit();return c.archive(root)


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=5400,
              single_use_containers=True,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def autotune(p):
    from .tune_worker import run
    import traceback
    root=Path('/evidence/nonreduction/n1_tune')/modal.current_function_call_id()
    try:run(p,root)
    except Exception:
        c.write_once(root/'uncaught_failure.json',dict(status='FAILED',error=traceback.format_exc()))
        if not (root/'result.json').exists():c.write_once(root/'result.json',dict(status='FAILED',error=traceback.format_exc()))
    cache.commit();evidence.commit();return c.archive(root)


@app.function(image=image,cpu=8,memory=32768,timeout=3600,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def g0(p,payload):
    from .gl_cpu import run
    import traceback
    root=Path('/evidence/nonreduction/n1_g0')/modal.current_function_call_id()
    try:run(p,payload,root)
    except Exception:
        c.write_once(root/'uncaught_failure.json',dict(status='FAILED',error=traceback.format_exc()))
        if not (root/'result.json').exists():c.write_once(root/'result.json',dict(status='FAILED',error=traceback.format_exc()))
    cache.commit();evidence.commit();return c.archive(root)


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=5400,
              single_use_containers=True,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def g0bind(p,payload):
    from .bind_g0_worker import run
    import traceback
    root=Path('/evidence/nonreduction/n1_actual_g0')/modal.current_function_call_id()
    try:run(p,payload,root)
    except Exception:
        c.write_once(root/'uncaught_failure.json',dict(status='FAILED',error=traceback.format_exc()))
        if not (root/'result.json').exists():c.write_once(root/'result.json',dict(status='FAILED',error=traceback.format_exc()))
    cache.commit();evidence.commit();return c.archive(root)


@app.local_entrypoint()
def main(action:str='bind',attempt:int=1):
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(ROOT/'preregistration/protocol.json');c.verify_snapshot(p,snapshot)
    assert action in ['bind','autotune','g0','g0bind']
    folder=ROOT/('binding' if action=='bind' else action)/f'attempt_{attempt:03}';assert not folder.exists(),folder
    fn={'bind':bind,'autotune':autotune,'g0':g0,'g0bind':g0bind}[action]
    args=(p,Path(p['reference_payload_path']).read_bytes()) if action in ['g0','g0bind'] else (p,)
    call=fn.spawn(*args);c.write_once(folder/'dispatch.json',dict(call_id=call.object_id,
        profile='chenmiaoming',protocol_SHA256=c.sha(c.encode(p))))
    blob=call.get();c.write_once(folder/'raw_export.zip',blob);c.extract(blob,folder/'export')
    r=c.read(folder/'export/result.json');print('N1 U binding',r['status'],flush=True)
    assert r['status']=='OK'
