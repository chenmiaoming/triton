"""Frozen protocols, one correctness worker and independent timing workers."""
from pathlib import Path
import os
import modal
from . import common as c

STAGE=os.environ.get('NONREDUCE_STAGE','stage_n1_baseline')
ROOT=c.OUT/STAGE
snapshot=Path('/tmp')/('nonreduction-'+STAGE)
app=modal.App('triton-nonreduction-frozen-runtime')
image=(modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
          'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(snapshot/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc']))
cache=modal.Volume.from_name('triton-build-cache');evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=7200,
    single_use_containers=True,include_source=False,volumes={'/cache':cache,'/evidence':evidence})
def measure(p,payload,invocation):
    from .runtime import run
    import traceback
    root=Path('/evidence/nonreduction/runtime')/modal.current_function_call_id()
    try:run(p,payload,root,invocation)
    except Exception:
        if not (root/'result.json').exists():c.write_once(root/'result.json',dict(status='FAILED',error=traceback.format_exc()))
    evidence.commit();return c.archive(root)


@app.function(image=image,cpu=8,memory=32768,timeout=7200,
    include_source=False,volumes={'/cache':cache,'/evidence':evidence})
def compile_candidates(p,payload):
    from .candidate_compile import run
    import traceback
    root=Path('/evidence/nonreduction/candidates')/modal.current_function_call_id()
    try:run(p,payload,root)
    except Exception:
        if not (root/'result.json').exists():c.write_once(root/'result.json',dict(status='FAILED',error=traceback.format_exc()))
    cache.commit();evidence.commit();return c.archive(root)


@app.local_entrypoint()
def main(action:str='check',attempt:int=1):
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    protocol=c.read(ROOT/'preregistration/protocol.json')
    c.verify_snapshot(protocol,snapshot);payload=Path(protocol['payload_path']).read_bytes()
    assert c.sha(payload)==protocol['payload_SHA256']
    assert action in ['check','timing','compile']
    target=ROOT/action/f'attempt_{attempt:03}';assert not target.exists(),target
    invocations=[0] if action=='check' else list(range(1,protocol['timing']['workers']+1))
    if action=='compile':invocations=[-1]
    calls=[]
    for invocation in invocations:
        call=compile_candidates.spawn(protocol,payload) if action=='compile' else measure.spawn(protocol,payload,invocation)
        c.write_once(target/f'invocation_{invocation:03}'/'dispatch.json',dict(call_id=call.object_id,
            profile='chenmiaoming',protocol_SHA256=c.sha(c.encode(protocol))))
        calls.append((invocation,call))
    okay=True
    for invocation,call in calls:
        folder=target/f'invocation_{invocation:03}';blob=call.get()
        c.write_once(folder/'raw_export.zip',blob);c.extract(blob,folder/'export')
        raw=c.read(folder/'export/result.json');c.write_once(folder/'raw.json',raw)
        print(STAGE,action,invocation,raw['status'],flush=True);okay &= raw['status']=='OK'
    assert okay,'Inspect preserved worker failures'
