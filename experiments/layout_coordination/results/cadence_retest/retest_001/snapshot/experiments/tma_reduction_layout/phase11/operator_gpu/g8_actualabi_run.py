"""Bounded Modal CPU compilation for the supplemental G8 actual-ABI audit."""
from pathlib import Path
import modal
from . import common as c
from .g8_actualabi import AUDIT

app = modal.App('triton-g8-exact-batch64-attribute-audit')
snapshot = Path('/tmp/operator-gpu-g8-worker-snapshot')
image = (modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
          'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(snapshot/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_file(c.SOURCE/'g8_actualabi.py','/opt/triton-src/experiments/tma_reduction_layout/phase11/operator_gpu/g8_actualabi.py',copy=True)
    .add_local_file(c.SOURCE/'g8_actualabi_run.py','/opt/triton-src/experiments/tma_reduction_layout/phase11/operator_gpu/g8_actualabi_run.py',copy=True))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,cpu=8,memory=32768,timeout=1800,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def compile_actual(protocol):
    from .g8_actualabi import run
    root = Path('/evidence/operator_gpu/g8_actual_abi')/modal.current_function_call_id()
    run(protocol,root); cache.commit(); evidence.commit(); return c.archive(root)


@app.local_entrypoint()
def main():
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol = c.read(AUDIT/'preregistration/protocol.json'); c.verify_sources(protocol)
    destination = AUDIT/'cpu/attempt_001'; assert not destination.exists()
    call = compile_actual.spawn(protocol)
    c.write_once(destination/'dispatch.json',dict(call_id=call.object_id,profile='chenmiaoming',GPU_allocated=False,
                                                protocol_SHA256=c.sha(c.encode(protocol))))
    blob = call.get(); c.write_once(destination/'raw_export.zip',blob); c.extract(blob,destination/'export')
    raw = c.read(destination/'export/result.json'); print('G8 exact batch64 attribute CPU:',raw['status'],flush=True)
    assert raw['status'] == 'OK'
