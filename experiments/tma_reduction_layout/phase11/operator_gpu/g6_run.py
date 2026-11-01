"""CPU-only Modal compiler runner; no backend hooks or GPU allocation."""
from pathlib import Path
import modal
from . import common as c
from .g6_prepare import OUT, SNAPSHOT

app = modal.App('triton-g6-source-only-baseline')
image = (modal.Image.from_id(c.BASE_IMAGE)
         .env({'PYTHONPATH': '/opt/triton-src',
               'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
               'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
         .add_local_dir(SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
                        ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, cpu=8, memory=32768, timeout=1800, include_source=False,
              volumes={'/cache': cache, '/evidence': evidence})
def run(p):
    from .g6_compile import run as compile_all
    root = Path('/evidence/operator_gpu/g6')/modal.current_function_call_id()
    compile_all(p, root); cache.commit(); evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(attempt: int):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    folder = OUT/f'attempt_{attempt:03}'
    p = c.read(folder/'preregistration/protocol.json'); c.verify_sources(p)
    assert not (folder/'raw_export.zip').exists()
    call = run.spawn(p)
    c.write_once(folder/'dispatch.json', dict(call_id=call.object_id, profile='chenmiaoming',
                 GPU_allocated=False, protocol_SHA256=c.sha(c.encode(p))))
    blob = call.get(); c.write_once(folder/'raw_export.zip', blob); c.extract(blob, folder/'export')
    print('G6 CPU compilation:', c.read(folder/'export/result.json')['status'])
