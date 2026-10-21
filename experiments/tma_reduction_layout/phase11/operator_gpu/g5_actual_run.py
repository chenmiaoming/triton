"""CPU-only verification of G4b's additional real-call ABI specializations."""
from pathlib import Path
import modal
from . import common as c
from .g5_prepare import OUT, SNAPSHOT

app = modal.App('triton-g5-real-call-ABI-audit')
image = (modal.Image.from_id(c.BASE_IMAGE)
         .env({'PYTHONPATH': '/opt/triton-src',
               'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache-g5-actual',
               'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
         .add_local_dir(SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
                        ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, cpu=8, memory=32768, timeout=1800, include_source=False,
              volumes={'/cache': cache, '/evidence': evidence})
def run(p):
    import os
    import subprocess
    c.verify_sources(p)
    root = Path('/evidence/operator_gpu/g5_actual') / modal.current_function_call_id()
    result = {'status': 'OK', 'programs': [], 'GPU_allocated': False, 'native_rebuilds': 0,
              'protocol_SHA256': c.sha(c.encode(p)),
              'ccache_before': subprocess.check_output(['ccache', '--print-stats'], text=True)}
    for item in p['cases']:
        for variant in ['U_actual', 'G0_actual_equivalent']:
            folder = root / item['case']['id'] / variant
            c.write_once(folder/'job.json', {'item': item, 'variant': variant})
            argv = [os.sys.executable, '-m', 'experiments.tma_reduction_layout.phase11.operator_gpu.g5_actual_run',
                    '--job', str(folder/'job.json'), '--root', str(folder)]
            proc = subprocess.run(argv, env=dict(os.environ, TRITON_ALWAYS_COMPILE='1'), text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            c.write_once(folder/'command.json', {'argv': argv, 'returncode': proc.returncode, 'output': proc.stdout})
            row = c.read(folder/'result.json') if (folder/'result.json').exists() else {'status': 'FAILED', 'error': proc.stdout}
            result['programs'].append({'path': str(folder.relative_to(root)), **row})
            print('G5 actual ABI', item['case']['id'], variant, row['status'], flush=True)
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    assert result['ccache_before'] == result['ccache_after']
    result['status'] = 'OK' if all(r['status'] == 'OK' for r in result['programs']) else 'FAILED'
    c.write_once(root/'result.json', result); cache.commit(); evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(attempt: int = 1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    p = c.read(OUT/'preregistration/actual_protocol.json'); c.verify_sources(p)
    folder = OUT/'actual'/f'attempt_{attempt:03}'; assert not folder.exists()
    call = run.spawn(p)
    c.write_once(folder/'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming',
                 'GPU_allocated': False, 'protocol_SHA256': c.sha(c.encode(p))})
    blob = call.get(); c.write_once(folder/'raw_export.zip', blob); c.extract(blob, folder/'export')
    result = c.read(folder/'export/result.json'); print('G5 actual ABI', result['status'])
    assert result['status'] == 'OK'


def child(job, root):
    from . import g5_worker
    from .g5_vllm_literal3 import reduce_segments
    original = g5_worker.function_and_constants
    def replace(item, variant, gl):
        fn, constants, attrs, old = original(item, variant, gl)
        if not variant.startswith('U') and item['case']['family'] == 'C06':
            fn = reduce_segments
        return fn, constants, attrs, old
    g5_worker.function_and_constants = replace
    install = g5_worker.install_intervention
    g5_worker.install_intervention = lambda v, path: install('U' if v.startswith('U') else 'G0_fullsuffix', path)
    g5_worker.child(job, root)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--job', required=True); parser.add_argument('--root', required=True)
    args = parser.parse_args(); child(c.read(args.job), args.root)
