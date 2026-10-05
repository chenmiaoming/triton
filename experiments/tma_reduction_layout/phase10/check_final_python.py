"""Verify the formatted Python test on the already checked native image.

Only Python formatting changed after the accepted native/lit run. Reuse that
exact compiled image without make, as required for Python-only changes.
"""
from pathlib import Path
import hashlib
import os

import modal

from experiments.tma_reduction_layout.phase10 import common as c

app = modal.App('triton-phase10-final-python-check')
base_image = 'im-KaEq6823Uj1qypGzBbwFr9'
checkout = Path('/tmp/triton-tma-current-main')
test_path = 'python/test/unit/cuda/test_tensor_descriptor_cuda.py'
image = (modal.Image.from_id(base_image)
         .add_local_file(checkout / test_path, '/opt/triton-src/' + test_path, copy=True)
         .add_local_file(Path(__file__),
                         '/opt/triton-src/experiments/tma_reduction_layout/phase10/check_final_python.py', copy=True))
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase10-evidence')


@app.function(image=image, gpu='H100!:1', timeout=1800,
              volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def check(expected_source, expected_native):
    import json
    import io
    import subprocess
    import zipfile
    import torch
    import triton._C.libtriton as native
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA

    root = Path('/opt/triton-src')
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in expected_source}
    assert actual == expected_source
    native_sha = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    assert native_sha == expected_native
    assert 'H100' in torch.cuda.get_device_name(0)
    assert torch.cuda.get_device_capability(0) == (9, 0)
    command = ['python3', '-m', 'pytest', '-s', '--tb=short',
               test_path + '::test_tensor_descriptor_partial_reduction_layout']
    cache_dir = Path('/tmp/phase10-final-python-cache')
    assert not cache_dir.exists()
    environment = dict(os.environ, TRITON_CACHE_DIR=str(cache_dir))
    run = subprocess.run(command, cwd=root, capture_output=True, text=True, env=environment)
    result = {'call_id': modal.current_function_call_id(),
              'status': 'OK' if run.returncode == 0 else 'FAILED',
              'return_code': run.returncode, 'command': command,
              'source': actual, 'native_SHA256': native_sha,
              'GPU_timing': False, 'native_build_reused': True,
              'base_image': base_image, 'gpu_environment': ExactCUDA().environment(),
              'fresh_disk_cache': str(cache_dir)}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(cache_dir.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(cache_dir))
    result['compiled_kernel_count'] = len(list(cache_dir.rglob('*.cubin')))
    out = Path('/evidence/stage_b') / result['call_id']
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (out / 'pytest.log').write_text(run.stdout + run.stderr)
    (out / 'test_source.py').write_bytes((root / test_path).read_bytes())
    (out / 'raw_compile_cache.zip').write_bytes(stream.getvalue())
    volume.commit()
    print(run.stdout + run.stderr, flush=True)
    return result, run.stdout + run.stderr, (root / test_path).read_bytes(), stream.getvalue()


@app.local_entrypoint()
def main():
    accepted = c.read(c.OUT / 'stage_b/accepted.json')
    expected = {name: c.sha((checkout / name).read_bytes()) for name in accepted['source']}
    for name in expected:
        if name != test_path:
            assert expected[name] == accepted['source'][name]
    call = check.spawn(expected, accepted['result']['native_SHA256'])
    root = c.OUT / 'stage_b/final_python'
    c.write_once(root / 'dispatch.json', {'call_id': call.object_id,
                 'profile': os.environ.get('MODAL_PROFILE'), 'source': expected})
    result, log, source, archive = call.get()
    c.write_once(root / 'result.json', result)
    c.write_once(root / 'pytest.log', log.encode())
    c.write_once(root / 'test_source.py', source)
    c.write_once(root / 'raw_compile_cache.zip', archive)
    assert result['status'] == 'OK'
    assert result['compiled_kernel_count'] == 54
