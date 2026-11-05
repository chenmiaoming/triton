"""CPU-only pinned preflight. This app intentionally defines no GPU function."""
from pathlib import Path
import traceback
import modal

from .prepare import SOURCE, write_once

app = modal.App('triton-operator-survey-cpu-preparation')
image = (modal.Image.from_id('im-qzTUZumWu2JXjWic8K82cZ')
         .env({'PYTHONPATH': '/opt/triton-src',
               'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
               'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache',
               'CUDA_VISIBLE_DEVICES': ''})
         .add_local_dir(SOURCE, '/opt/triton-src/experiments/tma_reduction_layout/phase11/operator_survey',
                        copy=True, ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, cpu=8, memory=16384, timeout=1800,
              volumes={'/cache': cache, '/evidence': evidence}, include_source=False)
def preflight(protocol):
    import io
    import zipfile
    from experiments.tma_reduction_layout.phase11.operator_survey.cpu_compile import compile_all
    call_id = modal.current_function_call_id()
    root = Path('/evidence/operator_survey_preparation/cpu') / call_id
    temporary = Path('/tmp') / ('operator-survey-' + call_id + '.json')
    write_once(temporary, protocol)
    try:
        result = compile_all(temporary, root, Path('/cache/phase11-upstream20d9-kernel-cache'), 'pinned')
        result['modal_call_id'] = call_id
        write_once(root / 'worker.json', {'call_id': call_id, 'GPU_allocated': False,
                                       'function_gpu_argument': None, 'profile': 'chenmiaoming'})
    except Exception:
        root.mkdir(parents=True, exist_ok=True)
        write_once(root / 'preflight_error.txt', traceback.format_exc().encode())
    cache.commit()
    evidence.commit()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return stream.getvalue()


def export(protocol_path, output):
    import io
    import json
    import zipfile
    output = Path(output)
    if output.exists():
        raise FileExistsError('Pinned preflight exports are write-once')
    protocol = json.loads(Path(protocol_path).read_text())
    payload = preflight.remote(protocol)
    output.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
            raise ValueError('Invalid export member')
        archive.extractall(output)
    result = output / 'result.json'
    if not result.exists():
        raise RuntimeError('CPU preflight failed; inspect preserved preflight_error.txt')
    result = json.loads(result.read_text())
    print(result['status'], len(result['records']), 'pinned CPU programs; GPU allocated:', result['GPU_allocated'])
    if result['status'] != 'OK':
        raise SystemExit(1)


@app.local_entrypoint()
def main(protocol: str, output: str):
    export(protocol, output)
