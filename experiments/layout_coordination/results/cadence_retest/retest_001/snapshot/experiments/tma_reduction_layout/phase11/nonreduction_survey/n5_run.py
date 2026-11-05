"""Launch only the frozen N5 binding/check job; never a performance run."""
from pathlib import Path
import os
import zipfile
import io

import modal
from . import common as c


REVISION = int(os.environ.get('NONREDUCE_N5_REVISION', '1'))
ROOT = c.OUT / 'stage_n5' / f'revision_{REVISION:03}'
snapshot = Path('/tmp') / f'nonreduction-n5-revision-{REVISION:03}'
app = modal.App('triton-nonreduction-footprint-tail-binding')
image = (modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH': '/opt/triton-src',
          'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
          'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(snapshot/'experiments', '/opt/triton-src/experiments', copy=True,
                   ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=5400,
    single_use_containers=True, include_source=False,
    volumes={'/cache': cache, '/evidence': evidence})
def bind(protocol):
    from .n5_worker import run
    import traceback
    root = Path('/evidence/nonreduction/n5_binding') / modal.current_function_call_id()
    try:
        run(protocol, root)
    except Exception:
        if not (root/'result.json').exists():
            c.write_once(root/'result.json', dict(status='FAILED', error=traceback.format_exc()))
    cache.commit()
    evidence.commit()
    return c.archive(root)


def timing_payload(root, result):
    """Package the two passed footprint cases for a separately approved run."""
    selected = set(result['timing_case_ids'])
    programs = [program for program in result['programs'] if program['geometry'] in selected]
    cases = [case for case in result['cases'] if case['id'] in selected]
    records = [record for record in result['records'] if record['case'] in selected]
    assert len(cases) == 2 and len(programs) == 6 and all(record['status']=='OK' for record in records)
    payload_index = dict(status='OK', cases=cases, programs=programs, records=records,
        source_binding_protocol_SHA256=result['protocol_SHA256'],
        role='Accepted frozen programs only; no timing authorized by this payload',
        correctness_checks=[check for check in result['checks'] if check['case'] in selected])
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('result.json', c.encode(payload_index))
        for program in programs:
            folder = root / program['artifact_directory']
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    return output.getvalue()


@app.local_entrypoint()
def main(attempt: int = 1):
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol = c.read(ROOT/'preregistration/protocol.json')
    c.verify_snapshot(protocol, snapshot)
    folder = ROOT/'binding'/f'attempt_{attempt:03}'
    assert not folder.exists(), folder
    call = bind.spawn(protocol)
    c.write_once(folder/'dispatch.json', dict(call_id=call.object_id,
        profile='chenmiaoming', protocol_SHA256=c.sha(c.encode(protocol)), formal_timing=False))
    blob = call.get()
    c.write_once(folder/'raw_export.zip', blob)
    c.extract(blob, folder/'export')
    result = c.read(folder/'export/result.json')
    print('N5 binding', result['status'], result.get('counts'), flush=True)
    if result['status'] == 'OK':
        c.write_once(folder/'accepted_timing_payload.zip', timing_payload(folder/'export', result))
    assert result['status']=='OK', 'Inspect preserved N5 baseline/check failures before any timing'
