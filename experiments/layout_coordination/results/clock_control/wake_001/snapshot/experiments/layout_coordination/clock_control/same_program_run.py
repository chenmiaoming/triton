"""Dispatch the preregistered same-program diagnostic, preserving all exports."""
from pathlib import Path
import os
import modal
from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c

ROOT = Path(os.environ.get('CLOCK_RUN_ROOT', '/tmp/same-program-unprepared'))
SNAPSHOT = ROOT/'snapshot'
BASE_IMAGE = 'im-qzTUZumWu2JXjWic8K82cZ'
app = modal.App('layout-coordination-same-program-repeatability')
image = (modal.Image.from_id(BASE_IMAGE)
    .env({'PYTHONPATH': '/opt/triton-src', 'OMP_NUM_THREADS': '4'})
    .add_local_dir(SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
                   ignore=['**/__pycache__/**','**/*.pyc']))
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, gpu='H100!:1', cpu=4, memory=16384, timeout=1800,
              single_use_containers=True, include_source=False,
              volumes={'/evidence': evidence})
def measure(protocol, payload, invocation, run_id):
    from experiments.layout_coordination.clock_control.same_program_worker import run
    import torch
    torch.set_num_threads(4)
    target = Path('/tmp')/run_id/f'invocation_{invocation:03}'
    result = run(protocol, payload, target, invocation)
    blob = c.archive(target)
    dest = Path('/evidence/layout_coordination')/run_id/f'invocation_{invocation:03}'
    c.write_once(dest/'raw_export.zip', blob)
    evidence.commit()
    return blob


@app.local_entrypoint()
def main():
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    p = c.read(ROOT/'preregistration/protocol.json')
    c.verify_snapshot(p, SNAPSHOT)
    payload = (ROOT/'payload.zip').read_bytes()
    assert c.sha(payload) == p['payload_SHA256']
    calls = []
    for invocation in range(p['timing']['workers']):
        folder = ROOT/'timing'/f'invocation_{invocation:03}'
        call = measure.spawn(p, payload, invocation, ROOT.name)
        c.write_once(folder/'dispatch.json', dict(call_id=call.object_id,
            protocol_SHA256=c.sha(c.encode(p)), payload_SHA256=c.sha(payload),
            profile='chenmiaoming', base_image=BASE_IMAGE))
        calls.append((folder, call))
    statuses = []
    for folder, call in calls:
        blob = call.get()
        c.write_once(folder/'raw_export.zip', blob)
        c.extract(blob, folder/'export')
        raw = c.read(folder/'export/result.json')
        c.write_once(folder/'raw.json', raw)
        statuses.append(raw['status'])
        print(folder.name, raw['status'], raw.get('environment',{}).get('gpu_uuid'), flush=True)
    assert statuses == ['OK']*p['timing']['workers'], statuses
