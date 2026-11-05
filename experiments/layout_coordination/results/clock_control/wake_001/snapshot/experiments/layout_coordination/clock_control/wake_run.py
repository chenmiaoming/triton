"""Modal dispatcher for the frozen adaptive wake-up diagnostic."""
from pathlib import Path
import os
import modal
from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c

ROOT=Path(os.environ.get('CLOCK_WAKE_ROOT','/tmp/wake-unprepared'))
SNAPSHOT=ROOT/'snapshot'
app=modal.App('layout-coordination-same-program-wake')
image=(modal.Image.from_id(c.BASE_IMAGE).env({'PYTHONPATH':'/opt/triton-src','OMP_NUM_THREADS':'4'})
    .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                   ignore=['**/__pycache__/**','**/*.pyc']))
evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',cpu=4,memory=16384,timeout=1800,
    single_use_containers=True,include_source=False,volumes={'/evidence':evidence})
def measure(p,payload,invocation,run_id):
    import random
    import time
    import torch
    from experiments.layout_coordination.clock_control.wake_worker import run
    torch.set_num_threads(4)
    root=Path('/tmp')/run_id/f'invocation_{invocation:03}'
    run(p,payload,root,invocation)
    # The historical harness guarded files on a Modal Volume; the first new
    # diagnostic used /tmp. Measure both read/hash paths separately, after GPU
    # teardown, without putting this host-only probe into target timing.
    local=root/'program_files'/p['programs'][0]['artifact_directory']/'kernel.cubin'
    blob=local.read_bytes()
    destination=Path('/evidence/layout_coordination')/run_id/f'invocation_{invocation:03}'
    volume_path=destination/'guard_probe/kernel.cubin'
    c.write_once(volume_path,blob)
    observations=[]
    rng=random.Random(p['timing']['seed']+invocation)
    for repeat in range(128):
        order=['local_tmp','modal_volume']
        rng.shuffle(order)
        for label in order:
            path=local if label=='local_tmp' else volume_path
            before=time.monotonic_ns()
            assert c.sha(path.read_bytes())==c.sha(blob)
            after=time.monotonic_ns()
            observations.append(dict(repeat=repeat,path_kind=label,duration_ns=after-before))
    c.write_once(root/'io_probe.json',dict(status='OK',CUBIN_SHA256=c.sha(blob),
        samples=observations,phase='host-only after GPU teardown',
        scope='Read plus hash of file and retained bytes, matching guard work. This does not reconstruct historical cold/remote filesystem latency.'))
    blob=c.archive(root)
    c.write_once(destination/'raw_export.zip',blob)
    evidence.commit()
    return blob


@app.local_entrypoint()
def main():
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(ROOT/'preregistration/protocol.json')
    c.verify_snapshot(p,SNAPSHOT)
    payload=(ROOT/'payload.zip').read_bytes()
    assert c.sha(payload)==p['payload_SHA256']
    calls=[]
    for i in range(p['timing']['workers']):
        call=measure.spawn(p,payload,i,ROOT.name)
        folder=ROOT/'timing'/f'invocation_{i:03}'
        c.write_once(folder/'dispatch.json',dict(call_id=call.object_id,profile='chenmiaoming',
            protocol_SHA256=c.sha(c.encode(p)),payload_SHA256=c.sha(payload)))
        calls.append((folder,call))
    statuses=[]
    for folder,call in calls:
        blob=call.get()
        c.write_once(folder/'raw_export.zip',blob)
        c.extract(blob,folder/'export')
        raw=c.read(folder/'export/result.json')
        c.write_once(folder/'raw.json',raw)
        statuses.append(raw['status'])
        print(folder.name,raw['status'],flush=True)
    assert statuses==['OK']*p['timing']['workers'],statuses
