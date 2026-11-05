"""Eligibility-first Modal execution of the preregistered cadence replays."""
from pathlib import Path
import os
import modal
from experiments.tma_reduction_layout.phase11.operator_gpu import common as c

ROOT=Path(os.environ.get('CADENCE_ROOT','/tmp/cadence-unprepared'))
SNAPSHOT=ROOT/'snapshot'
app=modal.App('layout-coordination-cadence-retest')
image=(modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH':'/opt/triton-src','OMP_NUM_THREADS':'4'})
    .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
        ignore=['**/__pycache__/**','**/*.pyc']))
evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=14400,
    single_use_containers=True,include_source=False,volumes={'/evidence':evidence})
def execute(protocol,payload,suite,action,attempt,invocation,run_id):
    import importlib
    import traceback
    import torch
    torch.set_num_threads(4)
    root=Path('/tmp')/run_id/suite/action/f'attempt_{attempt:03}'/f'invocation_{invocation:03}'
    try:
        assert c.sha(payload)==protocol['suites'][suite]['payload_SHA256']
        module=importlib.import_module('experiments.layout_coordination.cadence_retest.'+suite)
        module.run(protocol,payload,root,invocation)
    except Exception:
        if not (root/'result.json').exists():
            c.write_once(root/'result.json',dict(status='FAILED',suite=suite,invocation=invocation,
                error=traceback.format_exc(),protocol_SHA256=c.sha(c.encode(protocol))))
    files={str(path.relative_to(root)):c.sha(path.read_bytes()) for path in sorted(root.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts}
    c.write_once(root/'export_manifest.json',dict(status='OK',files=files,
        result_status=c.read(root/'result.json')['status'],
        scope='Byte integrity manifest; result_status separately determines GPU success.'))
    # Payloads already have immutable byte provenance; retain all shards and
    # raw traces including failed attempts, without editing old experiments.
    archive=c.archive(root)
    destination=Path('/evidence/layout_coordination/cadence_retest')/run_id/suite/action/f'attempt_{attempt:03}'/f'invocation_{invocation:03}'
    c.write_once(destination/'raw_export.zip',archive)
    evidence.commit()
    return archive


@app.local_entrypoint()
def main(suite:str='nonreduction',action:str='check',attempt:int=1):
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(ROOT/'preregistration/protocol.json')
    assert suite in p['suites'] and action in ('check','timing')
    for name,digest in p['source_SHA256'].items():
        assert c.sha((SNAPSHOT/name).read_bytes())==digest,name
    payload=(ROOT/p['suites'][suite]['payload_path']).read_bytes()
    assert c.sha(payload)==p['suites'][suite]['payload_SHA256']
    if action=='timing':
        gate_path=ROOT/suite/'check/attempt_001/validation.json'
        gate=c.read(gate_path)
        assert gate['status']=='OK' and gate['protocol_SHA256']==c.sha(c.encode(p))
    invocations=[0] if action=='check' else list(range(1,p['timing']['workers']+1))
    target=ROOT/suite/action/f'attempt_{attempt:03}'
    assert not target.exists(),('Refusing to overwrite an attempt',target)
    calls=[]
    for invocation in invocations:
        call=execute.spawn(p,payload,suite,action,attempt,invocation,ROOT.name)
        folder=target/f'invocation_{invocation:03}'
        c.write_once(folder/'dispatch.json',dict(call_id=call.object_id,profile='chenmiaoming',
            suite=suite,action=action,attempt=attempt,invocation=invocation,
            protocol_SHA256=c.sha(c.encode(p)),payload_SHA256=c.sha(payload),
            eligibility_gate_SHA256=c.sha(gate_path.read_bytes()) if action=='timing' else None))
        calls.append((folder,call))
    statuses=[]
    for folder,call in calls:
        archive=call.get()
        c.write_once(folder/'raw_export.zip',archive)
        c.extract(archive,folder/'export')
        raw=c.read(folder/'export/result.json')
        c.write_once(folder/'raw.json',raw)
        statuses.append(raw['status'])
        print(suite,action,folder.name,raw['status'],flush=True)
    assert statuses==['OK']*len(invocations),statuses
