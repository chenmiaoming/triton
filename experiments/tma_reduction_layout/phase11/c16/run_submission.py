"""One additional paired worker, no new CUBIN or compiler invocation."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11.c16 import common as c

app=modal.App('triton-phase11-c16-submission-control')
snapshot=Path('/tmp/tma-phase11-c16-submission-snapshot')
image=(modal.Image.from_id('im-qzTUZumWu2JXjWic8K82cZ')
    .env({'PYTHONPATH':'/opt/triton-src'})
    .add_local_dir(snapshot/'experiments','/opt/triton-src/experiments',copy=True,
                   ignore=['**/__pycache__/**','**/*.pyc']))
volume=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',timeout=900,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def execute(protocol,diagnostic,gate,payload):
    from experiments.tma_reduction_layout.phase11.c16.submission import execute as run
    raw=run(protocol,diagnostic,gate,payload,modal.current_function_call_id())
    c.write_once(Path('/evidence/stage_c16/submission')/raw['call_id']/'raw.json',raw)
    volume.commit()
    return raw


@app.local_entrypoint()
def main():
    c.protect_prior()
    folder=c.OUT/'submission/attempt_1'
    assert not folder.exists()
    protocol=c.load_protocol()
    diagnostic=c.read(c.OUT/'preregistration/submission_protocol.json')
    gate=c.read(c.OUT/'artifact_gate.json')
    payload=(c.OUT/'frozen_payload.zip').read_bytes()
    call=execute.spawn(protocol,diagnostic,gate,payload)
    c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),
                 'diagnostic_SHA256':c.sha(c.encode(diagnostic))})
    try:
        raw=call.get()
    except Exception as error:
        c.write_once(folder/'retrieval_failure.json',{'call_id':call.object_id,'error':repr(error)})
        raise
    c.write_once(folder/'raw.json',raw)
    print('C1.6 submission diagnostic',raw['status'],flush=True)
