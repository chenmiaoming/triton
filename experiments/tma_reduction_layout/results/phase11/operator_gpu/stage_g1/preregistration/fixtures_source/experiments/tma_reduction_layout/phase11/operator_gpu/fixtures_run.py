"""Reuse archived preparation CUBINs for an H100 correctness gate."""
from pathlib import Path
import modal
from . import common as c

SNAPSHOT = Path('/tmp/operator-gpu-fixtures-snapshot-v1')
app = modal.App('triton-operator-gpu-fixture-correctness')
image = (modal.Image.from_id(c.BASE_IMAGE).env({'PYTHONPATH': '/opt/triton-src'})
         .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                        ignore=['**/__pycache__/**', '**/*.pyc']))
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, gpu='H100!:1', timeout=1800, single_use_containers=True,
              volumes={'/evidence': evidence}, include_source=False)
def collect(protocol, payload):
    from experiments.tma_reduction_layout.phase11.operator_gpu.fixtures import execute
    c.verify_sources(protocol)
    call_id = modal.current_function_call_id()
    result = execute(protocol, payload, call_id)
    c.write_once(Path('/evidence/operator_gpu/fixtures') / call_id / 'raw.json', result)
    evidence.commit()
    return result


@app.local_entrypoint()
def main(attempt: int = 1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    out = c.OUT / 'stage_g1/fixtures' / f'attempt_{attempt:03}'
    assert not out.exists(), out
    protocol = c.read(c.OUT / 'stage_g1/preregistration/fixtures_protocol.json')
    c.verify_sources(protocol)
    payload = (c.OUT / 'stage_g1/preregistration/fixtures_payload.zip').read_bytes()
    call = collect.spawn(protocol, payload)
    c.write_once(out / 'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming',
                                       'protocol_SHA256': c.sha(c.encode(protocol))})
    result = call.get()
    c.write_once(out / 'raw.json', result)
    print('GPU FIXTURES', result['status'], len(result['checks']), 'checks', flush=True)
    assert result['status'] == 'OK', result.get('error', 'Numerical mismatch; inspect retained checks')
