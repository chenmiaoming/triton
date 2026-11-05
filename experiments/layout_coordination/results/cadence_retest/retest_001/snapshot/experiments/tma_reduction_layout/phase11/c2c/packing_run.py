"""Small same-worker packing supplement; main C2c raw data stay immutable."""
import os
from pathlib import Path
import subprocess
import modal
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2c import run as original

OUT = c.OUT/'packing_supplement'
app = modal.App('triton-phase11-c2c-packing-contrast')


@app.function(image=original.image, cpu=4, timeout=1200,
              volumes={'/cache': original.cache, '/evidence': original.evidence}, include_source=False)
def preflight(protocol, payload):
    import traceback
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from experiments.tma_reduction_layout.phase11.c2c.packing_kernels import fixed_scalar
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c2c/packing_preflight')/modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    c.extract(payload, root)
    result = {'status': 'FAILED', 'GPU_allocated': False, 'metadata': {}, 'new_compiles': 0, 'reused': 0,
              'protocol_SHA256': c.sha(c.encode(protocol))}
    try:
        result['native_SHA256'] = c.sha(Path(native.__file__).read_bytes())
        assert result['native_SHA256'] == protocol['baseline']['native_SHA256']
        result['ccache_before'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
        for case in protocol['cases']:
            folder = root/case['tag'].replace(':', '/')
            if case.get('reuse_source'):
                assert c.sha((folder/'kernel.cubin').read_bytes()) == case['cubin_SHA256']
                meta = case
                result['reused'] += 1
            else:
                constants = {'R': case['R'], 'P': 2, 'C': case['columns'],
                             'L': gl.BlockedLayout(**case['layout'])}
                signature = {'arg': '*bf16', 'out': '*fp32', 'K': 'i32', **{key: 'constexpr' for key in constants}}
                source = gluon.GluonASTSource(fixed_scalar, signature, constants, {(1,): [['tt.divisibility', 16]]})
                compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32), options={'num_warps': 4})
                for ext in ('glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                    value = compiled.asm[ext]
                    c.write_once(folder/('kernel.'+ext), value if isinstance(value, bytes) else value.encode())
                for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
                    value = subprocess.check_output(['cuobjdump', flag, str(folder/'kernel.cubin')], text=True)
                    c.write_once(folder/('kernel.'+ext), value.encode())
                meta = {**case, 'cubin_SHA256': c.sha(compiled.asm['cubin']), 'function_name': compiled.name,
                        'dynamic_smem_bytes': compiled.metadata.shared,
                        'global_scratch_size': compiled.metadata.global_scratch_size,
                        'tensordesc_meta': compiled.metadata.tensordesc_meta}
                result['new_compiles'] += 1
            result['metadata'][case['tag']] = meta
            c.write_once(folder/'c2c_metadata.json', meta)
        result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
        assert result['ccache_before'] == result['ccache_after']
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root/'result.json', result)
    original.cache.commit()
    original.evidence.commit()
    return c.archive(root)


@app.function(image=original.image, gpu='H100!:1', timeout=1800,
              volumes={'/evidence': original.evidence}, single_use_containers=True, include_source=False)
def collect(protocol, metadata, payload):
    return original.collect.local(protocol, metadata, payload)


@app.function(image=original.image, gpu='H100!:1', timeout=3600,
              volumes={'/evidence': original.evidence}, single_use_containers=True, include_source=False)
def timing(invocation, protocol, gate, payload):
    return original.timing.local(invocation, protocol, gate, payload)


@app.function(image=original.tools_image, gpu='H100!:1', timeout=2400,
              volumes={'/evidence': original.evidence}, single_use_containers=True, include_source=False)
def profiles(protocol, gate, payload):
    return original.profiles.local(protocol, gate, payload, 0)


@app.local_entrypoint()
def main(action: str = 'preflight', invocation: int = 1, attempt: int = 1):
    protocol = c.read(OUT/'preregistration/protocol.json')
    c.protect_prior()
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    if action == 'preflight':
        folder, fn = OUT/'cpu_preflight'/f'attempt_{attempt}', preflight
        args = protocol, (OUT/'preregistration/reuse.zip').read_bytes()
    elif action == 'collect':
        folder, fn = OUT/'eligibility'/f'attempt_{attempt}', collect
        args = protocol, c.read(OUT/'compile_gate.json')['metadata'], (OUT/'compiled_payload.zip').read_bytes()
    elif action == 'profiles':
        folder, fn = OUT/'profiles/group_0'/f'attempt_{attempt}', profiles
        args = protocol, c.read(OUT/'artifact_gate.json'), (OUT/'compiled_payload.zip').read_bytes()
    else:
        assert action == 'timing'
        folder, fn = OUT/'timing'/f'invocation_{invocation}', timing
        args = invocation, protocol, c.read(OUT/'artifact_gate.json'), (OUT/'compiled_payload.zip').read_bytes()
    assert not folder.exists(), folder
    call = fn.spawn(*args)
    c.write_once(folder/'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming', 'action': action,
                                        'protocol_SHA256': c.sha(c.encode(protocol))})
    value = call.get()
    if isinstance(value, bytes):
        c.write_once(folder/'raw_export.zip', value)
        c.extract(value, folder/'export')
        status = c.read(folder/'export/result.json')['status']
    else:
        c.write_once(folder/'raw.json', value)
        status = value['status']
    print('C2c packing', action, status, flush=True)
