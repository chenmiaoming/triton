"""CPU compilation of unchanged upstream bodies, before paid correctness/timing."""
from pathlib import Path
import shutil
import subprocess
import traceback
from . import common as c
from .production_cases import configuration


def compile_all(protocol, root):
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from . import vllm_triton, sglang_triton
    c.verify_sources(protocol)
    root = Path(root)
    result = {'status': 'FAILED', 'GPU_allocated': False, 'records': [],
              'native_SHA256': c.sha(Path(native.__file__).read_bytes()),
              'protocol_SHA256': c.sha(c.encode(protocol)), 'native_rebuilds': 0}
    assert result['native_SHA256'] == c.NATIVE
    result['ccache_before'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    for case in protocol['geometries']:
        for stage2 in ([False, True] if case['family'] == 'C02' else [False]):
            fn = vllm_triton.reduce_segments if case['family'] == 'C06' else (
                sglang_triton._fwd_kernel_stage2 if stage2 else sglang_triton._fwd_kernel_stage1)
            constants, pointers, values, types = configuration(case, stage2)
            sig = {name: ('constexpr' if name in constants else pointers.get(name, types.get(name))) for name in fn.arg_names}
            assert all(sig.values()), sig
            attrs = {(i,): [['tt.divisibility', 16]] for i, name in enumerate(fn.arg_names)
                     if name in pointers or (isinstance(values.get(name), int) and values[name] % 16 == 0)}
            tag = case['id'] + (':stage2' if stage2 else ':upstream')
            folder = root / tag.replace(':', '/')
            row = {'tag': tag, 'geometry': case['id'], 'stage2': stage2, 'status': 'FAILED'}
            try:
                source = ASTSource(fn, sig, constants, attrs)
                compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32),
                                          options={'num_warps': 4, 'num_stages': 2 if stage2 else case['num_stages']})
                hashes = {}
                for ext in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
                    blob = compiled.asm[ext]
                    blob = blob if isinstance(blob, bytes) else blob.encode()
                    c.write_once(folder / ('kernel.' + ext), blob)
                    hashes['kernel.' + ext] = c.sha(blob)
                for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
                    blob = subprocess.check_output(['cuobjdump', flag, str(folder / 'kernel.cubin')])
                    c.write_once(folder / ('kernel.' + ext), blob)
                    hashes['kernel.' + ext] = c.sha(blob)
                row.update(status='OK', artifact_SHA256=hashes, cubin_SHA256=c.sha(compiled.asm['cubin']),
                           function_name=compiled.name, compile_key=compiled.hash,
                           signature=sig, constexprs={key: str(value) for key, value in constants.items()},
                           attrs={str(key): value for key, value in attrs.items()},
                           dynamic_smem_bytes=compiled.metadata.shared,
                           global_scratch_size=compiled.metadata.global_scratch_size)
            except Exception:
                row['error'] = traceback.format_exc()
                c.write_once(folder / 'error.txt', row['error'].encode())
            c.write_once(folder / 'record.json', row)
            result['records'].append(row)
            print('CPU production', tag, row['status'], flush=True)
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    result['native_SHA256_after'] = c.sha(Path(native.__file__).read_bytes())
    assert result['ccache_before'] == result['ccache_after'] and result['native_SHA256_after'] == c.NATIVE
    result['status'] = 'OK' if all(row['status'] == 'OK' for row in result['records']) else 'FAILED'
    c.write_once(root / 'result.json', result)
    return result
