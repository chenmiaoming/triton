"""Compile source-aligned Gluon with the unmodified pinned backend."""
import ast
import inspect
from pathlib import Path
import subprocess
import traceback
from . import common as c
from .layout_parsing import instantiate
from .production_compile import controls, export
from .production_cases import configuration


def function_for_family(family):
    """Return the source-aligned entry point without modifying any backend."""
    if family == 'C06':
        from .g6_vllm import reduce_segments as fn
    elif family == 'C02':
        from .g6_sglang import _fwd_kernel_stage1 as fn
    elif family == 'C01':
        from .g6_layernorm import merge as fn
    else:
        raise ValueError(f'No verified source baseline for family {family!r}')
    return fn


def arguments(item, gl, candidate_layout=None):
    """Prepare the frozen call ABI; candidate layouts require their own gate."""
    case = item['case']; meta = item['record']; plan = item['plan']
    fn = function_for_family(case['family'])
    if case['family'] == 'C01':
        constants = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128,
                         L=instantiate(candidate_layout or plan['input'], gl),
                         LOUT=instantiate(plan['output'], gl), NATIVE_OUT=False)
    else:
        constants = {**configuration(case)[0], **controls(case, plan, candidate_layout or plan['V'], gl)}
    constants = {name: value for name, value in constants.items() if meta['JIT_signature'][name] == 'constexpr'}
    constants.update(meta['specialized_runtime_values'])
    attrs = {ast.literal_eval(key): value for key, value in meta['JIT_attributes'].items()}
    return fn, constants, attrs


def run(protocol, root):
    import triton
    from triton import knobs
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    root = Path(root); c.verify_sources(protocol)
    backend = Path(inspect.getfile(CUDABackend))
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    assert c.sha(backend.read_bytes()) == protocol['backend_SHA256']
    method = CUDABackend.gluon_to_ttgir
    assert Path(inspect.getsourcefile(method)).resolve() == backend.resolve()
    assert knobs.runtime.add_stages_inspection_hook is None
    result = dict(status='FAILED', records=[], GPU_allocated=False, native_rebuilds=0,
                  backend_SHA256=c.sha(backend.read_bytes()), native_SHA256=c.NATIVE,
                  compiler_intervention=False, protocol_SHA256=c.sha(c.encode(protocol)),
                  ccache_before=subprocess.check_output(['ccache', '--print-stats'], text=True))
    for item in protocol['cases']:
        case = item['case']; tag = case['id'] + ':source_aligned'
        row = dict(case=case['id'], status='FAILED', tag=tag)
        try:
            fn, constants, attrs = arguments(item, gl)
            compiled = triton.compile(gluon.GluonASTSource(fn, dict(item['record']['JIT_signature']), constants, attrs),
                                     target=GPUTarget('cuda', 90, 32),
                                     options={'num_warps': case['W'], 'num_stages': case['num_stages']})
            row.update(status='OK', metadata=export(compiled, fn, case, tag, root, constants))
        except Exception:
            row['error'] = traceback.format_exc()
            c.write_once(root/'errors'/(case['id']+'.txt'), row['error'].encode())
        result['records'].append(row)
        print('G6 source-only', case['id'], row['status'], flush=True)
    assert CUDABackend.gluon_to_ttgir is method
    assert knobs.runtime.add_stages_inspection_hook is None
    assert c.sha(backend.read_bytes()) == result['backend_SHA256']
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    assert result['ccache_before'] == result['ccache_after']
    result['status'] = 'OK' if all(row['status'] == 'OK' for row in result['records']) else 'FAILED'
    c.write_once(root/'result.json', result)
    return result
