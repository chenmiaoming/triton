"""CPU feasibility, then real CUDA-call-site specialization and immutable exports."""
from pathlib import Path
import subprocess
import traceback
from . import common as c
from .production_cases import configuration
from .layout_parsing import instantiate, plan


def variants(case, baseline):
    rows = [{'name': 'gluon_anchor', 'layout': baseline['V']}]
    rows += [{'name': row['name'], 'layout': {'kind': 'blocked', 'fields': row['layout']}}
             for row in case['layouts']]
    return rows


def controls(case, baseline, layout, gl):
    result = {'L': instantiate(layout, gl), 'LOUT': instantiate(baseline['output'], gl)}
    if case['family'] == 'C02':
        result.update(LK=instantiate(baseline['K'], gl), LQ=instantiate(baseline['Q'], gl))
    else:
        result['LSTAT'] = instantiate(baseline['statistics'], gl)
    return result


def export(compiled, fn, case, tag, root, constants, is_stage2=False):
    folder = Path(root) / 'artifacts' / tag.replace(':', '_')
    hashes = {}
    for ext in ('ttir', 'glir', 'ttgir', 'llir', 'ptx', 'cubin'):
        if ext not in compiled.asm:
            continue
        blob = compiled.asm[ext]
        blob = blob if isinstance(blob, bytes) else blob.encode()
        c.write_once(folder / ('kernel.' + ext), blob)
        hashes['kernel.' + ext] = c.sha(blob)
    for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
        blob = subprocess.check_output(['cuobjdump', flag, str(folder / 'kernel.cubin')])
        c.write_once(folder / ('kernel.' + ext), blob)
        hashes['kernel.' + ext] = c.sha(blob)
    src = compiled.src
    specialized = {path[0]: value for path, value in src.constants.items() if len(path) == 1}
    runtime_names = [name for i, name in enumerate(fn.arg_names) if i not in specialized]
    runtime_constants = {fn.arg_names[i]: value for i, value in specialized.items()
                         if fn.arg_names[i] not in constants and isinstance(value, (int, float, bool, type(None)))}
    meta = {'tag': tag, 'geometry': case['id'], 'family': case['family'], 'num_warps': case['W'],
            'stage2': is_stage2, 'artifact_directory': str(folder.relative_to(root)),
            'artifact_SHA256': hashes, 'cubin_SHA256': c.sha(compiled.asm['cubin']),
            'function_name': compiled.name, 'compile_key': compiled.hash,
            'dynamic_smem_bytes': compiled.metadata.shared,
            'global_scratch_size': compiled.metadata.global_scratch_size,
            'runtime_argument_names': runtime_names, 'specialized_runtime_values': runtime_constants,
            'JIT_signature': src.signature, 'JIT_constants': {str(key): str(value) for key, value in src.constants.items()},
            'JIT_attributes': {str(key): value for key, value in src.attrs.items()}}
    assert meta['global_scratch_size'] == 0
    c.write_once(folder / 'metadata.json', meta)
    return meta


def run_cpu(protocol, root):
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from . import vllm_controlled, sglang_controlled
    c.verify_sources(protocol)
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    result = {'status': 'FAILED', 'GPU_allocated': False, 'records': [], 'native_rebuilds': 0}
    result['ccache_before'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    for case in protocol['geometries']:
        baseline = protocol['baseline_layout_plans'][case['id']]
        fn = vllm_controlled.reduce_segments if case['family'] == 'C06' else sglang_controlled._fwd_kernel_stage1
        original, pointers, values, types = configuration(case)
        for variant in variants(case, baseline):
            tag = case['id'] + ':' + variant['name']
            row = {'tag': tag, 'status': 'FAILED'}
            try:
                constants = {**original, **controls(case, baseline, variant['layout'], gl)}
                sig = {name: 'constexpr' if name in constants else pointers.get(name, types.get(name)) for name in fn.arg_names}
                attrs = {(i,): [['tt.divisibility', 16]] for i, name in enumerate(fn.arg_names)
                         if name in pointers or (isinstance(values.get(name), int) and values[name] % 16 == 0)}
                source = gluon.GluonASTSource(fn, sig, constants, attrs)
                compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32),
                                          options={'num_warps': case['W'], 'num_stages': case['num_stages']})
                row.update(status='OK', metadata=export(compiled, fn, case, tag, root, constants))
            except Exception:
                row['error'] = traceback.format_exc()
                c.write_once(Path(root) / 'errors' / (tag.replace(':', '_') + '.txt'), row['error'].encode())
            result['records'].append(row)
            print('CPU controlled', tag, row['status'], flush=True)
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    result['native_SHA256'] = c.sha(Path(native.__file__).read_bytes())
    assert result['native_SHA256'] == c.NATIVE and result['ccache_before'] == result['ccache_after']
    result['status'] = 'OK' if all(row['status'] == 'OK' for row in result['records']) else 'FAILED'
    c.write_once(Path(root) / 'result.json', result)
    return result


def run_gpu(protocol, root):
    import numpy as np
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.experimental.gluon import language as gl
    from . import vllm_triton, sglang_triton, vllm_controlled, sglang_controlled
    from .production_data import make_data
    c.verify_sources(protocol)
    result = {'status': 'FAILED', 'formal_timing': False, 'records': [], 'native_rebuilds': 0,
              'native_SHA256': c.sha(Path(native.__file__).read_bytes()), 'plans': {},
              'torch_version': torch.__version__, 'ccache_before': subprocess.check_output(['ccache', '--print-stats'], text=True)}
    assert result['native_SHA256'] == c.NATIVE
    torch.cuda.init()
    assert torch.cuda.get_device_capability() == (9, 0)
    for case in protocol['geometries']:
        data = make_data(case, batch=4, length=129, with_reference=False)
        dtype_map = {'*bf16': torch.bfloat16, '*fp32': torch.float32, '*i32': torch.int32}
        tensors = {name: torch.from_numpy(array).to(device='cuda', dtype=dtype_map[data['types'][name]])
                   for name, array in data['arrays'].items()}
        for name, shape in data['output_shapes'].items():
            tensors[name] = torch.empty(shape, device='cuda', dtype=dtype_map[data['types'][name]])
        original, _, values, _ = configuration(case)
        values.update(data['scalars'])
        fn = vllm_triton.reduce_segments if case['family'] == 'C06' else sglang_triton._fwd_kernel_stage1
        kwargs = {name: (tensors[name] if name in tensors else original[name] if name in original else values[name])
                  for name in fn.arg_names}
        compiled = fn.warmup(**kwargs, grid=data['grid'], num_warps=case['W'], num_stages=case['num_stages'])
        baseline = plan(compiled.asm['ttgir'], case)
        result['plans'][case['id']] = baseline
        result['records'].append(export(compiled, fn, case, case['id'] + ':upstream', root, original))
        fn = vllm_controlled.reduce_segments if case['family'] == 'C06' else sglang_controlled._fwd_kernel_stage1
        for variant in variants(case, baseline):
            constants = {**original, **controls(case, baseline, variant['layout'], gl)}
            current = {**kwargs, **constants}
            compiled = fn.warmup(**current, grid=data['grid'], num_warps=case['W'], num_stages=case['num_stages'])
            tag = case['id'] + ':' + variant['name']
            row = export(compiled, fn, case, tag, root, constants)
            row['intervention_layout'] = variant['layout']
            result['records'].append(row)
            print('GPU real JIT binding', tag, flush=True)
        if case['family'] == 'C02':
            fn = sglang_triton._fwd_kernel_stage2
            constants, _, values, _ = configuration(case, True)
            aliases = {'Mid_O': 'Att_Out', 'Mid_O_1': 'Att_Lse'}
            kwargs2 = {name: tensors[aliases.get(name, name)] if aliases.get(name, name) in tensors else
                       constants[name] if name in constants else values[name] for name in fn.arg_names}
            compiled = fn.warmup(**kwargs2, grid=(4, case['H']), num_warps=4, num_stages=2)
            result['records'].append(export(compiled, fn, case, case['id'] + ':stage2', root, constants, True))
        torch.cuda.synchronize()
        del tensors, kwargs
        torch.cuda.empty_cache()
    result['native_SHA256_after'] = c.sha(Path(native.__file__).read_bytes())
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    assert result['native_SHA256_after'] == c.NATIVE
    result['status'] = 'OK'
    c.write_once(Path(root) / 'result.json', result)
    return result
