"""CPU exact-ABI compilation, then independent actual CUDA tensor JIT comparison."""
from pathlib import Path
import ast
import subprocess
import traceback
from . import common as c
from .production_cases import configuration
from .production_compile import controls, export
from .layout_parsing import instantiate, plan


def spec(group, variant, baseline, gl):
    case = group['case']
    constants, pointers, values, types = configuration(case)
    if case['family'] == 'C06':
        values['num_seqs'] = group['binding_batch']
    if variant.get('stage2'):
        return configuration(case, True)
    if not variant.get('upstream'):
        name = variant['pv']
        layout = baseline['V'] if name == 'default' else {'kind': 'blocked', 'fields': next(
            x['layout'] for x in case['layouts'] if x['name'] == name or name in x.get('aliases', []))}
        constants.update(controls(case, baseline, layout, gl))
        if case['family'] == 'C02' and variant['qk'] != 'default':
            layout = next(x['layout'] for x in case['layouts'] if x['name'] == variant['qk'])
            constants['LK'] = instantiate({'kind': 'blocked', 'fields': layout}, gl)
    return constants, pointers, values, types


def function(case, variant):
    from . import sglang_triton, sglang_controlled, vllm_triton, vllm_controlled
    if variant.get('stage2'):
        return sglang_triton._fwd_kernel_stage2
    if case['family'] == 'C02':
        return sglang_triton._fwd_kernel_stage1 if variant.get('upstream') else sglang_controlled._fwd_kernel_stage1
    return vllm_triton.reduce_segments if variant.get('upstream') else vllm_controlled.reduce_segments


def cpu(protocol, old_gate, root):
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    c.verify_sources(protocol)
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    result = {'status': 'FAILED', 'records': [], 'plans': {}, 'GPU_allocated': False,
              'native_rebuilds': 0, 'native_SHA256': c.NATIVE,
              'protocol_SHA256': c.sha(c.encode(protocol)),
              'ccache_before': subprocess.check_output(['ccache', '--print-stats'], text=True)}
    originals = {x['tag']: x for x in old_gate['records']}
    try:
        for group in protocol['groups']:
            case = group['case']; baseline = protocol['baseline_plans'][case['id']]
            variants = group['variants'] + ([{'name': 'stage2', 'stage2': True}] if case['family'] == 'C02' else [])
            for variant in variants:
                fn = function(case, variant)
                constants, pointers, values, types = spec(group, variant, baseline, gl)
                old = originals[case['id'] + (':stage2' if variant.get('stage2') else ':upstream')]
                # Use the previously verified original call site's explicit i64 annotations.
                sig = {name: old['JIT_signature'].get(name, 'constexpr') for name in fn.arg_names}
                specialized = {name: value for name, value in constants.items() if sig[name] == 'constexpr'}
                runtime_values = {**constants, **values}
                for name, value in runtime_values.items():
                    if isinstance(value, int) and value == 1 and name in sig and sig[name] != 'constexpr':
                        sig[name] = 'constexpr'; specialized[name] = value
                attrs = {(i,): [['tt.divisibility', 16]] for i, name in enumerate(fn.arg_names)
                         if name in pointers or (name not in specialized and isinstance(runtime_values.get(name), int)
                                                 and runtime_values[name] % 16 == 0)}
                cls = triton.compiler.ASTSource if variant.get('upstream') or variant.get('stage2') else gluon.GluonASTSource
                code = triton.compile(cls(fn, sig, specialized, attrs), target=GPUTarget('cuda', 90, 32),
                    options={'num_warps': case['W'], 'num_stages': 2 if variant.get('stage2') else case['num_stages']})
                tag = group['id'] + ':' + ('actual:' if protocol['stage'] == 'g4b' else '') + variant['name']
                meta = export(code, fn, case, tag, root, constants, variant.get('stage2', False))
                meta.update(group=group['id'], variant=variant, binding_batch=group['binding_batch'],
                            binding_length=group['binding_length'], provenance='CPU_actual_ABI',
                            stage2_tag=group['id'] + (':actual:' if protocol['stage'] == 'g4b' else ':') + 'stage2')
                result['records'].append(meta)
                if variant.get('upstream'):
                    baseline = plan(code.asm['ttgir'], case)
                    result['plans'][group['id']] = baseline
                print('G4 CPU', tag, 'PASS', flush=True)
        result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
        assert result['ccache_before'] == result['ccache_after']
        assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc(); print(result['error'], flush=True)
    c.write_once(Path(root) / 'result.json', result)
    return result


def bind(protocol, compiled):
    import torch
    import triton._C.libtriton as native
    from triton.experimental.gluon import language as gl
    from .production_data import make_data
    c.verify_sources(protocol)
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    torch.cuda.init(); assert torch.cuda.get_device_capability() == (9, 0)
    result = {'status': 'FAILED', 'records': [], 'native_rebuilds': 0, 'native_SHA256': c.NATIVE,
              'protocol_SHA256': c.sha(c.encode(protocol)),
              'ccache_before': subprocess.check_output(['ccache', '--print-stats'], text=True)}
    dtype = {'*bf16': torch.bfloat16, '*fp32': torch.float32, '*i32': torch.int32}
    try:
        for group in protocol['groups']:
            case = group['case']; baseline = compiled['plans'][group['id']]
            data = make_data(case, batch=group['binding_batch'], length=group['binding_length'], with_reference=False)
            arrays = {k: torch.from_numpy(v).to(device='cuda', dtype=dtype[data['types'][k]]) for k, v in data['arrays'].items()}
            arrays.update({k: torch.empty(v, device='cuda', dtype=dtype[data['types'][k]]) for k, v in data['output_shapes'].items()})
            for meta in compiled['records']:
                if meta['group'] != group['id'] or meta['provenance'] != 'CPU_actual_ABI': continue
                variant = meta['variant']; fn = function(case, variant)
                constants, pointers, values, types = spec(group, variant, baseline, gl)
                values.update(data['scalars'])
                aliases = {'Mid_O': 'Att_Out', 'Mid_O_1': 'Att_Lse'}
                kwargs = {n: arrays[aliases.get(n, n)] if aliases.get(n, n) in arrays else
                          constants[n] if n in constants else values[n] for n in fn.arg_names}
                code = fn.warmup(**kwargs, grid=(group['binding_batch'], case['H']) if variant.get('stage2') else data['grid'],
                    num_warps=case['W'], num_stages=2 if variant.get('stage2') else case['num_stages'])
                digest = c.sha(code.asm['cubin'])
                runtime = [name for i, name in enumerate(fn.arg_names) if (i,) not in code.src.constants]
                row = {'tag': meta['tag'], 'CPU_CUBIN': meta['cubin_SHA256'], 'real_JIT_CUBIN': digest,
                       'match': digest == meta['cubin_SHA256'] and runtime == meta['runtime_argument_names'],
                       'runtime_argument_names': runtime, 'JIT_signature': code.src.signature,
                       'JIT_constants': {str(k): str(v) for k, v in code.src.constants.items()},
                       'JIT_attributes': {str(k): v for k, v in code.src.attrs.items()},
                       'batch': data['batch'], 'length': data['length']}
                result['records'].append(row)
                print('G4 actual tensor JIT', meta['tag'], row['match'], flush=True)
            del arrays
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
        assert result['ccache_before'] == result['ccache_after']
        assert all(x['match'] for x in result['records'])
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc(); print(result['error'], flush=True)
    return result
