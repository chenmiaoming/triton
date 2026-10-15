"""A bounded A10 diagnostic, separate from the unapproved Phase 11 C plan.

collect imports the installed Triton; timing/profile load only archived CUBINs.
All original evidence is written once, including failed collector attempts.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from experiments.tma_reduction_layout.phase8.launch import Driver

SOURCE = Path(__file__).resolve().parent
GPU_UUID = 'GPU-69464998-e901-74ba-63ff-ae7022f8aed9'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    with path.open('xb') as f:
        f.write(data)


def read(path):
    return json.loads(Path(path).read_text())


def make_layout(bn, warps, vec, within=False):
    if within:
        tn, wn = bn // (vec * warps), warps
    else:
        tn = min(32, bn // vec)
        wn = min(warps, bn // (vec * tn))
    return {'size_per_thread': [1, 1, vec], 'threads_per_warp': [1, 32 // tn, tn],
            'warps_per_cta': [1, warps // wn, wn], 'order': [2, 1, 0]}


def prepare(root):
    cases = []
    for bm, bn, w in ((32, 128, 4), (32, 64, 8)):
        base = {'BM': bm, 'BN': bn, 'num_warps': w, 'grid': (8192 // bm) * (8192 // bn)}
        cases.append({**base, 'tag': f'M{bm}_N{bn}_w{w}:canonical', 'path': 'direct', 'variant': 'canonical'})
        for path in ('direct', 'cp_async'):
            for vec, within in [(1, False), (2, False), (4, False), (8, False), (8, True)]:
                variant = f'vec{vec}' + ('_within' if within else '')
                cases.append({**base, 'tag': f'M{bm}_N{bn}_w{w}:{path}:{variant}', 'path': path,
                              'variant': variant, 'layout': make_layout(bn, w, vec, within),
                              'copy_layout': make_layout(bn, w, 8),
                              'shared_layout': {'swizzle_byte_width': 128, 'element_bitwidth': 16,
                                                'rank': 3, 'transposed': False}})
    sources = {}
    for p in sorted(SOURCE.glob('*.py')):
        data = p.read_bytes()
        write(root / 'source' / p.name, data)
        sources[p.name] = digest(data)
    dependencies = {}
    for name in ('phase8/launch.py', 'phase4/exact_cuda.py'):
        data = (SOURCE.parent / name).read_bytes()
        write(root / 'source' / name, data)
        dependencies[name] = digest(data)
    protocol = {'status': 'FROZEN_BEFORE_COMPILATION_AND_TIMING', 'GPU_UUID': GPU_UUID,
                'physical_device_index': 0, 'CC': [8, 6], 'input_shape': [1, 8192, 8192],
                'output_shape': [256, 8192], 'dtype': 'bfloat16', 'output': 'float32', 'axis': 1,
                'role': 'Non-TMA path diagnostic; not Phase 11 C1 or cross-architecture model calibration',
                'compiler': 'Installed unmodified Triton 3.7.1, hexcute environment; not pinned Phase 11 native',
                'cases': cases, 'source_SHA256': sources, 'dependency_SHA256': dependencies,
                'correctness_patterns': ['quarter', 'random_seed42_finite'],
                'timing': {'invocations': 3, 'rounds': 6, 'samples_per_visit': 5, 'warmups': 3,
                           'single_launch_events': True, 'graphs': False, 'batch_divide': False,
                           'outcome_based_repeats': False, 'unit': 'three temporal repeats on one physical A10',
                           'interpretation': 'descriptive diagnostic; no unique-winner or broad non-regression claim'},
                'native_builds': 0, 'PR_authorized': False}
    assert len(cases) == 22
    write(root / 'protocol.json', protocol)
    print('Prepared', len(cases), 'logical artifacts; no GPU execution', flush=True)


def inputs(torch, pattern):
    if pattern == 'quarter':
        return ((torch.arange(8192 * 8192, device='cuda', dtype=torch.int32) % 97 - 48)
                .to(torch.bfloat16).reshape(1, 8192, 8192) / 4)
    torch.manual_seed(42)
    return torch.randn((1, 8192, 8192), device='cuda', dtype=torch.bfloat16)


def checked(torch, out, ref):
    return {'pass': bool(torch.equal(out, ref)), 'max_abs_error': float((out-ref).abs().max().item()),
            'elements': out.numel()}


def check_sources(protocol):
    for name, expected in protocol['source_SHA256'].items():
        assert digest((SOURCE / name).read_bytes()) == expected, name
    for name, expected in protocol['dependency_SHA256'].items():
        assert digest((SOURCE.parent / name).read_bytes()) == expected, name


def environment(torch, driver):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 6)
    assert torch.cuda.get_device_name(0) == 'NVIDIA A10'
    result = driver.environment()
    assert result['gpu_uuid'] == GPU_UUID, result
    return {**result, 'name': torch.cuda.get_device_name(0), 'CC': [8, 6],
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'physical_device_index': 0, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}


def collect(root, attempt):
    protocol = read(root / 'protocol.json')
    check_sources(protocol)
    raw = {'status': 'FAILED', 'attempt': attempt, 'artifacts': {}, 'errors': {}, 'correctness': []}
    target = root / 'collect' / str(attempt)
    driver = None
    try:
        import torch
        import triton
        from triton.experimental.gluon import language as gl
        from experiments.tma_reduction_layout.local_a10 import kernels
        assert triton.__version__ == '3.7.1'
        torch.cuda.init()
        driver = Driver()
        raw['environment'] = environment(torch, driver)
        native = Path(triton.__file__).parent / '_C/libtriton.so'
        raw['compiler'] = {'version': triton.__version__, 'path': triton.__file__,
                           'native_path': str(native), 'native_SHA256': digest(native.read_bytes()),
                           'cache_directory': os.environ['TRITON_CACHE_DIR'], 'native_builds': 0}
        assembler = Path(triton.__file__).parent / 'backends/nvidia/bin/ptxas'
        raw['compiler']['ptxas_SHA256'] = digest(assembler.read_bytes())
        raw['compiler']['ptxas_version'] = subprocess.check_output([str(assembler), '--version'], text=True)
        raw['compiler']['cache_CUBINs_before'] = len(list(Path(os.environ['TRITON_CACHE_DIR']).rglob('*.cubin')))
        x = inputs(torch, 'quarter')
        out = torch.empty((256, 8192), device='cuda', dtype=torch.float32)
        compiled = {}
        for case in protocol['cases']:
            tag = case['tag']
            try:
                bm, bn, w = case['BM'], case['BN'], case['num_warps']
                if case['variant'] == 'canonical':
                    function = kernels.canonical
                    consts = (bm, bn)
                else:
                    function = kernels.explicit
                    consts = (bm, bn, w, case['path'] == 'cp_async', gl.BlockedLayout(**case['layout']),
                              gl.BlockedLayout(**case['copy_layout']), gl.NVMMASharedLayout(**case['shared_layout']))
                kernel = function.warmup(x, out, *consts, grid=(case['grid'],), num_warps=w)
                kernel._init_handles()
                folder = target / 'artifacts' / tag.replace(':', '/')
                for ext in list(kernel.asm):
                    value = kernel.asm[ext]
                    if isinstance(value, (str, bytes)):
                        name = 'triton_sass' if ext == 'sass' else ext
                        write(folder / ('kernel.' + name), value if isinstance(value, bytes) else value.encode())
                write(folder / 'kernel.sass', subprocess.check_output(
                    ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(folder / 'kernel.cubin')]))
                write(folder / 'resources.txt', subprocess.check_output(
                    ['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(folder / 'kernel.cubin')]))
                meta = {'function_name': kernel.metadata.name, 'dynamic_smem_bytes': kernel.metadata.shared,
                        'num_regs': kernel.n_regs, 'num_spills': kernel.n_spills,
                        'global_scratch_size': getattr(kernel.metadata, 'global_scratch_size', 0),
                        'CUBIN_SHA256': digest(kernel.asm['cubin']), 'case': case,
                        'archive_path': str(folder.relative_to(root))}
                assert meta['global_scratch_size'] == 0
                write(folder / 'metadata.json', meta)
                raw['artifacts'][tag] = meta
                compiled[tag] = kernel
            except Exception:
                raw['errors'][tag] = traceback.format_exc()
                print('artifact FAIL', tag, flush=True)
        for pattern in protocol['correctness_patterns']:
            x = inputs(torch, pattern)
            ref = x.reshape(256, 32, 8192).float().amax(dim=1)
            for case in protocol['cases']:
                tag = case['tag']
                if tag not in compiled:
                    continue
                out.fill_(float('nan'))
                compiled[tag][(case['grid'], 1, 1)](x, out)
                torch.cuda.synchronize()
                check = checked(torch, out, ref)
                raw['correctness'].append({'tag': tag, 'pattern': pattern, **check})
                assert check['pass'], (tag, pattern, check)
        assert not raw['errors'] and len(raw['artifacts']) == 22 and len(raw['correctness']) == 44
        raw['compiler']['cache_CUBINs_after'] = len(list(Path(os.environ['TRITON_CACHE_DIR']).rglob('*.cubin')))
        assert digest(native.read_bytes()) == raw['compiler']['native_SHA256']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        if driver:
            raw['CUDA_return_codes'] = sorted(driver.codes)
        write(target / 'raw.json', raw)
    print('Collector', attempt, raw['status'], len(raw['artifacts']), 'artifacts', flush=True)
    return raw['status'] == 'OK'


class Loaded:
    def __init__(self, driver, root, meta):
        self.driver, self.meta = driver, meta
        self.path = root / meta['archive_path'] / 'kernel.cubin'
        data = self.path.read_bytes()
        assert digest(data) == meta['CUBIN_SHA256'] and data.startswith(b'\x7fELF')
        self.data = C.create_string_buffer(data)
        self.module, self.function = C.c_void_p(), C.c_void_p()
        driver.call('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p], C.byref(self.module), self.data)
        driver.call('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                    C.byref(self.function), self.module, meta['function_name'].encode())
        ptx = self.path.with_suffix('.ptx').read_text()
        entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
        assert entry
        self.abi = [line.strip() for line in entry[1].splitlines() if '.param' in line]
        assert len(self.abi) in (2, 3, 4) and all('.u64' in line for line in self.abi), self.abi
        self.launches = self.guards = 0

    def parameters(self, x, out):
        values = [C.c_uint64(x.data_ptr()), C.c_uint64(out.data_ptr())]
        values += [C.c_uint64(0) for _ in range(len(self.abi) - 2)]
        return values, (C.c_void_p * len(values))(*[C.addressof(value) for value in values])

    def guard(self):
        assert digest(self.path.read_bytes()) == self.meta['CUBIN_SHA256'] == digest(self.data.raw[:-1])
        self.guards += 1

    def launch(self, params, stream):
        case = self.meta['case']
        self.driver.call('cuLaunchKernel', [C.c_void_p] + [C.c_uint] * 7 +
                         [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p],
                         self.function, case['grid'], 1, 1, case['num_warps'] * 32, 1, 1,
                         self.meta['dynamic_smem_bytes'], stream, params, None)
        self.launches += 1

    def binding(self):
        return {'CUBIN_SHA256': self.meta['CUBIN_SHA256'], 'loaded_SHA256': digest(self.data.raw[:-1]),
                'function_name': self.meta['function_name'], 'ABI': self.abi,
                'launches': self.launches, 'sha_guards': self.guards}


def frozen_run(root, attempt, invocation, profile_tag):
    protocol = read(root / 'protocol.json')
    check_sources(protocol)
    collected = read(root / 'collect' / str(attempt) / 'raw.json')
    assert collected['status'] == 'OK'
    raw = {'status': 'FAILED', 'invocation': invocation, 'profile_tag': profile_tag, 'visits': [],
           'protocol_SHA256': digest((root / 'protocol.json').read_bytes()),
           'Triton_imported': False, 'compile_calls': 0}
    folder = root / ('profile_workers' if profile_tag else 'timing') / str(invocation)
    driver, loaded = None, {}
    try:
        import torch
        torch.cuda.init()
        driver = Driver()
        raw['environment'] = environment(torch, driver)
        x = inputs(torch, 'quarter')
        out = torch.empty((256, 8192), device='cuda', dtype=torch.float32)
        ref = x.reshape(256, 32, 8192).float().amax(dim=1)
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        selected = [profile_tag] if profile_tag else sorted(collected['artifacts'])
        params = {}
        for tag in selected:
            k = Loaded(driver, root, collected['artifacts'][tag])
            loaded[tag] = k
            params[tag] = k.parameters(x, out)
            out.fill_(float('nan'))
            for _ in range(protocol['timing']['warmups']):
                k.guard()
                k.launch(params[tag][1], stream)
            driver.call('cuStreamSynchronize', [C.c_void_p], stream)
            assert checked(torch, out, ref)['pass'], tag
        if profile_tag:
            k = loaded[profile_tag]
            k.guard()
            driver.call('cuProfilerStart', [])
            k.launch(params[profile_tag][1], stream)
            driver.call('cuStreamSynchronize', [C.c_void_p], stream)
            driver.call('cuProfilerStop', [])
        else:
            first, last = driver.event(), driver.event()
            for round_index in range(protocol['timing']['rounds']):
                shift = (round_index + int(invocation)) % len(selected)
                order = selected[shift:] + selected[:shift]
                if (round_index + int(invocation)) % 2:
                    order = order[::-1]
                for tag in order:
                    k = loaded[tag]
                    visit = {'tag': tag, 'round': round_index, 'samples_us': []}
                    raw['visits'].append(visit)
                    for _ in range(protocol['timing']['samples_per_visit']):
                        k.guard()
                        driver.record(first, stream)
                        k.launch(params[tag][1], stream)
                        driver.record(last, stream)
                        driver.synchronize_event(last)
                        value = driver.elapsed_us(first, last)
                        assert math.isfinite(value) and value > 0
                        visit['samples_us'].append(value)
            driver.call('cuEventDestroy_v2', [C.c_void_p], first)
            driver.call('cuEventDestroy_v2', [C.c_void_p], last)
        raw['Triton_imported'] = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
        assert not raw['Triton_imported']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        raw['bindings'] = {tag: k.binding() for tag, k in loaded.items()}
        if driver:
            raw['CUDA_return_codes'] = sorted(driver.codes)
            raw['CUDA_call_counts'] = dict(driver.counts)
            for k in loaded.values():
                driver.call('cuModuleUnload', [C.c_void_p], k.module)
        write(folder / 'raw.json', raw)
    print('Frozen run', invocation, raw['status'], len(raw['visits']), 'visits', flush=True)
    return raw['status'] == 'OK'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'collect', 'timing', 'profile'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt', default='1')
    parser.add_argument('--invocation', default='0')
    parser.add_argument('--tag')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'prepare':
        prepare(root)
        return
    if args.mode == 'collect':
        success = collect(root, args.attempt)
    else:
        success = frozen_run(root, args.attempt, args.invocation, args.tag if args.mode == 'profile' else None)
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
