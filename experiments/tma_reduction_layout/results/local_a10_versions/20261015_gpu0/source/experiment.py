"""Default-layout release comparison; collection and timing are separate processes."""
import argparse
import ast
import ctypes as C
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.tma_reduction_layout.local_a10.experiment import (
    Driver, GPU_UUID, Loaded, checked, digest, environment, inputs, read, write,
)

SOURCE = Path(__file__).resolve().parent
DEPENDENCIES = ('local_a10/experiment.py', 'local_a10/kernels.py',
                'phase8/launch.py', 'phase4/exact_cuda.py')


def canonical_ast(path):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'canonical')
    return ast.dump(node, include_attributes=False)


def prepare(root, provenance):
    assert canonical_ast(SOURCE / 'kernels.py') == canonical_ast(SOURCE.parent / 'local_a10/kernels.py')
    sources = {}
    for path in sorted(SOURCE.glob('*.py')):
        data = path.read_bytes()
        write(root / 'source' / path.name, data)
        sources[path.name] = digest(data)
    dependencies = {}
    for name in DEPENDENCIES:
        data = (SOURCE.parent / name).read_bytes()
        write(root / 'source' / name, data)
        dependencies[name] = digest(data)
    toolchains = read(provenance)
    write(root / 'toolchains.json', provenance.read_bytes())
    cases = [{'version': version, 'BM': 32, 'BN': bn, 'num_warps': w,
              'grid': (8192 // 32) * (8192 // bn), 'tag': f'{version}:M32_N{bn}_w{w}'}
             for version in ('3.5.1', '3.6.0', '3.7.1') for bn, w in ((128, 4), (64, 8))]
    protocol = {
        'status': 'FROZEN_BEFORE_COMPILATION_AND_TIMING',
        'question': 'Does ordinary tl.load change its compiler-selected layout or regress across the original release boundary?',
        'GPU_UUID': GPU_UUID, 'physical_device_index': 0, 'CC': [8, 6],
        'input_shape': [1, 8192, 8192], 'output_shape': [256, 8192],
        'input_dtype': 'bfloat16', 'output_dtype': 'float32', 'reduction_axis': 1,
        'cases': cases, 'source_SHA256': sources, 'dependency_SHA256': dependencies,
        'kernel_AST_identical_to_previous_plain_Triton_anchor': True,
        'toolchains_SHA256': digest((root / 'toolchains.json').read_bytes()),
        'common_ptxas': toolchains['common_ptxas'],
        'compile_options': 'num_warps per case; other compiler options use release defaults',
        'correctness_patterns': ['quarter', 'random_seed42_finite'],
        'timing': {'invocations': 7, 'rounds': 12, 'samples_per_visit': 5, 'warmups': 6,
                   'single_launch_events': True, 'graphs': False, 'batch_divide': False,
                   'order': 'rotate by round+invocation; reverse on odd parity',
                   'unit': 'seven temporal process repeats on one physical A10',
                   'primary_contrast': '3.6.0 / 3.5.1', 'secondary_contrast': '3.7.1 / 3.5.1',
                   'estimator': 'geometric mean of paired process median time ratios',
                   'interval': '95% Student t on seven paired process log ratios, df=6',
                   'practical_slowdown_threshold': 1.03,
                   'decision': 'lower bound >1.03: detected; upper bound <1.03: excludes 3% in this workload; otherwise inconclusive',
                   'outcome_based_repeats': False},
        'native_builds': 0, 'manual_layouts': False, 'TMA': False,
        'Phase11_C_started': False, 'PR_authorized': False,
    }
    write(root / 'protocol.json', protocol)
    print('Prepared six default-layout artifacts; no GPU execution', flush=True)


def check_sources(root):
    protocol = read(root / 'protocol.json')
    for name, expected in protocol['source_SHA256'].items():
        assert digest((SOURCE / name).read_bytes()) == expected, name
    for name, expected in protocol['dependency_SHA256'].items():
        assert digest((SOURCE.parent / name).read_bytes()) == expected, name
    assert digest((root / 'toolchains.json').read_bytes()) == protocol['toolchains_SHA256']
    return protocol


def collect(root, version, attempt):
    protocol = check_sources(root)
    provenance = read(root / 'toolchains.json')
    target = root / 'collect' / version / str(attempt)
    raw = {'status': 'FAILED', 'version': version, 'artifacts': {}, 'correctness': [], 'errors': {},
           'protocol_SHA256': digest((root / 'protocol.json').read_bytes())}
    driver = None
    try:
        import torch
        import triton
        from experiments.tma_reduction_layout.local_a10_versions.kernels import canonical
        assert triton.__version__ == version
        torch.cuda.init()
        driver = Driver()
        raw['environment'] = environment(torch, driver)
        native = Path(triton.__file__).parent / '_C/libtriton.so'
        ptxas = Path(os.environ['TRITON_PTXAS_PATH'])
        assert Path(triton.knobs.nvidia.ptxas.path).resolve() == ptxas.resolve()
        assert str(ptxas) == protocol['common_ptxas']['path']
        assert digest(ptxas.read_bytes()) == protocol['common_ptxas']['SHA256']
        assert digest(native.read_bytes()) == provenance['versions'][version]['native_SHA256']
        assert Path(triton.__file__).resolve() == Path(provenance['versions'][version]['module_path']).resolve()
        cache = Path(os.environ['TRITON_CACHE_DIR'])
        raw['compiler'] = {'version': version, 'module_path': triton.__file__, 'native_path': str(native),
                           'native_SHA256': digest(native.read_bytes()), 'ptxas_path': str(ptxas),
                           'ptxas_SHA256': digest(ptxas.read_bytes()),
                           'compiler_resolved_ptxas': triton.knobs.nvidia.ptxas.path,
                           'ptxas_version': subprocess.check_output([str(ptxas), '--version'], text=True),
                           'cache_directory': str(cache), 'native_builds': 0,
                           'cache_CUBINs_before': len(list(cache.rglob('*.cubin')))}
        x = inputs(torch, 'quarter')
        out = torch.empty((256, 8192), device='cuda', dtype=torch.float32)
        compiled = {}
        cases = [case for case in protocol['cases'] if case['version'] == version]
        for case in cases:
            tag = case['tag']
            try:
                kernel = canonical.warmup(x, out, case['BM'], case['BN'],
                                          grid=(case['grid'],), num_warps=case['num_warps'])
                kernel._init_handles()
                folder = target / 'artifacts' / tag.replace(':', '/')
                for ext, value in kernel.asm.items():
                    if isinstance(value, (str, bytes)):
                        ext = 'triton_sass' if ext == 'sass' else ext
                        write(folder / ('kernel.' + ext), value if isinstance(value, bytes) else value.encode())
                for suffix, option in (('sass', '--dump-sass'), ('resources.txt', '--dump-resource-usage')):
                    result = subprocess.run(['/usr/local/cuda/bin/cuobjdump', option, str(folder / 'kernel.cubin')],
                                            capture_output=True, check=True)
                    write(folder / ('kernel.' + suffix), result.stdout)
                    write(folder / (suffix + '.stderr'), result.stderr)
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
            for case in cases:
                tag = case['tag']
                if tag not in compiled:
                    continue
                out.fill_(float('nan'))
                compiled[tag][(case['grid'], 1, 1)](x, out)
                torch.cuda.synchronize()
                check = checked(torch, out, ref)
                raw['correctness'].append({'tag': tag, 'pattern': pattern, **check})
                assert check['pass'], (tag, pattern, check)
        assert not raw['errors'] and len(raw['artifacts']) == 2 and len(raw['correctness']) == 4
        raw['compiler']['cache_CUBINs_after'] = len(list(cache.rglob('*.cubin')))
        assert digest(native.read_bytes()) == raw['compiler']['native_SHA256']
        assert digest(ptxas.read_bytes()) == raw['compiler']['ptxas_SHA256']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        if driver:
            raw['CUDA_return_codes'] = sorted(driver.codes)
        write(target / 'raw.json', raw)
    print('Collector', version, attempt, raw['status'], flush=True)
    return raw['status'] == 'OK'


def timing(root, invocation):
    protocol = check_sources(root)
    artifacts = {}
    compilers = {}
    for version in ('3.5.1', '3.6.0', '3.7.1'):
        collected = read(root / 'collect' / version / '1/raw.json')
        assert collected['status'] == 'OK'
        artifacts.update(collected['artifacts'])
        compilers[version] = collected['compiler']
    assert len(artifacts) == 6
    raw = {'status': 'FAILED', 'invocation': invocation, 'visits': [], 'correctness': [],
           'protocol_SHA256': digest((root / 'protocol.json').read_bytes()),
           'Triton_imported': False, 'compile_calls': 0}
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
        selected = sorted(artifacts)
        params = {}
        for tag in selected:
            k = Loaded(driver, root, artifacts[tag])
            loaded[tag] = k
            params[tag] = k.parameters(x, out)
            out.fill_(float('nan'))
            for _ in range(protocol['timing']['warmups']):
                k.guard()
                k.launch(params[tag][1], stream)
            driver.call('cuStreamSynchronize', [C.c_void_p], stream)
            check = checked(torch, out, ref)
            raw['correctness'].append({'tag': tag, **check})
            assert check['pass'], tag
        first, last = driver.event(), driver.event()
        for round_index in range(protocol['timing']['rounds']):
            shift = (round_index + invocation) % len(selected)
            order = selected[shift:] + selected[:shift]
            if (round_index + invocation) % 2:
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
        raw['compiler_native_SHA256'] = {v: digest(Path(c['native_path']).read_bytes()) for v, c in compilers.items()}
        assert all(raw['compiler_native_SHA256'][v] == c['native_SHA256'] for v, c in compilers.items())
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        raw['bindings'] = {tag: k.binding() for tag, k in loaded.items()}
        if driver:
            for k in loaded.values():
                driver.call('cuModuleUnload', [C.c_void_p], k.module)
            raw['CUDA_return_codes'] = sorted(driver.codes)
            raw['CUDA_call_counts'] = dict(driver.counts)
        write(root / 'timing' / str(invocation) / 'raw.json', raw)
    print('Timing', invocation, raw['status'], len(raw['visits']), 'visits', flush=True)
    return raw['status'] == 'OK'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'collect', 'timing'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--version', choices=['3.5.1', '3.6.0', '3.7.1'])
    parser.add_argument('--attempt', default='1')
    parser.add_argument('--invocation', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'prepare':
        prepare(root, args.provenance)
        return
    success = collect(root, args.version, args.attempt) if args.mode == 'collect' else timing(root, args.invocation)
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
