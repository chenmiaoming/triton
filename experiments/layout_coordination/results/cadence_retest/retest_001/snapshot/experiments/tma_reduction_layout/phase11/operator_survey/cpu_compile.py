"""Compile-only gate. No CUDA context, runtime launch, correctness or timing."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import traceback

from .cases import constants
from .prepare import SOURCE, sha, write_once


def load_layouts(ir, bm, bn):
    definitions = {}
    for name, body in re.findall(r'(#\w+)\s*=\s*#ttg\.blocked<\{(.*?)\}>', ir, re.S):
        fields = {}
        for key, values in re.findall(r'(sizePerThread|threadsPerWarp|warpsPerCTA|order)\s*=\s*\[([^]]+)\]', body):
            translated = {'sizePerThread': 'size_per_thread', 'threadsPerWarp': 'threads_per_warp',
                          'warpsPerCTA': 'warps_per_cta', 'order': 'order'}[key]
            fields[translated] = [int(v.strip()) for v in values.split(',')]
        if len(fields) == 4 and len(fields['order']) == 2:
            definitions[name] = fields
    found = []
    for line in ir.splitlines():
        if 'tt.load' not in line:
            continue
        for alias in re.findall(r'tensor<' + str(bm) + 'x' + str(bn) + r'x.*?,\s*(#\w+)>', line):
            if alias in definitions and definitions[alias] not in found:
                found.append(definitions[alias])
    return found


def compile_all(protocol_path, output, cache_dir, policy='pinned'):
    output, cache_dir = Path(output), Path(cache_dir)
    if output.exists():
        raise FileExistsError('Compile attempts are write-once; use a new output directory')
    protocol = json.loads(Path(protocol_path).read_text())
    for name, digest in protocol['source_SHA256'].items():
        if sha((SOURCE / name).read_bytes()) != digest:
            raise ValueError(f'Changed source after preparation: {name}')
    os.environ['TRITON_CACHE_DIR'] = str(cache_dir)
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from . import kernels_triton, kernels_gluon
    if hasattr(gluon, 'GluonASTSource'):
        gluon_source = gluon.GluonASTSource
    else:
        # Installed 3.7.1 moved the class; the formal pinned image exports it.
        from triton.experimental.gluon._runtime import GluonASTSource
        gluon_source = GluonASTSource

    native_digest = sha(Path(native.__file__).read_bytes())
    if policy == 'pinned' and native_digest != protocol['formal_compiler']['native_SHA256']:
        raise ValueError('Native SHA does not match the frozen Phase 11 compiler')
    output.mkdir(parents=True)
    before_cache = list(cache_dir.rglob('*.cubin'))
    ccache = shutil.which('ccache')
    stats_before = subprocess.check_output([ccache, '--print-stats'], text=True) if ccache else None
    result = {'status': 'FAILED', 'GPU_allocated': False, 'GPU_correctness_verified': False,
              'timing_collected': False, 'compiler_policy': policy,
              'protocol_SHA256': sha(Path(protocol_path).read_bytes()), 'target': ['cuda', 90, 32],
              'triton_version': triton.__version__, 'triton_path': triton.__file__,
              'native_path': native.__file__, 'native_SHA256': native_digest,
              'native_rebuilds': 0, 'records': [], 'input_layout_anchors': {},
              'cache_directory': str(cache_dir), 'cache_CUBINs_before': len(before_cache)}
    from triton.backends.nvidia.compiler import get_ptxas
    assembler = Path(get_ptxas(90).path)
    result['ptxas_path'] = str(assembler)
    result['ptxas_SHA256'] = sha(assembler.read_bytes())
    result['ptxas_version'] = subprocess.check_output([str(assembler), '--version'], text=True)
    result['python_source_SHA256'] = {
        name: sha((Path(triton.__file__).parent / name).read_bytes())
        for name in ('compiler/compiler.py', 'backends/nvidia/compiler.py',
                     'experimental/gluon/_runtime.py', 'experimental/gluon/language/_layouts.py')
        if (Path(triton.__file__).parent / name).exists()
    }
    write_once(output / 'protocol.json', protocol)
    for name in protocol['source_SHA256']:
        write_once(output / 'source' / name, (SOURCE / name).read_bytes())
    keys = {}

    def one(case, variant, layout=None):
        tag = case['id'] + ':' + variant
        folder = output / 'artifacts' / case['id'] / variant.replace(':', '_')
        record = {'tag': tag, 'case': case['id'], 'variant': variant, 'status': 'FAILED',
                  'layout': layout, 'scope': 'semantic_fixture', 'production_adapter_verified': False}
        try:
            con = constants(case)
            sig = {**case['signature'], **{k: 'constexpr' for k in con}}
            if layout is None:
                fn = getattr(kernels_triton, case['kernel'])
                source = ASTSource(fn, sig, con)
            else:
                fn = getattr(kernels_gluon, case['kernel'])
                con['L'] = gl.BlockedLayout(**layout)
                sig['L'] = 'constexpr'
                source = gluon_source(fn, sig, con)
            compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32),
                                      options={'num_warps': case['W']})
            artifact_hashes = {}
            for extension in ('ttir', 'glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                if extension not in compiled.asm:
                    continue
                blob = compiled.asm[extension]
                if isinstance(blob, str):
                    blob = blob.encode()
                write_once(folder / ('kernel.' + extension), blob)
                artifact_hashes['kernel.' + extension] = sha(blob)
            ir = compiled.asm['ttgir']
            record.update(status='OK', function_name=compiled.name, compile_hash=compiled.hash,
                          cubin_SHA256=sha(compiled.asm['cubin']),
                          dynamic_smem_bytes=compiled.metadata.shared,
                          global_scratch_size=compiled.metadata.global_scratch_size,
                          artifact_SHA256=artifact_hashes,
                          artifact_directory=str(folder.relative_to(output)),
                          load_layouts=load_layouts(ir, case['BM'], case['BN']),
                          layout_conversion_sites=ir.count('ttg.convert_layout'),
                          dynamic_counts_available=False)
            if compiled.hash in keys:
                record['identical_compile_alias_of'] = keys[compiled.hash]
            else:
                keys[compiled.hash] = tag
            if layout is None:
                result['input_layout_anchors'][case['id']] = record['load_layouts']
            cuobjdump = shutil.which('cuobjdump')
            if cuobjdump:
                for extension, flag in (('resource.txt', '-res-usage'), ('sass', '-sass')):
                    blob = subprocess.check_output([cuobjdump, flag, str(folder / 'kernel.cubin')], stderr=subprocess.STDOUT)
                    write_once(folder / ('kernel.' + extension), blob)
                    artifact_hashes['kernel.' + extension] = sha(blob)
        except Exception:
            record['status'] = 'FAILED'
            record['error'] = traceback.format_exc()
            write_once(folder / 'error.txt', record['error'].encode())
        write_once(folder / 'record.json', record)
        result['records'].append(record)
        print('CPU', tag, record['status'], flush=True)

    for case in protocol['cases']:
        one(case, 'canonical_triton')
        anchors = result['input_layout_anchors'].get(case['id'], [])
        for i, layout in enumerate(anchors):
            one(case, f'gluon_input_anchor_{i}', layout)
        for variant in case['layouts']:
            one(case, 'gluon:' + variant['name'], variant['layout'])
    result['ccache_before'] = stats_before
    result['ccache_after'] = subprocess.check_output([ccache, '--print-stats'], text=True) if ccache else None
    result['native_SHA256_after'] = sha(Path(native.__file__).read_bytes())
    result['cache_CUBINs_after'] = len(list(cache_dir.rglob('*.cubin')))
    result['new_compile_keys'] = len(keys)
    result['all_canonical_inputs_parsed'] = all(result['input_layout_anchors'].get(c['id']) for c in protocol['cases'])
    result['whole_graph_bridge_verified'] = False
    if all(r['status'] == 'OK' for r in result['records']) and result['all_canonical_inputs_parsed']:
        result['status'] = 'OK'
    if stats_before != result['ccache_after'] or native_digest != result['native_SHA256_after']:
        result['status'] = 'FAILED'
        result['invariant_error'] = 'Unexpected native/ccache mutation'
    write_once(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--compiler-policy', choices=['pinned', 'development'], default='pinned')
    args = parser.parse_args()
    result = compile_all(args.protocol, args.output, args.cache_dir, args.compiler_policy)
    print(result['status'], len(result['records']), 'CPU programs; GPU work not performed', flush=True)
    if result['status'] != 'OK':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
