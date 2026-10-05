"""Read-only evidence and exact-source validation; no CUDA or compilation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import zipfile

from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
from experiments.tma_reduction_layout.phase4.artifact_gate import ttgir_contract
from experiments.tma_reduction_layout.phase10 import common as c, audit_stage_a


def git_tree(commit, prefix):
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', commit, '--', prefix], cwd=c.ROOT)
    return {name.decode(): header.decode().split()[2]
            for item in tree.split(b'\0') if item
            for header, name in [item.split(b'\t', 1)]}


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def validate_manifest(path):
    manifest = c.read(path)
    for name, digest in manifest.items():
        assert c.sha((c.ROOT / name).read_bytes()) == digest, name
    return len(manifest)


def committed_originals(manifest_path):
    relative = str(manifest_path.relative_to(c.ROOT))
    commits = subprocess.check_output(
        ['git', 'log', '--reverse', '--format=%H', '--', relative], cwd=c.ROOT, text=True).splitlines()
    assert commits, 'Original-byte manifest has no commit: ' + relative
    first = commits[0]
    original = subprocess.check_output(['git', 'show', first + ':' + relative], cwd=c.ROOT)
    assert original == manifest_path.read_bytes()
    tree = git_tree(first, 'experiments/tma_reduction_layout/results/phase10')
    for name in c.read(manifest_path):
        assert git_blob((c.ROOT / name).read_bytes()) == tree[name], name
    return {'first_commit': first, 'files': len(c.read(manifest_path))}


def invalid_run():
    root = c.OUT / 'invalid_invocations/jit_cache_alias'
    rejected = []
    for target in ('sm90', 'sm100', 'sm120'):
        export = root / 'artifacts' / target / 'research/export'
        for row in c.read(export / 'attempts.json'):
            if row['variant'] != 'research_vector4':
                continue
            directory = export / row['tag'].replace(':', '/')
            observed = ttgir_contract((directory / 'kernel.ttgir').read_text(),
                                      [1, row['M'], row['N']], row['num_warps'])
            assert observed['blocked']['sizePerThread'][-1] == 8
            rejected.append({'tag': row['tag'], 'target': target,
                             'requested_vector': 4, 'observed_vector': 8, 'status': 'REJECTED'})
    assert len(rejected) == 18
    assert c.read(c.OUT / 'stage_a_validation/checks.json')['invalid_vector4_conditions_rejected'] == rejected
    return rejected


def archive_equal(path):
    with zipfile.ZipFile(path / 'raw_export.zip') as z:
        for name in z.namelist():
            assert z.read(name) == (path / 'export' / name).read_bytes(), name


def cache_observations(path):
    observations = []
    with zipfile.ZipFile(path) as z:
        irs = sorted(name for name in z.namelist() if name.endswith('/kernel.ttgir'))
        assert len(irs) == 54
        assert len([name for name in z.namelist() if name.endswith('/kernel.cubin')]) == 54
        for name in irs:
            directory = name.rsplit('/', 1)[0]
            ir = z.read(name).decode()
            info = analyze_ttgir(ir)
            local = [line for line in ir.splitlines() if 'ttg.local_load ' in line]
            assert len(local) == 1
            n, dtype = re.search(r'tensor<1x32x(64|128)x(bf16|f16|f32),', local[0]).groups()
            source = z.read(directory + '/kernel.ttir').decode()
            path_kind = 'device' if 'tt.make_tensor_descriptor ' in source else 'host'
            combines = re.findall(r'arith\.(maximumf|minimumf|maxnumf|minnumf|addf) ', ir)
            assert len(combines) == 1
            op = {'maximumf': 'max', 'maxnumf': 'max', 'minimumf': 'min',
                  'minnumf': 'min', 'addf': 'sum'}[combines[0]]
            metadata = json.loads(z.read(directory + '/kernel.json'))
            w = info['module_attributes']['num_warps']
            assert w == metadata['num_warps'] and metadata['target']['arch'] == 90
            layout = info['blocked_encodings'][info['local_load_dest_layout']]
            # The complete GPU pipeline applies the prototype to these cases.
            assert layout['sizePerThread'] == [1, 1, 1 if int(n) == 64 else 2]
            artifact_hashes = {ext: c.sha(z.read(directory + '/kernel.' + ext))
                               for ext in ('source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'json')}
            observations.append({'cache_directory': directory, 'path': path_kind,
                                 'N': int(n), 'num_warps': w, 'dtype': dtype, 'op': op,
                                 'combine': combines[0],
                                 'layout': layout, 'artifacts_SHA256': artifact_hashes})
    actual = Counter((r['path'], r['N'], r['num_warps'], r['dtype'], r['op']) for r in observations)
    expected = Counter((p, n, w, d, op) for p in ('host', 'device')
                       for n, w in ((64, 8), (128, 4), (128, 8))
                       for d in ('bf16', 'f16', 'f32') for op in ('max', 'min', 'sum'))
    assert actual == expected
    return observations


def prototype():
    stage = c.OUT / 'stage_b'
    accepted = c.read(stage / 'accepted.json')
    cpu_root = c.ROOT / accepted['attempt']
    final = c.read(stage / 'final_python/result.json')
    cpu = accepted['result']
    assert cpu == c.read(cpu_root / 'return.json') == c.read(cpu_root / 'export/result.json')
    assert cpu['status'] == final['status'] == 'OK'
    assert not cpu['GPU_timing'] and not final['GPU_timing']
    assert cpu['native_SHA256'] == final['native_SHA256']
    assert cpu['image_id'] == final['base_image']
    assert final['compiled_kernel_count'] == 54 and final['return_code'] == 0
    assert final['native_build_reused']
    assert c.read(stage / 'final_python/dispatch.json')['source'] == final['source']
    assert c.read(stage / 'final_python/dispatch.json')['call_id'] == final['call_id']
    assert final['gpu_environment']['gpu_uuid']
    assert all(call['return_code'] == 0 for call in final['gpu_environment']['checked_cuda_calls'])
    assert '54 passed' in (stage / 'final_python/pytest.log').read_text()
    assert c.sha((stage / 'final_python/test_source.py').read_bytes()) == final['source'][
        'python/test/unit/cuda/test_tensor_descriptor_cuda.py']
    assert [r['tag'] for r in cpu['checks']] == ['targeted', 'coalesce', 'full_lit']
    assert all(r['return_code'] == 0 for r in cpu['checks'])
    lit = (cpu_root / 'export/full_lit.log').read_text()
    assert re.search(r'Total Discovered Tests:\s+305', lit)
    assert re.search(r'Passed\s+:\s+303', lit)
    assert re.search(r'Unsupported:\s+2', lit)
    assert c.read(cpu_root / 'export/commands.json') == [['make'], ['make', 'triton-opt']]
    for name, digest in accepted['source'].items():
        assert c.sha((cpu_root / 'export/source' / name).read_bytes()) == digest
        if name.startswith(('lib/', 'test/')):
            assert digest == final['source'][name]
    for attempt in sorted((stage / 'attempts').iterdir()):
        archive_equal(attempt)
        assert not c.read(attempt / 'return.json')['GPU_timing']
    patch = c.ROOT / 'experiments/tma_reduction_layout/phase10/compiler_prototype.patch'
    for name in ('build_prototype.py', 'run_prototype.py', 'check_final_python.py'):
        assert (stage / 'final_check_scripts' / name).read_bytes() == (
            c.ROOT / 'experiments/tma_reduction_layout/phase10' / name).read_bytes()
    # Apply the stored patch to files from the immutable clean source archive.
    # No dependence on the temporary development worktree or a compiler build.
    with tempfile.TemporaryDirectory(prefix='tma-phase10-apply-') as directory:
        root = Path(directory)
        with tarfile.open(c.OUT / 'stage_a/upstream_source.tar') as tar:
            for name in final['source']:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(tar.extractfile(name).read())
        subprocess.run(['git', 'apply', '--check', '--whitespace=error', str(patch)],
                       cwd=root, check=True, capture_output=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True, capture_output=True)
        assert {name: c.sha((root / name).read_bytes()) for name in final['source']} == final['source']
    assert c.sha((stage / 'design.patch').read_bytes()) == c.read(stage / 'design.json')['patch_SHA256']
    observations = cache_observations(stage / 'final_python/raw_compile_cache.zip')
    stats = {}
    for when in ('before', 'after'):
        stats[when] = dict(line.split() for line in (cpu_root / f'export/ccache_{when}.txt').read_text().splitlines())
    delta = {key: int(stats['after'][key]) - int(stats['before'][key])
             for key in ('direct_cache_hit', 'preprocessed_cache_hit', 'cache_miss')}
    return {'patch_SHA256': c.sha(patch.read_bytes()), 'source': final['source'],
            'native_SHA256': final['native_SHA256'], 'CPU_attempt': accepted['attempt'],
            'lit_passed': 303, 'lit_unsupported': 2, 'GPU_correctness_passed': 54,
            'GPU_compile_artifacts': observations, 'ccache_final_build_delta': delta,
            'final_build_CPP_compile_count': (cpu_root / 'export/build_0.log').read_text().count('Building CXX object'),
            'prototype_timing': False, 'held_out_validation': False}


def stage_b_manifest():
    stage = c.OUT / 'stage_b'
    derived = {stage / name for name in ('results.json', 'summary.md', 'raw_manifest.json')}
    return {str(p.relative_to(c.ROOT)): c.sha(p.read_bytes()) for p in sorted(stage.rglob('*'))
            if p.is_file() and p not in derived}


def run(committed=False):
    assert __debug__, 'Validator must not run under Python -O'
    result, summary, manifest = audit_stage_a.run()
    stage = c.OUT / 'stage_a'
    assert (stage / 'analysis.json').read_bytes() == c.encode(result)
    assert (stage / 'summary.md').read_text() == summary
    assert c.read(stage / 'raw_manifest.json') == manifest
    protocol = c.read(stage / 'protocol.json')
    prior_tree = git_tree(protocol['baseline_experiment_HEAD'], 'experiments/tma_reduction_layout')
    prior = c.read(stage / 'prior_manifest.json')
    # Two pre-existing ignored NCU diagnostic JSONs are protected by the
    # pre-dispatch SHA manifest in addition to every baseline-tracked file.
    assert set(prior_tree) <= set(prior)
    for name, blob in prior_tree.items():
        assert git_blob((c.ROOT / name).read_bytes()) == blob, name
    invalid_count = validate_manifest(c.OUT / 'invalid_invocations/raw_manifest.json')
    invalid_run()
    closure = {'stage_a': committed_originals(stage / 'raw_manifest.json'),
               'invalid_run': committed_originals(c.OUT / 'invalid_invocations/raw_manifest.json')}
    b = prototype()
    assert not (c.OUT / 'stage_c').exists()
    if committed:
        assert c.read(c.OUT / 'stage_b/raw_manifest.json') == stage_b_manifest()
        closure['stage_b'] = committed_originals(c.OUT / 'stage_b/raw_manifest.json')
    return {'status': 'PASS', 'read_only': True, 'prior_files_protected': len(prior),
            'prior_tracked_files': len(prior_tree),
            'prior_existing_ignored_files': sorted(set(prior) - set(prior_tree)),
            'stage_a_samples': result['samples'], 'stage_a_compiler_attempts': 54,
            'invalid_files_preserved': invalid_count, 'invalid_vector4_conditions_rejected': 18,
            'committed_originals': closure, 'prototype': b, 'stop_after_stage_b': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--committed', action='store_true')
    args = parser.parse_args()
    assert not (args.generate and args.committed)
    result = run(args.committed)
    if args.generate:
        c.write_once(c.OUT / 'stage_b/results.json', result['prototype'])
        c.write_once(c.OUT / 'stage_b/raw_manifest.json', stage_b_manifest())
        c.write_once(c.OUT / 'final_validation/suite.json', result)
    else:
        assert c.read(c.OUT / 'stage_b/results.json') == result['prototype']
        assert c.read(c.OUT / 'stage_b/raw_manifest.json') == stage_b_manifest()
        expected_suite = dict(result)
        expected_suite['committed_originals'] = dict(result['committed_originals'])
        expected_suite['committed_originals'].pop('stage_b', None)
        assert c.read(c.OUT / 'final_validation/suite.json') == expected_suite
    print(json.dumps({k: v for k, v in result.items() if k != 'prototype'}, indent=2, sort_keys=True))
    print('Phase10 final validator PASS: 8,100 baseline samples; 303 lit + 54 GPU correctness; no prototype timing')


if __name__ == '__main__':
    main()
