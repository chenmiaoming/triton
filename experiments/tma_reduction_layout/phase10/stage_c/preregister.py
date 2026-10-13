"""Freeze actual-patch validation before any new compiler/GPU dispatch."""
import argparse
import hashlib
import re
import subprocess
import tarfile

from experiments.tma_reduction_layout.phase10.stage_c import common as c

SHAPES = [(8, 256, 4), (16, 256, 4), (16, 512, 8), (32, 512, 4), (32, 1024, 8), (64, 512, 8), (128, 256, 2),
          (256, 128, 2)]


def prior_files():
    files = subprocess.check_output(['git', 'ls-tree', '-rz', c.BASELINE, '--', 'experiments/tma_reduction_layout'],
                                    cwd=c.ROOT)
    result = {}
    for item in files.split(b'\0'):
        if not item:
            continue
        header, name = item.split(b'\t', 1)
        name = name.decode()
        data = (c.ROOT / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert blob == header.decode().split()[2], name
        result[name] = c.sha(data)
    for name in ('query_metrics.json', 'result.json'):
        path = c.ROOT / 'experiments/tma_reduction_layout/results/diagnostics/ncu_bank_capability' / name
        result[str(path.relative_to(c.ROOT))] = c.sha(path.read_bytes())
    return result


def history(files):
    identities = set()
    for name in files:
        identities.update(tuple(map(int, v)) for v in re.findall(r'M(\d+)_N(\d+)_w(\d+)', name))
        if name.endswith('.json'):
            # Structural identities only; numeric performance fields do not
            # enter selection. This also includes the complete early sweep.
            identities.update(
                tuple(map(int, v)) for v in re.findall(r'M(\d+)_N(\d+)_w(\d+)', (c.ROOT / name).read_text()))
    return sorted(identities)


def cases(seen):
    result = []

    def add(m, n, w, dtype, op, mode, cohort):
        for path in ('host', 'device'):
            result.append({
                'case_id': f'M{m}_N{n}_w{w}:{dtype}:{op}:{mode}:{path}', 'M': m, 'N': n, 'num_warps': w, 'dtype': dtype,
                'op': op, 'mode': mode, 'cohort': cohort, 'path': path
            })

    for m, n, w in SHAPES:
        assert (m, n, w) not in seen
        for dtype, op in [('bfloat16', 'max'), ('float16', 'min'), ('float32', 'sum')]:
            add(m, n, w, dtype, op, 'reduce', 'held_out')
    for m, n, w in [(32, 64, 8), (32, 128, 4), (32, 128, 8)]:
        add(m, n, w, 'bfloat16', 'max', 'reduce', 'anchor')
    for mode in ('copy', 'multi_use', 'innermost'):
        add(16, 256, 4, 'bfloat16', 'max', mode, 'guard_control')
    add(1024, 64, 4, 'bfloat16', 'max', 'reduce', 'cost_fallback_probe')
    return sorted(result, key=lambda row: row['case_id'])


def render():
    files = prior_files()
    seen = history(files)
    root = c.ROOT / 'experiments/tma_reduction_layout'
    final = c.read(root / 'results/phase10/stage_b/final_python/result.json')
    archive = root / 'results/phase10/stage_a/upstream_source.tar'
    with tarfile.open(archive) as tar:
        original = {name: c.sha(tar.extractfile(name).read()) for name in final['source']}
    workers = {
        name: c.sha((c.SOURCE / name).read_bytes())
        for name in ('common.py', 'kernels.py', 'data.py', 'launch.py', 'run_artifacts.py', 'run_timing.py')
    }
    p = {
        'stage':
        'Phase10 C', 'baseline_experiment_HEAD':
        c.BASELINE, 'upstream_HEAD':
        '20d9d98fead78c2f65e1b3ba5a7881d86a836dad', 'patch_SHA256':
        c.sha((root / 'phase10/compiler_prototype.patch').read_bytes()), 'variants': ['baseline', 'patch'],
        'global_shape': [1, 8192, 8192], 'cases':
        cases(set(seen)), 'held_out_definition':
        'tile/warp triple absent from every prior JSON/path structural identity; dtype/op coverage is the declared 3-cell subset, not a full factorial',
        'compilers': {
            'baseline': {
                'image': 'im-qzTUZumWu2JXjWic8K82cZ', 'native_SHA256':
                'fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb', 'source': original
            }, 'patch':
            {'image': 'im-KaEq6823Uj1qypGzBbwFr9', 'native_SHA256': final['native_SHA256'], 'source': final['source']}
        }, 'worker_source_SHA256':
        workers, 'targets': {
            'sm90': {'gpu': 'H100!:1', 'name': 'H100', 'cc':
                     [9, 0]}, 'sm100': {'gpu': 'B200:1', 'name': 'B200', 'cc': [10, 0]}, 'sm120':
            {'gpu': 'RTX-PRO-6000:1', 'name': 'RTX PRO 6000', 'cc': [12, 0]}
        }, 'artifact_gate':
        'all 124 attempts per architecture retained; each case pair enters timing iff both compile, full-output exact oracle, positive checked occupancy and complete frozen ABI pass; no outcome exclusion or replacement',
        'native_build':
        'no C++ edits or make; reuse exact already-built images/native SHA and persistent ccache', 'assembler':
        'require same architecture-selected ptxas binary SHA for baseline and patch', 'input':
        'contiguous exact quarter-multiples: (arange %97 -48)/4; independent eager PyTorch tile oracle; no floating tolerance',
        'timing': {
            'invocations':
            3, 'rounds':
            10, 'samples_per_visit':
            5, 'warmups':
            3, 'grid_fractions': [1, 4, 16], 'primary_fraction':
            1, 'order':
            'sort case:fraction pairs, cyclic offsets 17*(invocation-1)+13*round; alternate baseline/patch order by (invocation+round)%2',
            'primitive':
            'one archived CUBIN per driver-event sample; SHA guard before every launch; no Triton/compiler/subprocess in worker',
            'repair':
            'infrastructure failures only, retain all attempts, at most 3 repairs per whole invocation; no outcome-driven reruns'
        }, 'decision': {
            'effect_percent':
            '100*(median_baseline-median_patch)/median_baseline per invocation', 'band':
            'mean +/-4.302652729911275*sampleSD/sqrt(3); descriptive 3-process band, no familywise significance',
            'practical_percent':
            3, 'BENEFIT':
            'lower bound >3', 'REGRESSION':
            'upper bound <-3', 'WITHIN_PRACTICAL_BAND':
            'both bounds in [-3,3]', 'UNRESOLVED':
            'all remaining valid pairs', 'UNAVAILABLE':
            'artifact pair or target unavailable; never imputed zero', 'scope':
            'report every primary and secondary effect by target, cohort, case and resource stratum; no pooled ranking or universal safety claim'
        }, 'quota':
        'stop and ask user to switch profile; no automatic profile fallback', 'stop':
        'after Stage C report/validator/one commit/push; no patch adjustment from outcomes, Stage D, PR creation/update or outbound reviewer message'
    }
    return p, files, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    p, files, seen = render()
    values = {'protocol.json': p, 'prior_manifest.json': files, 'historical_structural_ids.json': seen}
    for name, value in values.items():
        path = c.OUT / 'preregistration' / name
        if args.validate:
            assert path.read_bytes() == c.encode(value), name
        else:
            c.write_once(path, value)
    print(f'Preregistration PASS: 48 held-out +6 anchor +8 control pairs/target; {len(files)} prior files protected')


if __name__ == '__main__':
    main()
