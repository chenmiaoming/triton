"""Offline preregistration, raw-byte, source and timing evidence closure."""
import argparse
import copy
import json
import zipfile

from experiments.tma_reduction_layout.phase10.validate_final import committed_originals
from experiments.tma_reduction_layout.phase10.stage_c import common as c, preregister, audit_artifacts, analyze, scope


def preflight():
    protocol, files, seen = preregister.render()
    root = c.OUT / 'preregistration'
    assert (root / 'protocol.json').read_bytes() == c.encode(protocol)
    assert (root / 'prior_manifest.json').read_bytes() == c.encode(files)
    assert (root / 'historical_structural_ids.json').read_bytes() == c.encode(seen)
    for name, digest in c.read(root / 'raw_manifest.json').items():
        assert c.sha((c.ROOT / name).read_bytes()) == digest, name
    with zipfile.ZipFile(root / 'worker_source_snapshot.zip') as z:
        for name, digest in protocol['worker_source_SHA256'].items():
            relative = 'experiments/tma_reduction_layout/phase10/stage_c/' + name
            assert c.sha(z.read(relative)) == digest
        assert c.sha(
            z.read('test_source.py')
        ) == protocol['compilers']['patch']['source']['python/test/unit/cuda/test_tensor_descriptor_cuda.py']
    proof = c.read(c.OUT / 'preflight/prior_validator.json')
    assert proof['commit'] == c.BASELINE and proof['return_code'] == 0
    assert proof['log_SHA256'] == c.sha((c.OUT / 'preflight/prior_validator.log').read_bytes())
    assert 'Phase10 final validator PASS' in (c.OUT / 'preflight/prior_validator.log').read_text()
    return {
        'prior_files_protected': len(files), 'held_out_pairs': 48, 'anchor_pairs': 6, 'control_pairs': 8,
        'native_builds_needed': 0, 'protocol_SHA256': c.sha(c.encode(protocol)), 'patch_SHA256':
        protocol['patch_SHA256']
    }


def raw_manifest():
    derived = {
        c.OUT / name
        for name in ('artifact_gate.json', 'results.json', 'summary.md', 'scope.json', 'scope.md', 'raw_manifest.json',
                     'validation.json', 'validation_precommit.log', 'completion.md')
    }
    return {
        str(p.relative_to(c.ROOT)): c.sha(p.read_bytes())
        for p in sorted(c.OUT.rglob('*'))
        if p.is_file() and p not in derived
    }


def probes():
    target = 'sm90'
    gate = c.read(c.OUT / 'artifact_gate.json')[target]
    root = c.OUT / 'timing' / target
    raw = c.read(root / 'raw_invocation_1.json')
    metadata = c.read(root / 'metadata.json')
    results = []
    for name in ('launch_SHA_flip', 'actual_parameter_flip', 'order_swap', 'grid_change'):
        changed = copy.deepcopy(raw)
        if name == 'launch_SHA_flip':
            changed['visits'][0]['launch_SHA256'][0] = '0' * 64
        elif name == 'actual_parameter_flip':
            changed['visits'][0]['actual_parameters'][-1] = 1
        elif name == 'order_swap':
            changed['visits'][0], changed['visits'][1] = changed['visits'][1], changed['visits'][0]
        else:
            changed['visits'][0]['grid'] += 1
        try:
            analyze.validate_raw(changed, target, 1, metadata, gate['eligible'])
        except AssertionError:
            results.append({'probe': name, 'status': 'REJECTED'})
        else:
            raise AssertionError('Protocol tamper was accepted: ' + name)
    return results


def run():
    assert __debug__, 'Do not run validator with Python -O'
    base = preflight()
    gate = audit_artifacts.run()
    assert (c.OUT / 'artifact_gate.json').read_bytes() == c.encode(gate)
    results, text = analyze.run()
    assert (c.OUT / 'results.json').read_bytes() == c.encode(results)
    assert (c.OUT / 'summary.md').read_text() == text
    scoped = scope.derive()
    assert (c.OUT / 'scope.json').read_bytes() == c.encode(scoped)
    assert (c.OUT / 'scope.md').read_text() == scope.summary(scoped)
    return {
        'status': 'PASS', 'preflight': base, 'compiler_attempts': 372, 'samples': results['total_samples'], 'targets':
        results['targets'], 'integrity_probes': probes(), 'no_patch_adjustment': True, 'stop_after_stage_c': True,
        'no_PR_created_or_updated': True, 'classified_regressions_all_grids': len(scoped['classified_regressions']),
        'unresolved_case_grid_effects': len(scoped['unresolved']), 'unresolved_bands_allowing_over_3_percent_loss':
        len(scoped['unresolved_bands_allowing_over_3_percent_loss']), 'universal_non_regression_established': False
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--committed', action='store_true')
    args = parser.parse_args()
    assert sum((args.preflight, args.freeze, args.committed)) <= 1
    if args.preflight:
        print(json.dumps({'status': 'PASS', 'GPU_data_required': False, **preflight()}, indent=2))
        return
    result = run()
    if args.freeze:
        c.write_once(c.OUT / 'raw_manifest.json', raw_manifest())
        c.write_once(c.OUT / 'validation.json', result)
    else:
        assert c.read(c.OUT / 'raw_manifest.json') == raw_manifest()
        assert c.read(c.OUT / 'validation.json') == result
    if args.committed:
        result['committed_originals'] = committed_originals(c.OUT / 'raw_manifest.json')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
