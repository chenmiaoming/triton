"""Prepare/analyze the p=2 packing contrast independently of main raw evidence."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import statistics as st
from experiments.tma_reduction_layout.phase11.c2c import common as c


def prepare():
    out = c.OUT/'packing_supplement'
    main = c.protocol()
    gate = c.read(c.OUT/'compile_gate.json')
    cases = []
    target = Path('/tmp/tma-phase11-c2c-snapshot')
    hashes = dict(main['worker_source_SHA256'])
    for name in ('packing_kernels.py', 'packing_run.py'):
        source = c.SOURCE/name
        relative = source.relative_to(c.ROOT)
        (target/relative).parent.mkdir(parents=True, exist_ok=True)
        (target/relative).write_bytes(source.read_bytes())
        hashes[str(relative)] = c.sha(source.read_bytes())
    reuse = out/'preregistration/reuse'
    for r in (4, 8, 16, 32):
        columns = r//2
        prefix = f'local_R{r}_p2_c{columns}'
        for variant in ('native', 'wide', 'fixture', 'fixture_wide'):
            tag = prefix+':'+variant
            meta = {**gate['metadata'][tag], 'reuse_source': gate['artifact_directory']+'/'+tag.replace(':', '/')}
            cases.append(meta)
            origin = c.ROOT/meta['reuse_source']
            destination = reuse/tag.replace(':', '/')
            destination.mkdir(parents=True)
            for path in origin.glob('kernel.*'):
                shutil.copyfile(path, destination/path.name)
        scalar = {**gate['metadata'][prefix+':native'], 'tag': prefix+':scalar', 'variant': 'scalar'}
        for key in ('reuse_source', 'cubin_SHA256', 'function_name', 'dynamic_smem_bytes',
                    'global_scratch_size', 'tensordesc_meta', 'compile_cache_key'):
            scalar.pop(key, None)
        cases.append(scalar)
    protocol = {**main, 'stage': 'C2c_packing_supplement', 'revision': 1,
        'cases': cases, 'worker_source_SHA256': hashes,
        'profiles': {**main['profiles'], 'group_limit': 20},
        'supplement_reason': 'Disentangle native BF16 packing from FP32 widening at a one-stage p=2 tree; preserve main evidence and use same-worker matched contrasts.',
        'main_protocol_SHA256': c.sha(c.encode(main)),
        'native_and_fixture_CUBIN_reused': 16, 'new_scalar_CUBINs': 4,
        'main_measurements_repeated': 'Only unchanged local contrast binaries rerun in the new matched cohort; required for a paired packing intervention.'}
    c.write_once(out/'preregistration/protocol.json', protocol)
    c.write_once(out/'preregistration/reuse.zip', c.archive(reuse))
    c.write_once(out/'preregistration/prior_manifest.json', c.read(c.OUT/'preregistration/prior_manifest.json'))
    c.write_once(out/'preregistration/source_snapshot.zip', c.archive(target))
    print('C2c packing frozen: 4 new scalar programs, 16 exact reused binaries, no D1')


def analyze(action):
    from experiments.tma_reduction_layout.phase11.c2c import analyze as a
    c.OUT = c.OUT/'packing_supplement'
    if action in ('compile', 'artifacts', 'counters', 'services'):
        {'compile': a.compile_gate, 'artifacts': a.artifact_gate,
         'counters': a.counters, 'services': a.services}[action]()
    elif action == 'import':
        from experiments.tma_reduction_layout.phase11.c2c.offline_import import main
        main()
    elif action == 'summary':
        data = c.read(c.OUT/'analysis.json')
        idx = {(row['tag'], row['q_condition'], row['invocation']): row for row in data['services']}
        rows = []
        for row in data['services']:
            if not row['tag'].endswith(':scalar'):
                continue
            prefix = row['tag'].rsplit(':', 1)[0]
            native = idx[prefix+':native', row['q_condition'], row['invocation']]
            rows.append({**row, 'native_service_ns': native['service_ns'],
                'scalar_minus_native_ns': row['service_ns']-native['service_ns'],
                'native_q': native['q'], 'matched_q': native['q'] == row['q']})
        c.derived(c.OUT/'packing_contrasts.json', {'rows': rows,
            'scope': 'p=2 removes ternary max-tree fusion; BF16 scalar vs native BF16 path. Verify actual SASS/dynamic max and conversion counts before attributing packing.'})
        print('Packing contrasts',len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'compile', 'artifacts', 'counters', 'services', 'import', 'summary'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else analyze(args.action)
