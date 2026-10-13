"""Recompute eligibility from original artifacts, without performance fields."""
import argparse
import re
import zipfile
from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
from experiments.tma_reduction_layout.gluon.artifact_checks import parse_resource
from experiments.tma_reduction_layout.phase10.stage_c import common as c


def run():
    protocol = c.protocol()
    gates = {}
    for target in protocol['targets']:
        rows, environments = {}, {}
        for variant in protocol['variants']:
            root = c.OUT / 'artifacts' / target / variant
            with zipfile.ZipFile(root / 'raw_export.zip') as z:
                for name in z.namelist():
                    assert z.read(name) == (root / 'export' / name).read_bytes(), name
            env = c.read(root / 'export/environment.json')
            assert env['variant'] == variant and env['target'] == target
            assert env['native_SHA256'] == protocol['compilers'][variant]['native_SHA256']
            assert env['source'] == protocol['compilers'][variant]['source']
            assert env['worker_source_SHA256'] == protocol['worker_source_SHA256']
            assert env['protocol_SHA256'] == c.sha(c.encode(protocol))
            assert env['patch_SHA256'] == protocol['patch_SHA256']
            assert env['compute_capability'] == protocol['targets'][target]['cc']
            assert protocol['targets'][target]['name'] in env['gpu_name']
            assert not env['GPU_timing'] and env['ccache_start'] == env['ccache_end']
            records = c.read(root / 'export/attempts.json')
            assert len(records) == len(protocol['cases']) == 62
            assert [r['case_id'] for r in records] == [r['case_id'] for r in protocol['cases']]
            assert c.read(root / 'return.json') == {'records': records, 'environment': env}
            for row in records:
                directory = root / 'export' / row['tag'].replace(':', '/')
                assert c.read(directory / 'attempt.json') == row
                for name, digest in row['files'].items():
                    assert c.sha((directory / name).read_bytes()) == digest, name
                if row['status'] == 'OK':
                    assert row['actual_jit_compile_events'] == 1 and row['correctness_max_abs_diff'] == 0
                    meta = c.read(directory / 'metadata.json')
                    assert meta['cubin_SHA256'] == row['cubin_SHA256'] == c.sha(
                        (directory / 'kernel.cubin').read_bytes())
                    assert meta['grid'] == c.grid(row)
                    ir = (directory / 'kernel.ttgir').read_text()
                    info = analyze_ttgir(ir)
                    assert row['observed_layout'] == info['blocked_encodings'].get(info['local_load_dest_layout'])
                    assert row['observed_axes'] == [int(x) for x in re.findall(r'"tt.reduce".*?<\{axis = (\d+)', ir)]
                    assert info['module_attributes'] == {
                        'num_ctas': 1, 'num_warps': row['num_warps'], 'threads_per_warp': 32
                    }
                    occupancy = c.read(directory / 'occupancy.json')
                    assert occupancy['queried_cubin_sha256'] == meta['cubin_SHA256']
                    assert occupancy['blocks_per_sm_actual_dynamic_smem'] > 0
                    assert all(call['return_code'] == 0 for call in occupancy['checked_cuda_calls'])
                else:
                    assert 'error' in row and (directory / 'error.txt').read_text() == row['error']
            rows[variant] = {r['case_id']: r for r in records}
            environments[variant] = env
        assert environments['baseline']['ptxas']['SHA256'] == environments['patch']['ptxas']['SHA256']
        eligible, unavailable, observations = [], [], []
        for case in protocol['cases']:
            identity = case['case_id']
            a, b = rows['baseline'][identity], rows['patch'][identity]
            if a['status'] != 'OK' or b['status'] != 'OK':
                unavailable.append({
                    'case_id': identity, 'cohort': case['cohort'], 'baseline_status': a['status'], 'patch_status':
                    b['status'], 'errors': {v: rows[v][identity].get('error')
                                            for v in protocol['variants']}
                })
                continue
            eligible.append(identity)
            occupancy, resources = {}, {}
            for variant in protocol['variants']:
                directory = c.OUT / 'artifacts' / target / variant / 'export' / c.tag(case, variant).replace(':', '/')
                occupancy[variant] = c.read(directory / 'occupancy.json')
                resources[variant] = parse_resource((directory / 'kernel.resource.txt').read_text())
            same_code = a['cubin_SHA256'] == b['cubin_SHA256']
            same_layout = a['observed_layout'] == b['observed_layout']
            if case['cohort'] == 'guard_control':
                assert same_code and same_layout, 'Eligibility guard unexpectedly changes code: ' + identity
            spilled = any(resources[v]['stack_bytes'] or resources[v]['local_bytes'] or occupancy[v]['local_bytes']
                          for v in protocol['variants'])
            stratum = ('local_memory_or_spill' if spilled else 'matched_residency' if occupancy['baseline']
                       ['active_warps_per_sm'] == occupancy['patch']['active_warps_per_sm'] else 'changed_residency')
            observations.append({
                'case_id': identity, 'cohort': case['cohort'], 'baseline_layout': a['observed_layout'], 'patch_layout':
                b['observed_layout'], 'same_CUBIN': same_code, 'same_layout': same_layout, 'occupancy': occupancy,
                'resources': resources, 'resource_stratum': stratum
            })
        gates[target] = {
            'eligible': eligible, 'unavailable': unavailable, 'observations': observations, 'environments':
            environments, 'compiler_attempts': 124
        }
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    gates = run()
    path = c.OUT / 'artifact_gate.json'
    if args.validate:
        assert path.read_bytes() == c.encode(gates)
    else:
        c.write_once(path, gates)
    print({t: {'eligible': len(g['eligible']), 'unavailable': len(g['unavailable'])} for t, g in gates.items()})


if __name__ == '__main__':
    main()
