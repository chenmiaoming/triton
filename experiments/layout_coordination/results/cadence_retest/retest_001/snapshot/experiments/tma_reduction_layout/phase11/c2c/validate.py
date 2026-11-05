"""Recompute evidence bindings and measurements; success is not model efficacy."""
import argparse
from collections import defaultdict
import statistics as st
from pathlib import Path
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2c.analyze import fit


def run(committed=False):
    c.protect_prior()
    protocol = c.protocol()
    c.verify_sources(protocol)
    gate = c.read(c.OUT/'artifact_gate.json')
    compiled = c.read(c.OUT/'compile_gate.json')
    counter = c.read(c.OUT/'counter_gate.json')
    analysis = c.read(c.OUT/'analysis.json')
    composition = c.read(c.OUT/'composition.json')
    assert all(value['status'] == 'OK' for value in (gate, compiled, counter, analysis))
    protected = {tuple(row) for row in protocol['reserved_D1']}
    assert len(protected) == 11
    raw_compile = c.read(c.ROOT/compiled['artifact_directory']/'result.json')
    assert not raw_compile['GPU_allocated'] and raw_compile['ccache_before'] == raw_compile['ccache_after']
    assert raw_compile['native_SHA256'] == protocol['baseline']['native_SHA256']
    for tag, meta in gate['metadata'].items():
        assert gate['resources'][tag]['loaded_CUBIN_SHA256'] == meta['cubin_SHA256']
        assert gate['resources'][tag]['attributes']['num_regs'] > 0
        folder = c.ROOT/compiled['artifact_directory']/tag.replace(':', '/')
        assert c.sha((folder/'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
        if meta['kind'] == 'local':
            assert meta['logical_axes'] == ['owner_tensor', 'local_reduction', 'local_columns']
            assert meta['logical_shape'] == [128, meta['p'], meta['columns']]
            assert meta['p']*meta['columns'] == meta['R']
            assert meta['ABI_kind'] == 'pointer_K'
            assert 'cp.async.bulk.tensor' not in (folder/'kernel.ptx').read_text()
        else:
            assert (meta['M'], meta['N'], meta['num_warps']) not in protected
    sources = {}
    samples = 0
    uuids = set()
    for invocation in (1, 2, 3):
        path = c.OUT/'timing'/f'invocation_{invocation}'/'raw.json'
        raw = c.read(path)
        assert raw['status'] == 'OK' and raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        assert not raw['Triton_imported'] and raw['profiler_attached'] is False
        assert raw['driver_return_codes'] == [0]
        uuids.add(raw['environment']['gpu_uuid'])
        values = {}
        for row in raw['cells']:
            assert len(row['samples_us']) == 12 and row['correctness']['pass']
            assert all(sorted(visit['node_types']) == [0, 7, 7] for visit in row['visits'])
            samples += len(row['samples_us'])
            for identity in [row['cell_id']]+row['aliases']:
                tag, q, grid, k = identity.split('|')
                values[tag, q[1:], int(grid[1:]), int(k[1:])] = st.median(row['samples_us'])
        sources[invocation] = values
    assert samples == 271188, samples
    for row in analysis['services']:
        source = sources[row['invocation']]
        grids = [32*132, 64*132, 96*132]
        slopes = [fit([16, 64, 256], [source[row['tag'], row['q_condition'], grid, k] for k in (16, 64, 256)])['slope']
                  for grid in grids]
        expected = 1000*fit([32, 64, 96], slopes)['slope']
        assert abs(expected-row['service_ns']) < 1e-8, row['tag']
        assert row['GPU_UUID']
    for row in analysis['oneshot']:
        source = sources[row['invocation']]
        grids = [row['full_grid']//divisor for divisor in (4, 2, 1)]
        expected = fit([grid/132 for grid in grids], [source[row['tag'], row['q_condition'], grid, 1] for grid in grids])
        assert abs(1000*expected['slope']-row['tile_slope_ns']) < 1e-8
    for row in analysis['local_pairs']:
        if row.get('contrast_available'):
            p = row['matched_service_ns']
            assert abs(row['native_net_ns']-(p['native']-p['fixture'])) < 1e-9
            assert abs(row['wide_net_ns']-(p['wide']-p['fixture_wide'])) < 1e-9
    local_affine = 0
    for row in counter['affine']:
        if gate['metadata'][row['tag']]['kind'] == 'local':
            local_affine += 1
            for family in ('shared_load', 'shared_store', 'shuffle', 'barrier', 'global_store'):
                assert abs(row['families'][family]['per_iteration_warp_instructions']) < 1e-9, (row['tag'], family, row['families'][family])
            # All max work in a local probe belongs to the repeated region.
            assert abs(row['families']['max']['fixed_warp_instructions']) < 1e-9, row['tag']
    assert local_affine >= 50, local_affine
    assert len(counter['reports']) == 22
    for report in counter['reports']:
        assert c.sha((c.ROOT/report['path']).read_bytes()) == report['SHA256']
    imported = c.read(c.OUT/'offline_import_validation.json')
    assert imported['status'] == 'OK' and len(imported['reports']) == 22
    for report in imported['reports']:
        assert report['all_metrics_and_source_match']
        assert c.sha((c.ROOT/report['local_import']).read_bytes()) == report['local_import_SHA256']
    assert composition['physical_base_identified'] is False
    assert not composition['global_intercepts_used_in_selector'] and not composition['historical_26_6_used']
    binding = c.read(c.OUT/'preregistration/C32_method_binding.json')
    assert binding['before_any_GPU_dispatch']
    assert c.sha(c.encode(binding['C32_protocol'])) == binding['C32_protocol_SHA256']
    assert c.sha((c.OUT/'preregistration/C32_method_binding.json').read_bytes()) == protocol['C32_method_binding_SHA256']
    # Frozen CPU and profiling exports must contain exactly the saved files.
    exports = list((c.OUT/'cpu_preflight').glob('attempt_*/raw_export.zip'))
    exports += list((c.OUT/'profiles').glob('group_*/attempt_*/raw_export.zip'))
    for archive in exports:
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                assert (archive.parent/'export'/name).read_bytes() == zipped.read(name), (archive, name)
    if committed:
        changed = subprocess.check_output(['git', 'diff', 'HEAD', '--', str(c.SOURCE.relative_to(c.ROOT)),
                                           str(c.OUT.relative_to(c.ROOT))], cwd=c.ROOT)
        assert not changed, 'Committed C2c changed'
    from experiments.tma_reduction_layout.phase11.c2c.packing_validate import run as packing_run
    packing = packing_run()
    result = {'status': 'OK', 'stage': 'C2c', 'programs': len(gate['metadata']),
        'correctness_checks': gate['checks'], 'graph_event_samples': samples,
        'timing_invocations': 3, 'physical_GPU_UUIDs': sorted(uuids),
        'NCU_reports': len(counter['reports']), 'local_dynamic_affine_programs': local_affine,
        'independent_local_NCU_reimports': len(imported['reports']),
        'prior_files_hash_checked': len(c.read(c.OUT/'preregistration/prior_manifest.json')['files']),
        'D1_protected_geometries': len(protected), 'D1_compiled': False, 'D1_timed': False,
        'native_rebuilds': 0, 'physical_components_identified': False,
        'packing_supplement': packing,
        'all_graph_event_samples': samples+packing['graph_event_samples'],
        'all_NCU_reports': len(counter['reports'])+packing['NCU_reports'],
        'independent_packing_cost_identified': False,
        'scope': 'Evidence integrity/recomputation PASS. A unique physical base/transfer decomposition was not established.'}
    c.derived(c.OUT/'validation.json', result)
    print('C2c validator OK;', result['all_graph_event_samples'], 'total event samples;', result['all_NCU_reports'], 'NCU reports; D1 untouched')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--committed', action='store_true')
    run(parser.parse_args().committed)
