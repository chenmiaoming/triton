"""Independent checks for the same-worker BF16 p=2 packing intervention."""
import re
import statistics as st
import zipfile
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2c.analyze import fit


def run():
    out = c.OUT/'packing_supplement'
    protocol = c.read(out/'preregistration/protocol.json')
    c.verify_sources(protocol)
    compiled = c.read(out/'compile_gate.json')
    gate = c.read(out/'artifact_gate.json')
    counter = c.read(out/'counter_gate.json')
    analysis = c.read(out/'analysis.json')
    imported = c.read(out/'offline_import_validation.json')
    assert len(compiled['metadata']) == 20 and compiled['reused'] == 16
    cpu = c.read(c.ROOT/compiled['artifact_directory']/'result.json')
    assert cpu['new_compiles'] == 4 and not cpu['GPU_allocated']
    assert cpu['ccache_before'] == cpu['ccache_after']
    assert cpu['native_SHA256'] == protocol['baseline']['native_SHA256']
    affine = {row['tag']: row for row in counter['affine']}
    identities = []
    for r in (4, 8, 16, 32):
        prefix = f'local_R{r}_p2_c{r//2}'
        sites = {}
        for variant in ('native', 'scalar', 'wide', 'fixture', 'fixture_wide'):
            tag = prefix+':'+variant
            meta = gate['metadata'][tag]
            folder = c.ROOT/compiled['artifact_directory']/tag.replace(':', '/')
            assert c.sha((folder/'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
            assert gate['resources'][tag]['loaded_CUBIN_SHA256'] == meta['cubin_SHA256']
            assert gate['resources'][tag]['attributes']['local_bytes'] == 0
            ptx = (folder/'kernel.ptx').read_text()
            sites[variant] = {opcode: len(re.findall(r'\b'+re.escape(opcode)+r'\b', ptx))
                              for opcode in ('max.bf16', 'max.bf16x2', 'max.f32')}
            # No shared-memory, shuffle, barrier or per-iteration output fixture.
            for family in ('shared_load', 'shared_store', 'shuffle', 'barrier', 'global_store'):
                assert abs(affine[tag]['families'][family]['per_iteration_warp_instructions']) < 1e-9
        assert sites['native']['max.bf16x2'] == r//4
        assert sites['native']['max.bf16'] == 0
        assert sites['scalar']['max.bf16'] == r//2
        assert sites['scalar']['max.bf16x2'] == 0
        counts = {variant: affine[prefix+':'+variant]['families']['max']['per_iteration_warp_instructions']
                  for variant in ('native', 'scalar', 'wide')}
        # The proposed scalar intervention FAILED: ptxas recombines the scalar
        # PTX into packed SASS. Preserve that result; evidence integrity is a
        # different question from successful mechanism identification.
        assert counts['native'] == r and counts['scalar'] == r, counts
        identities.append({'R': r, 'PTX_max_sites': sites, 'dynamic_max_warp_instructions_per_iteration': counts})
    samples, uuids = 0, set()
    source = {}
    for invocation in (1, 2, 3):
        raw = c.read(out/'timing'/f'invocation_{invocation}'/'raw.json')
        assert raw['status'] == 'OK' and not raw['Triton_imported'] and not raw['profiler_attached']
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        assert raw['driver_return_codes'] == [0]
        uuids.add(raw['environment']['gpu_uuid'])
        cells = {}
        for row in raw['cells']:
            assert row['correctness']['pass'] and len(row['samples_us']) == 12
            assert all(sorted(visit['node_types']) == [0, 7, 7] for visit in row['visits'])
            samples += len(row['samples_us'])
            for identity in [row['cell_id']]+row['aliases']:
                tag, q, grid, k = identity.split('|')
                cells[tag, q[1:], int(grid[1:]), int(k[1:])] = st.median(row['samples_us'])
        source[invocation] = cells
    for row in analysis['services']:
        values = source[row['invocation']]
        slopes = [fit([16, 64, 256], [values[row['tag'], row['q_condition'], g, k] for k in (16, 64, 256)])['slope']
                  for g in (32*132, 64*132, 96*132)]
        expected = 1000*fit([32, 64, 96], slopes)['slope']
        assert abs(expected-row['service_ns']) < 1e-8
    assert len(counter['reports']) == len(imported['reports']) == 2
    assert imported['status'] == 'OK' and all(row['all_metrics_and_source_match'] for row in imported['reports'])
    for row in counter['reports']:
        assert c.sha((c.ROOT/row['path']).read_bytes()) == row['SHA256']
    for row in imported['reports']:
        assert c.sha((c.ROOT/row['local_import']).read_bytes()) == row['local_import_SHA256']
    for archive in list((out/'cpu_preflight').glob('attempt_*/raw_export.zip'))+list((out/'profiles').glob('group_*/attempt_*/raw_export.zip')):
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                assert (archive.parent/'export'/name).read_bytes() == zipped.read(name)
    contrast = c.read(out/'packing_contrasts.json')
    assert len(contrast['rows']) == 48
    for row in contrast['rows']:
        assert abs(row['scalar_minus_native_ns']-(row['service_ns']-row['native_service_ns'])) < 1e-9
    feasibility = c.read(out/'scalar_feasibility/attempt_3/export/result.json')
    assert not feasibility['GPU_allocated']
    assert feasibility['ccache_before'] == feasibility['ccache_after']
    assert len(feasibility['cases']) == 8 and all(row['status'] == 'OK' for row in feasibility['cases'])
    for attempt in (2, 3):
        folder = out/'scalar_feasibility'/f'attempt_{attempt}'
        frozen = c.read(folder/'protocol.json')
        for name in ('packing_cpu.py', 'packing_cpu_kernels.py'):
            source = folder/'worker_sources'/name if attempt == 2 else c.SOURCE/name
            assert c.sha(source.read_bytes()) == frozen['worker_source_SHA256'][str((c.SOURCE/name).relative_to(c.ROOT))]
        with zipfile.ZipFile(folder/'raw_export.zip') as zipped:
            for name in zipped.namelist():
                assert (folder/'export'/name).read_bytes() == zipped.read(name)
    result = {'status': 'OK', 'programs': 20, 'new_scalar_programs': 4,
        'exact_reused_CUBINs': 16, 'correctness_checks': gate['checks'],
        'graph_event_samples': samples, 'physical_GPU_UUIDs': sorted(uuids),
        'NCU_reports': 2, 'same_worker_packing_contrasts': 48,
        'packing_instruction_validation': identities,
        'packing_intervention_status': 'FAILED_REPACKED_BY_BACKEND',
        'independent_packing_cost_identified': False,
        'CPU_only_alternative_implementations': 8,
        'D1_compiled': False, 'D1_timed': False, 'native_rebuilds': 0,
        'scope': 'Evidence integrity PASS. Scalar PTX was repacked, so this is not an isolated packing-cost contrast; no naked opcode latency is identified.'}
    c.derived(out/'validation.json', result)
    print('C2c packing evidence validator OK;', samples, 'samples; intervention FAILED: scalar PTX repacked')
    return result


if __name__ == '__main__':
    run()
