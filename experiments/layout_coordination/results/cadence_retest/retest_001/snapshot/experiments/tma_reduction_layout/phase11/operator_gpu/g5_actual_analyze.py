"""Reimport actual-call CPU compiles and bind all ten frozen G4b workloads."""
from . import common as c
from .g5_prepare import OUT
from .g5_analyze import elf, program, save


def main():
    folder = OUT/'actual/attempt_001/export'
    raw = c.read(folder/'result.json')
    assert raw['status'] == 'OK' and not raw['GPU_allocated']
    rows = [program(folder, r) for r in raw['programs']]
    by = {(r['case'], r['variant']): r for r in rows}; comparisons = []
    for case in sorted({r['case'] for r in rows}):
        u = by[(case, 'U_actual')]; r = by[(case, 'G0_actual_equivalent')]
        comparisons.append({'case': case,
            'all_allocated_section_bytes_equal_U': r['allocated_sections'] == u['allocated_sections'],
            'ABI_equal_U': r['ABI'] == u['ABI'] and r['metadata']['runtime_argument_names'] == u['metadata']['runtime_argument_names'],
            'resources_equal_U': r['resources'] == u['resources'] and r['metadata']['dynamic_smem_bytes'] == u['metadata']['dynamic_smem_bytes']})
    assert all(all(r[k] for k in ['all_allocated_section_bytes_equal_U', 'ABI_equal_U', 'resources_equal_U']) for r in comparisons)
    assert all(r['matches_original_CUBIN'] for r in rows if r['variant'] == 'U_actual')
    original = c.read(c.OUT/'stage_g4b/compile_gate_v2.json')
    groups = c.read(c.OUT/'stage_g4b/preregistration/protocol.json')['groups']
    core = c.read(OUT/'compiler_analysis.json')['programs']
    equivalent = c.read(OUT/'equivalence_gate.json')['cases']
    bindings = []
    for group in groups:
        gid = group['id']; case = group['case']['id']
        if case.startswith('C02'):
            key = case + '_b8_l2048'; r = by[(key, 'G0_actual_equivalent')]
        elif '_b1_' in gid:
            r = by[(gid, 'G0_actual_equivalent')]
        else:
            gate = next(x for x in equivalent if x['case'] == case)
            r = next(x for x in core if x['case'] == case and x['variant'] == gate['repair_variant'])
        u = next(x for x in original['records'] if x['tag'] == gid+':actual:upstream')
        root = c.ROOT/original['CPU_export_directory']/u['artifact_directory']
        matches = r['allocated_sections'] == elf((root/'kernel.cubin').read_bytes())
        assert matches and all(u[k] == r['metadata'][k] for k in ['runtime_argument_names', 'dynamic_smem_bytes', 'num_warps', 'global_scratch_size'])
        bindings.append({'group': gid, 'original_U_CUBIN_SHA256': u['cubin_SHA256'],
                         'repair_artifacts': r['source_path'], 'matches_original_U_allocated_sections_ABI_resources': matches})
    save(OUT/'actual_ABI_analysis.json', {'status': 'OK', 'programs': rows,
         'comparisons': comparisons, 'G4b_workload_bindings': bindings,
         'GPU_allocated': False, 'timing_samples': 0})
    print('G5 real-call ABI audit: 4/4 new compiled pairs; 10/10 G4b workload bindings equal U')


if __name__ == '__main__':
    main()
