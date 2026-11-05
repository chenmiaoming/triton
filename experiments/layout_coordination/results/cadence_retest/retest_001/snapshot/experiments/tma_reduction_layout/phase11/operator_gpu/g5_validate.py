"""Validate G5 byte equivalence, CPU provenance, cache repair and prior evidence."""
from pathlib import Path
import argparse
import gzip
import io
import json
import subprocess
import zipfile
from . import common as c
from .g5_prepare import OUT, PIN
from .g5_analyze import elf, facts, normalize


RUNS = {
    'pilot/attempt_001': 'protocol',
    'pilot/attempt_002': 'protocol_v2',
    'matrix/attempt_001': 'protocol_v2',
    'literal/attempt_001': 'literal_protocol',
    'literal2/attempt_001': 'literal2_protocol',
    'literal3/attempt_001': 'literal3_protocol',
    'actual/attempt_001': 'actual_protocol',
}
SNAPSHOTS = {
    'protocol': 'source_snapshot', 'protocol_v2': 'source_snapshot_v2',
    'literal_protocol': 'literal_source_snapshot',
    'literal2_protocol': 'literal2_source_snapshot',
    'literal3_protocol': 'literal3_source_snapshot',
    'actual_protocol': 'actual_source_snapshot',
}


def validate(write=False):
    protocols = {}
    for name, snapshot in SNAPSHOTS.items():
        p = c.read(OUT / 'preregistration' / (name + '.json'))
        protocols[name] = p
        assert not p['GPU_allocated'] and p['compiler'] == PIN and p['native_SHA256'] == c.NATIVE
        with zipfile.ZipFile(OUT / 'preregistration' / (snapshot + '.zip')) as archive:
            for path, digest in p['worker_source_SHA256'].items():
                assert c.sha(archive.read(path)) == digest, (snapshot, path)
        for path, digest in {**p['old_evidence_SHA256'], **p.get('prior_attempt_SHA256', {})}.items():
            assert c.sha((c.ROOT / path).read_bytes()) == digest, path
        for path, digest in p['compiler_source_SHA256'].items():
            assert c.sha((OUT / 'compiler_source' / path).read_bytes()) == digest, path
    # V1's worker is intentionally retained only in its pre-intervention snapshot.
    for p in list(protocols.values())[1:]:
        c.verify_sources(p)
    manifest = c.read(OUT / 'compiler_source_manifest.json')
    assert manifest['compiler'] == PIN
    for path, digest in manifest['source_SHA256'].items():
        blob = (OUT / 'compiler_source' / path).read_bytes()
        assert c.sha(blob) == digest
        assert blob == subprocess.check_output(['git', 'show', PIN + ':' + path], cwd=c.ROOT)

    programs = {}; trace_count = 0
    for namespace, protocol_name in RUNS.items():
        target = OUT / namespace; folder = target / 'export'
        with zipfile.ZipFile(target / 'raw_export.zip') as archive:
            expected = {n for n in archive.namelist() if not n.endswith('/')}
            assert expected == {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
            for path in expected:
                assert archive.read(path) == (folder / path).read_bytes(), (namespace, path)
        result = c.read(folder / 'result.json')
        assert result['status'] == 'OK' and not result['GPU_allocated'] and result['native_rebuilds'] == 0
        assert result['ccache_before'] == result['ccache_after']
        assert result['protocol_SHA256'] == c.sha(c.encode(protocols[protocol_name]))
        dispatch = c.read(target / 'dispatch.json')
        assert dispatch['profile'] == 'chenmiaoming' and not dispatch['GPU_allocated']
        assert dispatch['protocol_SHA256'] == result['protocol_SHA256']
        for r in result['programs']:
            assert r['status'] == 'OK' and not r['GPU_allocated'] and r['native_SHA256'] == c.NATIVE
            root = folder / r['path']; meta = r['metadata']; artifacts = root / meta['artifact_directory']
            for path, digest in meta['artifact_SHA256'].items():
                assert c.sha((artifacts / path).read_bytes()) == digest, artifacts / path
            assert c.sha((artifacts / 'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
            assert c.read(root / 'command.json')['returncode'] == 0
            if (root / 'passes.mlir.gz').exists():
                blob = gzip.decompress((root / 'passes.mlir.gz').read_bytes())
                integrity = c.read(root / 'trace_integrity.json')
                assert c.sha(blob) == integrity['uncompressed_SHA256'] and len(blob) == integrity['uncompressed_bytes']
                trace_count += 1
            programs[(r['case'], r['variant'])] = r, artifacts
    assert len(programs) == 59 and trace_count == 18
    for (case, variant), (r, _) in programs.items():
        if variant in ['U', 'G0', 'U_actual', 'U_restore_cache', 'G0_restore_cache']:
            assert r['matches_original_CUBIN'] and r['metadata']['cubin_SHA256'] == r['original_CUBIN_SHA256']
        if variant.endswith('_restore_cache'):
            affected = ['U_no_rmlc'] if variant.startswith('U') else ['G0_cleanup', 'G0_rmlc', 'G0_fullsuffix']
            for old in affected:
                assert r['metadata']['compile_key'] == programs[(case, old)][0]['metadata']['compile_key']
        if 'kernel_cache_directory' in r and variant not in ['U', 'G0', 'U_restore_cache', 'G0_restore_cache']:
            assert '-g5-' in r['kernel_cache_directory']

    analysis = c.read(OUT / 'compiler_analysis.json')
    assert analysis['status'] == 'OK' and len(analysis['programs']) == 45 and len(analysis['traces']) == 18
    for row in analysis['programs']:
        r, root = programs[(row['case'], row['variant'])]
        assert row['allocated_sections'] == elf((root / 'kernel.cubin').read_bytes())
        text = (root / 'kernel.ttgir').read_text()
        assert row['facts'] == facts(text)
        assert row['normalized_TTGIR_SHA256'] == c.sha(normalize(text).encode())
    equal = c.read(OUT / 'equivalence_gate.json')
    assert equal['status'] == 'OK' and len(equal['cases']) == 9
    comparisons = {(r['case'], r['variant']): r for r in analysis['comparisons']}
    for row in equal['cases']:
        assert row['repair_variant'] == ('G0_literal3' if row['case'].startswith('C06') else 'G0_fullsuffix')
        gate = comparisons[(row['case'], row['repair_variant'])]
        assert all(gate[k] for k in ['all_allocated_section_bytes_equal_U', 'instruction_bytes_equal_U', 'ABI_equal_U', 'resources_equal_U'])
        u, uroot = programs[(row['case'], 'U')]
        repaired, rroot = programs[(row['case'], row['repair_variant'])]
        assert elf((uroot / 'kernel.cubin').read_bytes()) == elf((rroot / 'kernel.cubin').read_bytes())
        for key in ['runtime_argument_names', 'dynamic_smem_bytes', 'num_warps', 'global_scratch_size']:
            assert u['metadata'][key] == repaired['metadata'][key]
        assert row['comparison'] == gate
        assert not comparisons[(row['case'], 'G0')]['instruction_bytes_equal_U']
    actual = c.read(OUT / 'actual_ABI_analysis.json')
    assert actual['status'] == 'OK' and len(actual['programs']) == 8
    assert len(actual['comparisons']) == 4 and len(actual['G4b_workload_bindings']) == 10
    for comparison in actual['comparisons']:
        assert all(comparison[k] for k in ['all_allocated_section_bytes_equal_U', 'ABI_equal_U', 'resources_equal_U'])
        _, uroot = programs[(comparison['case'], 'U_actual')]
        _, rroot = programs[(comparison['case'], 'G0_actual_equivalent')]
        assert elf((uroot / 'kernel.cubin').read_bytes()) == elf((rroot / 'kernel.cubin').read_bytes())
    for binding in actual['G4b_workload_bindings']:
        assert binding['matches_original_U_allocated_sections_ABI_resources']
    assert (OUT / 'REPORT.md').is_file() and (OUT / 'REPRODUCE.md').is_file()
    inventory = {str(p.relative_to(OUT)): c.sha(p.read_bytes()) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and p.name != 'final_validation.json'}
    scripts = {str(p.relative_to(c.ROOT)): c.sha(p.read_bytes()) for p in sorted(c.SOURCE.glob('g5_*.py'))}
    final = {'status': 'OK', 'stage': 'G5', 'CPU_program_compilations': len(programs),
             'original_U_G0_CUBIN_matches': 18, 'restored_affected_cache_keys': 6,
             'complete_native_pass_traces': trace_count, 'equivalent_repaired_baselines': 9,
             'additional_actual_ABI_U_CUBIN_matches': 4, 'additional_equivalent_actual_ABI_pairs': 4,
             'equivalent_G4b_workload_bindings': 10,
             'equivalence_unit': 'all ELF SHF_ALLOC sections, runtime ABI and launch resources',
             'GPU_allocations': 0, 'new_timing_samples': 0, 'native_rebuilds': 0,
             'prior_files_unchanged': len(protocols['protocol']['old_evidence_SHA256']),
             'REPORT_SHA256': c.sha((OUT / 'REPORT.md').read_bytes()),
             'source_SHA256': scripts, 'evidence_files': len(inventory),
             'evidence_inventory_SHA256': c.sha(c.encode(inventory))}
    path = OUT / 'final_validation.json'
    if write: c.write_once(path, final)
    if path.exists(): assert c.read(path) == final
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--write', action='store_true')
    validate(parser.parse_args().write)
