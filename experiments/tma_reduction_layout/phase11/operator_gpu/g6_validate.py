"""Validate all retained attempts and the accepted immutable source baseline."""
import argparse
import io
import json
from pathlib import Path
import zipfile
from . import common as c
from .g6_analyze import analyze, artifact, references
from .g6_prepare import OUT


def validate(write=False):
    attempts = sorted(OUT.glob('attempt_*')); assert attempts
    count = 0
    refs = references()
    for folder in attempts:
        attempt = int(folder.name.split('_')[1]); p = c.read(folder/'preregistration/protocol.json')
        assert p['native_SHA256'] == c.NATIVE and not p['compiler_intervention'] and not p['GPU_allocated']
        with zipfile.ZipFile(folder/'preregistration/source_snapshot.zip') as source:
            for name, digest in p['worker_source_SHA256'].items():
                assert c.sha(source.read(name)) == digest, (attempt, name)
        for name, digest in p['protected_SHA256'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
        with zipfile.ZipFile(folder/'raw_export.zip') as archive:
            names = {n for n in archive.namelist() if not n.endswith('/')}
            assert names == {str(path.relative_to(folder/'export')) for path in (folder/'export').rglob('*') if path.is_file()}
            for name in names:
                assert archive.read(name) == (folder/'export'/name).read_bytes()
        raw = c.read(folder/'export/result.json'); dispatch = c.read(folder/'dispatch.json')
        assert raw['protocol_SHA256'] == dispatch['protocol_SHA256'] == c.sha(c.encode(p))
        assert dispatch['profile'] == 'chenmiaoming' and not dispatch['GPU_allocated']
        assert raw['GPU_allocated'] is False and raw['compiler_intervention'] is False and raw['native_rebuilds'] == 0
        assert raw['native_SHA256'] == c.NATIVE and raw['backend_SHA256'] == p['backend_SHA256']
        assert raw['ccache_before'] == raw['ccache_after']
        observed = [r['case'] for r in raw['records']]; expected = [item['case']['id'] for item in p['cases']]
        assert len(set(observed)) == len(observed) == len(expected) and set(observed) == set(expected)
        comparison = analyze(attempt)
        assert comparison['total'] == len(expected)
        count += len(observed)
        for r in comparison['rows']:
            if not r['compiled']:
                continue
            refroot, refmeta = refs[r['case']]
            reference = artifact(refroot, refmeta)
            actual = artifact(folder/'export', r['metadata'])
            assert reference == r['U_reference'] and actual == r['artifact']
            assert r['allocated_sections_equal_U'] == (actual['allocated_sections'] == reference['allocated_sections'])
    manifest = c.read(OUT/'baseline_manifest.json')
    accepted = OUT/f"attempt_{manifest['attempt']:03}"
    protocol = c.read(accepted/'preregistration/protocol.json'); c.verify_sources(protocol)
    assert manifest['status'] == 'OK' and manifest['default_compiler_pipeline'] and not manifest['compiler_intervention']
    analysis = c.read(accepted/'comparison.json')
    assert analysis['total'] == analysis['equivalent'] == len(manifest['cases']) == 13
    assert len(manifest['actual_G4b_workloads']) == 10
    expected_groups = {g['id'] for g in c.read(c.OUT/'stage_g4b/preregistration/protocol.json')['groups']}
    assert {r['group'] for r in manifest['actual_G4b_workloads']} == expected_groups
    bycase = {}
    for row in manifest['cases']:
        match = next(r for r in analysis['rows'] if r['case'] == row['case'])
        assert row['metadata'] == match['metadata'] and row['artifacts'] == match['artifact']['artifacts']
        assert match['equivalent']
        bycase[row['case']] = match
    g4 = c.read(c.OUT/'stage_g4b/compile_gate_v2.json')
    for binding in manifest['actual_G4b_workloads']:
        repaired = bycase[binding['source_baseline_case']]
        u = next(r for r in g4['records'] if r['tag'] == binding['group']+':actual:upstream')
        reference = artifact(c.ROOT/g4['CPU_export_directory'], u)
        assert reference['allocated_sections'] == repaired['artifact']['allocated_sections']
        assert reference['ABI'] == repaired['artifact']['ABI']
        assert reference['resources'] == repaired['artifact']['resources']
        assert all(u[k] == repaired['metadata'][k] for k in ['runtime_argument_names', 'num_warps', 'dynamic_smem_bytes', 'global_scratch_size'])
        assert binding['original_U_CUBIN_SHA256'] == u['cubin_SHA256']
    for name, digest in manifest['source_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest
    assert (OUT/'REPORT.md').exists() and (OUT/'REPRODUCE.md').exists()
    inventory = {str(p.relative_to(OUT)): c.sha(p.read_bytes()) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and p.name != 'final_validation.json'}
    value = dict(status='OK', stage='G6', retained_attempts=len(attempts), CPU_compile_requests=count,
                 accepted_source_only_compilation_forms=13, actual_workload_bindings=10,
                 compiler_interventions=0, GPU_allocations=0, new_timing_samples=0, native_rebuilds=0,
                 old_files_unchanged=len(protocol['protected_SHA256']), accepted_attempt=manifest['attempt'],
                 source_SHA256={str(p.relative_to(c.ROOT)): c.sha(p.read_bytes()) for p in sorted(c.SOURCE.glob('g6_*.py'))},
                 REPORT_SHA256=c.sha((OUT/'REPORT.md').read_bytes()), evidence_files=len(inventory),
                 evidence_inventory_SHA256=c.sha(c.encode(inventory)))
    final = OUT/'final_validation.json'
    if write: c.write_once(final, value)
    if final.exists(): assert c.read(final) == value
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--write', action='store_true')
    validate(parser.parse_args().write)
