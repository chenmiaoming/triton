"""Independently close the CPU ABI audit after its final-print failure.

The frozen compiler/audit source is preserved.  Its analysis JSON was written
before a missing ``json`` import broke only the terminal summary.  This module
recomputes and asserts every execution-equivalence check and retains that
original CLI failure as evidence.
"""
import argparse
import json
from pathlib import Path
import zipfile
from . import common as c
from .g8_actualabi import AUDIT, OUT, program


def verify(write=False):
    protocol = c.read(AUDIT/'preregistration/protocol.json'); c.verify_sources(protocol)
    for name,digest in protocol['protected_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest,name
    with zipfile.ZipFile(AUDIT/'preregistration/source_snapshot.zip') as archive:
        for name,digest in protocol['worker_source_SHA256'].items():
            assert c.sha(archive.read(name)) == digest,name
    folder = AUDIT/'cpu/attempt_001'; root = folder/'export'
    with zipfile.ZipFile(folder/'raw_export.zip') as archive:
        names = [name for name in archive.namelist() if not name.endswith('/')]
        assert len(names) == len(set(names))
        assert set(names) == {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
        for name in names: assert archive.read(name) == (root/name).read_bytes()
    raw = c.read(root/'result.json'); analysis = c.read(AUDIT/'analysis.json')
    gate = c.read(OUT/'compile_gate.json'); metas = {meta['tag']:meta for meta in gate['records']}
    assert protocol['parent_compile_gate_SHA256'] == c.sha(c.encode(gate))
    assert raw['status'] == analysis['status'] == 'OK'
    assert raw['protocol_SHA256'] == analysis['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert not raw['GPU_allocated'] and not raw['compiler_intervention']
    assert raw['GPU_kernel_launches'] == raw['native_rebuilds'] == 0
    assert raw['native_SHA256'] == protocol['native_SHA256'] == c.NATIVE
    assert raw['backend_SHA256'] == protocol['backend_SHA256']
    assert raw['ccache_before'] == raw['ccache_after']
    assert analysis['raw_result_SHA256'] == c.sha((root/'result.json').read_bytes())
    assert len(raw['records']) == len(protocol['jobs']) == len(analysis['records']) == 4
    original_root = c.ROOT/gate['CPU_export_directory']
    checked = []
    for job,record,row in zip(protocol['jobs'],raw['records'],analysis['records']):
        assert job['tag'] == record['tag'] == row['tag'] and record['status'] == 'OK'
        assert job['expected_G8_metadata'] == metas[job['tag']]
        meta = record['metadata']; expected = metas[job['tag']]
        assert meta['JIT_attributes'] == job['exact_attributes']
        assert meta['JIT_signature'] == expected['JIT_signature'] and meta['JIT_constants'] == expected['JIT_constants']
        actual = program(root,meta); generic = program(original_root,expected)
        fields = {name:actual[name] == generic[name] for name in actual if name != 'CUBIN_SHA256'}
        assert all(fields.values()) and fields == row['checks'] and row['all_execution_fields_equal']
        assert actual == row['exact_batch64'] and generic == row['generic_G8']
        proof = dict(tag=job['tag'],checks=fields,all_execution_fields_equal=True)
        if job['variant']['name'].endswith('gluon_anchor'):
            frozen = job['G7_actual_G0']
            prior = program(c.ROOT/frozen['real_JIT_export_root'],frozen['real_JIT_metadata'])
            assert all(actual[name] == prior[name] for name in fields)
            assert row['G7_real_JIT_G0_execution_fields_equal']
            proof['G7_real_JIT_G0_execution_fields_equal'] = True
        checked.append(proof)
    assert len(protocol['reference_pairs']) == len(analysis['upstream_reference_pairs']) == 2
    upstream = []
    for pair,row in zip(protocol['reference_pairs'],analysis['upstream_reference_pairs']):
        assert pair['G8_upstream'] == metas[pair['group']+':actual:upstream']
        generic = program(original_root,pair['G8_upstream']); real = pair['G7_actual_U']
        actual = program(c.ROOT/real['real_JIT_export_root'],real['real_JIT_metadata'])
        fields = {name:actual[name] == generic[name] for name in actual if name != 'CUBIN_SHA256'}
        assert all(fields.values()) and row['all_execution_fields_equal']
        assert generic == row['generic_G8'] and actual == row['G7_real_JIT_actual']
        upstream.append(dict(group=pair['group'],checks=fields,all_execution_fields_equal=True))
    failure = AUDIT/'analysis_cli_failure.txt'
    assert "NameError: name 'json' is not defined" in failure.read_text()
    result = dict(status='OK',CPU_only=True,GPU_allocations=0,GPU_kernel_launches=0,native_rebuilds=0,
        protocol_SHA256=c.sha(c.encode(protocol)),analysis_SHA256=c.sha((AUDIT/'analysis.json').read_bytes()),
        raw_result_SHA256=c.sha((root/'result.json').read_bytes()),
        analysis_cli_failure_SHA256=c.sha(failure.read_bytes()),
        verification_source_SHA256=c.sha(Path(__file__).read_bytes()),
        records=checked,upstream_reference_pairs=upstream,reuse_G8_evidence_supported=True,
        correction='The original analysis wrote its JSON, then failed at a terminal-only json.dumps call. '
                   'Frozen source and failure are retained; this independent CPU verifier recomputes and asserts all four Gluon and two U execution-equivalence comparisons.')
    destination = AUDIT/'verification.json'
    if write: c.write_once(destination,result)
    if destination.exists(): assert c.read(destination) == result
    print(json.dumps(dict(status='OK',Gluon_execution_equivalence_pairs=4,upstream_execution_equivalence_pairs=2,
                          reuse_G8_evidence_supported=True),indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--write',action='store_true')
    verify(parser.parse_args().write)
