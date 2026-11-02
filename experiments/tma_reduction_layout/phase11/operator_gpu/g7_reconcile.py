"""Audit all actual-call artifacts, preserving literal attribute differences."""
from . import common as c
from .g7_prepare import OUT


def comparable(meta):
    return {k:v for k,v in meta['JIT_attributes'].items() if v}


def main():
    p = c.read(OUT/'preregistration/protocol_v2.json')
    frozen = {r['tag']:r for r in c.read(OUT/'compile_gate_v2.json')['records']}
    selected = {}; sources = []
    for attempt in [1,2]:
        folder = OUT/'binding'/f'attempt_{attempt:03}'/'export'
        raw = c.read(folder/'returned_result.json')
        assert raw['native_SHA256'] == raw['native_SHA256_after'] == c.NATIVE
        assert raw['backend_SHA256'] == raw['backend_SHA256_after'] == p['backend_SHA256']
        assert raw['ccache_before'] == raw['ccache_after']
        assert raw['environment']['cc'] == [9,0] and 'H100' in raw['environment']['name']
        sources.append(dict(path=str((folder/'returned_result.json').relative_to(c.ROOT)),
                            SHA256=c.sha((folder/'returned_result.json').read_bytes()), status=raw['status']))
        for row in raw['records']:
            if 'real_JIT_metadata' not in row:
                continue
            original = frozen[row['tag']]; actual = row['real_JIT_metadata']
            for name,digest in actual['artifact_SHA256'].items():
                path = folder/'real_jit'/actual['artifact_directory']/name
                assert c.sha(path.read_bytes()) == digest
            checks = {key:actual[key] == original[key] for key in [
                'cubin_SHA256','runtime_argument_names','JIT_signature','JIT_constants',
                'num_warps','dynamic_smem_bytes','global_scratch_size']}
            assert all(checks.values()), (row['tag'],checks)
            aa, bb = comparable(actual), comparable(original)
            compatible = aa == bb; reason = 'identical_nonempty_assumptions'
            if not compatible:
                # The real batch64 call exposes an additional true alignment
                # fact absent from the reused generic batch8 reference.
                assert original['family'] == 'C06' and row['batch'] == 64
                assert aa == {**bb, '(5,)':[['tt.divisibility',16]]}
                assert row['batch'] % 16 == 0
                compatible = True; reason = 'batch64_runtime_num_seqs_divisible_by_16; full_CUBIN_and_ABI_identical'
            selected[row['tag']] = dict(tag=row['tag'], match=True, checks=checks,
                literal_attributes_equal=actual['JIT_attributes'] == original['JIT_attributes'],
                attributes_compatible=compatible, attribute_basis=reason,
                attempt=attempt, real_JIT_metadata=actual,
                real_JIT_export_root=str((folder/'real_jit').relative_to(c.ROOT)))
    expected = {g['id']+':'+role for g in p['binding_groups'] for role in ['U','G0']}
    expected |= {g['id']+':'+role for g in p['training_cases'] for role in ['U','G0']}
    assert set(selected) == expected and len(selected) == 26
    result = dict(status='OK', records=[selected[tag] for tag in sorted(selected)],
        sources=sources, protocol_SHA256=c.sha(c.encode(p)), GPU_form_bindings=26,
        reused_existing_bindings=18, new_SGLang_bindings=8,
        timing_outcomes_used=False, source_or_candidate_changes=False)
    c.write_once(OUT/'binding_gate.json', result)
    print('G7 actual binding eligibility: 26/26; literal metadata differences retained')


if __name__ == '__main__':
    main()
