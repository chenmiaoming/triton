"""Offline C1 gates and descriptive slope/counter analysis, without GPU calls."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import io
import json
import math
import re
import statistics
import zipfile

from experiments.tma_reduction_layout.phase11.c1 import common as c
from experiments.tma_reduction_layout.phase11.validate_a import csv_metrics

FAMILIES = {
    'shared_load': r'\bLDS(?:M)?(?:\.|\s)',
    'shared_store': r'\bSTS(?:\.|\s)',
    'shuffle': r'\bSHFL(?:\.|\s)',
    'barrier': r'\bBAR\.SYNC(?:\.|\s)',
    'max': r'\b(?:F|H)MNMX(?:2)?(?:\.|\s)',
    'conversion': r'\b(?:F2F|F2FP|F2I|I2F)(?:\.|\s)',
}


def protocol_for_digest(digest):
    paths = [c.OUT / 'preregistration/protocol.json']
    paths += list((c.OUT / 'preregistration/revisions').glob('*/protocol.json'))
    return next(protocol for path in paths if c.sha(c.encode(protocol := c.read(path))) == digest)


def artifact_gate():
    protocol = c.load_protocol()
    accepted, attempts = [], []
    for root in sorted((c.OUT / 'artifacts').glob('attempt_*/export')):
        result = c.read(root / 'result.json')
        attempts.append({'attempt': root.parent.name, 'status': result['status'], 'records': len(result['records'])})
        if result['status'] == 'OK':
            accepted.append((root, result))
    assert len(accepted) == 1, attempts
    root, result = accepted[0]
    assert len(result['records']) == 16
    original_protocol = protocol_for_digest(result['protocol_SHA256'])
    # Tool/runner dependency revisions may reuse identical already-checked
    # artifacts. Kernel, reference, cases and measurement semantics must match.
    for key in ('cases', 'upstream', 'baseline', 'K', 'D', 'ML', 'MR', 'correctness', 'timing'):
        assert original_protocol[key] == protocol[key], key
    assert {k:v for k,v in original_protocol['profiles'].items() if k != 'gate'} == {
            k:v for k,v in protocol['profiles'].items() if k != 'gate'}
    for name in ('kernels.py', 'data.py'):
        path = (c.SOURCE / name).relative_to(c.ROOT).as_posix()
        assert original_protocol['worker_source_SHA256'][path] == protocol['worker_source_SHA256'][path]
    env = result['environment']
    assert env['native_SHA256'] == protocol['baseline']['native_SHA256']
    assert env['ccache_before'] == env['ccache_after'] and env['native_rebuild'] is False
    assert env['cc'] == [9, 0] and 'H100' in env['name']
    metadata, resources, static = {}, {}, {}
    for case, record in zip(protocol['cases'], result['records']):
        assert case['tag'] == record['tag'] and record['status'] == 'OK'
        checks = record['correctness']
        assert len(checks) == 12 and all(row['pass'] for row in checks)
        assert {(row['pattern'], row['K']) for row in checks} == {
            (p, k) for p in protocol['correctness']['patterns'] for k in protocol['K']}
        folder = root / case['tag'].replace(':', '/')
        meta = c.read(folder / 'metadata.json')
        assert c.sha((folder / 'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
        occupancy = c.read(folder / 'occupancy.json')
        assert occupancy['queried_cubin_sha256'] == meta['cubin_SHA256']
        assert occupancy['blocks_per_sm_actual_dynamic_smem'] > 0
        assert occupancy['local_bytes'] == 0, ('spill/resource boundary; stop C1', case['tag'], occupancy)
        assert meta['global_scratch_size'] == 0
        ir = (folder / 'kernel.ttgir').read_text()
        expected = dict(zip(('sizePerThread', 'threadsPerWarp', 'warpsPerCTA', 'order'),
            (case['layout'][key] for key in ('size_per_thread', 'threads_per_warp', 'warps_per_cta', 'order'))))
        encodings = {}
        for name, body in re.findall(r'(#\w+)\s*=\s*#ttg.blocked<\{(.*?)\}>', ir):
            attrs = {key: ast.literal_eval(value) for key, value in re.findall(r'(\w+)\s*=\s*(\[[^]]*\])', body)}
            encodings[name] = attrs
        assert expected in encodings.values(), (case['tag'], expected, encodings)
        assert 'scf.for' in ir and 'ttg.local_load' in ir
        if case['family'] == 'MR':
            assert '"tt.reduce"' in ir
        else:
            assert '"tt.reduce"' not in ir
        ptx = (folder / 'kernel.ptx').read_text()
        opcodes = Counter()
        for line in ptx.splitlines():
            match = re.match(r'\s*(?:@!?%\w+\s+)?([\w.]+)\s+.*;', line)
            if match:
                opcodes[match[1]] += 1
        metadata[case['tag']], resources[case['tag']] = meta, occupancy
        static[case['tag']] = dict(opcodes)
    return {'status': 'OK', 'accepted_attempt': root.parent.name, 'metadata': metadata,
            'resources': resources, 'static_PTX_opcode_sites': static, 'attempts': attempts,
            'environment': env, 'full_output_correctness_checks': 192,
            'static_sites_are_not_hardware_counts': True}


def freeze_artifacts():
    gate = artifact_gate()
    c.write_once(c.OUT / 'artifact_gate.json', gate)
    root = c.OUT / 'artifacts' / gate['accepted_attempt'] / 'export'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for tag in gate['metadata']:
            for ext in ('ptx', 'cubin'):
                name = tag.replace(':', '/') + '/kernel.' + ext
                z.writestr(name, (root / name).read_bytes())
    c.write_once(c.OUT / 'frozen_payload.zip', buffer.getvalue())
    print('C1 artifact gate PASS: 16 binaries, 192 complete-output checks, no spill')


def source_rows(text):
    rows = list(csv.reader(io.StringIO(text)))
    h = next(i for i, row in enumerate(rows) if 'Source' in row and 'Instructions Executed' in row)
    head = rows[h]
    result = {}
    for row in rows[h + 1:]:
        if len(row) != len(head):
            continue
        value = dict(zip(head, row))
        if not value.get('Address') or not value['Source'].strip():
            continue
        count = value['Instructions Executed'].replace(',', '').strip()
        if count in ('', 'N/A', 'nan'):
            continue
        families = [name for name, pattern in FAMILIES.items() if re.search(pattern, value['Source'])]
        result[value['Address']] = {'absolute_PC': value['Address'], 'instruction': value['Source'].strip(),
            'count': int(count), 'families': families}
    assert result, 'NCU per-PC dynamic instruction data missing'
    # Each NCU subprocess loads the same CUBIN into a fresh CUDA context.
    # Normalize its function's first PC to zero; process VA is not a binary PC.
    base = min(int(pc, 16) for pc in result)
    return {f'0x{int(pc,16)-base:04x}': {**row, 'PC': f'0x{int(pc,16)-base:04x}'} for pc, row in result.items()}


def counter_gate():
    protocol = c.load_protocol()
    artifact = c.read(c.OUT / 'artifact_gate.json')
    accepted = [(p, c.read(p / 'result.json')) for p in sorted((c.OUT / 'profiles').glob('attempt_*/export'))
                if c.read(p / 'result.json')['status'] == 'OK']
    assert len(accepted) == 1
    root, result = accepted[0]
    assert len(result['records']) == 128 and all(r['pass'] for r in result['records'])
    assert result['formal_timing_sample'] is False
    collected_protocol = protocol_for_digest(result['protocol_SHA256'])
    for key in ('cases', 'upstream', 'baseline', 'K', 'D', 'ML', 'MR', 'correctness', 'timing'):
        assert collected_protocol[key] == protocol[key]
    assert {k:v for k,v in collected_protocol['profiles'].items() if k != 'gate'} == {
            k:v for k,v in protocol['profiles'].items() if k != 'gate'}
    assert result['payload_SHA256'] == c.sha((c.OUT / 'frozen_payload.zip').read_bytes())
    binaries, failures = {}, []
    for tag, meta in artifact['metadata'].items():
        measured = {}
        for k in protocol['K']:
            base = root / tag.replace(':', '/') / f'K{k}'
            for group in ('software', 'hardware'):
                folder = base / group
                collected = c.read(folder / 'collect.json')
                assert collected['returncode'] == 0
                line = next(line for line in collected['output'].splitlines() if line.startswith('PROFILE_BINDING '))
                binding = json.loads(line[len('PROFILE_BINDING '):])
                assert binding['binding']['CUBIN_SHA256'] == meta['cubin_SHA256']
                assert binding['binding']['loaded_CUBIN_SHA256'] == meta['cubin_SHA256']
                assert binding['binding']['launches'] == binding['binding']['sha_guards'] == 2
                assert binding['correctness']['pass'] and binding['K'] == k and binding['grid'] == 1
                assert binding['Triton_imported'] is False and binding['compile_calls'] == 0
                assert binding['environment']['gpu_uuid'] == result['environment']['gpu_uuid']
                assert len(list(folder.glob('report.ncu-rep*'))) == 1
            measured[k] = {'PCs': source_rows(c.read(base / 'software/source.json')['output']),
                           'hardware': {name: float(value.replace(',', '')) for name, value in
                            csv_metrics(c.read(base / 'hardware/import.json')['output']).items()
                            if name in protocol['profiles']['metrics']}}
            assert len(measured[k]['hardware']) == 5
        pc_keys = set(measured[1]['PCs'])
        assert all(set(measured[k]['PCs']) == pc_keys for k in protocol['K']), tag
        loop_pcs, fixed_pcs, other_pcs = [], [], []
        for pc in sorted(pc_keys):
            first = measured[1]['PCs'][pc]
            counts = {k: measured[k]['PCs'][pc]['count'] for k in protocol['K']}
            row = {**first, 'counts_by_K': counts}
            if len(set(counts.values())) == 1:
                fixed_pcs.append(row)
            else:
                loop_pcs.append(row)
                if not row['families']:
                    other_pcs.append(row)
        invalid_targets = []
        # Bound the K-dependent machine loops, including unrolled copies and
        # separate tails. Branch counts may be K/4 rather than K.
        pc_base = min(int(row['absolute_PC'], 16) - int(pc, 16) for pc, row in measured[1]['PCs'].items())
        loop_ranges = []
        for row in loop_pcs:
            if re.search(r'\bBRA\b', row['instruction']):
                counts = row['counts_by_K']
                if not (counts[256] > counts[16] and
                        (counts[64] - counts[16]) * 192 == (counts[256] - counts[64]) * 48):
                    continue
                targets = re.findall(r'0x[0-9a-fA-F]+', row['instruction'])
                if targets:
                    target, current = int(targets[-1], 16) - pc_base, int(row['PC'], 16)
                    if 0 <= target < current:
                        loop_ranges.append([target, current])
        assert loop_ranges, ('Runtime K loop backedge missing', tag)
        fixed_inside_loop = [row for row in fixed_pcs if row['families'] and row['count'] and
                             any(a <= int(row['PC'],16) <= b for a,b in loop_ranges)]
        invalid_targets += fixed_inside_loop
        summary = {}
        for name in FAMILIES:
            body = {k: sum(r['counts_by_K'][k] for r in loop_pcs if name in r['families']) for k in protocol['K']}
            exact = all(body[k] == k * body[1] for k in protocol['K'])
            if not exact:
                invalid_targets.append({'family': name, 'body_counts_by_K': body})
            summary[name] = {'loop_per_iteration_warp_instructions': body[1],
                'body_counts_by_K': body, 'strict_K_multiple': exact,
                'fixed_warp_instructions': sum(r['count'] for r in fixed_pcs if name in r['families']),
                'total_by_K': {k: sum(r['count'] for r in measured[k]['PCs'].values() if name in r['families']) for k in protocol['K']}}
        opcode_groups = {}
        for row in loop_pcs:
            if row['families']:
                tokens = row['instruction'].split()
                opcode = tokens[1] if tokens[0].startswith('@') else tokens[0]
                counts = opcode_groups.setdefault(opcode, {k: 0 for k in protocol['K']})
                for k in protocol['K']:
                    counts[k] += row['counts_by_K'][k]
        for opcode, counts in opcode_groups.items():
            if not all(counts[k] == k * counts[1] for k in protocol['K']):
                invalid_targets.append({'opcode': opcode, 'counts_by_K': counts})
        valid = not invalid_targets
        valid &= summary['shared_load']['loop_per_iteration_warp_instructions'] > 0 if meta['family'] == 'ML' else summary['max']['loop_per_iteration_warp_instructions'] > 0
        if meta['family'] == 'ML':
            valid &= summary['shared_store']['loop_per_iteration_warp_instructions'] == 0
            valid &= summary['shuffle']['loop_per_iteration_warp_instructions'] == 0
            valid &= summary['shared_load']['fixed_warp_instructions'] == 0
        if not valid:
            failures.append({'tag': tag, 'invalid_target_PCs': invalid_targets, 'summary': summary})
        hardware_slopes = {name: (measured[256]['hardware'][name] - measured[1]['hardware'][name]) / 255
                           for name in protocol['profiles']['metrics']}
        binaries[tag] = {'pass': bool(valid), 'families': summary, 'loop_PCs': loop_pcs, 'fixed_PCs': fixed_pcs,
            'other_iteration_PCs': other_pcs, 'hardware_by_K': {k: measured[k]['hardware'] for k in protocol['K']},
            'K_dependent_loop_PC_ranges': loop_ranges, 'fixed_target_PCs_inside_loop': fixed_inside_loop,
            'expanded_body_opcode_counts_by_K': opcode_groups,
            'hardware_delta_per_iteration': hardware_slopes,
            'wavefronts_are_not_all_bank_conflicts': True}
    return {'status': 'OK' if not failures else 'FAILED', 'accepted_attempt': root.parent.name,
            'binaries': binaries, 'failures': failures, 'report_count': 128,
            'environment': result['environment'], 'GPU_UUID': result['environment']['gpu_uuid'],
            'strict_K_check_scope': 'logical body opcode groups including every unrolled/tail PC; setup/output PCs K-invariant; normalized PC detail retained',
            'analysis_revision_protocol_SHA256': c.sha(c.encode(protocol)),
            'direct_shared_requests': 'unavailable, not inferred from vector width',
            'durations_used_for_timing': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['artifacts', 'counters'])
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    c.protect_prior()
    if args.mode == 'artifacts':
        if args.validate:
            assert artifact_gate() == c.read(c.OUT / 'artifact_gate.json')
        else:
            freeze_artifacts()
    else:
        gate = counter_gate()
        if args.validate:
            assert json.loads(c.encode(gate)) == c.read(c.OUT / 'counter_gate.json')
        else:
            c.write_once(c.OUT / 'counter_gate.json', gate)
        print('C1 counter gate', gate['status'], len(gate['binaries']), 'binaries', gate['report_count'], 'reports')
        if gate['failures']:
            print(json.dumps(gate['failures'], indent=2))


if __name__ == '__main__':
    main()
