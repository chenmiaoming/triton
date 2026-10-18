"""Refine historical exposure without treating a proposed pool as execution."""
import re
from experiments.tma_reduction_layout.phase11.c2c import common as c

TOKEN = re.compile(r'M(\d+)_N(\d+)_w(\d+)')


def audit():
    old = c.read(c.BASE / 'results/phase11/stage_c3/history_audit.json')
    supplemental = c.read(c.BASE / 'results/phase11/stage_c3/supplemental_source_audit.json')
    literal_hits = supplemental['hits']
    phase2 = c.read(c.BASE / 'results/phase2/sweep/results.json')
    phase2_proof = {}

    def visit(value):
        if isinstance(value, dict):
            for name, row in value.items():
                if TOKEN.fullmatch(name) and isinstance(row, dict):
                    candidates = row.get('candidates', {})
                    proofs = [key for key, candidate in candidates.items()
                              if candidate.get('raw', {}).get('raw_samples_us') or
                              candidate.get('raw', {}).get('ptx_sha256')]
                    if proofs:
                        phase2_proof[name] = proofs
                visit(row)
        elif isinstance(value, list):
            for row in value:
                visit(row)

    visit(phase2)
    rows = []
    for original in old['pool']:
        row = dict(original)
        m, n, w = row['BM'], row['BN'], row['W']
        hard = [name for name in row['evidence'] if
                name.endswith(('.ptx', '.cubin', '.sass', '.ncu-rep', '.ttgir', '.llir')) or
                '/cpu_preflight/' in name or '/canonical/' in name or '/artifacts/' in name]
        literals = literal_hits.get('_'.join(map(str, (m, n, w))), [])
        if hard or row['case'] in phase2_proof:
            category = 'executed_or_inspected'
        elif row['evidence'] or literals:
            category = 'unexecuted_code_only'
        else:
            category = 'strict_zero_mention'
        row.update(category=category, binary_or_inspection_proof=hard,
                   phase2_sample_or_PTX_proof=phase2_proof.get(row['case'], []),
                   source_literal_evidence=literals, R=m*n//(32*w),
                   core_domain=m in (8, 16, 32, 64) and n >= 64 and
                               w in (4, 8) and m*n//(32*w) in (4, 8, 16, 32))
        if category == 'unexecuted_code_only':
            # This is more precise than "literal only": record prior analytic
            # legality/layout predictions; they are not compiled observations.
            row['prior_static_rule_prediction'] = any(
                'exclusions' in name or 'eligibility' in name for name in row['evidence'])
        rows.append(row)
    return {
        'cutoff': 'Pre-C3 history snapshot; C3/C3.1 prediction/preregistration mentions are design records, not GPU exposure.',
        'original_audit_SHA256': c.sha((c.BASE / 'results/phase11/stage_c3/history_audit.json').read_bytes()),
        'source_literal_audit_SHA256': c.sha((c.BASE / 'results/phase11/stage_c3/supplemental_source_audit.json').read_bytes()),
        'scope': old['scope'], 'rows': rows,
        'classification': 'Any compiled artifact/NCU or historical raw timing/PTX evidence excludes from blind testing. Code-only records retain analytic-rule flags.',
        'outside_recorded_history': 'Unrecorded experiments cannot be ruled out.',
    }


def reserve(value):
    legacy = c.read(c.BASE / 'results/phase11/stage_c31/preregistration/protocol.json')['blind_cases']
    geometries = {(row['BM'], row['BN'], row['W']) for row in legacy}
    # All empirically unexposed legal core combinations, not chosen by a model
    # score. Preserve all legacy boundaries rather than discarding abstentions.
    core = [row for row in value['rows'] if row['core_domain'] and
            row['category'] != 'executed_or_inspected']
    geometries.update((row['BM'], row['BN'], row['W']) for row in core)
    rows = [row for row in value['rows'] if
            (row['BM'], row['BN'], row['W']) in geometries]
    assert all(row['category'] != 'executed_or_inspected' for row in rows)
    return {
        'reserved_geometries': [[row['BM'], row['BN'], row['W']] for row in rows],
        'rows': rows,
        'selection': 'All unexposed core combinations plus all eight legacy holdouts; no timing/selector score used.',
        'coverage_limit': 'No empirically unexposed R=8 or BM=32/64 core combinations remain under the recorded history. Balanced retrospective coverage must be reported separately, never called blind.',
        'no_D1_compilation_or_timing': True,
    }
