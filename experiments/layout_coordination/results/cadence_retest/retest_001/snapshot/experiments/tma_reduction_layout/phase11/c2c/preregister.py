"""Freeze calibration, controls and protected holdouts before compilation."""
import argparse
import shutil
import subprocess
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2c.history import audit, reserve
from experiments.tma_reduction_layout.phase11.c3.static import candidates

GEOMETRIES = ((8, 64, 8), (8, 128, 4), (8, 256, 4), (16, 64, 4),
              (16, 128, 4), (32, 64, 8), (32, 128, 4), (64, 64, 8))
RESOURCE_GEOMETRIES = ((128, 256, 8), (128, 512, 8))
SHARED = {'swizzle_byte_width': 128, 'element_bitwidth': 16,
          'rank': 3, 'transposed': False}


def cases():
    result = []
    for m, n, w in GEOMETRIES:
        for case in candidates(m, n, w):
            if case['placement'] != 'n_first' and m != 32:
                continue
            common = {'M': m, 'N': n, 'num_warps': w, 'D': 1, 'kind': 'composition',
                      'dtype': 'bfloat16', 'path': 'host', 'ABI_kind': 'descriptor_K',
                      'input_kind': 'bf16_matrix', 'layout': case['register_layout'],
                      'layout_id': case['id'], 'placement': case['placement'],
                      'vec': case['register_layout']['size_per_thread'][2],
                      'shared_layout': SHARED}
            for family in ('MC', 'FULL'):
                result.append({**common, 'family': family,
                               'tag': case['id'] + ':' + family + ':D1'})
            if m != 32 or case['placement'] != 'n_first':
                result.append({**common, 'family': 'ONE_FULL', 'original_once': True,
                               'one_pass': True, 'ABI_kind': 'descriptor',
                               'tag': case['id']+':ONE_FULL'})
        for family in ('TRANSFER', 'ROW', 'SCAFFOLD'):
            result.append({'tag': f'M{m}_N{n}_w{w}:control:{family}',
                           'M': m, 'N': n, 'num_warps': w, 'D': 1,
                           'kind': 'composition', 'family': family, 'dtype': 'bfloat16',
                           'path': 'host' if family != 'SCAFFOLD' else 'direct',
                           'ABI_kind': 'descriptor_K' if family != 'SCAFFOLD' else 'pointer_K',
                           'input_kind': 'bf16_matrix', 'shared_layout': SHARED})
        if m != 32:
            result.append({'tag': f'M{m}_N{n}_w{w}:control:ONE_TRANSFER',
                           'M': m, 'N': n, 'num_warps': w, 'D': 1,
                           'kind': 'composition', 'family': 'ONE_TRANSFER', 'dtype': 'bfloat16',
                           'path': 'host', 'ABI_kind': 'descriptor', 'original_once': True,
                           'one_pass': True, 'input_kind': 'bf16_matrix', 'shared_layout': SHARED})
    for r in (4, 8, 16, 32):
        for p in (1, 2, 4, 8, 16, 32):
            if p > r:
                continue
            columns = r // p
            for variant in ('native', 'wide', 'fixture', 'fixture_wide'):
                result.append({'tag': f'local_R{r}_p{p}_c{columns}:{variant}',
                    'kind': 'local', 'family': 'LOCAL', 'variant': variant,
                    'R': r, 'p': p, 'columns': columns, 'num_warps': 4, 'D': 1,
                    'dtype': 'bfloat16', 'path': 'direct', 'ABI_kind': 'pointer_K',
                    'input_kind': 'bf16_matrix', 'M': 1, 'N': 128*r,
                    'logical_axes': ['owner_tensor', 'local_reduction', 'local_columns'],
                    'logical_shape': [128, p, columns],
                    'layout': {'size_per_thread': [1, p, columns],
                               'threads_per_warp': [32, 1, 1],
                               'warps_per_cta': [4, 1, 1], 'order': [2, 1, 0]}})
    for m, n, w in RESOURCE_GEOMETRIES:
        for case in candidates(m, n, w):
            result.append({'tag': case['id']+':RESOURCE', 'kind': 'resource',
                'family': 'FULL', 'resource_only': True, 'M': m, 'N': n,
                'num_warps': w, 'D': 1, 'dtype': 'bfloat16', 'path': 'host',
                'ABI_kind': 'descriptor', 'one_pass': True, 'input_kind': 'bf16_matrix',
                'layout': case['register_layout'], 'layout_id': case['id'],
                'placement': case['placement'], 'vec': case['register_layout']['size_per_thread'][2],
                'shared_layout': SHARED})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--reason', default='Initial C2c authorized calibration')
    parser.add_argument('--refresh-cases', action='store_true')
    args = parser.parse_args()
    _, hashes = c.snapshot()
    dest = c.OUT / 'preregistration'
    if args.revision > 1:
        value = c.protocol()
        value = {**value, 'revision': args.revision, 'technical_reason': args.reason,
                 'worker_source_SHA256': hashes,
                 'previous_protocol_SHA256': c.sha(c.encode(value))}
        binding = c.OUT/'preregistration/C32_method_binding.json'
        if binding.exists():
            value['C32_method_binding_SHA256'] = c.sha(binding.read_bytes())
        if args.refresh_cases:
            old = value['cases']
            value['cases'] = cases()+[row for row in old if row.get('reuse_source') and row.get('original_once')]
            prior = {row['tag']: row for row in old if row.get('reuse_source')}
            for row in value['cases']:
                if row['tag'] in prior:
                    row.update({**prior[row['tag']], **row})
                    # Keep binary metadata, including reused C2a metadata.
                    for key in ('cubin_SHA256', 'function_name', 'dynamic_smem_bytes', 'global_scratch_size', 'tensordesc_meta', 'reuse_source'):
                        row[key] = prior[row['tag']][key]
        dest = dest / 'revisions' / str(args.revision)
    else:
        history = audit()
        protected = reserve(history)
        c.write_once(dest / 'history_audit.json', history)
        c.write_once(dest / 'D1_reservation.json', protected)
        value = {'stage': 'C2c', 'date': '2026-10-17', 'revision': 1,
            'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
            'authorization': 'User approved C2c composition/fixed-R supplementation and C3.2 model update; stop before D1 compilation/timing.',
            'pinned_upstream': c.PIN,
            'baseline': c.read(c.BASE/'results/phase11/stage_c2a/preregistration/protocol.json')['baseline'],
            'profile': 'chenmiaoming', 'target': {'GPU': 'H100!:1', 'cc': [9, 0], 'SM_count': 132},
            'worker_source_SHA256': hashes, 'cases': cases(),
            'calibration_geometries': GEOMETRIES, 'resource_geometries': RESOURCE_GEOMETRIES,
            'reserved_D1': protected['reserved_geometries'],
            'K': [1, 2, 4, 16, 64, 256], 'fit_K': [16, 64, 256],
            'timing': {'workers': 3, 'samples_per_cell': 12, 'visits': 3,
                'samples_per_visit': 4, 'warmups': 2, 'grid_SM_multipliers': [32, 64, 96],
                'order_seed': 20261017, 'local_q': ['natural', '1', '2', '4'],
                'composition_q': ['natural', '4'], 'single_kernel_graph': True,
                'full_footprint_control_K': [1], 'outcome_based_repeats': False},
            'correctness': {'patterns': ['quarter', 'random', 'boundary'], 'grid': 2},
            'profiles': {'software_K': [16, 64], 'hardware_K': 64,
                'local_q': ['natural', '1', '2', '4'], 'group_limit': 12,
                'metrics': ['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
                    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
                    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
                    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
                    'smsp__inst_executed.sum', 'sm__warps_active.avg.pct_of_peak_sustained_active',
                    'dram__bytes_read.sum', 'dram__bytes_write.sum',
                    'dram__throughput.avg.pct_of_peak_sustained_elapsed'],
                'duration_used_for_timing': False},
            'analysis': {
                'units': 'ns/tile/SM from grid slope; global launch intercept kept separate',
                'old_26_6_offset_used': False,
                'candidate_envelopes': ['max(control, MC)', 'max(control, B_tile + MC)',
                                       'nonnegative shared-feature full-service regression'],
                'identification': 'B_tile/transfer are separately reported only if matched controls and held-geometry tests support a common additive decomposition; otherwise retain a set/bound and declare nonidentification.',
                'local_semantics': 'All variants share BF16 state update, final-state observation and identical checksum consumer for each fixed (R,p,columns). Across p the intended local grouping changes; differences are net subgraph costs, not naked opcode latency.',
                'packed_contrast': 'native vs exact positive scale before wide max, with corresponding fixture vs fixture_wide controls; inspect final lowering and retain extra-operation costs.',
                'transfer_scope': 'Fresh-origin repeated TMA plus mandatory slot-reuse barrier; not pure hardware TMA latency. Full-footprint K1 controls measure small-BM behavior.',
                'cross_validation': 'Hold complete geometry out; include pairing within worker/q; never train on D1.',
                'selector_inputs': ['BM', 'BN', 'W', 'dtype', 'reduce_axis', 'register_layout', 'shared_layout'],
            }, 'native_rebuilds': 0, 'D1_compiled': False, 'D1_timed': False}
        protected_set = {tuple(geometry) for geometry in value['reserved_D1']}
        assert all((meta['M'], meta['N'], meta['num_warps']) not in protected_set
                   for meta in value['cases'] if meta['kind'] != 'local')
        old = c.read(c.BASE/'results/phase11/stage_c2a/artifact_gate.json')
        c16 = c.read(c.BASE/'results/phase11/stage_c16/artifact_gate.json')
        for tag, original in c16['metadata'].items():
            if original['family'] not in ('FULL', 'TMA'):
                continue
            family = 'ORIGINAL_FULL' if original['family'] == 'FULL' else 'ORIGINAL_TRANSFER'
            folder = c.BASE/'results/phase11/stage_c16/artifacts'/c16['accepted_attempt']/'export'/tag.replace(':', '/')
            value['cases'].append({**original, 'tag': tag+':original_once',
                'family': family, 'kind': 'composition', 'original_once': True,
                'ABI_kind': 'descriptor', 'input_kind': 'bf16_matrix', 'D': 1,
                'layout_id': f'M{original["M"]}_N{original["N"]}_w{original["num_warps"]}:vec{original.get("vec", 1)}:n_first',
                'placement': 'n_first', 'reuse_source': str(folder.relative_to(c.ROOT))})
        reuse = c.OUT / 'preregistration/reuse'
        for meta in value['cases']:
            if meta.get('reuse_source'):
                folder = c.ROOT/meta['reuse_source']
            elif meta['tag'] in old['metadata']:
                folder = c.BASE/'results/phase11/stage_c2a/cpu_preflight/attempt_4/export'/meta['tag'].replace(':', '/')
                meta.update({**old['metadata'][meta['tag']], **meta,
                             'reuse_source': str(folder.relative_to(c.ROOT))})
            else:
                continue
            assert folder.exists()
            target = reuse/meta['tag'].replace(':', '/')
            target.mkdir(parents=True)
            for path in folder.glob('kernel.*'):
                shutil.copyfile(path, target/path.name)
        c.write_once(dest/'reuse.zip', c.archive(reuse))
        c.write_once(dest/'prior_manifest.json', {'files': c.prior_manifest()})
    c.write_once(dest/'protocol.json', value)
    c.write_once(dest/'source_snapshot.zip', c.archive(c.snapshot()[0]))
    print('C2c frozen:', len(value['cases']), 'programs;',
          len(value['reserved_D1']), 'protected holdouts; revision', value['revision'])


if __name__ == '__main__':
    main()
