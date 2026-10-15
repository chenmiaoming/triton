"""Freeze the user-approved C1, including the corrected no-broadcast MR."""
import argparse
import subprocess
from experiments.tma_reduction_layout.phase11.c1 import common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--reason', default='Pinned Gluon has no gl.range; use supported runtime Python range(K). No timing data collected.')
    args = parser.parse_args()
    baseline = c.read(c.base.OUT / 'stage_a/preregistration/revisions/3/protocol.json')
    cases = []
    for m, n, w in ((32, 64, 8), (32, 128, 4)):
        for vec in (1, 2, 4, 8):
            tn = min(32, n // vec)
            wn = min(w, n // (vec * tn))
            layout = {'size_per_thread': [1, 1, vec], 'threads_per_warp': [1, 32 // tn, tn],
                      'warps_per_cta': [1, w // wn, wn], 'order': [2, 1, 0]}
            for family in ('ML', 'MR'):
                cases.append({'tag': f'M{m}_N{n}_w{w}:vec{vec}:{family}',
                    'M': m, 'N': n, 'num_warps': w, 'vec': vec, 'family': family,
                    'path': 'host', 'dtype': 'bfloat16', 'layout': layout,
                    'shared_layout': {'swizzle_byte_width': 128, 'element_bitwidth': 16,
                                      'rank': 3, 'transposed': False}})
    assert len(cases) == 16
    snapshot = '/tmp/tma-phase11-c1-snapshot'
    sources = c.snapshot(snapshot)
    protocol = {'stage': 'C1', 'date': '2026-10-15', 'status': 'FROZEN_BEFORE_GPU_EXECUTION',
        'user_authorization': 'H100; eight layouts; ML/MR; at most sixteen logical programs; stop after C1',
        'upstream': baseline['upstream'], 'baseline': baseline['baseline'],
        'target': {'gpu': 'H100!:1', 'cc': [9, 0]}, 'profile': 'chenmiaoming',
        'cases': cases, 'logical_program_limit': 16, 'K': [1, 16, 64, 256], 'D': 1,
        'input_shape': [1, 8192, 8192], 'worker_source_SHA256': sources,
        'MR': {'input_state': 'BF16; each element privately wraps at >=8 by -16, otherwise adds row-dependent quarter step',
               'step': '((local_m*13 + local_n*7) % 7 + 1)/4',
               'result': 'FP32 max over M, accumulated in native SliceLayout; one final global store',
               'reduction_result_feedback': False, 'explicit_smem_store_in_source': False,
               'fixture_costs_retained': ['private stepping', 'BF16 rounding', 'result accumulation', 'loop control'],
               'not_a_serial_reduction_result_dependency_chain': True},
        'ML': {'TMA_fills_before_loop': 2, 'slots': 'current tile and tile offset 128 in M-tile coordinate',
               'runtime_memdesc_view': 'slot=i&1', 'consumer': 'FP32 elementwise accumulator, full-tile final store',
               'reduction': False, 'extra_buffer_occupancy_limit_recorded': True},
        'correctness': {'patterns': ['quarter', 'random', 'boundary'], 'K': [1, 16, 64, 256],
                        'full_output_grid': 2, 'oracle': 'CPU numpy reference with round-to-nearest-even BF16 state',
                        'timing_worker_checks': 'all output for grid1; first two and last CTA for throughput grid'},
        'timing': {'worker_invocations': 3, 'samples_per_cell': 20, 'warmups_per_cell': 3,
                   'grids': ['1', '64*actual_SM_count'], 'occupancy': 'natural only in C1',
                   'one_kernel_per_event_sample': True, 'graphs': False, 'batch_divide': False,
                   'K_is_runtime_same_CUBIN': True, 'subprocess_or_JIT_in_timing': False,
                   'outcome_based_repeats': False, 'order': 'seeded shuffle, 4 visits x 5 samples, rotated by worker'},
        'profiles': {'K': [1, 16, 64, 256], 'grid': 1, 'repeats': 1,
                     'hardware_software_separate': True, 'report_limit': 128,
                     'source_section': 'InstructionStats',
                     'metrics': ['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
                                 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
                                 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
                                 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
                                 'smsp__inst_executed.sum'],
                     'gate': 'per-PC loop instructions strictly K times their K=1 counts; fixed setup/output separately identified',
                     'families': ['LDS/LDSM', 'STS', 'SHFL', 'BAR.SYNC', 'max', 'conversion'],
                     'no_profiler_duration_in_timing': True},
        'analysis': {'slopes': 'per-worker OLS T=a+b*K over all four K, plus all adjacent deltaT/deltaK',
                     'uncertainty': 'descriptive spread across three invocations; not independent devices',
                     'trend': 'relative slopes bML+bMR versus vec1 and adjacent vec2->vec4->vec8; compare existing host BF16 Stage B only',
                     'additivity_is_hypothesis': True, 'no_cost_model_fit_in_C1': True,
                     'intercepts_not_TMA_latency': True, 'unmatched_templates_or_loop_artifacts_reported': True},
        'stop_after': 'C1; user review before C2/C3/D', 'PR_authorized': False,
        'native_rebuild': False, 'old_draft_overrides': ['C1 layout list now all four natural vecs per geometry',
            'MR result-broadcast feedback removed; iteration recurrence is private input state only'],
        'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip()}
    target = c.OUT / 'preregistration'
    if args.revision > 1:
        protocol['revision'] = args.revision
        protocol['previous_protocol_SHA256'] = c.sha(c.encode(c.load_protocol()))
        protocol['technical_revision_reason'] = args.reason
        target = target / 'revisions' / str(args.revision)
    else:
        c.write_once(target / 'prior_manifest.json', {'files': c.freeze_prior()})
        for name in ('FOLLOWUP_PLAN.md', 'followup_review_matrix.json'):
            c.write_once(target / ('prior_draft_' + name), (c.base.SOURCE / name).read_bytes())
    if args.revision >= 4:
        protocol['profiles']['gate'] = 'Normalize PC to function offsets; retain every PC, group unrolled/tail copies of logical body by opcode; group totals strictly K times K=1, with K-invariant setup/output PCs separately audited'
        protocol['counter_analysis_revision'] = {'reason': 'Compiler emitted 4-way loop unrolling and a K=1 tail; per-PC copies do not individually scale with K, while logical body totals do.',
            'fixed_before_event_timing': True, 'new_GPU_runs_required': False,
            'kernel_or_layout_change': False, 'user_requested_K_fold_dynamic_counts_retained': True,
            'audit': 'analysis_revisions/1/expanded_PC_group_audit.json'}
    c.write_once(target / 'protocol.json', protocol)
    c.write_once(target / 'source_snapshot.zip', c.archive(snapshot))
    c.write_once(target / 'preregister_source.py', __import__('pathlib').Path(__file__).read_bytes())
    print('C1 frozen: 8 layouts, 16 programs, runtime K=1/16/64/256; no GPU calls')


if __name__ == '__main__':
    main()
