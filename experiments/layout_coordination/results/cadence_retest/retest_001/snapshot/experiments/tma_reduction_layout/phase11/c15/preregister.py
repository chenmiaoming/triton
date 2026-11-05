"""Freeze C1.5 before compilation; eight existing ML and eight repaired MR."""
import argparse
import copy
import io
from pathlib import Path
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.c15 import common as c


def scaling_audit():
    old = c.read(c.prior.OUT / 'analysis.json')
    rows = []
    for row in old['trends']:
        if row['grid'] != 8448:
            continue
        grid = (8192 // 32) * (8192 // row['N'])
        factor = grid / row['grid']
        a = row['sum_of_median_slopes_us_per_iteration']['1'] * factor
        b = row['sum_of_median_slopes_us_per_iteration']['8'] * factor
        full = row['full_kernel_times_us']
        rows.append({'N': row['N'], 'full_CTA_count': grid, 'scaling_factor': factor,
            'scaled_vec1_component_us': a, 'scaled_vec8_component_us': b,
            'scaled_delta_us': b-a, 'assumed_HBM_base_us': 45,
            'predicted_vec1_us': 45+a, 'predicted_vec8_us': 45+b,
            'time_reduction_percent_of_vec8': 100*(b-a)/(45+b),
            'slowdown_percent_of_vec1': 100*(b-a)/(45+a),
            'observed_StageB_vec1_us': full['1'], 'observed_StageB_vec8_us': full['8'],
            'observed_StageB_delta_us': full['8']-full['1']})
    return {'source': 'C1 four-point OLS, not C1.5 results', 'rows': rows,
        'CTA_linear_scaling_measured': False, 'HBM_base_45us_measured': False,
        'interpretation': 'Arithmetic audit of proposed explanation only; neither linear scaling nor fixed-base additivity validated'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--reason', default='Initial approved C1.5')
    args = parser.parse_args()
    protocol = copy.deepcopy(c.prior.load_protocol())
    protocol.update({'stage': 'C1.5', 'status': 'FROZEN_BEFORE_C15_COMPILATION',
        'user_authorization': 'Correct private BF16 MR update and ML steady-K slope on the current eight layouts; report before C2',
        'new_logical_program_limit': 8, 'logical_program_limit': 16,
        'stop_after': 'C1.5, no C2/C3/D or PR', 'revision': args.revision,
        'parent_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()})
    protocol.pop('counter_analysis_revision', None)
    protocol.pop('previous_protocol_SHA256', None)
    protocol['technical_revision_reason'] = args.reason
    protocol['MR']['input_state'] = 'BF16 typed private state, step, threshold and wrap; update before widening/max'
    protocol['MR']['widening'] = 'updated BF16 state -> FP32 cast single-use directly by max'
    protocol['MR']['reference_order'] = 'update BF16 first, reduce updated state, accumulate FP32 result'
    protocol['MR']['template_gate'] = 'All eight max opcode histograms match original Stage B host BF16 Gluon; seven have BF16 max, N64 vec8 legitimately has no local BF16 combine'
    protocol['ML']['CUBIN_reused_from_C1'] = True
    protocol['timing']['worker_invocations'] = 3
    protocol['timing']['order'] = 'seeded shuffle, 4 visits x 5 samples, rotated by worker'
    protocol['analysis']['fit_K'] = [16, 64, 256]
    protocol['analysis']['slopes'] = 'per-worker OLS over K16/64/256 only; both steady adjacent slopes and residuals retained'
    protocol['analysis']['K1'] = 'separate measured T1 and reference T1-b; neither T1 nor the fitted intercept is pure HBM/TMA latency'
    protocol['analysis']['original_C1_OLS'] = 'recompute K>=16 slopes in C1.5 derived output without modifying C1'
    protocol['profiles']['report_limit'] = 64
    protocol['profiles']['new_profile_family'] = 'MR only; reuse exact-CUBIN ML counter records from C1'
    protocol['profiles']['reused_ML_reports'] = 64
    snapshot = '/tmp/tma-phase11-c15-snapshot'
    protocol['worker_source_SHA256'] = c.snapshot(snapshot)
    prior_gate = c.read(c.prior.OUT / 'artifact_gate.json')
    old_root = c.prior.OUT / 'artifacts' / prior_gate['accepted_attempt'] / 'export'
    protocol['reused_ML'] = {}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as z:
        for case in protocol['cases']:
            if case['family'] != 'ML':
                continue
            tag = case['tag']
            path = tag.replace(':','/')
            folder = old_root / path
            protocol['reused_ML'][tag] = {'CUBIN_SHA256': prior_gate['metadata'][tag]['cubin_SHA256'],
                'original_path': folder.relative_to(c.ROOT).as_posix(),
                'source_manifest_SHA256': c.sha((c.prior.OUT / 'raw_manifest.json').read_bytes())}
            for f in folder.iterdir():
                if f.is_file():
                    name = 'prior_' + f.name if f.name in ('metadata.json','attempt.json','occupancy.json') else f.name
                    z.writestr(path + '/' + name, f.read_bytes())
    target = c.OUT / 'preregistration'
    if args.revision == 1:
        c.write_once(target / 'prior_manifest.json', {'files': c.freeze_prior()})
        c.write_once(target / 'reuse_ml_artifacts.zip', payload.getvalue())
        c.write_once(target / 'C1_full_grid_hypothesis_audit.json', scaling_audit())
    else:
        protocol['previous_protocol_SHA256'] = c.sha(c.encode(c.load_protocol()))
        target = target / 'revisions' / str(args.revision)
    c.write_once(target / 'protocol.json', protocol)
    c.write_once(target / 'source_snapshot.zip', c.archive(snapshot))
    c.write_once(target / 'preregister_source.py', Path(__file__).read_bytes())
    print('C1.5 frozen: 8 new MR programs, 8 reused ML CUBINs, runtime K1/16/64/256, fit K>=16')


if __name__ == '__main__':
    main()
