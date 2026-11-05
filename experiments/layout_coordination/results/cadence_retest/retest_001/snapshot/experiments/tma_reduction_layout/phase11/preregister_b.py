"""Freeze the H100 layout matrix, timing/statistics and source before runs."""
import subprocess
from experiments.tma_reduction_layout.phase11 import common as c
from experiments.tma_reduction_layout.phase11.artifact_audit import FAMILIES


def main():
    draft = c.read(c.SOURCE/'draft_protocol.json')
    a = c.read(c.OUT/'stage_a/preregistration/revisions/3/protocol.json')
    snapshot = '/tmp/tma-phase11-stage-b-snapshot'
    cases = []
    for case in draft['cases']:
        m,n,w = case['BM'],case['BN'],case['num_warps']
        candidates = [{'variant':row['variant'],'layout':row['layout']['gluon_blocked_layout']}
                      for row in draft['candidate_layouts'] if row['geometry']==[m,n,w] and row['variant'].startswith('fixed_vec')]
        assert len(candidates)==4
        cases.append({'case_id':case['case_id'],'M':m,'N':n,'num_warps':w,'dtype':case['dtype'],
                      'path':case['path'],'op':'max','full_grid':case['full_grid'],'candidates':candidates})
    sources = c.source_snapshot(snapshot)
    protocol = {'stage':'B','date':'2026-10-15','status':'FROZEN_BEFORE_ARTIFACTS_AND_TIMING',
        'upstream':a['upstream'],'baseline':a['baseline'],'cases':cases,'targets':{'sm90':a['targets']['sm90']},
        'worker_source_SHA256':sources,'stage_A_analysis_SHA256':c.sha((c.OUT/'stage_a/analysis.json').read_bytes()),
        'stage_A_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'layout_control':'Gluon explicit BlockedLayout + smem.load; unmodified native compiler',
        'correctness':{'patterns':['quarter','random_seed42','NaN/inf/mixed-zero/all-negative-zero'],
                       'oracle':'Pairwise PyTorch fp32 fmax over 32 rows, complete output; pinned tl.max maxnum semantics',
                       'NaN_policy':'PropagateNan.NONE; compare mask and all non-NaN values',
                       'signed_zero':'record actual/reference f32 bits and zero signs; no fixed bit-sign oracle'},
        'timing':{'worker_invocations':7,'warmup_launches_per_binary':3,'samples_per_visit':5,
                  'fractions':[1,16],'sample_kernel_launches':1,'batch_divide':False,'graphs':False,
                  'gluon_orders':'2K Williams/cyclic and reverse rows for K unique physical candidates',
                  'bridge_orders':'4 alternating/reverse rows per two-binary bridge, 5 samples per visit',
                  'main_population':'four Gluon layouts; canonical bridge is a separate cohort',
                  'input':'finite quarter multiples (linear index % 97 - 48)/4',
                  'outcome_based_repeats':False,'fresh_processes':True,'physical_GPUs_independence_assumed':False},
        'statistics':{'unit':'worker median, paired log time ratios across 7 invocations',
                      'interval':'95% Bonferroni simultaneous t, df6, 6 unordered pairs in each 4-layout family',
                      'bridge_interval':'95% paired t, df6, descriptive within each separately reported bridge',
                      'practical_margin_percent':3,'global_significance_or_non_regression_claim':False,
                      'normal_log_ratio_and_independent_invocation_assumptions_reported':True,
                      'winner':'advantage lower bounds >3% against all candidates; otherwise possible-best set'},
        'reproduction':{'fingerprint_families':list(FAMILIES),'fingerprint_scope':'complete filtered whole-kernel sequence; includes output',
                        'input_layout_alone_establishes_equivalence':False,'NCU_bridge_repeats_per_binary':3,
                        'hardware_software_profiles_separate':True,'no_C1_C2_intervention_this_stage':True},
        'stop_after':'B; report and await discussion before C/D', 'PR_authorized':False,'PPT_modified':False}
    root = c.OUT/'stage_b/preregistration'
    c.write_once(root/'protocol.json',protocol)
    c.write_once(root/'worker_source_snapshot.zip',c.archive(snapshot))
    c.write_once(root/'draft_protocol.json',(c.SOURCE/'draft_protocol.json').read_bytes())
    print('Stage B frozen',len(cases),'conditions, 40 core identities, 7 worker invocations')


if __name__=='__main__':
    main()
