"""Freeze instrument plan, prior hashes, baseline native and worker source."""
from experiments.tma_reduction_layout.phase11 import common as c
from experiments.tma_reduction_layout.phase11.calibration import KINDS, BLOCKS


def main():
    old = c.read(c.ROOT/'experiments/tma_reduction_layout/results/phase10/stage_c/preregistration/protocol.json')
    snapshot = '/tmp/tma-phase11-stage-a-snapshot'
    sources = c.source_snapshot(snapshot)
    protocol = {'stage':'A', 'status':'FROZEN_BEFORE_GPU_COLLECTION', 'date':'2026-10-15',
                'upstream':'20d9d98fead78c2f65e1b3ba5a7881d86a836dad',
                'baseline':old['compilers']['baseline'], 'targets':old['targets'],
                'worker_source_SHA256':sources, 'fixtures':list(KINDS),
                'fixture_grid':BLOCKS,'fixture_threads':32,'matrix_values':'0x3f80 BF16 bits',
                'ordinary_contiguous_expected':'one conflict-free scalar shared read per warp',
                'ordinary_conflict32_expected':'32 distinct shared words on the same bank per warp',
                'matrix_expected':'one warp collective ldmatrix.x4 per warp; separate metric coverage test',
                'instrument_fixtures_are_compiler_reproducers':False,
                'hardware_and_software_passes_separate':True,'profile_durations_are_timing_samples':False,
                'max_infrastructure_retries_per_target':2,'outcome_based_retries':False,
                'missing_metrics_are_zero':False,'NCU_worker_UUID_required_from_driver':True,
                'scope_after_completion':'B H100 only; stop after B; C and D not executed this turn',
                'accepted_plan_SHA256':c.sha((c.SOURCE/'PLAN.md').read_bytes())}
    root = c.OUT/'stage_a/preregistration'
    c.write_once(root/'prior_manifest.json', {'HEAD':c.BASELINE,'files':c.prior_inventory()})
    c.write_once(root/'protocol.json', protocol)
    c.write_once(root/'worker_source_snapshot.zip', c.archive(snapshot))
    c.write_once(root/'accepted_plan.md', (c.SOURCE/'PLAN.md').read_bytes())
    print('Stage A frozen', len(sources), 'worker files', flush=True)


if __name__ == '__main__':
    main()
