"""Pre-freeze bounded neighboring-layout and real-call-site experiments."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import subprocess
from . import common as c

PIN = '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'


def out(stage):
    assert stage in ('g4a', 'g4b')
    return c.OUT / ('stage_' + stage)


def snapshot(stage):
    return Path('/tmp') / ('operator-gpu-' + stage + '-snapshot-v1')


def protocol_path(stage):
    revised = out(stage) / 'preregistration/protocol_v2.json'
    return revised if revised.exists() else out(stage) / 'preregistration/protocol.json'


def gate_path(stage):
    return out(stage) / 'compile_gate_v2.json'


def payload_path(stage):
    return out(stage) / 'compiled_payload_v2.zip'


def main(stage, revision=1):
    old = c.read(c.OUT / 'stage_g2/preregistration/protocol.json')
    gate = c.read(c.OUT / 'stage_g2/compile_gate.json')
    cases = {x['id']: x for x in old['geometries']}
    groups = []
    if stage == 'g4a':
        for geom, target in [('C02_n64_d128', 'vec4:n_first'), ('C02_n64_d64', 'vec1:n_first')]:
            groups.append({'id': geom, 'case': cases[geom], 'binding_batch': 4, 'binding_length': 129,
                'variants': [{'name': 'upstream', 'upstream': True},
                    {'name': 'gluon_anchor', 'pv': 'default', 'qk': 'default'},
                    {'name': 'pv_only', 'pv': target, 'qk': 'default'},
                    {'name': 'qk_only', 'pv': 'default', 'qk': target},
                    {'name': 'coordinated', 'pv': target, 'qk': target}],
                'workloads': [{'batch': 8, 'length': n} for n in (256, 2048)]})
        hypotheses = [
            'PV-only improvements or regressions need not survive a coordinated QK/PV layout change.',
            'A QK intervention also changes K load/reduction; a factorial interaction is not a pure conversion latency.',
            'Fewer communication instructions, wavefronts or registers individually need not mean a faster decode.',
            'The previous d128 vec4 positive and d64 vec1 negative remain named controls, with both upstream and Gluon anchors.']
    else:
        for geom, target in [('C06_s16_d128', 'vec1:n_first'), ('C06_s32_d256', 'vec4:within'),
                             ('C02_n64_d128', 'vec4:n_first'), ('C02_n64_d64', 'vec1:n_first')]:
            works = [(b, 8192) for b in (1, 8, 64)] if geom.startswith('C06') else [(1, 256), (8, 2048)]
            for batch, length in works:
                groups.append({'id': f'{geom}_b{batch}_l{length}', 'case': cases[geom],
                    'binding_batch': batch, 'binding_length': length,
                    'variants': [{'name': 'upstream', 'upstream': True},
                        {'name': 'gluon_anchor', 'pv': 'default', 'qk': 'default'},
                        {'name': 'target', 'pv': target, 'qk': 'default'}],
                    'old_B4_variants': {'upstream': 'upstream', 'gluon_anchor': 'gluon_anchor', 'target': target},
                    'workloads': [{'batch': batch, 'length': length}]})
        hypotheses = [
            'Actual per-batch JIT can change equal-to-one or divisibility specialization, even with the same algorithm.',
            'A real source call-site specialization may also yield an identical binary; retain this null outcome.',
            'Compare actual JIT and old B4 replay on the same worker/workload, preserving U-to-G0 differences.',
            'This is source-level kernel/pipeline evidence, not an installed-framework or whole-model benchmark.']
    names = list(old['worker_source_SHA256'])
    names += [p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.glob('g4_*.py')
              if p.name not in ('g4_analyze.py', 'g4_validate.py', 'g4_report.py')]
    hashes = c.snapshot(names, snapshot(stage))
    prior = ('stage_g2', 'stage_g3a', 'stage_g3b')
    manifest = {str(p.relative_to(c.ROOT)): c.sha(p.read_bytes()) for name in prior
                for p in sorted((c.OUT / name).rglob('*')) if p.is_file()}
    g3 = c.read(c.OUT / 'stage_g3a/preregistration/protocol.json')
    p = {'stage': stage, 'frozen_at_UTC': datetime.now(timezone.utc).isoformat(),
         'parent_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         'compiler': PIN, 'native_SHA256': c.NATIVE, 'native_rebuilds': 0,
         'target': 'H100!', 'profile': 'chenmiaoming', 'groups': groups,
         'baseline_plans': gate['plans'], 'worker_source_SHA256': hashes,
         'old_evidence_SHA256': manifest,
         'old_payload_SHA256': c.sha((c.OUT / 'stage_g2/compiled_payload.zip').read_bytes()),
         'hypotheses': hypotheses,
         'timing': {'workers': 3, 'visits': 5, 'samples': 31, 'warmups': 8, 'seed': 20261024,
                    'bootstrap_replicates': 4096, 'practical_threshold_percent': 3,
                    'no_result_dependent_repeats': True},
         'correctness': {'patterns': ['finite', 'quarter', 'masked'], 'all_outputs': True,
                         'canaries_bytes': 256, 'all_inputs_unchanged': True,
                         'FP32_rtol': .0002, 'FP32_atol': .00002, 'BF16_rtol': .008, 'BF16_atol': .0001},
         'mandatory_metrics': g3['mandatory_metrics'], 'optional_metrics': g3['optional_metrics'],
         'profile_modes': ['software', 'warm_application'], 'profile_length': 2048,
         'profile_batch': 8, 'profile_all_g4a_variants': stage == 'g4a',
         'official_documentation': g3['official_documentation'], 'terminology': 'default-layout optimization opportunity'}
    suffix = '_v2' if revision == 2 else ''
    if revision == 2:
        p['revision'] = 2
        p['revision_reason'] = 'Actual tensor-JIT gate rejected the CPU auxiliary-parameter ABI before any formal timing. Keep three zero auxiliary integers as runtime parameters, matching the original JIT signature.'
        p['prior_attempt_SHA256'] = {str(f.relative_to(c.ROOT)): c.sha(f.read_bytes())
            for f in out(stage).rglob('*') if f.is_file()}
    c.write_once(out(stage) / f'preregistration/source_snapshot{suffix}.zip', c.archive(snapshot(stage)))
    c.write_once(out(stage) / f'preregistration/protocol{suffix}.json', p)
    print(stage, 'frozen:', len(groups), 'binding groups;', len(manifest), 'prior files protected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['g4a', 'g4b'])
    parser.add_argument('--revision', type=int, choices=[1, 2], default=1)
    args = parser.parse_args(); main(args.stage, args.revision)
