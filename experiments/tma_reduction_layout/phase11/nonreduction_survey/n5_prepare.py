"""Freeze the bounded footprint and tail/stride follow-up before GPU binding."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from . import common as c
from . import quant_adapter


def cases():
    templates = {case['family']: case for case in quant_adapter.cases()}
    scale = copy.deepcopy(templates['N01'])
    scale.update(id='N01_m4096_n8192', rows=4096, cols=8192,
                 followup_role='footprint', selected_variant='coordinated_fully_coalesced',
                 evidence_scope='Same pinned scale-rearrangement kernel at a larger synthetic footprint; no production dispatch claim.')
    awq = copy.deepcopy(templates['N02'])
    awq.update(id='N02_k8192_p1024_g128', rows=8192, packed_cols=1024,
               cols=8192, group_size=128, followup_role='footprint',
               selected_variant='packed_compute_vec1',
               evidence_scope='Same pinned AWQ dequantization kernel at a larger footprint; not AWQ GEMM or whole-model performance.')
    tail = copy.deepcopy(templates['N01'])
    tail.update(id='N01_tail_stride_m131_n7', rows=131, cols=7,
                row_stride=19, col_stride=2, followup_role='correctness_only',
                selected_variant='all_eight_N01',
                evidence_scope='Padded tail and injective positive-stride correctness stress; no performance claim.')
    return [scale, awq, tail]


def run(revision):
    stage = c.OUT / 'stage_n5' / f'revision_{revision:03}'
    assert not stage.exists(), stage
    names = [str(path.relative_to(c.ROOT)) for path in c.SOURCE.rglob('*')
             if path.is_file() and '__pycache__' not in path.parts]
    names += [str((c.old.SOURCE / name).relative_to(c.ROOT)) for name in
              ['common.py', 'driver.py', 'layout_parsing.py']]
    names += ['experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    snapshot = Path('/tmp') / f'nonreduction-n5-revision-{revision:03}'
    hashes = c.old.snapshot(names, snapshot)
    previous = c.read(c.old.OUT / 'stage_g8/preregistration/protocol.json')
    protected_root = c.OUT / 'stage_n2_compile_r2'
    reference_path = protected_root / 'compile/attempt_001/invocation_-01/export/result.json'
    reference = c.read(reference_path)
    assert reference['status'] == 'OK'
    definitions = {}
    for case in cases():
        reference_id = 'N01_m4096_n128' if case['family']=='N01' else 'N02_k3584_p448_g128'
        # Freeze the already declared candidate definitions, including aliases
        # of a new case's default. The old development adapter deliberately
        # drops equal-default choices and assumes its original cardinality.
        choices = quant_adapter.variants(case, reference['plans'][reference_id])
        if case['followup_role']=='footprint':
            choices = [choice for choice in choices if choice['name']==case['selected_variant']]
            assert len(choices)==1
        else:
            assert len(choices)==8
        definitions[case['id']] = choices
    protocol = dict(stage='stage_n5', revision=revision,
        phase='N5_footprint_and_tail_binding', profile='chenmiaoming',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        compiler=previous['compiler'], native_SHA256=c.NATIVE,
        backend_SHA256=previous['backend_SHA256'], source_SHA256=hashes,
        snapshot=str(snapshot), cases=cases(),
        candidate_definitions=definitions,
        candidate_reference_path=str(reference_path.relative_to(c.ROOT)),
        candidate_reference_SHA256=c.sha(reference_path.read_bytes()),
        candidate_alias_policy='Retain all original declared choices, even when one equals a new case default; no runtime-driven layout selection.',
        authorization='Bounded adaptive follow-up authorized by parent from user-approved GPU follow-ups; no formal timing in this stage.',
        selected_candidates={case['id']: case['selected_variant'] for case in cases()},
        correctness=dict(patterns=['finite', 'structured', 'edge'], full_outputs=True,
            guards_each_storage_bytes=256, readonly_and_storage_gaps=True,
            padded_output=True, preserve_strides=True, restore_before_each_launch=True),
        baseline_gate='Real Tensor U/G0 JIT; allocated ELF, runtime ABI, launch/shared resources, and all three full oracle/guard checks must agree.',
        formal_timing=False, timing_events_created=0,
        interpretation='Footprint and masks/strides stress only; cache residency, L2 and DRAM regimes require NCU and are not inferred from allocation sizes.',
        protected_SHA256={str(path.relative_to(c.ROOT)): c.sha(path.read_bytes())
            for path in protected_root.rglob('*') if path.is_file()})
    c.write_once(stage / 'preregistration/protocol.json', protocol)
    c.write_once(stage / 'preregistration/source_snapshot.zip', c.archive(snapshot))
    print('N5 frozen', revision, len(protocol['cases']), 'cases; 16 programs / 48 checks; no formal timing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, required=True)
    run(parser.parse_args().revision)
