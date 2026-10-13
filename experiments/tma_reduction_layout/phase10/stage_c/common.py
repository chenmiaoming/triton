"""Stage C evidence, kept separate from the immutable Stage A/B records."""
from experiments.tma_reduction_layout.phase10.common import ROOT, sha as sha, encode as encode, read, write_once as write_once

OUT = ROOT / 'experiments/tma_reduction_layout/results/phase10/stage_c'
SOURCE = ROOT / 'experiments/tma_reduction_layout/phase10/stage_c'
BASELINE = '67bccffd0a8f58f5bc9648735756a2623243ee5d'


def tag(case, variant):
    return case['case_id'] + ':' + variant


def protocol():
    return read(OUT / 'preregistration/protocol.json')


def grid(case, fraction=1):
    return (8192 // case['M']) * (8192 // case['N']) // fraction


def output_per_cta(case):
    if case['mode'] == 'copy':
        return case['M'] * case['N']
    return case['M'] if case['mode'] == 'innermost' else case['N']


def schedule(tags, invocation):
    return [tags[o:] + tags[:o] for r in range(10) for o in [((invocation - 1) * 17 + r * 13) % len(tags)]]
