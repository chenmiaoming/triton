"""Validate the CPU-only M4 plan without importing a compiler or CUDA."""
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates
from .static import GEOMETRIES, CONSUMERS, plan


def main():
    root = c.BASE / 'results/phase11/stage_m4_preparation'
    frozen = c.read(root/'freeze.json')
    for name, digest in frozen['files'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest, name
    prior = c.read(root/'prior_manifest.json')['files']
    for name, digest in prior.items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest, name
    prediction = c.read(root/'operation_plans_v2.json')
    expected = [plan(case, consumer) for m,n,w in GEOMETRIES
                for case in candidates(m,n,w) for consumer in CONSUMERS]
    assert expected == prediction['plans'] and len(expected) == 88
    assert prediction['not_a_cost_model'] and not prediction['new_GPU_observations_used']
    print('M4 CPU VALIDATOR OK:', len(expected), 'plans;', len(prior), 'prior files unchanged; no GPU')


if __name__ == '__main__':
    main()
