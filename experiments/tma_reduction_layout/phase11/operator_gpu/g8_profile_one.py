"""Compiler-free G8 NCU entry point using the validated family fixtures."""
import argparse
from . import common as c


def validate_fixture(job):
    assert len(job['metadata']) == 1
    assert job['mode'] in ['software', 'warm_application', 'cold_isolated']
    assert job.get('warmups', 8) == 8
    case = job['case']; meta = job['metadata'][0]
    assert meta['family'] == case['family']
    # The serving fixtures use one CTA/batch1 for instruction profiling and
    # the full batch8 grid for hardware profiling.  Do not apply an actual
    # batch1 specialization or a batch64-only alignment fact to another grid.
    if case['family'] == 'C06':
        batch = 1 if job['mode'] == 'software' else 8
        if meta['JIT_signature'].get('num_seqs') == 'constexpr':
            assert int(meta['JIT_constants']['(5,)']) == batch, (meta['tag'], batch)
        for name, divisor in meta.get('JIT_attributes', {}).get('(5,)', []):
            assert name == 'tt.divisibility' and batch % int(divisor) == 0, (meta['tag'], batch, name, divisor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = c.read(args.job); validate_fixture(job)
    if job['case']['family'] == 'C01':
        from .g3b_profile_one import main as execute
    else:
        assert job['case']['family'] in ['C02', 'C06']
        from .g3_profile_one import main as execute
    # Both existing entry points take exactly --root/--job from sys.argv.
    # Their own Device/Loaded checks, complete output checks, canaries and
    # PROFILE_BINDING output remain in effect, with the new G8 metadata.
    execute()


if __name__ == '__main__':
    main()
