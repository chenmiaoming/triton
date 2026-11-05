"""NCU entrypoint: checked frozen binaries, no compiler/JIT imports."""
import argparse
import ctypes as C
import json
import sys
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c2d.runtime import Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = c.read(args.job)
    session = Session(Path(args.root), job['metadata'],
                      maxgrid=max(row['grid'] for row in job['cells']),
                      maxk=max(row['K'] for row in job['cells']))
    records = []
    for cell in job['cells']:
        owner, params, _ = session.params(cell['tag'], cell['K'])
        kernel = session.kernels[cell['tag']]
        kernel.guard()
        kernel.launch(cell['grid'], params, session.stream, cell['q'])
    session.driver.call('cuStreamSynchronize', [C.c_void_p], session.stream)
    session.driver.call('cuProfilerStart', [])
    for cell in job['cells']:
        owner, params, serialized = session.params(cell['tag'], cell['K'])
        kernel = session.kernels[cell['tag']]
        kernel.guard()
        kernel.launch(cell['grid'], params, session.stream, cell['q'])
        session.driver.call('cuStreamSynchronize', [C.c_void_p], session.stream)
        check = session.check(cell['tag'], cell['K'], [0])
        assert check['pass'], check
        records.append({**cell, 'correctness': check, 'parameters': serialized,
                        'binding': kernel.binding()})
    session.driver.call('cuProfilerStop', [])
    imported = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
    assert not imported
    print('PROFILE_BINDING '+json.dumps({'records': records, 'environment': session.environment(),
          'Triton_imported': imported, 'compile_calls': 0,
          'driver_return_codes': sorted(session.driver.codes)}), flush=True)
    session.close()


if __name__ == '__main__':
    main()
