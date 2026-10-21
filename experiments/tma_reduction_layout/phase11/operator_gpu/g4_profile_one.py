"""One exact binary per NCU report, target-specific priming, no JIT imports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from . import common as c
from .driver import Device, Loaded
from .production_buffers import Buffers
from .production_data import make_data


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--job', required=True)
    args = parser.parse_args(); job = c.read(args.job); software = job['mode'] == 'software'
    data = make_data(job['case'], batch=job['batch'], length=job['length'])
    if software:
        for name, old in data['masks'].items():
            mask = np.zeros_like(old, dtype=bool)
            if name == 'Att_Out': mask[0, 0, 0, :] = old[0, 0, 0, :]
            elif name == 'Att_Lse': mask[0, 0, 0] = old[0, 0, 0]
            data['masks'][name] = mask
    device = Device(); buffers = Buffers(device, job['case'], data)
    kernel = Loaded(device, Path(args.root), job['metadata'])
    grid = (1, 1, 1) if software else data['grid']
    owners, params = kernel.parameters(buffers.arguments(kernel.meta))
    for _ in range(job['warmups']): kernel.guard(); kernel.launch(grid, params, device.stream)
    device.synchronize(); device.driver.call('cuProfilerStart', [])
    kernel.guard(); kernel.launch(grid, params, device.stream)
    device.synchronize(); device.driver.call('cuProfilerStop', [])
    check = buffers.check(stage2=False, include_inputs=True); assert check['pass'], check
    assert not any(n == 'triton' or n.startswith('triton.') for n in sys.modules)
    assert sorted(device.driver.codes) == [0]
    print('PROFILE_BINDING ' + json.dumps({'mode': job['mode'], 'environment': device.environment,
        'Triton_imported': False, 'compile_calls': 0, 'driver_return_codes': [0],
        'records': [{'tag': kernel.meta['tag'], 'grid': list(grid), 'correctness': check, 'binding': kernel.binding()}]}, allow_nan=False), flush=True)
    device.close()


if __name__ == '__main__': main()
