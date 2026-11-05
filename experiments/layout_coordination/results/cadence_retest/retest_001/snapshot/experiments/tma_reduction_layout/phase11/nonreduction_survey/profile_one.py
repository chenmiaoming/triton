"""One frozen binary per NCU process; no Triton or compiler imports.

Software instruction profiling launches CTA (0,0,0) with the original complete
arguments.  A logical output mask checks that CTA's values while HostFixture
also verifies every inactive storage byte and both allocation canaries.
"""
import argparse
import json
from pathlib import Path
import sys

from . import common as c
from .fixtures import DriverFixture
from ..operator_gpu.driver import Device, Loaded


MODES = ('software', 'warm_application', 'cold_isolated')


def validate_fixture(job):
    assert job['mode'] in MODES
    assert job.get('warmups', 8) == 8
    assert len(job['metadata']) == 1
    case, meta = job['case'], job['metadata'][0]
    assert case['family'] in c.ADAPTERS
    assert meta['family'] == case['family'] and meta['geometry'] == case['id']
    assert meta['global_scratch_size'] == 0
    return case, meta


def software_spec(spec, case):
    """Only change the oracle's writable mask; never resize a runtime tensor."""
    import torch
    masks = {name: torch.zeros_like(spec['expected'][name], dtype=torch.bool)
             for name in spec['outputs']}
    family = case['family']
    if family == 'N01':
        assert set(masks) == {'output_ptr'}
        masks['output_ptr'].reshape(-1)[:512] = True
    elif family == 'N02':
        assert set(masks) == {'result_ptr'}
        masks['result_ptr'][:32, :256] = True
    elif family == 'N03' and case['operator'] == 'flash_rope':
        assert set(masks) == {'OUT'}
        masks['OUT'][0, :case['BLOCK_M'], :case['BLOCK_H'], :] = True
    elif family == 'N03':
        assert case['operator'] == 'mrope' and set(masks) == {'q_ptr', 'k_ptr'}
        masks['q_ptr'][0, :] = True
        masks['k_ptr'][0, :] = True
    elif family == 'N04':
        assert set(masks) == {'q_fp8', 'k_fp8', 'v_fp8'}
        masks['q_fp8'][0, :64, :] = True
        masks['k_fp8'][0, :64, :] = True
        masks['v_fp8'][0, :, :64] = True
    elif family == 'N05':
        assert set(masks) == {'yg'} and case['BT'] is not None
        masks['yg'][:case['BT'], 0, :] = True
    else:
        assert family == 'N06' and set(masks) == {'key_cache_ptr', 'value_cache_ptr'}
        assert not spec['constants']['HAS_SWA']
        slot = int(spec['tensors']['slot_mapping_ptr'][0])
        assert slot >= 0, 'The finite profiling fixture must execute CTA zero'
        block, offset = divmod(slot, int(spec['scalars']['block_size']))
        masks['key_cache_ptr'][block, offset, :4, :] = True
        masks['value_cache_ptr'][block, offset, :4, :] = True
    for name, mask in masks.items():
        if name in spec.get('masks', {}):
            mask &= spec['masks'][name]
        assert bool(mask.any()), (family, name)
    # Preserve the original storage objects and their aliases.  deepcopy would
    # risk changing physical Q/K/V alias provenance in the N04 fixture.
    return {**spec, 'masks': masks}


def execute(artifact_root, job):
    case, meta = validate_fixture(job)
    assert not any(name=='triton' or name.startswith('triton.') for name in sys.modules)
    original = c.adapter(case).fixture(case, 'finite')
    software = job['mode'] == 'software'
    spec = software_spec(original, case) if software else original
    # No scalar, stride, size, constexpr or buffer changes accompany grid=1.
    assert spec['tensors'] is original['tensors']
    assert spec.get('scalars') is original.get('scalars')
    assert spec['constants'] is original['constants']
    device = None
    profiler_started = False
    try:
        device = Device()
        buffers = DriverFixture(spec, device)
        kernel = Loaded(device, Path(artifact_root), meta)
        owners, params = kernel.parameters(buffers.arguments(meta))
        grid = (1, 1, 1) if software else buffers.grid
        for _ in range(8):
            buffers.reset(only_inplace=True)
            kernel.guard()
            kernel.launch(grid, params, device.stream)
            device.synchronize()
        device.synchronize()
        # In-place inputs must describe the same operation in the target
        # launch as in every warmup; restoration stays outside the NCU region.
        buffers.reset(only_inplace=True)
        device.synchronize()
        device.driver.call('cuProfilerStart', [])
        profiler_started = True
        kernel.guard()
        kernel.launch(grid, params, device.stream)
        device.synchronize()
        device.driver.call('cuProfilerStop', [])
        profiler_started = False
        check = buffers.check()
        assert check['pass'], check
        assert all(check['guards'])
        assert not any(name=='triton' or name.startswith('triton.') for name in sys.modules)
        assert sorted(device.driver.codes) == [0]
        binding = kernel.binding()
        assert binding['cubin_SHA256'] == binding['loaded_SHA256'] == meta['cubin_SHA256']
        masks = spec.get('masks', {})
        mask_evidence = {name: dict(elements=int(mask.sum()), shape=list(mask.shape),
            SHA256=c.sha(mask.contiguous().numpy().tobytes())) for name, mask in masks.items()}
        result = dict(mode=job['mode'], warmups=8, compile_calls=0, Triton_imported=False,
            profiler_attached=True, formal_timing=False, driver_return_codes=[0],
            environment=device.environment,
            records=[dict(tag=meta['tag'], grid=list(grid), original_grid=list(buffers.grid),
                correctness=check, binding=binding, fixture=buffers.metadata(),
                software_active_masks=mask_evidence,
                inplace_restored_before_each_launch=True,
                runtime_arguments_unchanged_from_full_fixture=True)])
        print('PROFILE_BINDING '+json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return result
    finally:
        if device is not None:
            if profiler_started:
                device.driver.call('cuProfilerStop', [])
            device.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    execute(args.root, c.read(args.job))


if __name__ == '__main__':
    main()
