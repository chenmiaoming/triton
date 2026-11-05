"""Actual Tensor-JIT baseline gates and bounded candidate correctness; no timing."""
import inspect
from pathlib import Path
import subprocess
import traceback

from . import common as c
from .candidate_compile import _audit_ir, _resources
from .elf import elf
from .export import export
from .fixtures import TorchFixture


def run(protocol, root):
    import torch
    import triton
    import triton._C.libtriton as native
    from triton import knobs
    from triton.backends.nvidia.compiler import CUDABackend

    root = Path(root)
    c.verify_sources(protocol)
    assert protocol['formal_timing'] is False and protocol['timing_events_created'] == 0
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE == protocol['native_SHA256']
    backend = Path(inspect.getfile(CUDABackend))
    assert c.sha(backend.read_bytes()) == protocol['backend_SHA256']
    assert knobs.runtime.add_stages_inspection_hook is None
    torch.set_num_threads(2)
    torch.cuda.init()
    assert torch.cuda.get_device_capability() == (9, 0)
    result = dict(status='FAILED', phase=protocol['phase'], cases=protocol['cases'],
        records=[], programs=[], checks=[], plans={}, GPU_allocated=True,
        formal_timing=False, timing_events_created=0, native_rebuilds=0,
        compiler_intervention=False, protocol_SHA256=c.sha(c.encode(protocol)),
        native_SHA256=c.NATIVE, backend_SHA256=protocol['backend_SHA256'],
        compiler_version=triton.__version__, GPU=torch.cuda.get_device_name(),
        CUDA_device_UUID=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        ccache_before=subprocess.check_output(['ccache', '--print-stats'], text=True))

    for case in protocol['cases']:
        row = dict(case=case['id'], status='FAILED', compiled_programs=[], checks=[])
        try:
            adapter = c.adapter(case)
            original, translated = adapter.upstream(case), adapter.gluon(case)
            assert translated.arg_names[:len(original.arg_names)] == original.arg_names
            options = dict(num_warps=case['W'], num_stages=case['stages'])
            spec = adapter.fixture(case, 'finite')
            buffers = TorchFixture(spec)
            fixture_metadata = buffers.metadata()
            arguments = buffers.arguments()
            u_kwargs = {name: arguments[name] for name in original.arg_names}
            u_code = original.warmup(**u_kwargs, grid=buffers.grid, **options)
            upstream = export(u_code, original, case, case['id'] + ':U', root, spec['constants'])
            upstream.update(case=case, variant='U', fixture=fixture_metadata,
                            compiler_options=options, timing_eligible=case['followup_role']=='footprint')
            plan = adapter.derive_plan(u_code.asm['ttgir'], case)
            result['plans'][case['id']] = plan
            g0_extra = adapter.layout_constants(case, plan, 'G0')
            g0_arguments = {**arguments, **g0_extra}
            g0_kwargs = {name: g0_arguments[name] for name in translated.arg_names}
            g0_code = translated.warmup(**g0_kwargs, grid=buffers.grid, **options)
            g0 = export(g0_code, translated, case, case['id'] + ':G0', root,
                        {**spec['constants'], **g0_extra})
            g0.update(case=case, variant='G0', fixture=fixture_metadata,
                      compiler_options=options, timing_eligible=case['followup_role']=='footprint')
            u_folder, g_folder = [root / meta['artifact_directory'] for meta in [upstream, g0]]
            same = elf((u_folder/'kernel.cubin').read_bytes()) == elf((g_folder/'kernel.cubin').read_bytes())
            abi = all(upstream[key] == g0[key] for key in
                      ['runtime_argument_names', 'num_warps', 'dynamic_smem_bytes', 'global_scratch_size'])
            resource_match = _resources(u_folder) == _resources(g_folder)
            for meta, folder in [(upstream, u_folder), (g0, g_folder)]:
                meta['resources'] = _resources(folder)
                meta['effective_layout_audit'] = _audit_ir(folder)
                result['programs'].append(meta)
            row.update(upstream=upstream, metadata=g0, allocated_sections_equal=same,
                       ABI_and_launch_equal=abi, resources_equal=resource_match)
            programs = [('U', original, {}, upstream), ('G0', translated, g0_extra, g0)]
            # Baseline mismatch prevents candidate compilation/execution. U/G0
            # correctness is still retained to diagnose any translation issue.
            if same and abi and resource_match:
                choices = protocol['candidate_definitions'][case['id']]
                if case['followup_role'] == 'footprint':
                    choices = [choice for choice in choices if choice['name'] == case['selected_variant']]
                    assert len(choices) == 1, case
                else:
                    assert case['selected_variant'] == 'all_eight_N01' and len(choices) == 8
                for choice in choices:
                    extra = adapter.layout_constants(case, plan, choice)
                    current = {**arguments, **extra}
                    kwargs = {name: current[name] for name in translated.arg_names}
                    code = translated.warmup(**kwargs, grid=buffers.grid, **options)
                    variant = 'GL_' + choice['name']
                    metadata = export(code, translated, case, case['id'] + ':' + variant,
                                      root, {**spec['constants'], **extra})
                    metadata.update(case=case, variant=variant, choice=choice,
                        fixture=fixture_metadata, compiler_options=options,
                        timing_eligible=case['followup_role']=='footprint')
                    folder = root / metadata['artifact_directory']
                    metadata['resources'] = _resources(folder)
                    metadata['effective_layout_audit'] = _audit_ir(folder,
                        {name: repr(value) for name, value in extra.items()})
                    assert metadata['runtime_argument_names'] == upstream['runtime_argument_names']
                    result['programs'].append(metadata)
                    programs.append((variant, translated, extra, metadata))
            for pattern in protocol['correctness']['patterns']:
                if pattern != 'finite':
                    del buffers, spec, arguments, u_kwargs, g0_arguments, g0_kwargs
                    torch.cuda.empty_cache()
                    spec = adapter.fixture(case, pattern)
                    buffers = TorchFixture(spec)
                    arguments = buffers.arguments()
                    # Keep cleanup variables defined for the next pattern;
                    # they never participate in compilation or launch here.
                    u_kwargs = g0_arguments = g0_kwargs = None
                for variant, function, extra, metadata in programs:
                    buffers.restore()
                    current = {**arguments, **extra}
                    kwargs = {name: current[name] for name in function.arg_names}
                    code = function[buffers.grid](**kwargs, **options)
                    torch.cuda.synchronize()
                    assert c.sha(code.asm['cubin']) == metadata['cubin_SHA256'], 'Pattern changed Tensor-JIT specialization'
                    check = buffers.check()
                    entry = dict(case=case['id'], variant=variant, tag=metadata['tag'],
                                 pattern=pattern, **check)
                    result['checks'].append(entry)
                    row['checks'].append(dict(variant=variant, pattern=pattern, pass_=check['pass']))
                    assert check['pass'], entry
                    print('N5 correctness', case['id'], variant, pattern, 'PASS', flush=True)
            row['compiled_programs'] = [metadata['tag'] for _, _, _, metadata in programs]
            row['status'] = 'OK' if same and abi and resource_match else 'MISMATCH'
            del buffers, spec, arguments, current, kwargs, u_kwargs, g0_arguments, g0_kwargs
            torch.cuda.empty_cache()
        except Exception:
            row['error'] = traceback.format_exc()
            print(row['error'], flush=True)
        result['records'].append(row)
        c.write_once(root / (case['id']+'_checkpoint.json'), row)
        print('N5 baseline/candidates', case['id'], row['status'], flush=True)

    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    assert c.sha(backend.read_bytes()) == protocol['backend_SHA256']
    assert knobs.runtime.add_stages_inspection_hook is None
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    result['status'] = 'OK' if all(row['status']=='OK' for row in result['records']) else 'NEEDS_ALIGNMENT'
    result['counts'] = dict(cases=len(result['records']), programs=len(result['programs']),
                            full_correctness_checks=len(result['checks']),
                            passing_checks=sum(check['pass'] for check in result['checks']))
    result['timing_case_ids'] = [case['id'] for case in protocol['cases'] if case['followup_role']=='footprint']
    if result['status'] == 'OK':
        assert result['counts'] == dict(cases=3, programs=16, full_correctness_checks=48, passing_checks=48)
    c.write_once(root / 'result.json', result)
    return result
