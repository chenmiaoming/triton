"""Real H100 tensor JIT binding for the frozen U/source-aligned G0 pair.

This stage only calls JIT warmup: it does not launch kernels or time them.
Correctness and timing use separately loaded frozen CUBINs in the driver stage.
"""
import inspect
import io
import json
from pathlib import Path
import subprocess
import traceback
import zipfile
from . import common as c


def _canonical(value):
    """Normalize tuples exactly as the persisted metadata JSON does."""
    return json.loads(c.encode(value))


def _upstream(family):
    if family == 'C06':
        from .vllm_triton import reduce_segments
        return reduce_segments
    if family == 'C02':
        from .sglang_triton import _fwd_kernel_stage1
        return _fwd_kernel_stage1
    if family == 'C01':
        from .g3b_triton import _layer_norm_bwd_dwdb
        return _layer_norm_bwd_dwdb
    raise ValueError(f'Unsupported binding family {family!r}')


def _environment(torch):
    props = torch.cuda.get_device_properties(0)
    result = dict(name=props.name, cc=list(torch.cuda.get_device_capability(0)),
                  total_memory=props.total_memory, SM_count=props.multi_processor_count,
                  torch_version=torch.__version__, torch_CUDA_version=torch.version.cuda)
    if hasattr(props, 'uuid'):
        result['UUID'] = str(props.uuid)
    try:
        result['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        result['nvidia_smi_error'] = str(exc)
    assert result['cc'] == [9, 0] and 'H100' in result['name'], result
    return result


def _groups(protocol):
    """Accept an explicit list of binding group dictionaries or frozen ids."""
    inference = {g['id']: g for g in protocol['inference_groups']}
    result = []
    for value in protocol['binding_groups']:
        group = inference[value] if isinstance(value, str) else value
        assert group['id'] in inference and group == inference[group['id']]
        result.append(group)
    assert len(result) == len({g['id'] for g in result})
    return result


def _inference_tensors(torch, case, group):
    from .production_data import make_data
    work = group.get('workloads', [])
    if work:
        assert len(work) == 1, 'A real-call binding group must have exactly one workload'
        batch, length = work[0]['batch'], work[0]['length']
    else:
        batch = group.get('binding_batch', group.get('batch'))
        length = group.get('binding_length', group.get('length'))
    assert batch is not None and length is not None
    data = make_data(case, batch=batch, length=length, with_reference=False)
    types = {'*bf16': torch.bfloat16, '*fp32': torch.float32, '*i32': torch.int32}
    tensors = {name: torch.from_numpy(array).to(device='cuda', dtype=types[data['types'][name]])
               for name, array in data['arrays'].items()}
    tensors.update({name: torch.empty(shape, device='cuda', dtype=types[data['types'][name]])
                    for name, shape in data['output_shapes'].items()})
    return tensors, data['scalars'], data['grid'], dict(batch=batch, length=length)


def _training_tensors(torch, case):
    tensors = {name: torch.empty((case['P'], case['N']), device='cuda', dtype=torch.bfloat16)
               for name in ['DW', 'DB']}
    tensors.update({name: torch.empty((case['N'],), device='cuda', dtype=torch.bfloat16)
                    for name in ['FINAL_DW', 'FINAL_DB']})
    return tensors, dict(M=case['P'], N=case['N']), ((case['N']+127)//128, 1, 1), dict(M=case['P'], N=case['N'])


def run(protocol, compiled, payload, root):
    root = Path(root)
    result = dict(status='FAILED', records=[], GPU_allocated=True, experiment_kernel_launches=0,
                  formal_timing=False, timing_samples=0, native_rebuilds=0,
                  compiler_intervention=False, protocol_SHA256=c.sha(c.encode(protocol)),
                  payload_SHA256=c.sha(payload))
    backend = None; method = None
    try:
        import torch
        import triton._C.libtriton as native
        from triton import knobs
        from triton.backends.nvidia.compiler import CUDABackend
        from triton.experimental.gluon import language as gl
        from .g6_compile import arguments
        from .production_compile import export
        from .production_cases import configuration

        c.verify_sources(protocol)
        result['ccache_before'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
        backend = Path(inspect.getfile(CUDABackend))
        method = CUDABackend.gluon_to_ttgir
        result['backend_SHA256'] = c.sha(backend.read_bytes())
        result['native_SHA256'] = c.sha(Path(native.__file__).read_bytes())
        assert result['native_SHA256'] == c.NATIVE == protocol['native_SHA256']
        assert result['backend_SHA256'] == protocol['backend_SHA256']
        assert Path(inspect.getsourcefile(method)).resolve() == backend.resolve()
        assert knobs.runtime.add_stages_inspection_hook is None
        torch.cuda.init()
        result['environment'] = _environment(torch)

        items = protocol['binding_items']
        if isinstance(items, list):
            items = {item['case']['id']: item for item in items}
        groups = _groups(protocol)
        training = protocol['training_cases']
        jobs = [(g['id'], g['case'], g, False) for g in groups]
        jobs += [(case['id'], case, None, True) for case in training]
        if protocol.get('binding_only_groups'):
            jobs = [job for job in jobs if job[0] in protocol['binding_only_groups']]
        assert len(jobs) == len({job[0] for job in jobs})
        records = compiled['records']
        expected_tags = []
        for group_id, _, _, _ in jobs:
            metas = [r for r in records if r['group'] == group_id and r['pair_role'] in ['U', 'G0']]
            assert len(metas) == 2 and {r['pair_role'] for r in metas} == {'U', 'G0'}, group_id
            expected_tags += [r['tag'] for r in metas]
        result['expected_tags'] = expected_tags
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for group_id, case, group, is_training in jobs:
                tensors = None
                try:
                    if is_training:
                        tensors, values, grid, condition = _training_tensors(torch, case)
                    else:
                        tensors, values, grid, condition = _inference_tensors(torch, case, group)
                    metas = [r for r in records if r['group'] == group_id and r['pair_role'] in ['U', 'G0']]
                    for meta in metas:
                        row = dict(tag=meta['tag'], group=group_id, form_id=meta['form_id'],
                                   pair_role=meta['pair_role'], status='FAILED', **condition)
                        try:
                            frozen_cubin = archive.read(str(Path(meta['artifact_directory'])/'kernel.cubin'))
                            assert c.sha(frozen_cubin) == meta['cubin_SHA256']
                            item = items[meta['form_id']]
                            fn, constants, _ = arguments(item, gl)
                            if meta['pair_role'] == 'U':
                                fn = _upstream(case['family'])
                                constants = {name: value for name, value in constants.items() if name in fn.arg_names}
                            if not is_training:
                                default_constants, _, default_values, _ = configuration(case)
                                # Some API defaults (aux0_stride_*) remain runtime
                                # scalars in this frozen ABI. They are not part
                                # of arguments()'s constexpr-only dictionary.
                                values = {**default_constants, **default_values, **values}
                            # A value specialized out of the ABI still has to
                            # describe this actual call (notably batch=1
                            # num_seqs).  Do not hide mismatched data behind a
                            # constant that takes precedence in kwargs below.
                            for name, value in constants.items():
                                if name in values and isinstance(value, (int, float, bool, type(None))):
                                    assert values[name] == value, (name, values[name], value)
                            kwargs = {name: tensors[name] if name in tensors else
                                      constants[name] if name in constants else values[name]
                                      for name in fn.arg_names}
                            code = fn.warmup(**kwargs, grid=grid, num_warps=case['W'],
                                             num_stages=case['num_stages'])
                            actual = export(code, fn, case, meta['tag']+':real_JIT', root/'real_jit', constants)
                            checks = {key: _canonical(actual[key]) == _canonical(meta[key]) for key in
                                      ['cubin_SHA256', 'runtime_argument_names', 'JIT_signature', 'JIT_constants',
                                       'JIT_attributes', 'num_warps', 'dynamic_smem_bytes', 'global_scratch_size']}
                            attributes_equal = {k:v for k,v in actual['JIT_attributes'].items() if v} == {
                                k:v for k,v in meta['JIT_attributes'].items() if v}
                            # Empty attribute lists encode no assumption. Keep
                            # literal equality as raw provenance, compare the
                            # nonempty assumptions for the eligibility gate.
                            match = attributes_equal and all(v for k,v in checks.items() if k != 'JIT_attributes')
                            row.update(status='OK' if match else 'MISMATCH',
                                       match=match, checks=checks, nonempty_attributes_equal=attributes_equal,
                                       CPU_CUBIN_SHA256=meta['cubin_SHA256'], real_JIT_metadata=actual)
                        except Exception:
                            row.update(match=False, error=traceback.format_exc())
                            c.write_once(root/'errors'/(meta['tag'].replace(':', '_')+'.txt'), row['error'].encode())
                        result['records'].append(row)
                        print('G7 real tensor JIT', row['tag'], row['status'], flush=True)
                except Exception:
                    error = traceback.format_exc()
                    for meta in [r for r in records if r['group'] == group_id and r['pair_role'] in ['U', 'G0']]:
                        result['records'].append(dict(tag=meta['tag'], group=group_id, form_id=meta['form_id'],
                                                      pair_role=meta['pair_role'], status='FAILED', match=False, error=error))
                finally:
                    if tensors is not None:
                        del tensors
                    torch.cuda.empty_cache()
        torch.cuda.synchronize()
        assert CUDABackend.gluon_to_ttgir is method and knobs.runtime.add_stages_inspection_hook is None
        result['backend_SHA256_after'] = c.sha(backend.read_bytes())
        result['native_SHA256_after'] = c.sha(Path(native.__file__).read_bytes())
        assert result['backend_SHA256_after'] == result['backend_SHA256']
        assert result['native_SHA256_after'] == result['native_SHA256']
        result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
        assert result['ccache_before'] == result['ccache_after']
        assert len(result['records']) == len(expected_tags) and {r['tag'] for r in result['records']} == set(expected_tags)
        result['status'] = 'OK' if all(r.get('match', False) for r in result['records']) else 'FAILED'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root/'raw.json', result)
    return result
