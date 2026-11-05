"""CPU-only compilation of every preregistered non-reduction layout candidate.

The input bundle contains accepted, actual-JIT U/G0 baselines.  Candidate
compilation inherits U's signature, attributes, scalar specialization and
launch configuration; only the adapter's layout constants change.  This module
does not allocate a CUDA tensor, initialize a driver, or launch a kernel.
"""
import ast
import copy
import inspect
from pathlib import Path
import re
import subprocess
import traceback

from . import common as c
from .elf import elf
from .export import export


def _resources(folder):
    text = (folder / 'kernel.resource.txt').read_text()
    result = {key: int(value) for key, value in re.findall(
        r'\b(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER):(\d+)', text)}
    assert 'REG' in result, folder
    return result


def _artifact_path(root, metadata):
    path = Path(metadata['artifact_directory'])
    assert not path.is_absolute() and '..' not in path.parts, path
    folder = Path(root) / path
    assert folder.is_dir(), folder
    for name, digest in metadata['artifact_SHA256'].items():
        assert c.sha((folder / name).read_bytes()) == digest, (folder, name)
    assert c.sha((folder / 'kernel.cubin').read_bytes()) == metadata['cubin_SHA256']
    return folder


def _audit_ir(folder, requested_constants=None):
    """Keep actual ownership and communication operations, including aliases.

    An adapter's derive_plan can contain assumptions specific to U.  The
    boundary audit here therefore preserves complete canonical encoding
    definitions and ordered memory/conversion operations without imposing U's
    relations on candidates that deliberately change those relations.
    """
    ir = (folder / 'kernel.ttgir').read_text()
    lowered = ir + '\n' + (folder / 'kernel.ttir').read_text() if (folder / 'kernel.ttir').exists() else ir
    prohibited = re.findall(
        r'\b(?:tt\.(?:reduce|scan|dot|atomic_[\w]+|descriptor_load|descriptor_store)'
        r'|ttng\.[\w]*(?:tma|mma)[\w]*)\b', lowered)
    assert not prohibited, ('Non-reduction, non-TMA candidate gate', prohibited)
    assembly = [line.strip() for line in lowered.splitlines() if 'inline_asm' in line]
    assert not any(re.search(r'%(?:laneid|warpid|tid\.[xyz]|ntid\.[xyz])\b', line)
                   for line in assembly), 'Source inline assembly exposes hardware IDs'
    definitions = [line.strip() for line in ir.splitlines()
                   if re.match(r'\s*#\w+\s*=\s*#ttg\.', line)]
    families = {
        'loads': r'\btt\.load\b',
        'stores': r'\btt\.store\b',
        'conversions': r'\bttg\.convert_layout\b',
        'reshapes_and_pairs': r'\btt\.(?:reshape|join|split|trans)\b',
    }
    rows = {key: [line.strip().split(' loc(')[0] for line in ir.splitlines()
                  if re.search(pattern, line)] for key, pattern in families.items()}
    return dict(status='OK', source_scope='No reduce/scan/dot/atomic/TMA operations in target IR',
                ttgir_SHA256=c.sha(ir.encode()), encoding_definitions=definitions,
                requested_layout_constants=requested_constants or {}, inline_assembly=assembly,
                **rows)


def _copy_program(metadata, case, variant, baseline, root):
    """Rebase bundle paths in the result without modifying imported metadata."""
    value = copy.deepcopy(metadata)
    original = value['artifact_directory']
    folder = _artifact_path(baseline, value)
    value.update(case=copy.deepcopy(case), variant=variant,
                 baseline_artifact_directory=original,
                 artifact_directory=str(folder.relative_to(root)),
                 imported_baseline=True, resources=_resources(folder))
    return value


def _single_constants(original, metadata):
    constants = {}
    for key, value in metadata['JIT_constants'].items():
        path = ast.literal_eval(key)
        assert isinstance(path, tuple) and len(path) == 1, ('Unsupported nested specialization', key)
        constants[original.arg_names[path[0]]] = ast.literal_eval(value)
    return constants


def run(protocol, payload, root):
    import triton
    import triton._C.libtriton as native
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.experimental.gluon import GluonASTSource

    c.verify_sources(protocol)
    root = Path(root)
    expected_payload = protocol.get('payload_SHA256', protocol.get('reference_payload_SHA256'))
    assert expected_payload and c.sha(payload) == expected_payload, 'Baseline payload SHA256'
    baseline = root / 'baseline'
    assert not baseline.exists(), baseline
    c.extract(payload, baseline)
    bundle = c.read(baseline / 'result.json')
    assert bundle['status'] == 'OK', 'Only accepted baseline bundles may enter candidate compilation'
    by_case = {}
    for record in bundle['records']:
        case = record['case']
        identity = case['id'] if isinstance(case, dict) else case
        assert identity not in by_case, identity
        assert record.get('status', 'OK') == 'OK', identity
        by_case[identity] = record

    native_path = Path(native.__file__)
    backend_path = Path(inspect.getfile(CUDABackend))
    assert c.sha(native_path.read_bytes()) == c.NATIVE == protocol['native_SHA256']
    assert c.sha(backend_path.read_bytes()) == protocol['backend_SHA256']
    assert knobs.runtime.add_stages_inspection_hook is None
    result = dict(status='RUNNING', programs=[], records=[], plans={}, GPU_allocated=False,
                  GPU_kernel_launches=0, native_rebuilds=0, native_SHA256=c.NATIVE,
                  backend_SHA256=protocol['backend_SHA256'],
                  compilation_target=dict(backend='cuda', capability=90, warp_size=32),
                  protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
                  logical_candidate_policy='Keep all declared candidates, including full-CUBIN aliases',
                  ccache_before=subprocess.check_output(['ccache', '--print-stats'], text=True))

    for case in protocol['cases']:
        case_id = case['id']
        try:
            record = by_case[case_id]
            upstream = record['upstream']
            g0 = record['metadata']
            u_folder = _artifact_path(baseline, upstream)
            g_folder = _artifact_path(baseline, g0)
            assert upstream['geometry'] == g0['geometry'] == case_id
            assert elf((u_folder / 'kernel.cubin').read_bytes()) == elf((g_folder / 'kernel.cubin').read_bytes())
            assert _resources(u_folder) == _resources(g_folder)
            assert all(upstream[key] == g0[key] for key in
                       ['runtime_argument_names', 'num_warps', 'dynamic_smem_bytes', 'global_scratch_size'])
            adapter = c.adapter(case)
            plan = adapter.derive_plan((u_folder / 'kernel.ttgir').read_text(), case)
            result['plans'][case_id] = plan
            for variant_name, metadata in [('U', upstream), ('G0', g0)]:
                program = _copy_program(metadata, case, variant_name, baseline, root)
                result['programs'].append(program)
                result['records'].append(dict(case=case_id, variant=variant_name, status='OK',
                    metadata=program, effective_layout_audit=_audit_ir(root / program['artifact_directory'])))

            original = adapter.upstream(case)
            fn = adapter.gluon(case)
            assert fn.arg_names[:len(original.arg_names)] == original.arg_names
            base_constants = _single_constants(original, upstream)
            signature = {name: upstream['JIT_signature'].get(name, 'constexpr') for name in fn.arg_names}
            attributes = {(fn.arg_names.index(original.arg_names[ast.literal_eval(key)[0]]),): value
                          for key, value in upstream['JIT_attributes'].items()}
            choices = adapter.variants(case, plan)
            assert len({choice['name'] for choice in choices}) == len(choices)
            for choice in choices:
                name = choice['name']
                variant_name = 'GL_' + name
                row = dict(case=case_id, variant=variant_name, choice=copy.deepcopy(choice), status='FAILED')
                try:
                    assert re.fullmatch(r'[A-Za-z0-9_]+', name), name
                    constants = dict(base_constants)
                    layout_constants = adapter.layout_constants(case, plan, choice)
                    assert not (set(layout_constants) & set(base_constants)), 'Candidate overrides a non-layout constant'
                    assert set(layout_constants) == set(fn.arg_names[len(original.arg_names):]), 'Only appended layout parameters may vary'
                    constants.update(layout_constants)
                    code = triton.compile(GluonASTSource(fn, signature, constants, attributes),
                        target=GPUTarget('cuda', 90, 32),
                        options={'num_warps': case['W'], 'num_stages': case['stages']})
                    metadata = export(code, fn, case, case_id + ':' + variant_name, root, constants)
                    metadata['specialized_runtime_values'] = copy.deepcopy(upstream['specialized_runtime_values'])
                    assert metadata['runtime_argument_names'] == upstream['runtime_argument_names']
                    folder = root / metadata['artifact_directory']
                    audit = _audit_ir(folder, {key: repr(value) for key, value in layout_constants.items()})
                    try:
                        audit['adapter_effective_plan'] = adapter.derive_plan((folder / 'kernel.ttgir').read_text(), case)
                        audit['adapter_effective_plan_status'] = 'OK'
                    except Exception:
                        audit['adapter_effective_plan_status'] = 'BASELINE_SPECIFIC_PARSER_NOT_APPLICABLE'
                        audit['adapter_effective_plan_diagnostic'] = traceback.format_exc()
                    program = {**metadata, 'case': copy.deepcopy(case), 'variant': variant_name,
                               'choice': copy.deepcopy(choice), 'imported_baseline': False,
                               'resources': _resources(folder)}
                    result['programs'].append(program)
                    row.update(status='OK', metadata=program, effective_layout_audit=audit)
                except Exception:
                    row['error'] = traceback.format_exc()
                result['records'].append(row)
                print('Non-reduction CPU candidate', case_id, variant_name, row['status'], flush=True)
        except Exception:
            result['records'].append(dict(case=case_id, status='FAILED', error=traceback.format_exc()))
            print('Non-reduction CPU case', case_id, 'FAILED', flush=True)

    assert c.sha(native_path.read_bytes()) == c.NATIVE
    assert c.sha(backend_path.read_bytes()) == protocol['backend_SHA256']
    assert knobs.runtime.add_stages_inspection_hook is None
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    result['status'] = 'OK' if all(record['status'] == 'OK' for record in result['records']) else 'FAILED'
    result['counts'] = dict(cases=len(protocol['cases']), logical_programs=len(result['programs']),
                            variants=sum(row.get('variant', '').startswith('GL_') for row in result['records']),
                            failed_records=sum(row['status'] != 'OK' for row in result['records']))
    c.write_once(root / 'result.json', result)
    return result
