"""CPU-only audit of immutable non-reduction evidence and accepted lineage.

Failed revisions remain evidence: only explicitly accepted case rows must pass
the exact-binary gate. No Triton import, compilation, or GPU access occurs here.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import random
import zipfile

from . import common as c
from .elf import elf


COMPILER = '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'
FORBIDDEN_IR = re.compile(r'\b(?:tt\.(?:reduce|scan|dot|atomic_rmw|atomic_cas)|'
                          r'ttg\.(?:warp_group_dot|mma)|ttng\.(?:tc_gen5_mma|warp_group_dot))\b')
FORBIDDEN_CALLS = {'sum', 'max', 'min', 'reduce', 'argmax', 'argmin', 'dot',
                   'associative_scan', 'cumsum', 'cumprod', 'histogram',
                   'atomic_add', 'atomic_max', 'atomic_min', 'atomic_xchg',
                   'atomic_and', 'atomic_or', 'atomic_xor', 'atomic_cas'}
RUNTIME_KEYS = ('runtime_argument_names', 'num_warps', 'dynamic_smem_bytes',
                'global_scratch_size', 'specialized_runtime_values')


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _blob_sha1(blob):
    return hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()


def _ast(node):
    return ast.dump(node, include_attributes=False)


def _functions(text):
    return {node.name: node for node in ast.walk(ast.parse(text))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _no_reduction_source(text, names=None):
    functions = _functions(text)
    selected = set(functions if names is None else names)
    # Audit the transitive local helper closure, including wrappers such as
    # softplus -> softplus_nv. Other kernels in the full upstream file are not
    # part of the selected program and must not contaminate this check.
    pending = list(selected)
    while pending:
        name = pending.pop()
        node = functions[name]
        for call in (x for x in ast.walk(node) if isinstance(x, ast.Call)):
            if isinstance(call.func, ast.Attribute):
                _assert(not (isinstance(call.func.value, ast.Name)
                             and call.func.value.id in {'tl', 'gl'}
                             and call.func.attr in FORBIDDEN_CALLS),
                        f'{name}: forbidden axis/atomic operation {call.func.attr}')
            elif isinstance(call.func, ast.Name) and call.func.id in functions:
                if call.func.id not in selected:
                    selected.add(call.func.id)
                    pending.append(call.func.id)
    _assert(not re.search(r'%(?:laneid|warpid|tid\.[xyz]|ntid\.[xyz])\b', text),
            'Source uses hidden hardware-ID scalar values')
    return sorted(selected)


def _zip_members(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _assert(len(names) == len(set(names)), f'Duplicate ZIP members: {path}')
        for name in names:
            _assert(not Path(name).is_absolute() and '..' not in Path(name).parts,
                    f'Unsafe ZIP member: {name}')
        return {name: archive.read(name) for name in names if not name.endswith('/')}


def _path(path):
    path = Path(path)
    if path.exists():
        return path
    # Frozen protocols retain their original absolute provenance when a
    # checkout is moved; resolve the repository-relative evidence suffix.
    marker = '/experiments/'
    if marker in str(path):
        candidate = c.ROOT/('experiments/'+str(path).split(marker, 1)[1])
        if candidate.exists():
            return candidate
    return c.ROOT/path


def _resources(blob):
    return dict(re.findall(r'\b(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER):(\d+)',
                           blob.decode()))


def _check_correctness(check):
    _assert(check.get('pass') is True, 'Correctness failed')
    _assert(check.get('details'), 'Missing output and untouched-storage checks')
    _assert(all(row.get('pass_') is True for row in check['details']),
            'A correctness/storage detail failed')
    _assert(check.get('guards') and all(check['guards']), 'Missing or failed guards')


def _derived_origin(meta,directory):
    index=c.read(directory/'result.json')
    _assert(meta in index['programs'], 'Program is absent from derived payload index')
    row=next(x for x in index['records'] if x['case']==meta['geometry'])
    lineage=row['lineage'];origin_path=_path(lineage['result_path'])
    _assert(c.sha(origin_path.read_bytes())==lineage['result_SHA256'], 'Derived payload origin SHA changed')
    original=next(x for x in c.read(origin_path)['records'] if x['case']==meta['geometry'])
    _assert(meta['variant'] in {'U','G0'}, 'Unsupported rebased baseline variant')
    source=original['upstream' if meta['variant']=='U' else 'metadata']
    _assert(meta['tag']==meta['geometry']+':'+meta['variant'], 'Rebased baseline tag mismatch')
    expected={**source,'tag':meta['tag'],'variant':meta['variant'],
              'geometry':meta['geometry'],'artifact_directory':meta['artifact_directory'],
              'case':original['metadata']['case']}
    _assert(meta==expected, 'Derived index changes more than tag/path/variant/case provenance')
    return source


def _check_artifacts(meta, directory, original_metadata=None, index_mode=False):
    directory=Path(directory)
    if index_mode:
        _derived_origin(meta,directory)
    folder = directory/meta['artifact_directory']
    _assert(meta['artifact_SHA256'], f'No artifacts for {meta.get("tag")}')
    for name, digest in meta['artifact_SHA256'].items():
        _assert(c.sha((folder/name).read_bytes()) == digest, f'Artifact SHA changed: {folder/name}')
    cubin = (folder/'kernel.cubin').read_bytes()
    _assert(c.sha(cubin) == meta['cubin_SHA256'], f'CUBIN SHA changed: {folder}')
    disk_meta = c.read(folder/'metadata.json')
    for key, value in disk_meta.items():
        if index_mode and key in {'artifact_directory','tag'}:
            # The index is derived and rebased; these bytes intentionally keep
            # the original exporter identity. All ABI, specialization, CUBIN,
            # and artifact inventory fields remain strict below.
            continue
        if original_metadata is not None:
            _assert(original_metadata.get(key) == value, f'Bundled metadata differs from original: {folder}/{key}')
            continue
        if key == 'specialized_runtime_values' and meta.get('tag', '').endswith(':G0'):
            # gl_cpu writes exporter metadata first, then deliberately copies
            # U's runtime-specialization contract into the result record.
            # The copied contract is checked against the actual U below.
            _assert(value == {} or value == meta.get(key), 'Unexpected G0 specialization metadata')
            continue
        _assert(meta.get(key) == value, f'Metadata mismatch: {folder}/{key}')
    _assert(meta['global_scratch_size'] == 0, f'Unexpected global scratch: {folder}')
    for extension in ('ttir', 'glir', 'ttgir'):
        ir_path = folder/('kernel.'+extension)
        if ir_path.exists():
            _assert(not FORBIDDEN_IR.search(ir_path.read_text()),
                    f'Reduction/scan/dot/atomic IR in {ir_path}')
    return folder


def validate_provenance(sources=c.SOURCE/'sources'):
    """Recompute Git blob and AST relationships without executing source code."""
    sources = Path(sources)
    rows = []
    quant = c.read(sources/'quant_provenance.json')
    extracted = (sources/'quant_original.py').read_text()
    for entry in quant['records']:
        blob = (sources/entry['raw_file']).read_bytes()
        _assert(c.sha(blob) == entry['raw_SHA256'], entry['raw_file'])
        _assert(_blob_sha1(blob) == entry['Git_blob_SHA1'], entry['raw_file']+' Git blob')
        a, b = _functions(blob.decode())[entry['function']], _functions(extracted)[entry['function']]
        _assert(_ast(a) == _ast(b), entry['function']+' AST')
        _assert(c.sha(_ast(a).encode()) == entry['AST_SHA256'], entry['function']+' AST SHA')
        license_blob = (sources/entry['license_file']).read_bytes()
        _assert(c.sha(license_blob) == entry['license_SHA256'], 'Quant license SHA')
        _assert(_blob_sha1(license_blob) == entry['license_Git_blob_SHA1'], 'Quant license Git blob')
        rows.append(dict(function=entry['function'], repository=entry['repository'],
                         helpers=_no_reduction_source(extracted, [entry['function']])))
    for entry in c.read(sources/'rotary_provenance.json'):
        raw = (sources/entry['full_source']).read_bytes()
        extracted = (sources/entry['extracted_source']).read_bytes()
        _assert(c.sha(raw) == entry['full_source_SHA256'], entry['full_source'])
        _assert(c.sha(extracted) == entry['extracted_source_SHA256'], entry['extracted_source'])
        _assert(_blob_sha1(raw) == entry['upstream_git_blob_SHA1'], entry['full_source']+' Git blob')
        a, b = _functions(raw.decode())[entry['function']], _functions(extracted.decode())[entry['function']]
        _assert(_ast(a) == _ast(b), entry['function']+' AST')
        _assert(c.sha(_ast(a).encode()) == entry['function_AST_SHA256'], entry['function']+' AST SHA')
        rows.append(dict(function=entry['function'], repository=entry['repository'],
                         helpers=_no_reduction_source(extracted.decode(), [entry['function']])))
    pack = c.read(sources/'pack_gate_provenance.json')
    originals = {}
    for entry in pack['sources']:
        blob = (sources/entry['local']).read_bytes()
        _assert(c.sha(blob) == entry['SHA256'], entry['local'])
        _assert(_blob_sha1(blob) == entry['github_blob_SHA'], entry['local']+' Git blob')
        originals[entry['local']] = _functions(blob.decode())
    for entry in pack['extractions']:
        blob = (sources/entry['local']).read_bytes()
        _assert(c.sha(blob) == entry['SHA256'], entry['local'])
        functions = _functions(blob.decode())
        raw = originals['pack_fused_prep_pinned.py' if entry['group']=='pack' else 'gate_gate_pinned.py']
        for record in entry['functions']:
            name = record['function']; a, b = raw[name], functions[name]
            _assert(_ast(a.args) == _ast(b.args), name+' signature')
            body = ast.Module(body=b.body, type_ignores=[])
            _assert(_ast(ast.Module(body=a.body, type_ignores=[])) == _ast(body), name+' body')
            _assert(c.sha(_ast(body).encode()) == record['body_SHA256'], name+' body SHA')
        rows.append(dict(function=entry['group'], helpers=_no_reduction_source(blob.decode())))
    # Helpers have the same body/signature even where their host dispatch
    # decorators were deliberately removed in the standalone extraction.
    gate = _functions((sources/'gate_original.py').read_text())
    soft = originals['gate_softplus_pinned.py']
    for name in ('_generate_softplus', '_generate_constraints', 'softplus_nv'):
        _assert(_ast(gate[name].args) == _ast(soft[name].args), name+' helper signature')
        _assert(_ast(ast.Module(body=gate[name].body, type_ignores=[])) ==
                _ast(ast.Module(body=soft[name].body, type_ignores=[])), name+' helper body')
    soft_tree = ast.parse((sources/'gate_softplus_pinned.py').read_text())
    branch = next(n for n in soft_tree.body if isinstance(n, ast.If) and ast.unparse(n.test)=='IS_NVIDIA')
    nv_softplus = next(n for n in branch.body if isinstance(n, ast.FunctionDef) and n.name=='softplus')
    _assert(_ast(gate['softplus']) == _ast(nv_softplus), 'NVIDIA softplus dispatch differs')
    op_tree = ast.parse((sources/'gate_op_pinned.py').read_text())
    branch = next(n for n in op_tree.body if isinstance(n, ast.If) and 'FLA_USE_FAST_OPS' in ast.unparse(n.test))
    default_exp = next(n for n in branch.orelse if isinstance(n, ast.FunctionDef) and n.name=='exp')
    _assert(_ast(gate['exp']) == _ast(default_exp), 'FLA_USE_FAST_OPS=0 exp differs')
    cache = c.read(sources/'cache_provenance.json')
    raw = (sources/cache['raw_file']).read_bytes()
    used = (sources/cache['used_file']).read_bytes()
    _assert(c.sha(raw) == cache['raw_SHA256'], 'Cache full source SHA')
    _assert(_blob_sha1(raw) == cache['Git_blob_SHA1'], 'Cache Git blob')
    _assert(c.sha(used) == cache['used_SHA256'], 'Cache bound source SHA')
    _assert(raw.rstrip(b'\n') == used.rstrip(b'\n'), 'Cache change exceeds terminal newlines')
    for entry in cache['AST_functions']:
        name = entry['function']
        a, b = _functions(raw.decode())[name], _functions(used.decode())[name]
        _assert(_ast(a) == _ast(b), name+' cache AST')
        _assert(c.sha(_ast(a).encode()) == entry['AST_SHA256'], name+' cache AST SHA')
    license_blob = (sources/cache['license_file']).read_bytes()
    _assert(c.sha(license_blob) == cache['license_SHA256'], 'Cache license SHA')
    _assert(_blob_sha1(license_blob) == cache['license_Git_blob_SHA1'], 'Cache license Git blob')
    rows.append(dict(function='reshape_and_cache_flash', repository=cache['repository'],
                     helpers=_no_reduction_source(used.decode(), ['reshape_and_cache_flash'])))
    return rows


class Audit:
    def __init__(self):
        self.checks = []
        self.errors = []

    def check(self, label, function):
        try:
            value = function()
            self.checks.append(dict(check=label, status='OK'))
            return value
        except Exception as error:
            self.errors.append(dict(check=label, error=f'{type(error).__name__}: {error}'))
            return None


def _revision_protocol(revision):
    p = c.read(revision/'preregistration/protocol.json')
    _assert(p['compiler'] == COMPILER and p['native_SHA256'] == c.NATIVE, 'Compiler/native changed')
    _assert(p['revision'] == int(revision.name.split('_')[1]), 'Revision number mismatch')
    members = _zip_members(revision/'preregistration/source_snapshot.zip')
    _assert(set(members) == set(p['source_SHA256']), 'Snapshot member set mismatch')
    for name, digest in p['source_SHA256'].items():
        _assert(c.sha(members[name]) == digest, f'Snapshot source changed: {name}')
    if 'reference_payload_path' in p:
        _assert(c.sha(_path(p['reference_payload_path']).read_bytes()) == p['reference_payload_SHA256'],
                'Reference payload changed')
    return p


def _attempt(attempt, protocol, allow_pending=False):
    dispatch = c.read(attempt/'dispatch.json')
    _assert(dispatch['protocol_SHA256'] == c.sha(c.encode(protocol)), 'Dispatch protocol mismatch')
    archive = attempt/'raw_export.zip'
    if archive.exists():
        members = _zip_members(archive)
        for name, blob in members.items():
            _assert((attempt/'export'/name).read_bytes() == blob, f'Extracted evidence changed: {name}')
        result = c.read(attempt/'export/result.json')
        if 'protocol_SHA256' in result:
            _assert(result['protocol_SHA256'] == dispatch['protocol_SHA256'], 'Result protocol mismatch')
        _assert(result.get('native_rebuilds', 0) == 0, 'A worker rebuilt the compiler')
        if 'native_SHA256' in result:
            _assert(result['native_SHA256'] == c.NATIVE, 'Worker native SHA changed')
        for row in result.get('records', []):
            if row.get('metadata'):
                _check_artifacts(row['metadata'], attempt/'export')
                if row['metadata'].get('tag', '').endswith(':G0'):
                    _assert(row['metadata']['specialized_runtime_values'] ==
                            row['upstream']['specialized_runtime_values'],
                            'G0 runtime specialization copy differs from U')
        return dict(path=str(attempt.relative_to(c.ROOT)), status=result.get('status'),
                    result_SHA256=c.sha((attempt/'export/result.json').read_bytes()),
                    raw_export_SHA256=c.sha(archive.read_bytes()),
                    failed_rows=sum(row.get('status') != 'OK' for row in result.get('records', [])))
    failure = attempt/'failure.json'
    if not failure.exists() and allow_pending:
        return dict(path=str(attempt.relative_to(c.ROOT)), status='RUNNING_OR_INCOMPLETE')
    _assert(failure.exists(), 'Dispatched attempt has neither raw export nor preserved failure')
    record = c.read(failure)
    _assert(record.get('status') == 'FAILED' and record.get('error'), 'Invalid preserved failure')
    return dict(path=str(attempt.relative_to(c.ROOT)), status='FAILED',
                failure_SHA256=c.sha(failure.read_bytes()))


def _accepted(entry, n1):
    case_id = entry.get('case', entry.get('id'))
    revision = n1/f'revision_{entry["g0_revision"]:03}'
    protocol = c.read(revision/'preregistration/protocol.json')
    result_dir = revision/'g0'/f'attempt_{entry.get("g0_attempt", 1):03}'/'export'
    result = c.read(result_dir/'result.json')
    row = next(row for row in result['records'] if row['case'] == case_id)
    _assert(row['status'] == 'OK', 'Accepted row did not pass')
    _assert(result['GPU_allocated'] is False and result['native_rebuilds'] == 0,
            'G0 gate must be CPU-only with zero native rebuilds')
    _assert(result['reference_payload_SHA256'] == protocol['reference_payload_SHA256'],
            'G0 reference hash mismatch')
    ref = result_dir/'reference'
    reference = c.read(ref/'result.json')
    upstreams = reference.get('selected_records') or [x['metadata'] for x in reference['records'] if x['status']=='OK']
    upstream = next(x for x in upstreams if x['geometry'] == case_id)
    _assert(row['upstream'] == upstream, 'Accepted row refers to another U record')
    original_metadata = None
    if 'lineage' in reference:
        lineage = next(x for x in reference['lineage'] if x['case']==case_id)
        origin_path = _path(lineage['result_path'])
        _assert(c.sha(origin_path.read_bytes()) == lineage['result_SHA256'], 'Bundled U origin SHA changed')
        original_metadata = lineage['original_metadata']
        expected_bundle = {**original_metadata, 'artifact_directory':upstream['artifact_directory'],
                           'case':upstream['case']}
        _assert(upstream == expected_bundle, 'Bundled U changes extend beyond path/case provenance')
        origin_reference = c.read(origin_path)
        origin_rows = origin_reference.get('selected_records') or [x['metadata'] for x in origin_reference['records'] if x['status']=='OK']
        _assert(original_metadata in origin_rows, 'Original U metadata not found in source result')
        origin_attempt = origin_path.parent.parent
    else:
        origin_reference = reference
        origin_attempt = _path(protocol['reference_payload_path']).parent
    gfolder = _check_artifacts(row['metadata'], result_dir)
    ufolder = _check_artifacts(upstream, ref, original_metadata)
    _assert(elf((gfolder/'kernel.cubin').read_bytes()) == elf((ufolder/'kernel.cubin').read_bytes()),
            'Allocated ELF sections differ')
    for key in RUNTIME_KEYS:
        _assert(row['metadata'][key] == upstream[key], 'ABI/launch differs: '+key)
    _assert(_resources((gfolder/'kernel.resource.txt').read_bytes()) ==
            _resources((ufolder/'kernel.resource.txt').read_bytes()), 'Compiler resources differ')
    _assert(all(row.get(key) is True for key in ('allocated_sections_equal',
              'ABI_and_launch_equal', 'resources_equal')), 'Recorded binary gate disagrees')
    _assert(any(value.startswith('*') for value in upstream['JIT_signature'].values()),
            'No actual tensor pointer signature')
    _assert(upstream['JIT_attributes'] and upstream['compile_key'], 'Missing actual JIT specialization')
    checks = [x for x in origin_reference.get('checks', []) if x['case']==case_id]
    if checks:
        _assert({x['pattern'] for x in checks} == {'finite','structured','edge'}, 'Missing U correctness patterns')
        for check in checks:
            _check_correctness(check)
        _assert(upstream.get('fixture', {}).get('views'), 'Missing real Tensor fixture metadata')
        patterns = sorted(x['pattern'] for x in checks)
    else:
        # Original-menu tuning is U-only and stores correctness per config;
        # selected configurations get all-pattern driver checks at N1 runtime.
        candidates = [x for x in origin_reference['records'] if x.get('metadata') == (original_metadata or upstream)]
        _assert(len(candidates)==1, 'Selected U menu record missing')
        _check_correctness(candidates[0]['correctness'])
        _assert(len(candidates[0]['visits'])==3 and
                all(len(x['samples_us'])==9 for x in candidates[0]['visits']), 'Incomplete U-only config samples')
        patterns = ['finite (U-only original-menu selection)']
    parent_protocol = c.read(origin_attempt.parent.parent/'preregistration/protocol.json')
    _assert(origin_reference['protocol_SHA256'] == c.sha(c.encode(parent_protocol)), 'U parent protocol mismatch')
    source = _zip_members(origin_attempt.parent.parent/'preregistration/source_snapshot.zip')
    worker = 'tune_worker.py' if 'selected_records' in origin_reference else 'bind_worker.py'
    worker_source = source[str((c.SOURCE/worker).relative_to(c.ROOT))].decode()
    _assert('TorchFixture' in worker_source and '.warmup(**kwargs' in worker_source,
            'U was not frozen with actual Tensor warmup path')
    return dict(case=case_id, g0_revision=protocol['revision'], status='OK',
                U_correctness_patterns=patterns, allocated_ELF_equal=True,
                ABI_equal=True, resources_equal=True,
                U_result_SHA256=c.sha((ref/'result.json').read_bytes()))


def default_acceptance(n1):
    """Known lineage for preflight only; final runs use the explicit manifest."""
    unified=n1/'revision_008/g0/attempt_001/export/result.json'
    if unified.exists():
        result=c.read(unified)
        expected={case['id'] for case in c.cases()}
        if result['status']=='OK' and {row['case'] for row in result['records']}==expected:
            return [dict(case=row['case'],g0_revision=8,g0_attempt=1) for row in result['records']]
    revisions = {'N01':7, 'N02':101, 'N03':201, 'N04':7, 'N05':301, 'N06':4}
    rows = []
    for case in c.cases():
        revision = revisions[case['family']]
        path = n1/f'revision_{revision:03}/g0/attempt_001/export/result.json'
        if path.exists():
            result = c.read(path)
            if any(x.get('case')==case['id'] and x.get('status')=='OK' for x in result['records']):
                rows.append(dict(case=case['id'], g0_revision=revision, g0_attempt=1))
    return rows


def validate_actual_g0(protocol_path, result_path):
    """Recompute actual Tensor-JIT G0 == CPU G0 == original U, with guards."""
    protocol_path, result_path = Path(protocol_path), Path(result_path)
    protocol, result = c.read(protocol_path), c.read(result_path)
    directory = result_path.parent
    _assert(result['status']=='OK' and result['native_rebuilds']==0, 'Actual G0 binding failed/rebuilt native')
    _assert(result['native_SHA256']==c.NATIVE, 'Actual G0 native SHA changed')
    _assert(result['formal_timing'] is False, 'Actual binding is not timing evidence')
    _assert(result['protocol_SHA256']==c.sha(c.encode(protocol)), 'Actual G0 protocol mismatch')
    _assert(result['reference_payload_SHA256']==protocol['reference_payload_SHA256'], 'Actual G0 reference mismatch')
    cpu = c.read(directory/'reference/result.json')
    _assert(cpu['status']=='OK', 'Actual G0 references a failed CPU gate')
    by = {row['case']:row for row in cpu['records']}
    expected = {case['id'] for case in protocol['cases']}
    _assert({row['case'] for row in result['records']}==expected, 'Actual G0 case coverage')
    rows = []
    for row in result['records']:
        case_id = row['case']; previous = by[case_id]
        _assert(row['status']=='OK', case_id+' actual G0 failed')
        _assert(row['CPU_G0']==previous['metadata'] and row['upstream']==previous['upstream'],
                case_id+' actual G0 origin mismatch')
        actual = _check_artifacts(row['metadata'],directory)
        cpu_folder = _check_artifacts(row['CPU_G0'],directory/'reference')
        ufolder = directory/'reference/reference'/row['upstream']['artifact_directory']
        fingerprints = [elf((folder/'kernel.cubin').read_bytes()) for folder in (actual,cpu_folder,ufolder)]
        _assert(fingerprints[0]==fingerprints[1]==fingerprints[2], case_id+' actual/CPU/U ELF mismatch')
        for key in RUNTIME_KEYS:
            _assert(row['metadata'][key]==row['upstream'][key], case_id+' actual G0 ABI '+key)
        resources = [_resources((folder/'kernel.resource.txt').read_bytes()) for folder in (actual,cpu_folder,ufolder)]
        _assert(resources[0]==resources[1]==resources[2], case_id+' actual/CPU/U resources mismatch')
        _assert(row['metadata'].get('fixture',{}).get('views'), case_id+' actual Tensor fixture absent')
        checks = [check for check in result['checks'] if check['case']==case_id]
        _assert(len(checks)==3 and {x['pattern'] for x in checks}=={'finite','structured','edge'},
                case_id+' actual G0 correctness coverage')
        for check in checks:
            _check_correctness(check)
        rows.append(dict(case=case_id,actual_CPU_U_ELF_equal=True,ABI_equal=True,
                         resources_equal=True,correctness_patterns=3))
    return dict(status='OK',cases=rows,result_SHA256=c.sha(result_path.read_bytes()))


def _programs(protocol):
    programs = protocol['programs']
    if isinstance(programs,dict):
        programs = [dict(row,tag=tag) for tag,row in programs.items()]
    by = {row['tag']:row for row in programs}
    _assert(len(by)==len(programs), 'Duplicate runtime program tag')
    return by


def _identity(meta):
    return (meta['cubin_SHA256'],tuple(meta['runtime_argument_names']),
            c.encode(meta.get('specialized_runtime_values',{})),meta['num_warps'],
            meta['dynamic_smem_bytes'],meta.get('global_scratch_size',0))


def validate_runtime(protocol, raw, payload_directory=None):
    """Check a frozen-driver correctness or timing invocation, including aliases."""
    programs = _programs(protocol)
    _assert(raw['status']=='OK', 'Frozen driver runtime failed')
    _assert(raw['protocol_SHA256']==c.sha(c.encode(protocol)), 'Runtime protocol mismatch')
    _assert(raw['payload_SHA256']==protocol['payload_SHA256'], 'Runtime payload mismatch')
    _assert(raw['phase']==protocol['phase'], 'Runtime phase mismatch')
    _assert(raw['compile_calls']==0 and raw['Triton_imported'] is False, 'Runtime compiled/imported Triton')
    _assert(raw['profiler_attached'] is False and raw['driver_return_codes']==[0],
            'Timing runtime has profiler or CUDA errors')
    environment = raw['environment']
    _assert(environment['cc']==[9,0] and 'H100' in environment['name'], 'Runtime is not H100')
    _assert(environment['SM_count']>0 and environment['gpu_uuid'].startswith('GPU-'), 'Missing GPU identity')
    _assert(set(raw['bindings'])==set(programs), 'Frozen runtime binding coverage')
    for tag, binding in raw['bindings'].items():
        _assert(binding['cubin_SHA256']==binding['loaded_SHA256']==programs[tag]['cubin_SHA256'],
                'Loaded CUBIN mismatch: '+tag)
        _assert(binding['attributes']['n_regs']>0, 'Missing physical register binding: '+tag)
        if payload_directory is not None:
            _check_artifacts(programs[tag],Path(payload_directory),index_mode=True)
    timing = int(raw['invocation'])>0
    _assert(raw['formal_timing'] is timing, 'Runtime timing flag mismatch')
    patterns = ['finite'] if timing else protocol['correctness']['patterns']
    expected = {(tag,pattern) for tag in programs for pattern in patterns}
    cells, checked = set(), set()
    raw_samples = 0
    representatives = {}
    for cell in raw['cells']:
        tag, pattern = cell['tag'], cell['pattern']
        _assert(tag in programs and cell['case']==programs[tag]['geometry'], 'Unknown runtime cell')
        _assert(pattern in patterns, 'Unexpected runtime pattern')
        tags = [tag]+cell.get('aliases',[])
        _assert(len(tags)==len(set(tags)), 'Duplicate aliases')
        for alias in tags:
            _assert(alias in programs and _identity(programs[alias])==_identity(programs[tag]),
                    'Alias changes physical program/ABI: '+alias)
            _assert((alias,pattern) not in cells, 'Duplicate physical/logical runtime cell')
            if alias != tag:
                _assert(programs[alias]['variant']!='U' and programs[tag]['variant']!='U',
                        'U must be an independent timing/control observation')
            cells.add((alias,pattern));representatives[(alias,pattern)]=tag
        _assert(raw['bindings'][tag]['guards']>0, 'Representative not checked before launch')
        if timing:
            visits = cell['visits'];cfg=protocol['timing']
            _assert(len(visits)==cfg['visits'] and {x['visit'] for x in visits}==set(range(cfg['visits'])),
                    'Incomplete/duplicate timing visits')
            for visit in visits:
                samples=visit['samples_us']
                _assert(len(samples)==cfg['samples'] and all(isinstance(x,(float,int)) and x>0 and
                        x<float('inf') for x in samples), 'Invalid raw timing samples')
                _assert(sorted(visit['node_types'])==[0,7,7], 'Graph is not one kernel + two events')
                raw_samples += len(samples)
        else:
            _assert(not cell['visits'], 'Correctness invocation contains timing samples')
    _assert(cells==expected, 'Runtime logical program/pattern coverage mismatch')
    for check in raw['checks']:
        _check_correctness(check)
        for tag in [check['tag']]+check.get('aliases',[]):
            key=(tag,check['pattern'])
            _assert(key not in checked, 'Duplicate correctness result')
            checked.add(key)
    _assert(checked==expected, 'Runtime correctness checks incomplete')
    for case in protocol['cases']:
        pair={row['variant']:row['tag'] for row in programs.values()
              if row['geometry']==case['id'] and row['variant'] in {'U','G0'}}
        _assert(set(pair)=={'U','G0'}, 'Runtime case lacks independent controls')
        for pattern in patterns:
            _assert(representatives[(pair['U'],pattern)]!=representatives[(pair['G0'],pattern)],
                    'U/G0 observations were aliased')
    if timing and protocol['timing'].get('sample_order')=='interleaved_rounds':
        rng=random.Random(protocol['timing']['seed']+int(raw['invocation']))
        for case in protocol['cases']:
            group=[cell for cell in raw['cells'] if cell['case']==case['id']]
            for visit in range(protocol['timing']['visits']):
                order=list(range(len(group)))
                rng.shuffle(order)
                expected_orders=[]
                for _ in range(protocol['timing']['samples']):
                    rng.shuffle(order)
                    expected_orders.append([group[index]['tag'] for index in order])
                for cell in group:
                    entry=next(x for x in cell['visits'] if x['visit']==visit)
                    _assert(entry['round_orders']==expected_orders,
                            'Interleaved order differs from frozen seeded permutation: '+cell['tag'])
    return dict(status='OK',invocation=raw['invocation'],logical_programs=len(programs),
                patterns=patterns,raw_timing_samples=raw_samples,GPU_uuid=environment['gpu_uuid'],
                compiler_invocations=0,loaded_binaries_verified=True)


def validate_timing_analysis(protocol, raw_records, analysis):
    """Recompute the registered estimator; do not demand favorable outcomes."""
    from .analyze import analyze
    expected = analyze(protocol,raw_records)
    for key,value in expected.items():
        _assert(analysis.get(key)==value, 'Derived timing analysis changed: '+key)
    if 'inputs' in analysis:
        inputs=analysis['inputs']
        for record in [inputs['protocol'],inputs['analysis_source']]+inputs['raw']:
            _assert(c.sha(_path(record['path']).read_bytes())==record['SHA256'],
                    'Derived analysis input/source hash changed: '+record['path'])
    for raw in raw_records:
        validate_runtime(protocol,raw)
    return dict(status='OK',logical_programs=expected['audit']['logical_programs'],
                raw_timing_samples=expected['audit']['raw_sample_count'],
                baseline_classifications=dict(Counter(x['classification'] for x in expected['baseline_equivalence'])))


def _final_baseline(manifest, n1):
    protocol_path=_path(manifest['baseline_protocol'])
    protocol=c.read(protocol_path)
    _assert(protocol['compiler']==COMPILER and protocol['native_SHA256']==c.NATIVE,
            'Baseline compiler/native changed')
    snapshot=_zip_members(protocol_path.parent/'source_snapshot.zip')
    _assert(set(snapshot)==set(protocol['source_SHA256']), 'Baseline source ZIP member set changed')
    for name,digest in protocol['source_SHA256'].items():
        _assert(c.sha(snapshot[name])==digest, 'Baseline frozen source changed: '+name)
    for name,digest in protocol['protected_SHA256'].items():
        _assert(c.sha((c.ROOT/name).read_bytes())==digest, 'Baseline protected history changed: '+name)
    payload=_path(manifest['baseline_payload'])
    _assert(c.sha(payload.read_bytes())==protocol['payload_SHA256'], 'Baseline payload SHA changed')
    members=_zip_members(payload)
    programs=_programs(protocol)
    accepted_ids={x.get('case',x.get('id')) for x in manifest['accepted']}
    _assert({x['geometry'] for x in programs.values()}==accepted_ids, 'Baseline case coverage differs from accepted gate')
    _assert(len(programs)==2*len(accepted_ids) and
            {x['variant'] for x in programs.values()}=={'U','G0'},
            'N1 baseline must contain only independent U/G0 controls')
    known={}
    for entry in manifest['accepted']:
        result=c.read(n1/f'revision_{entry["g0_revision"]:03}'/'g0'/
                      f'attempt_{entry.get("g0_attempt",1):03}'/'export/result.json')
        row=next(x for x in result['records'] if x['case']==entry['case'])
        known[(entry['case'],'U')]={row['upstream']['cubin_SHA256']}
        known[(entry['case'],'G0')]={row['metadata']['cubin_SHA256']}
    actual=manifest['actual_G0']
    actual_result=c.read(n1/f'revision_{actual["revision"]:03}'/'g0bind'/
                         f'attempt_{actual.get("attempt",1):03}'/'export/result.json')
    for row in actual_result['records']:
        known[(row['case'],'G0')].add(row['metadata']['cubin_SHA256'])
    for meta in programs.values():
        _assert(meta['cubin_SHA256'] in known[(meta['geometry'],meta['variant'])],
                'Baseline uses a CUBIN outside accepted lineage: '+meta['tag'])
        for name,digest in meta['artifact_SHA256'].items():
            _assert(c.sha(members[meta['artifact_directory']+'/'+name])==digest,
                    'Baseline payload artifact changed: '+meta['tag']+'/'+name)
    raws=[]
    for path in manifest['baseline_raw']:
        raw_path=_path(path);folder=raw_path.parent
        dispatch=c.read(folder/'dispatch.json')
        _assert(dispatch['protocol_SHA256']==c.sha(c.encode(protocol)), 'Baseline dispatch protocol mismatch')
        exported=_zip_members(folder/'raw_export.zip')
        _assert(exported['result.json']==raw_path.read_bytes(), 'Baseline raw copy changed')
        for name,blob in exported.items():
            _assert((folder/'export'/name).read_bytes()==blob, 'Baseline extracted evidence changed: '+name)
        raws.append(c.read(raw_path))
    correctness=[raw for raw in raws if int(raw['invocation'])==0]
    timing=[raw for raw in raws if int(raw['invocation'])>0]
    _assert(len(correctness)==1, 'Require one dedicated all-pattern driver check')
    checks=[validate_runtime(protocol,raw) for raw in raws]
    analysis=c.read(_path(manifest['baseline_analysis']))
    analyzed=validate_timing_analysis(protocol,timing,analysis)
    return dict(status='OK',runtime_checks=checks,timing=analyzed,
                scientific_equivalence_is_required=False,
                equivalence_boundary='CI classifications are reported as observed, not forced to pass.')


def validate_stage_n1(root=c.OUT, accepted=None, require_complete=True, require_final=False):
    root = Path(root); n1 = root/'stage_n1'; audit = Audit()
    provenance = audit.check('public_source_provenance', validate_provenance)
    protocols = {}; attempts = []; protected = {}
    for revision in sorted(n1.glob('revision_*')):
        protocol = audit.check(revision.name+'/frozen_snapshot', lambda r=revision: _revision_protocol(r))
        if protocol is None:
            continue
        protocols[revision.name] = protocol
        for name, digest in protocol.get('protected_SHA256', {}).items():
            if name in protected and protected[name] != digest:
                audit.errors.append(dict(check='protected_SHA256',error='Conflicting baseline hash: '+name))
            protected[name] = digest
        for attempt in sorted(revision.glob('*/attempt_*')):
            if not (attempt/'dispatch.json').exists():
                continue
            row = audit.check(str(attempt.relative_to(n1)),
                              lambda a=attempt,p=protocol: _attempt(a,p,not require_complete))
            if row:
                attempts.append(row)
    def historical():
        _assert(protected, 'No protected historical artifact hashes')
        for name, digest in protected.items():
            _assert(c.sha((c.ROOT/name).read_bytes()) == digest, 'Historical artifact changed: '+name)
        return len(protected)
    historical_count = audit.check('historical_G8_SHA256', historical)
    if accepted is None:
        manifest = n1/'accepted_manifest.json'
        accepted = c.read(manifest) if manifest.exists() else default_acceptance(n1)
    manifest=accepted if isinstance(accepted,dict) else {}
    if isinstance(accepted, dict):
        accepted = accepted.get('accepted', accepted.get('accepted_rows'))
    accepted = accepted or []
    accepted_ids = [x.get('case', x.get('id')) for x in accepted]
    audit.check('unique_accepted_cases', lambda: _assert(len(accepted_ids)==len(set(accepted_ids)), 'Duplicate accepted cases'))
    expected = {x['id'] for x in c.cases()}
    pending = sorted(expected-set(accepted_ids))
    audit.check('accepted_case_scope', lambda: _assert(set(accepted_ids)<=expected, 'Unknown accepted case'))
    if require_complete:
        audit.check('all_cases_accepted', lambda: _assert(not pending, 'Pending accepted cases: '+', '.join(pending)))
    accepted_rows = []
    for entry in accepted:
        row = audit.check('accepted/'+str(entry.get('case',entry.get('id'))), lambda e=entry: _accepted(e,n1))
        if row:
            accepted_rows.append(row)
    final={};pending_gates=[]
    if manifest.get('actual_G0'):
        actual=manifest['actual_G0']
        actual_revision=n1/f'revision_{actual["revision"]:03}'
        result_path=actual_revision/'g0bind'/f'attempt_{actual.get("attempt",1):03}'/'export/result.json'
        final['actual_G0']=audit.check('actual_Tensor_G0',lambda:
            validate_actual_g0(actual_revision/'preregistration/protocol.json',result_path))
    else:
        pending_gates.append('actual_Tensor_G0')
    baseline_keys={'baseline_protocol','baseline_payload','baseline_raw','baseline_analysis'}
    if baseline_keys<=set(manifest):
        final['frozen_driver_baseline']=audit.check('frozen_driver_baseline',lambda:_final_baseline(manifest,n1))
    else:
        pending_gates.append('frozen_driver_baseline')
    if require_final:
        audit.check('final_N1_gate_coverage',lambda:_assert(not pending_gates,'Pending final gates: '+', '.join(pending_gates)))
    return dict(status='FAILED' if audit.errors else ('PREFLIGHT_OK' if pending else 'OK'),
        scope='N1 immutable source, actual-Tensor U binding, exact G0 binary lineage, and any final gates explicitly listed below; candidate timing/NCU need their stage validators',
        GPU_access=False, compiler_invocations=0, checks_passed=len(audit.checks),
        errors=audit.errors, pending_cases=pending, accepted=accepted_rows,
        source_provenance=provenance, historical_G8_files=historical_count,
        pending_final_gates=pending_gates,final_gates=final,
        revision_count=len(protocols), attempts=attempts,
        attempt_status_counts=dict(Counter(x['status'] for x in attempts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=c.OUT)
    parser.add_argument('--accepted-manifest',type=Path)
    parser.add_argument('--preflight',action='store_true',help='Allow explicitly pending acceptance rows')
    parser.add_argument('--final',action='store_true',help='Also require actual-Tensor G0 and frozen-driver baseline evidence')
    parser.add_argument('--output',type=Path,help='Write a new validation record; never overwrite')
    args = parser.parse_args()
    accepted = c.read(args.accepted_manifest) if args.accepted_manifest else None
    report = validate_stage_n1(args.root,accepted,not args.preflight,args.final)
    if args.output:
        c.write_once(args.output,report)
    print(json.dumps({key:report[key] for key in ('status','checks_passed','errors','pending_cases',
                     'pending_final_gates','revision_count','attempt_status_counts')},indent=2))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
