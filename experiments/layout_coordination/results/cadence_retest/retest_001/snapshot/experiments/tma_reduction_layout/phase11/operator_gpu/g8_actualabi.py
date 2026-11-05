"""Supplemental CPU audit of batch-64 call-site divisibility attributes.

The original G8 matrix is immutable.  This audit compiles its two batch-64
vLLM Gluon anchors and targets with the exact attributes observed by G7's real
tensor JIT calls, then compares all allocated ELF sections, ABI and resources.
No kernel is launched and no performance observation selects an attribute.
"""
import argparse
import copy
from datetime import datetime, timezone
import inspect
import io
from pathlib import Path
import re
import subprocess
import traceback
import zipfile

from . import common as c
from .g5_analyze import elf
from .g6_compile import arguments
from .g8_compile import layout
from .production_compile import export

OUT = c.OUT/'stage_g8'
AUDIT = OUT/'actual_abi_audit'


def prepare():
    original = c.read(OUT/'preregistration/protocol.json')
    gate = c.read(OUT/'compile_gate.json')
    binding_path = c.OUT/'stage_g7/binding_gate.json'
    binding = c.read(binding_path)
    bytag = {row['tag']:row for row in gate['records']}
    actual = {row['tag']:row for row in binding['records']}
    jobs = []; reference_pairs = []
    for group in original['inference_groups']:
        if group['panel'] != 'g4b' or group['case']['family'] != 'C06' or group['workloads'] != [{'batch':64,'length':8192}]:
            continue
        call_name = group['id'].removeprefix('g4b_')
        jit = actual[call_name+':G0']; real = jit['real_JIT_metadata']
        original_item = original['items'][group['form_id']]
        clean = lambda attrs:{name:value for name,value in attrs.items() if value}
        assert clean(real['JIT_attributes']) == {**clean(original_item['record']['JIT_attributes']),
                                                '(5,)':[['tt.divisibility',16]]}
        assert real['JIT_signature'] == original_item['record']['JIT_signature']
        for name in ['actual:gluon_anchor','actual:target']:
            variant = next(row for row in group['variants'] if row['name'] == name)
            item = copy.deepcopy(original_item)
            item['record']['JIT_attributes'] = copy.deepcopy(real['JIT_attributes'])
            tag = group['id']+':'+name
            jobs.append(dict(tag=tag,case=group['case'],variant=variant,item=item,
                expected_G8_metadata=bytag[tag],exact_attributes=real['JIT_attributes'],
                G7_actual_G0=jit,G7_actual_U=actual[call_name+':U']))
        reference_pairs.append(dict(group=group['id'],G8_upstream=bytag[group['id']+':actual:upstream'],
                                    G7_actual_U=actual[call_name+':U']))
    assert len(jobs) == 4 and len(reference_pairs) == 2
    additions = [c.SOURCE/'g8_actualabi.py',c.SOURCE/'g8_actualabi_run.py']
    hashes = {**original['worker_source_SHA256'],
              **{str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for path in additions}}
    protected = {**original['protected_SHA256'],
                 **{str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for path in
                    [OUT/'preregistration/protocol.json',OUT/'compile_gate.json',OUT/'compiled_payload.zip',binding_path]}}
    protocol = dict(stage='g8_actual_abi_cpu_audit',GPU_allocated=False,compiler_intervention=False,
        compiler=original['compiler'],native_SHA256=original['native_SHA256'],backend_SHA256=original['backend_SHA256'],
        profile='chenmiaoming',frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        worker_source_SHA256=hashes,protected_SHA256=protected,jobs=jobs,reference_pairs=reference_pairs,
        parent_protocol_SHA256=c.sha(c.encode(original)),parent_compile_gate_SHA256=c.sha(c.encode(gate)),
        purpose='Exact real batch64 num_seqs divisibility attribute versus generic G8 specialization; no GPU or timing.',
        policy='Allocated ELF sections, runtime argument names/types, launch resources and compiler resource usage must all match to reuse G8 measurements. Any mismatch requires replacement GPU evidence for the exact call-site form.')
    stream = io.BytesIO()
    with zipfile.ZipFile(OUT/'preregistration/source_snapshot.zip') as source, zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            if not name.endswith('/'): target.writestr(name,source.read(name))
        for path in additions: target.writestr(str(path.relative_to(c.ROOT)),path.read_bytes())
    c.write_once(AUDIT/'preregistration/source_snapshot.zip',stream.getvalue())
    c.write_once(AUDIT/'preregistration/protocol.json',protocol)
    print('G8 exact batch64 attribute audit frozen: four Gluon compilations; zero GPU work')


def run(protocol, root):
    import triton
    from triton import knobs
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    c.verify_sources(protocol); root = Path(root)
    backend = Path(inspect.getfile(CUDABackend)); method = CUDABackend.gluon_to_ttgir
    assert c.sha(Path(native.__file__).read_bytes()) == protocol['native_SHA256'] == c.NATIVE
    assert c.sha(backend.read_bytes()) == protocol['backend_SHA256']
    assert Path(inspect.getsourcefile(method)).resolve() == backend.resolve()
    assert knobs.runtime.add_stages_inspection_hook is None
    result = dict(status='FAILED',records=[],GPU_allocated=False,GPU_kernel_launches=0,native_rebuilds=0,
        compiler_intervention=False,protocol_SHA256=c.sha(c.encode(protocol)),native_SHA256=c.NATIVE,
        backend_SHA256=protocol['backend_SHA256'],ccache_before=subprocess.check_output(['ccache','--print-stats'],text=True))
    for index,job in enumerate(protocol['jobs']):
        row = dict(tag=job['tag'],status='FAILED')
        try:
            variant = job['variant']; candidate = None if variant.get('pv','default') == 'default' else layout(job['case'],variant['pv'])
            fn,constants,attrs = arguments(job['item'],gl,candidate)
            code = triton.compile(gluon.GluonASTSource(fn,dict(job['item']['record']['JIT_signature']),constants,attrs),
                target=GPUTarget('cuda',90,32),options={'num_warps':job['case']['W'],'num_stages':job['case']['num_stages']})
            meta = export(code,fn,job['case'],'exact_batch64_'+str(index),root,constants)
            assert meta['JIT_attributes'] == job['exact_attributes']
            row.update(status='OK',metadata=meta)
        except Exception:
            row['error'] = traceback.format_exc()
            c.write_once(root/'errors'/str(index),row['error'].encode())
        result['records'].append(row); print('G8 exact ABI CPU',job['tag'],row['status'],flush=True)
    assert CUDABackend.gluon_to_ttgir is method and knobs.runtime.add_stages_inspection_hook is None
    assert c.sha(Path(native.__file__).read_bytes()) == c.NATIVE
    assert c.sha(backend.read_bytes()) == protocol['backend_SHA256']
    result['ccache_after'] = subprocess.check_output(['ccache','--print-stats'],text=True)
    assert result['ccache_before'] == result['ccache_after']
    result['status'] = 'OK' if all(row['status'] == 'OK' for row in result['records']) else 'FAILED'
    c.write_once(root/'result.json',result); return result


def program(root, meta):
    folder = Path(root)/meta['artifact_directory']
    for name,digest in meta['artifact_SHA256'].items(): assert c.sha((folder/name).read_bytes()) == digest
    cubin = (folder/'kernel.cubin').read_bytes(); assert c.sha(cubin) == meta['cubin_SHA256']
    ptx = (folder/'kernel.ptx').read_text()
    entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid',ptx,re.S); assert entry
    abi = re.findall(r'\.param\s+\.(\w+)',entry[1])
    resources = dict(re.findall(r'\b(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER):(\d+)',(folder/'kernel.resource.txt').read_text()))
    assert resources
    return dict(CUBIN_SHA256=c.sha(cubin),allocated_sections=elf(cubin),ABI=abi,resources=resources,
        runtime_argument_names=meta['runtime_argument_names'],num_warps=meta['num_warps'],
        dynamic_smem_bytes=meta['dynamic_smem_bytes'],global_scratch_size=meta['global_scratch_size'])


def analyze():
    protocol = c.read(AUDIT/'preregistration/protocol.json'); c.verify_sources(protocol)
    for name,digest in protocol['protected_SHA256'].items(): assert c.sha((c.ROOT/name).read_bytes()) == digest,name
    root = AUDIT/'cpu/attempt_001/export'; raw = c.read(root/'result.json')
    assert raw['status'] == 'OK' and len(raw['records']) == len(protocol['jobs']) == 4
    assert raw['protocol_SHA256'] == c.sha(c.encode(protocol)) and not raw['GPU_allocated']
    assert raw['GPU_kernel_launches'] == raw['native_rebuilds'] == 0 and not raw['compiler_intervention']
    assert raw['native_SHA256'] == c.NATIVE and raw['ccache_before'] == raw['ccache_after']
    original_gate = c.read(OUT/'compile_gate.json'); original_root = c.ROOT/original_gate['CPU_export_directory']
    rows = []
    for job,record in zip(protocol['jobs'],raw['records']):
        assert job['tag'] == record['tag'] and record['status'] == 'OK'
        assert record['metadata']['JIT_attributes'] == job['exact_attributes']
        actual = program(root,record['metadata']); generic = program(original_root,job['expected_G8_metadata'])
        fields = {name:actual[name] == generic[name] for name in actual if name != 'CUBIN_SHA256'}
        row = dict(tag=job['tag'],exact_batch64=actual,generic_G8=generic,checks=fields,
                   all_execution_fields_equal=all(fields.values()),full_CUBIN_identical=actual['CUBIN_SHA256'] == generic['CUBIN_SHA256'])
        if job['variant']['name'].endswith('gluon_anchor'):
            frozen = job['G7_actual_G0']; prior = program(c.ROOT/frozen['real_JIT_export_root'],frozen['real_JIT_metadata'])
            row['G7_real_JIT_G0_execution_fields_equal'] = all(actual[name] == prior[name] for name in fields)
            assert row['G7_real_JIT_G0_execution_fields_equal']
        rows.append(row)
    upstream = []
    for pair in protocol['reference_pairs']:
        generic = program(original_root,pair['G8_upstream']); real = pair['G7_actual_U']
        actual = program(c.ROOT/real['real_JIT_export_root'],real['real_JIT_metadata'])
        same = all(actual[name] == generic[name] for name in actual if name != 'CUBIN_SHA256')
        upstream.append(dict(group=pair['group'],generic_G8=generic,G7_real_JIT_actual=actual,all_execution_fields_equal=same))
    okay = all(row['all_execution_fields_equal'] for row in rows+upstream)
    result = dict(status='OK' if okay else 'NEEDS_REMEASUREMENT',CPU_only=True,GPU_allocations=0,GPU_kernel_launches=0,
        protocol_SHA256=c.sha(c.encode(protocol)),raw_result_SHA256=c.sha((root/'result.json').read_bytes()),
        records=rows,upstream_reference_pairs=upstream,compiled_Gluon_programs=4,native_rebuilds=0,
        reuse_G8_evidence_supported=okay,
        boundary='This is CPU code/ABI/resource equivalence for an actual-JIT-observed attribute delta. It adds no independent GPU timing observations or actual-JIT calls for candidate layouts.')
    c.write_once(AUDIT/'analysis.json',result)
    print(json.dumps(dict(status=result['status'],compiled_Gluon_programs=4,reuse_G8_evidence_supported=okay),indent=2))
    assert okay, 'Exact batch64 attributes changed execution fields; replace affected GPU evidence.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--action',choices=['prepare','analyze'],required=True)
    args = parser.parse_args(); {'prepare':prepare,'analyze':analyze}[args.action]()
