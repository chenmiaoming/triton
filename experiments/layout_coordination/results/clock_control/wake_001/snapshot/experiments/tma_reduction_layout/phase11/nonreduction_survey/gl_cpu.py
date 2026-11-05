"""Compile G0 from actual U JIT attributes; compare every allocated ELF section."""
import ast
import inspect
from pathlib import Path
import subprocess
import traceback
import re
from . import common as c
from .export import export
from .elf import elf


def run(protocol,payload,root):
    import triton
    import triton._C.libtriton as native
    from triton import knobs
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon import GluonASTSource
    c.verify_sources(protocol);root=Path(root);reference=root/'reference';c.extract(payload,reference)
    assert c.sha(payload)==protocol['reference_payload_SHA256']
    raw=c.read(reference/'result.json');assert raw['status']=='OK'
    refs=raw.get('selected_records') or [r['metadata'] for r in raw['records'] if r['status']=='OK']
    by={r['geometry']:r for r in refs};assert len(by)==len(refs)
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    assert c.sha(Path(inspect.getfile(CUDABackend)).read_bytes())==protocol['backend_SHA256']
    assert knobs.runtime.add_stages_inspection_hook is None
    result=dict(status='NEEDS_ALIGNMENT',records=[],plans={},GPU_allocated=False,native_rebuilds=0,
        native_SHA256=c.NATIVE,protocol_SHA256=c.sha(c.encode(protocol)),
        reference_payload_SHA256=c.sha(payload),
        ccache_before=subprocess.check_output(['ccache','--print-stats'],text=True))
    for case in protocol['cases']:
        row=dict(case=case['id'],status='FAILED')
        try:
            adapter=c.adapter(case);up=by[case['id']];ufolder=reference/up['artifact_directory']
            ir=(ufolder/'kernel.ttgir').read_text()
            plan=adapter.derive_plan(ir,case);result['plans'][case['id']]=plan
            fn=adapter.gluon(case);original=adapter.upstream(case)
            const={original.arg_names[ast.literal_eval(key)[0]]:ast.literal_eval(value)
                   for key,value in up['JIT_constants'].items() if len(ast.literal_eval(key))==1}
            extra=adapter.layout_constants(case,plan,'G0');const.update(extra)
            assert fn.arg_names[:len(original.arg_names)]==original.arg_names
            sig={name:up['JIT_signature'][name] if name in up['JIT_signature'] else 'constexpr' for name in fn.arg_names}
            attrs={(fn.arg_names.index(original.arg_names[ast.literal_eval(key)[0]]),):value
                   for key,value in up['JIT_attributes'].items()}
            code=triton.compile(GluonASTSource(fn,sig,const,attrs),target=GPUTarget('cuda',90,32),
                options={'num_warps':case['W'],'num_stages':case['stages']})
            meta=export(code,fn,case,case['id']+':G0',root,const);meta['case']=case
            meta['specialized_runtime_values']=up['specialized_runtime_values']
            folder=root/meta['artifact_directory']
            same=elf((folder/'kernel.cubin').read_bytes())==elf((ufolder/'kernel.cubin').read_bytes())
            runtime_keys=['runtime_argument_names','num_warps','dynamic_smem_bytes','global_scratch_size']
            abi=all(meta[k]==up[k] for k in runtime_keys)
            resource=lambda f:dict(re.findall(r'\b(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER):(\d+)',f.read_text()))
            resources=resource(folder/'kernel.resource.txt')==resource(ufolder/'kernel.resource.txt')
            row.update(status='OK' if same and abi and resources else 'MISMATCH',metadata=meta,upstream=up,
                allocated_sections_equal=same,ABI_and_launch_equal=abi,resources_equal=resources)
        except Exception:row['error']=traceback.format_exc();print(row['error'],flush=True)
        result['records'].append(row);print('N1 G0 CPU',case['id'],row['status'],flush=True)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    result['status']='OK' if all(r['status']=='OK' for r in result['records']) else 'NEEDS_ALIGNMENT'
    c.write_once(root/'result.json',result);return result
