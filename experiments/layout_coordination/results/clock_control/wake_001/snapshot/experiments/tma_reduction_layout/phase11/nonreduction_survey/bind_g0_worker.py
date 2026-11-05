"""Verify real Gluon Tensor-JIT specialization, not only manually supplied AST attrs."""
from pathlib import Path
import traceback
import inspect
import subprocess
from . import common as c
from .fixtures import TorchFixture
from .export import export
from .elf import elf


def run(protocol,payload,root):
    import torch
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import CUDABackend
    c.verify_sources(protocol);root=Path(root);reference=root/'reference';c.extract(payload,reference)
    assert c.sha(payload)==protocol['reference_payload_SHA256']
    raw=c.read(reference/'result.json');assert raw['status']=='OK'
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    assert c.sha(Path(inspect.getfile(CUDABackend)).read_bytes())==protocol['backend_SHA256']
    torch.set_num_threads(2);assert torch.cuda.get_device_capability()==(9,0)
    by={x['case']:x for x in raw['records']};result=dict(status='FAILED',records=[],checks=[],
        native_SHA256=c.NATIVE,native_rebuilds=0,formal_timing=False,
        protocol_SHA256=c.sha(c.encode(protocol)),reference_payload_SHA256=c.sha(payload),
        ccache_before=subprocess.check_output(['ccache','--print-stats'],text=True))
    for case in protocol['cases']:
        row=dict(case=case['id'],status='FAILED')
        try:
            previous=by[case['id']];cpu=previous['metadata'];up=previous['upstream']
            source=reference/'reference'/up['artifact_directory']
            adapter=c.adapter(case);plan=adapter.derive_plan((source/'kernel.ttgir').read_text(),case)
            spec=adapter.fixture(case,'finite');buffers=TorchFixture(spec)
            extra=adapter.layout_constants(case,plan,'G0');args={**buffers.arguments(),**extra}
            fn=adapter.gluon(case);kwargs={n:args[n] for n in fn.arg_names}
            options=dict(num_warps=case['W'],num_stages=case['stages'])
            code=fn.warmup(**kwargs,grid=buffers.grid,**options)
            meta=export(code,fn,case,case['id']+':G0_actual',root,{**spec['constants'],**extra})
            meta.update(case=case,fixture=buffers.metadata())
            actual=elf((root/meta['artifact_directory']/'kernel.cubin').read_bytes())
            u_sections=elf((source/'kernel.cubin').read_bytes())
            cpu_sections=elf((reference/cpu['artifact_directory']/'kernel.cubin').read_bytes())
            same=actual==u_sections==cpu_sections
            abi=all(meta[k]==up[k] for k in ['runtime_argument_names','num_warps','dynamic_smem_bytes','global_scratch_size'])
            fn[buffers.grid](**kwargs,**options);torch.cuda.synchronize();check=buffers.check()
            assert check['pass'],check
            result['checks'].append(dict(case=case['id'],pattern='finite',**check))
            for pattern in ['structured','edge']:
                other=TorchFixture(adapter.fixture(case,pattern));data={**other.arguments(),**extra}
                fn[other.grid](**{n:data[n] for n in fn.arg_names},**options)
                torch.cuda.synchronize();check=other.check();assert check['pass'],(pattern,check)
                result['checks'].append(dict(case=case['id'],pattern=pattern,**check))
                del other,data
            row.update(status='OK' if same and abi else 'MISMATCH',metadata=meta,
                upstream=up,CPU_G0=cpu,allocated_sections_equal=same,ABI_and_launch_equal=abi,
                full_CUBIN_matches_CPU=meta['cubin_SHA256']==cpu['cubin_SHA256'])
            del buffers,args,kwargs,spec;torch.cuda.empty_cache()
        except Exception:row['error']=traceback.format_exc();print(row['error'],flush=True)
        result['records'].append(row);print('N1 actual G0 bind',case['id'],row['status'],flush=True)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    result['status']='OK' if all(r['status']=='OK' for r in result['records']) else 'NEEDS_ALIGNMENT'
    c.write_once(root/'result.json',result);return result
