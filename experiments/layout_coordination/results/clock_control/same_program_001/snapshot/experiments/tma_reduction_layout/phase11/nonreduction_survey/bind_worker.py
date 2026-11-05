"""Actual-Tensor upstream binding, full source artifacts, and independent oracles."""
from pathlib import Path
import inspect
import shutil
import subprocess
import traceback
from . import common as c


def run(protocol,root):
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import CUDABackend
    from .export import export
    from .fixtures import TorchFixture
    c.verify_sources(protocol);root=Path(root)
    backend=Path(inspect.getfile(CUDABackend))
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    assert c.sha(backend.read_bytes())==protocol['backend_SHA256']
    torch.set_num_threads(2);torch.cuda.init()
    assert torch.cuda.get_device_capability()==(9,0)
    result=dict(status='FAILED',records=[],checks=[],plans={},native_rebuilds=0,formal_timing=False,
        compiler_intervention=False,native_SHA256=c.NATIVE,
        protocol_SHA256=c.sha(c.encode(protocol)),ncu=shutil.which('ncu'),
        GPU=torch.cuda.get_device_name(),compiler_version=triton.__version__,
        ccache_before=subprocess.check_output(['ccache','--print-stats'],text=True))
    for case in protocol['cases']:
        row=dict(case=case['id'],status='FAILED')
        try:
            adapter=c.adapter(case);spec=adapter.fixture(case,'finite');fixture=TorchFixture(spec)
            fn=adapter.upstream(case);kwargs=fixture.arguments()
            kwargs={name:kwargs[name] for name in fn.arg_names}
            options=dict(num_warps=case['W'],num_stages=case['stages'])
            code=fn.warmup(**kwargs,grid=fixture.grid,**options)
            export_case={**case,'num_stages':case['stages']}
            meta=export(code,fn,export_case,case['id']+':U',root,spec['constants'])
            meta.update(case=case,fixture=fixture.metadata(),compiler_options=options)
            fn[fixture.grid](**kwargs,**options);torch.cuda.synchronize()
            check=fixture.check();assert check['pass'],check
            result['checks'].append(dict(case=case['id'],pattern='finite',**check))
            for pattern in ['structured','edge']:
                other=TorchFixture(adapter.fixture(case,pattern));current=other.arguments()
                fn[other.grid](**{name:current[name] for name in fn.arg_names},**options)
                torch.cuda.synchronize();check=other.check();assert check['pass'],(pattern,check)
                result['checks'].append(dict(case=case['id'],pattern=pattern,**check))
                del other,current
            try:result['plans'][case['id']]=adapter.derive_plan(code.asm['ttgir'],case)
            except Exception:result.setdefault('plan_diagnostics',{})[case['id']]=traceback.format_exc()
            row.update(status='OK',metadata=meta)
            del fixture,kwargs,spec;torch.cuda.empty_cache()
        except Exception:
            row['error']=traceback.format_exc();print(row['error'],flush=True)
        result['records'].append(row);print('N1 U bind',case['id'],row['status'],flush=True)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert result['ccache_before']==result['ccache_after']
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    result['status']='OK' if all(r['status']=='OK' for r in result['records']) else 'FAILED'
    c.write_once(root/'result.json',result);return result
