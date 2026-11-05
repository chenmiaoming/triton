"""U-only selection from the public KDA configuration menu, before any GL timing."""
from pathlib import Path
import inspect
import random
import subprocess
import traceback
import numpy as np
from . import common as c


def run(protocol,root):
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import CUDABackend
    from .fixtures import TorchFixture
    from .export import export
    root=Path(root);c.verify_sources(protocol)
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    assert c.sha(Path(inspect.getfile(CUDABackend)).read_bytes())==protocol['backend_SHA256']
    assert torch.cuda.get_device_capability()==(9,0);torch.set_num_threads(2)
    result=dict(status='FAILED',records=[],selected_cases=[],selected_records=[],plans={},
        native_rebuilds=0,native_SHA256=c.NATIVE,formal_layout_timing=False,
        protocol_SHA256=c.sha(c.encode(protocol)),GPU=torch.cuda.get_device_name(),
        selection_policy='All 24 public BT/W/stages configurations, U only, 3 visits x 9 graph samples, 3 warmups, randomized order; minimum mean visit median.',
        scope='Source configuration-menu selection; not a recreation of the full external FLA cache/autotuner implementation.',
        ccache_before=subprocess.check_output(['ccache','--print-stats'],text=True))
    rng=random.Random(20261105)
    for base in protocol['cases']:
        assert base['family']=='N05';adapter=c.adapter(base);fn=adapter.upstream(base)
        entries=[]
        for i,config in enumerate(base['autotune_configs']):
            case={**base,**config,'autotune_resolved':True};row=dict(case=case,config_index=i,status='FAILED',visits=[])
            try:
                spec=adapter.fixture(case,'finite');fixture=TorchFixture(spec);kwargs=fixture.arguments()
                kwargs={n:kwargs[n] for n in fn.arg_names};options=dict(num_warps=case['W'],num_stages=case['stages'])
                code=fn.warmup(**kwargs,grid=fixture.grid,**options)
                meta=export(code,fn,case,base['id']+f':config_{i}:U',root,spec['constants'])
                fn[fixture.grid](**kwargs,**options);torch.cuda.synchronize()
                check=fixture.check();assert check['pass'],check
                first=torch.cuda.Event(enable_timing=True,external=True);last=torch.cuda.Event(enable_timing=True,external=True)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    first.record();fn[fixture.grid](**kwargs,**options);last.record()
                row.update(status='OK',metadata=meta,correctness=check)
                entries.append(dict(row=row,fixture=fixture,graph=graph,first=first,last=last,code=code))
            except Exception:row['error']=traceback.format_exc();print(row['error'],flush=True)
            result['records'].append(row);print('N1 U config',base['id'],i,row['status'],flush=True)
        assert entries,('No public configuration compiled and passed correctness',base['id'])
        for visit in range(3):
            order=list(range(len(entries)));rng.shuffle(order)
            for i in order:
                e=entries[i]
                for _ in range(3):e['graph'].replay()
                torch.cuda.synchronize();samples=[]
                for _ in range(9):
                    e['graph'].replay();torch.cuda.synchronize()
                    samples.append(e['first'].elapsed_time(e['last'])*1000.)
                assert all(np.isfinite(v) and v>0 for v in samples)
                e['row']['visits'].append(dict(visit=visit,samples_us=samples))
        best=min(entries,key=lambda e:np.mean([np.median(v['samples_us']) for v in e['row']['visits']]))
        selected={**best['row']['case'],'configuration_selection':'U-only original menu'}
        result['selected_cases'].append(selected);result['selected_records'].append(best['row']['metadata'])
        result['plans'][base['id']]=adapter.derive_plan(best['code'].asm['ttgir'],selected)
        for e in entries:e['graph'].reset()
        del entries,fixture,kwargs,graph;torch.cuda.empty_cache()
        c.write_once(root/(base['id']+'_checkpoint.json'),result)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    result['status']='OK';c.write_once(root/'result.json',result);return result
