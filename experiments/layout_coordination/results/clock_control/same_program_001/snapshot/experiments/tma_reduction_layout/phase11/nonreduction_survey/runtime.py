"""Driver-only frozen-binary execution; CPU Torch builds deterministic fixtures."""
from pathlib import Path
import random
import sys
import traceback
import numpy as np
from . import common as c
from .fixtures import DriverFixture
from ..operator_gpu.driver import Device,Loaded,Graph


def identity(meta):
    return (meta['cubin_SHA256'],tuple(meta['runtime_argument_names']),
        c.encode(meta.get('specialized_runtime_values',{})),meta['num_warps'],
        meta['dynamic_smem_bytes'],meta.get('global_scratch_size',0))


def run(protocol,payload,root,invocation=0):
    c.verify_sources(protocol);root=Path(root);artifact=root/'program_files';c.extract(payload,artifact)
    assert c.sha(payload)==protocol['payload_SHA256']
    result=dict(status='FAILED',phase=protocol['phase'],invocation=invocation,
        formal_timing=invocation>0,profiler_attached=False,compile_calls=0,
        protocol_SHA256=c.sha(c.encode(protocol)),payload_SHA256=c.sha(payload),
        cells=[],checks=[],bindings={})
    device=None
    try:
        device=Device();result['environment']=device.environment
        kernels={r['tag']:Loaded(device,artifact,r) for r in protocol['programs']}
        rng=random.Random(protocol['timing']['seed']+invocation)
        for case in protocol['cases']:
            metas=[r for r in protocol['programs'] if r['geometry']==case['id']]
            patterns=['finite'] if invocation else protocol['correctness']['patterns']
            for pattern in patterns:
                spec=c.adapter(case).fixture(case,pattern);buffers=DriverFixture(spec,device)
                unique=[];seen={}
                for meta in metas:
                    # U and G0 remain independently measured controls even if
                    # their entire CUBINs happen to be identical.
                    key=(identity(meta),'U' if meta.get('variant')=='U' else 'Gluon')
                    if key in seen:seen[key]['aliases'].append(meta['tag']);continue
                    cell=dict(case=case['id'],tag=meta['tag'],aliases=[],pattern=pattern,visits=[])
                    unique.append(cell);seen[key]=cell
                graphs={};owners={}
                for cell in unique:
                    kernel=kernels[cell['tag']];owner,params=kernel.parameters(buffers.arguments(kernel.meta))
                    owners[cell['tag']]=(owner,params)
                    if invocation:graphs[cell['tag']]=Graph(kernel,buffers.grid,params,device.stream)
                for visit in range(protocol['timing']['visits'] if invocation else 1):
                    order=list(range(len(unique)));rng.shuffle(order)
                    if invocation and protocol['timing'].get('sample_order')=='interleaved_rounds':
                        # Balance temporal drift at the individual observation
                        # level rather than measuring an entire variant block.
                        # All kernels retain the original one-kernel graph.
                        for i in order:
                            cell=unique[i];buffers.reset();graph=graphs[cell['tag']]
                            for _ in range(protocol['timing']['warmups']):
                                buffers.reset(only_inplace=True);graph.run()
                            if visit==0:
                                check=buffers.check();assert check['pass'],(case['id'],cell['tag'],pattern,check)
                                result['checks'].append(dict(case=case['id'],tag=cell['tag'],aliases=cell['aliases'],pattern=pattern,**check))
                        observations={cell['tag']:[] for cell in unique};round_orders=[]
                        for _ in range(protocol['timing']['samples']):
                            rng.shuffle(order);round_orders.append([unique[i]['tag'] for i in order])
                            for i in order:
                                cell=unique[i];buffers.reset(only_inplace=True)
                                observations[cell['tag']].append(graphs[cell['tag']].run())
                        for cell in unique:
                            samples=observations[cell['tag']]
                            assert all(np.isfinite(x) and x>0 for x in samples)
                            cell['visits'].append(dict(visit=visit,samples_us=samples,
                                node_types=graphs[cell['tag']].node_types,round_orders=round_orders))
                        device.driver.calls.clear()
                        continue
                    for i in order:
                        cell=unique[i];kernel=kernels[cell['tag']];params=owners[cell['tag']][1]
                        buffers.reset()
                        if invocation:
                            graph=graphs[cell['tag']]
                            for _ in range(protocol['timing']['warmups']):
                                buffers.reset(only_inplace=True);graph.run()
                            samples=[]
                            for _ in range(protocol['timing']['samples']):
                                buffers.reset(only_inplace=True);samples.append(graph.run())
                            assert all(np.isfinite(x) and x>0 for x in samples)
                            cell['visits'].append(dict(visit=visit,samples_us=samples,node_types=graph.node_types))
                        else:
                            kernel.guard();kernel.launch(buffers.grid,params,device.stream);device.synchronize()
                        if visit==0:
                            check=buffers.check();assert check['pass'],(case['id'],cell['tag'],pattern,check)
                            result['checks'].append(dict(case=case['id'],tag=cell['tag'],aliases=cell['aliases'],pattern=pattern,**check))
                        device.driver.calls.clear()
                for graph in graphs.values():graph.close()
                result['cells']+=unique
                device.free_buffers()
                print('Nonreduce runtime',protocol['phase'],invocation,case['id'],pattern,'PASS',flush=True)
            c.write_once(root/(case['id']+'_checkpoint.json'),dict(cells=result['cells'],checks=result['checks']))
        result['bindings']={tag:k.binding() for tag,k in kernels.items()}
        result['driver_return_codes']=sorted(device.driver.codes)
        result['Triton_imported']=any(x=='triton' or x.startswith('triton.') for x in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes']==[0]
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if device:device.close()
    c.write_once(root/'result.json',result);return result
