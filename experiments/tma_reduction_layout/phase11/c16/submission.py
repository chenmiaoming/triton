"""Paired direct events / one-kernel graph with event nodes inside graph."""
import ctypes as C
import math
from pathlib import Path
import random
import sys
import traceback
from experiments.tma_reduction_layout.phase11.c16 import common as c
from experiments.tma_reduction_layout.phase11.c16 import runtime as r
from experiments.tma_reduction_layout.phase11.c16.profile_one import Pointer, resources, evict, correctness


def execute(protocol, diagnostic, gate, payload, call_id):
    raw = {'status': 'FAILED', 'call_id': call_id, 'samples': [], 'checks': [], 'graphs': {},
           'diagnostic_SHA256': c.sha(c.encode(diagnostic)), 'payload_SHA256': c.sha(payload),
           'Triton_imported': False, 'subprocess_calls': 0}
    def audit(event,args):
        if event=='subprocess.Popen':
            raw['subprocess_calls'] += 1
            raise RuntimeError('No subprocess/JIT in submission diagnostic')
    sys.addaudithook(audit)
    for name,digest in diagnostic['worker_source_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes())==digest,name
    driver, kernels, graphs = None, {}, {}
    try:
        root=Path('/tmp/c16-submission-frozen')
        c.extract(payload,root)
        metadata={tag:meta for tag,meta in gate['metadata'].items() if meta['family'] in ('FULL','TMA')}
        assert len(metadata)==10
        driver=r.Driver()
        context,pointers=resources(driver,protocol,metadata)
        assert [driver.attr(75),driver.attr(76)]==[9,0]
        raw['environment']=driver.environment()
        stream=C.c_void_p()
        driver.call('cuStreamCreate',[C.POINTER(C.c_void_p),C.c_uint],C.byref(stream),1)
        params={}
        for tag,meta in metadata.items():
            kernel=kernels[tag]=r.load(driver,root,meta)
            params[tag]=kernel.parameters(*[Pointer(p.value) for p in pointers[:3]])
            first,last=driver.event(),driver.event()
            for _ in range(3):
                kernel.guard(); kernel.launch(meta['full_grid'],params[tag][1],stream)
            driver.call('cuStreamSynchronize',[C.c_void_p],stream)
            raw['checks'].append({'tag':tag,'method':'direct',**correctness(driver,meta,pointers[1])})
            graph,executable=C.c_void_p(),C.c_void_p()
            driver.call('cuStreamBeginCapture',[C.c_void_p,C.c_int],stream,0)
            driver.call('cuEventRecordWithFlags',[C.c_void_p,C.c_void_p,C.c_uint],first,stream,1)
            kernel.guard(); kernel.launch(meta['full_grid'],params[tag][1],stream)
            driver.call('cuEventRecordWithFlags',[C.c_void_p,C.c_void_p,C.c_uint],last,stream,1)
            driver.call('cuStreamEndCapture',[C.c_void_p,C.POINTER(C.c_void_p)],stream,C.byref(graph))
            if hasattr(driver.lib,'cuGraphInstantiateWithFlags'):
                driver.call('cuGraphInstantiateWithFlags',[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint64],
                            C.byref(executable),graph,0)
            else:
                error,log=C.c_void_p(),C.create_string_buffer(2048)
                driver.call('cuGraphInstantiate_v2',[C.POINTER(C.c_void_p),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t],
                            C.byref(executable),graph,C.byref(error),C.cast(log,C.c_void_p),len(log))
            count=C.c_size_t()
            driver.call('cuGraphGetNodes',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],graph,None,C.byref(count))
            nodes=(C.c_void_p*count.value)()
            driver.call('cuGraphGetNodes',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],graph,nodes,C.byref(count))
            types=[]
            for node in nodes:
                kind=C.c_int()
                driver.call('cuGraphNodeGetType',[C.c_void_p,C.POINTER(C.c_int)],node,C.byref(kind))
                types.append(kind.value)
            assert sorted(types)==[0,7,7], ('Graph must contain one kernel and two event records',types)
            raw['graphs'][tag]={'node_types':types,'kernel_nodes':1,'external_event_record_nodes':2,
                                'CUBIN_SHA256':kernel.digest,'parameters':params[tag][2]}
            graphs[tag]=(graph,executable,first,last)
            kernel.guard()
            driver.call('cuGraphLaunch',[C.c_void_p,C.c_void_p],executable,stream)
            driver.synchronize_event(last)
            raw['checks'].append({'tag':tag,'method':'graph',**correctness(driver,meta,pointers[1])})
        cells=[(tag,mode,method) for tag in kernels for mode in ('warm','cold') for method in ('direct','graph')]
        for visit in range(4):
            order=cells[:]
            random.Random(2026101690+visit).shuffle(order)
            for tag,mode,method in order:
                kernel=kernels[tag]
                graph,executable,first,last=graphs[tag]
                for _ in range(3):
                    kernel.guard()
                    if method=='graph':
                        driver.call('cuGraphLaunch',[C.c_void_p,C.c_void_p],executable,stream)
                    else:
                        kernel.launch(kernel.metadata['full_grid'],params[tag][1],stream)
                driver.call('cuStreamSynchronize',[C.c_void_p],stream)
                row={'tag':tag,'cache':mode,'method':method,'visit':visit,'CUBIN_SHA256':kernel.digest,'samples_us':[]}
                raw['samples'].append(row)
                for _ in range(5):
                    if mode=='cold':
                        evict(driver,pointers,protocol)
                        driver.call('cuCtxSynchronize',[])
                    kernel.guard()
                    if method=='graph':
                        driver.call('cuGraphLaunch',[C.c_void_p,C.c_void_p],executable,stream)
                    else:
                        driver.record(first,stream)
                        kernel.launch(kernel.metadata['full_grid'],params[tag][1],stream)
                        driver.record(last,stream)
                    driver.synchronize_event(last)
                    value=driver.elapsed_us(first,last)
                    assert math.isfinite(value) and value>0
                    row['samples_us'].append(value)
            print('C1.6 paired submission diagnostic visit',visit+1,'/4',flush=True)
        raw['Triton_imported']=any(m=='triton' or m.startswith('triton.') for m in sys.modules)
        assert not raw['Triton_imported']
        raw['status']='OK'
    except Exception:
        raw['error']=traceback.format_exc()
        print(raw['error'],flush=True)
    finally:
        raw['binaries']={tag:kernel.binding() for tag,kernel in kernels.items()}
        if driver:
            for graph,executable,first,last in graphs.values():
                driver.call('cuGraphExecDestroy',[C.c_void_p],executable)
                driver.call('cuGraphDestroy',[C.c_void_p],graph)
                for event in (first,last):
                    driver.call('cuEventDestroy_v2',[C.c_void_p],event)
            for kernel in kernels.values():
                driver.call('cuModuleUnload',[C.c_void_p],kernel.module)
            raw['CUDA_call_counts']=dict(driver.counts)
            raw['CUDA_return_codes']=sorted(driver.codes)
    return raw
