"""Exact-binary training merge and original two-kernel backward device pipeline."""
import ctypes as C
from pathlib import Path
import random,sys,traceback
import numpy as np
from . import common as c
from .driver import Device,Loaded,Graph
from .production_buffers import pack,unpack
from experiments.tma_reduction_layout.phase11.operator_survey.oracles import bf16_round


class Buffers:
    def __init__(self,device,case,mode='merge',m=128,pattern='finite'):
        self.device,self.case,self.mode=device,case,mode
        self.p,self.n=case['P'],case['N']
        self.rows={};self.expected={}
        self.scalars={'M':self.p if mode=='merge' else min(self.p,m),'N':self.n,'stride':self.n}
        rng=np.random.default_rng(20261022+self.p+self.n+m)
        def tensor(shape,scale=.125):
            x=(rng.standard_normal(shape)*scale).astype(np.float32)
            if pattern=='quarter':x=((np.arange(x.size).reshape(shape)%19-9)/8).astype(np.float32)
            if pattern=='cancellation':x=np.where(np.arange(x.size).reshape(shape)%2==0,x,-x)
            return bf16_round(x)
        if mode=='merge':
            for name in ('DW','DB'):
                arr=tensor((self.p,self.n));self.add(name,arr,'*bf16')
                self.expected['FINAL_'+name]=bf16_round(arr.sum(0,dtype=np.float64).astype(np.float32))
        else:
            x,dy,w=tensor((m,self.n),.5),tensor((m,self.n)),bf16_round(rng.uniform(.5,1.5,self.n).astype(np.float32))
            mean=x.mean(1,dtype=np.float64).astype(np.float32)
            var=((x.astype(np.float64)-mean[:,None])**2).mean(1)
            rstd=(1/np.sqrt(var+1.e-5)).astype(np.float32)
            for name,arr,typ in [('X',x,'*bf16'),('DY',dy,'*bf16'),('W',w,'*bf16'),('Mean',mean,'*fp32'),('Rstd',rstd,'*fp32')]:
                self.add(name,arr,typ)
            xhat=(x.astype(np.float64)-mean[:,None])*rstd[:,None]
            wdy=dy.astype(np.float64)*w[None,:]
            dx=(wdy-xhat*(xhat*wdy).mean(1)[:,None]-wdy.mean(1)[:,None])*rstd[:,None]
            self.expected['DX']=bf16_round(dx.astype(np.float32))
            self.add('DX',np.zeros_like(x),'*bf16',writable=True)
            for name in ('DW','DB'):self.add(name,np.zeros((self.p,self.n),np.float32),'*bf16',writable=True,zero=True)
            self.add('Lock',np.zeros(2*self.p,np.int32),'*i32',writable=True,zero=True)
        for name in ('FINAL_DW','FINAL_DB'):self.add(name,np.zeros(self.n,np.float32),'*bf16',writable=True)
        self.scratch=device.allocate(256)

    def add(self,name,array,typ,writable=False,zero=False):
        blob=pack(array,typ)
        if writable and not zero:blob=bytes([0xcd])*len(blob)
        raw=bytes([0xa5])*256+blob+bytes([0x5a])*256
        base=self.device.allocate(len(raw));self.device.upload(base,raw)
        self.rows[name]={'base':base,'ptr':base+256,'shape':array.shape,'dtype':typ,'size':len(blob),
                         'initial':raw,'writable':writable,'zero':zero}

    def args(self,meta):
        result=[self.rows[name]['ptr'] if name in self.rows else self.scalars[name] for name in meta['runtime_argument_names']]
        for name,value in meta['specialized_runtime_values'].items():
            if name in self.scalars:assert self.scalars[name]==value,(name,value)
        return result+[self.scratch,0]

    def reset(self):
        for row in self.rows.values():
            if row['writable']:self.device.upload(row['base'],row['initial'])

    def zero_for_pipeline(self):
        if self.mode!='pipeline':return
        for name in ('DW','DB','Lock'):
            row=self.rows[name]
            self.device.driver.call('cuMemsetD32Async',[C.c_uint64,C.c_uint,C.c_size_t,C.c_void_p],
                                    row['ptr'],0,row['size']//4,self.device.stream)

    def check(self,include_inputs=True):
        downloaded={};checks=[]
        for name,row in self.rows.items():
            if not include_inputs and not row['writable']:continue
            raw=self.device.download(row['base'],row['size']+512)
            okay=raw[:256]==row['initial'][:256] and raw[-256:]==row['initial'][-256:]
            if not row['writable']:okay=okay and raw==row['initial']
            else:downloaded[name]=unpack(raw[256:-256],row['dtype'],row['shape'])
            checks.append({'name':name,'canaries_and_inputs_pass':okay})
        if self.mode=='pipeline':
            for name in ('DW','DB'):
                actual=downloaded[name][:self.scalars['M']]
                self.expected['FINAL_'+name]=bf16_round(actual.sum(0,dtype=np.float64).astype(np.float32))
        outputs=[]
        for name,expected in self.expected.items():
            row=self.rows[name]
            actual=downloaded[name]
            okay=bool(np.allclose(actual,expected,rtol=.008,atol=.001))
            outputs.append({'name':name,'pass':okay,'elements':actual.size,
                            'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-expected))),
                            'actual_SHA256':c.sha(actual.tobytes()),'expected_SHA256':c.sha(expected.tobytes())})
        return {'pass':all(x['canaries_and_inputs_pass'] for x in checks) and all(x['pass'] for x in outputs),
                'buffers':checks,'outputs':outputs,'basis':'FP64 merge of exact partials; independent dX formula for pipeline'}


class TrainingGraph(Graph):
    def __init__(self,stages,buffers):
        self.kernel=stages[0][0];self.driver=self.kernel.driver;self.stream=buffers.device.stream
        self.stages=stages;d=self.driver
        self.first,self.last=d.event(),d.event();self.graph,self.executable=C.c_void_p(),C.c_void_p()
        d.call('cuStreamBeginCapture',[C.c_void_p,C.c_int],self.stream,0)
        d.call('cuEventRecordWithFlags',[C.c_void_p,C.c_void_p,C.c_uint],self.first,self.stream,1)
        buffers.zero_for_pipeline()
        for kernel,grid,params in stages:kernel.guard();kernel.launch(grid,params,self.stream)
        d.call('cuEventRecordWithFlags',[C.c_void_p,C.c_void_p,C.c_uint],self.last,self.stream,1)
        d.call('cuStreamEndCapture',[C.c_void_p,C.POINTER(C.c_void_p)],self.stream,C.byref(self.graph))
        d.call('cuGraphInstantiateWithFlags',[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint64],C.byref(self.executable),self.graph,0)
        count=C.c_size_t();d.call('cuGraphGetNodes',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],self.graph,None,C.byref(count))
        nodes=(C.c_void_p*count.value)();d.call('cuGraphGetNodes',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],self.graph,nodes,C.byref(count))
        self.node_types=[]
        for node in nodes:
            val=C.c_int();d.call('cuGraphNodeGetType',[C.c_void_p,C.POINTER(C.c_int)],node,C.byref(val));self.node_types.append(val.value)
        assert sorted(self.node_types)==sorted([0]*len(stages)+[7,7]+([2]*3 if buffers.mode=='pipeline' else []))

    def run(self):
        for kernel,_,_ in self.stages:kernel.guard()
        self.driver.call('cuGraphLaunch',[C.c_void_p,C.c_void_p],self.executable,self.stream)
        for kernel,_,_ in self.stages:kernel.graph_launches+=1
        self.driver.call('cuStreamSynchronize',[C.c_void_p],self.stream)
        return self.driver.elapsed_us(self.first,self.last)


def stages_for(kernels,meta,case,buffers,m):
    rows=[];owners=[]
    names=([case['id']+':dx'] if buffers.mode=='pipeline' else [])+[meta['tag']]
    for tag in names:
        k=kernels[tag];owner,params=k.parameters(buffers.args(k.meta));owners.append(owner)
        rows.append((k,(m,1,1) if k.meta['role']=='dx' else ((case['N']+127)//128,1,1),params))
    return rows,owners


def run(protocol,compiled,payload,root,invocation=0):
    c.verify_sources(protocol)
    folder=Path('/tmp')/('g3b-runtime-'+root.name);c.extract(payload,folder)
    result={'status':'FAILED','invocation':invocation,'formal_timing':invocation>0,
            'profiler_attached':False,'protocol_SHA256':c.sha(c.encode(protocol)),
            'payload_SHA256':c.sha(payload),'cells':[],'checks':[]}
    device=None
    try:
        device=Device();result['environment']=device.environment
        kernels={r['tag']:Loaded(device,folder,r) for r in compiled['records']}
        rng=random.Random(protocol['timing']['seed']+invocation)
        for case in protocol['cases']:
            metas=[r for r in compiled['records'] if r['geometry']==case['id'] and r['role']=='merge']
            conditions=[('merge',128)]+[('pipeline',m) for m in case['pipeline_M']]
            for mode,m in conditions:
                patterns=['finite'] if invocation else protocol['correctness']['patterns']
                for pattern in patterns:
                    buffers=Buffers(device,case,mode,m,pattern)
                    unique=[];seen={}
                    for meta in metas:
                        key=(meta['cubin_SHA256'],tuple(meta['runtime_argument_names']))
                        if key in seen:seen[key]['aliases'].append(meta['tag']);continue
                        cell={'tag':meta['tag'],'geometry':case['id'],'mode':mode,'M':m,'pattern':pattern,'aliases':[],'visits':[]}
                        unique.append(cell);seen[key]=cell
                    for visit in range(protocol['timing']['visits'] if invocation else 1):
                        order=list(range(len(unique)));rng.shuffle(order)
                        for j in order:
                            cell=unique[j];meta=kernels[cell['tag']].meta;buffers.reset()
                            stages,owners=stages_for(kernels,meta,case,buffers,m)
                            if invocation:
                                graph=TrainingGraph(stages,buffers)
                                for _ in range(protocol['timing']['warmups']):graph.run()
                                samples=[graph.run() for _ in range(protocol['timing']['samples'])]
                                cell['visits'].append({'visit':visit,'samples_us':samples,'node_types':graph.node_types})
                                graph.close()
                            else:
                                buffers.zero_for_pipeline()
                                for k,g,p in stages:k.guard();k.launch(g,p,device.stream)
                                device.synchronize()
                            if visit==0:
                                check=buffers.check(include_inputs=not bool(invocation))
                                assert check['pass'],(cell['tag'],mode,check)
                                result['checks'].append({'tag':cell['tag'],'aliases':cell['aliases'],'mode':mode,'M':m,'pattern':pattern,**check})
                            device.driver.calls.clear()
                    result['cells']+=unique;device.free_buffers()
                    print('G3b runtime',invocation,case['id'],mode,m,pattern,'PASS',flush=True)
                    c.write_once(root/f'checkpoint_{len(result["cells"])}.json',{'cells':result['cells'],'checks':result['checks']})
        result['bindings']={k:v.binding() for k,v in kernels.items()}
        result['driver_return_codes']=sorted(device.driver.codes)
        result['Triton_imported']=any(x=='triton' or x.startswith('triton.') for x in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes']==[0]
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if device:device.close()
    c.write_once(root/'result.json',result)
    return result
