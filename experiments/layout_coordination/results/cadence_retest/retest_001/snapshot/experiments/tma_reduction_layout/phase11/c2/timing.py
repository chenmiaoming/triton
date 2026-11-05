"""Driver-only correctness, resource census and single-kernel graph timing."""
import ctypes as C
from pathlib import Path
import random
import statistics
import sys
import traceback
import numpy as np
from experiments.tma_reduction_layout.phase11.c2 import common as c
from experiments.tma_reduction_layout.phase11.c2 import data
from experiments.tma_reduction_layout.phase11.c2.driver import Driver,Loaded,Pointer,allocate,upload,Graph


class Session:
    def __init__(self,root,metadata,maxgrid=2,maxk=256):
        self.driver=Driver();d=self.driver
        self.context=C.c_void_p()
        d.call('cuCtxCreate_v2',[C.POINTER(C.c_void_p),C.c_uint,C.c_int],C.byref(self.context),0,d.device)
        self.cc=[d.attr(75),d.attr(76)];self.sm_count=d.attr(16)
        name=C.create_string_buffer(256)
        d.call('cuDeviceGetName',[C.c_void_p,C.c_int,C.c_int],C.cast(name,C.c_void_p),256,d.device)
        self.device_name=name.value.decode()
        assert self.cc==[9,0] and self.sm_count==132 and 'H100' in self.device_name
        self.stream=C.c_void_p()
        d.call('cuStreamCreate',[C.POINTER(C.c_void_p),C.c_uint],C.byref(self.stream),0)
        self.metadata=metadata
        self.kernels={tag:Loaded(d,Path(root),meta) for tag,meta in metadata.items()}
        outsize=max(maxgrid*c.stride(m,maxk)*4 for m in metadata.values())
        self.pointers=[allocate(d,n) for n in (8192*8192*2,256*8192*4,outsize,
                       max(1,max(m['global_scratch_size']*maxgrid for m in metadata.values())))]
        self.allocated_bytes=8192*8192*2+256*8192*4+outsize
        assert self.allocated_bytes<8*1024**3,self.allocated_bytes
        self.pattern=None
        self.set_pattern('quarter')

    def environment(self):
        return {**self.driver.environment(),'cc':self.cc,'SM_count':self.sm_count,'name':self.device_name}

    def set_pattern(self,pattern):
        if self.pattern==pattern:return
        upload(self.driver,self.pointers[0],data.input_bytes(pattern))
        upload(self.driver,self.pointers[1],data.result_bytes(pattern))
        self.pattern=pattern

    def params(self,tag,k):
        kernel=self.kernels[tag];m=kernel.metadata
        x=self.pointers[1 if m['input_kind']=='result_f32' else 0]
        return kernel.parameters(Pointer(x.value),Pointer(self.pointers[2].value),Pointer(self.pointers[3].value),k)

    def check(self,tag,k,pids,strict=True):
        meta=self.metadata[tag];stride=c.stride(meta,k)
        actual=np.empty((len(pids),stride),dtype=np.float32)
        for i,pid in enumerate(pids):
            row=actual[i]
            self.driver.call('cuMemcpyDtoH_v2',[C.c_void_p,C.c_uint64,C.c_size_t],C.c_void_p(row.ctypes.data),
                             self.pointers[2].value+pid*stride*4,row.nbytes)
        if meta['family']=='TMA':
            expected=data.tiles(meta,pids,pattern=self.pattern)[:,0,:]
        else:expected=data.reference(meta,k,pids,self.pattern)
        checked=data.compare(actual,expected)
        if not checked['pass']:
            indices=np.flatnonzero(actual.ravel()!=expected.ravel())
            checked['mismatch_elements']=len(indices)
            checked['first_mismatch_flat_indices']=indices[:32].tolist()
            checked['actual_at_mismatches']=actual.ravel()[indices[:32]].tolist()
            checked['expected_at_mismatches']=expected.ravel()[indices[:32]].tolist()
        if strict:assert checked['pass'],(tag,k,self.pattern,checked)
        return {'tag':tag,'K':k,'pids':pids,'pattern':self.pattern,**checked}

    def close(self):
        d=self.driver
        for k in self.kernels.values():d.call('cuModuleUnload',[C.c_void_p],k.module)
        for ptr in self.pointers:d.call('cuMemFree_v2',[C.c_uint64],ptr)
        d.call('cuStreamDestroy_v2',[C.c_void_p],self.stream)
        d.call('cuCtxDestroy_v2',[C.c_void_p],self.context)


def eligibility(protocol,metadata,root,previous_checks=None):
    previous_checks=[r for r in (previous_checks or []) if r['pass']]
    already={(r['tag'],r['K'],r['pattern']) for r in previous_checks}
    result={'status':'FAILED','checks':list(previous_checks),'resources':{},'formal_timing':False,
            'reused_passed_checks':len(previous_checks),'reused_checks_SHA256':c.sha(c.encode(previous_checks))}
    s=None
    try:
        s=Session(root,metadata)
        assert [s.driver.attr(75),s.driver.attr(76)]==[9,0]
        result['environment']=s.environment()
        for tag,kernel in s.kernels.items():result['resources'][tag]=kernel.binding()
        for pattern in protocol['correctness']['patterns']:
            s.set_pattern(pattern)
            for tag,meta in metadata.items():
                ks=[1] if meta['family'] in ('MO','TMA') else protocol['K']
                # The original 16 D1 correctness suites remain valid, and are
                # reused; verify ABI/input again at K16 in this worker.
                if 'reused_tag' in meta:ks=[16]
                for k in ks:
                    if (tag,k,pattern) in already:continue
                    owner,params,_=s.params(tag,k);kernel=s.kernels[tag]
                    kernel.guard();kernel.launch(2,params,s.stream)
                    s.driver.call('cuStreamSynchronize',[C.c_void_p],s.stream)
                    result['checks'].append(s.check(tag,k,[0,1],strict=False))
            print(protocol['stage'],'correctness',pattern,'checked',len(result['checks']),
                  'failures',sum(not r['pass'] for r in result['checks']),flush=True)
        result['resources']={t:k.binding() for t,k in s.kernels.items()}
        result['driver_return_codes']=sorted(s.driver.codes)
        result['Triton_imported']=any(m=='triton' or m.startswith('triton.') for m in sys.modules)
        assert result['Triton_imported'] is False
        result['status']='OK' if all(r['pass'] for r in result['checks']) else 'CORRECTNESS_FAILED'
    except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if s:s.close()
    return result


def cells_for(protocol,gate):
    metadata=gate['metadata'];resources=gate['resources']
    cells=[];seen={}
    for tag,meta in metadata.items():
        if resources[tag]['attributes']['local_bytes']>0:continue
        if meta['family']=='TMA':
            if protocol['stage']=='C2b':continue
            variants=[('natural',meta['full_grid'],1)]
        else:
            grids=[1]+[x*protocol['target']['SM_count'] for x in protocol['timing']['grid_SM_multipliers']]
            ks=[1] if meta['family']=='MO' else protocol['K']
            variants=[('natural',g,k) for g in grids for k in ks]
            if protocol['stage']=='C2b':
                # All previous natural curves are reused verbatim. New probes
                # obtain natural curves plus q controls; q never reruns aliases.
                if meta.get('previous_batch'):variants=[]
                for q in ('1','2','4'):
                    if not resources[tag]['reservations'][q].get('feasible'):continue
                    variants += [(q,g,k) for g in grids if g>1 for k in ([1] if meta['family']=='MO' else protocol['fit_K'])]
        for q,g,k in variants:
            key=(meta['cubin_SHA256'],resources[tag]['reservations'][q]['dynamic_bytes'],g,k)
            cell={'tag':tag,'q':q,'grid':g,'K':k,'cell_id':f'{tag}|q{q}|G{g}|K{k}'}
            if key in seen:
                seen[key]['aliases'].append(cell['cell_id']);continue
            cell['aliases']=[];seen[key]=cell;cells.append(cell)
    return cells


def execute(invocation,protocol,gate,payload,call_id):
    root=Path('/tmp')/('c2-timing-'+call_id);c.extract(payload,root)
    c.verify_sources(protocol)
    result={'status':'FAILED','call_id':call_id,'invocation':invocation,'cells':[],
            'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload),
            'formal_timing':True,'profiler_attached':False,'samples_per_cell':20,'single_kernel_per_sample':True}
    s=None
    try:
        cells=cells_for(protocol,gate)
        maxgrid=max(x['grid'] for x in cells)
        s=Session(root,gate['metadata'],maxgrid=maxgrid)
        result['environment']=s.environment();result['allocated_bytes']=s.allocated_bytes
        rng=random.Random(protocol['timing']['order_seed']+invocation)
        # All cells have 4 interleaved visits, with fixed per-visit samples.
        order=list(range(len(cells)));rng.shuffle(order)
        rows=[{**cell,'samples_us':[],'visits':[],'correctness':None} for cell in cells]
        for visit in range(4):
            rng.shuffle(order)
            for index in order:
                cell=cells[index];tag=cell['tag'];kernel=s.kernels[tag]
                owners,params,serialized=s.params(tag,cell['K'])
                graph=Graph(kernel,cell['grid'],params,s.stream,cell['q'])
                for _ in range(protocol['timing']['warmups']):graph.run()
                samples=[graph.run() for _ in range(5)]
                row=rows[index];row['samples_us']+=samples
                row['visits'].append({'visit':visit,'samples_us':samples,'node_types':graph.node_types})
                graph.close()
                if visit==0:
                    pids=sorted(set([0,1,cell['grid']//2,cell['grid']-1])) if cell['grid']>1 else [0]
                    row['correctness']=s.check(tag,cell['K'],pids)
                    row['parameters']=serialized
                s.driver.calls.clear()  # counts/status retained; do not emit millions of duplicate API rows.
            print(protocol['stage'],'timing',invocation,'visit',visit+1,'/',4,'cells',len(cells),flush=True)
        result['cells']=rows
        result['bindings']={tag:kernel.binding() for tag,kernel in s.kernels.items()}
        result['driver_return_codes']=sorted(s.driver.codes);result['driver_call_counts']=dict(s.driver.counts)
        result['Triton_imported']=any(m=='triton' or m.startswith('triton.') for m in sys.modules)
        assert result['Triton_imported'] is False and result['driver_return_codes']==[0]
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if s:s.close()
    return result
