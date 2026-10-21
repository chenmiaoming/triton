"""Frozen-binary timing and full-output correctness; no compiler imports."""
import ctypes as C
from pathlib import Path
import random
import sys
import traceback
from experiments.tma_reduction_layout.phase11.m4 import common as c
from experiments.tma_reduction_layout.phase11.d1.runtime import Session as BaseSession, reference
from experiments.tma_reduction_layout.phase11.d1.driver import allocate, Graph
from .driver import Driver, Loaded
from experiments.tma_reduction_layout.phase11.c2 import data


class Session(BaseSession):
    def __init__(self, root, metadata, maxgrid=2, maxk=1):
        self.driver=Driver()
        d=self.driver
        self.context=C.c_void_p()
        d.call('cuCtxCreate_v2',[C.POINTER(C.c_void_p),C.c_uint,C.c_int],C.byref(self.context),0,d.device)
        self.cc=[d.attr(75),d.attr(76)]
        self.sm_count=d.attr(16)
        name=C.create_string_buffer(256)
        d.call('cuDeviceGetName',[C.c_void_p,C.c_int,C.c_int],C.cast(name,C.c_void_p),256,d.device)
        self.device_name=name.value.decode()
        assert self.cc==[9,0] and self.sm_count==132 and 'H100' in self.device_name
        self.stream=C.c_void_p()
        d.call('cuStreamCreate',[C.POINTER(C.c_void_p),C.c_uint],C.byref(self.stream),0)
        self.metadata=metadata
        self.kernels={tag:Loaded(d,Path(root),meta) for tag,meta in metadata.items()}
        self.input_rows=max(meta.get('global_rows',8192) for meta in metadata.values())
        sizes=(self.input_rows*8192*2,maxgrid*max(c.stride(meta) for meta in metadata.values())*4+64,256)
        assert sum(sizes)<12*1024**3
        self.pointers=[allocate(d,size) for size in sizes]
        self.pattern=None
        self.set_pattern('quarter')

    def check(self,tag,k,pids):
        meta=self.metadata[tag]
        if meta.get('output_policy') not in ('broadcast','centered','copy'):
            return super().check(tag,k,pids)
        import numpy as np
        width=c.stride(meta,k)
        actual=np.empty((len(pids),width),np.float32)
        for i,pid in enumerate(pids):
            self.driver.call('cuMemcpyDtoH_v2',[C.c_void_p,C.c_uint64,C.c_size_t],
                C.c_void_p(actual[i].ctypes.data),self.pointers[1].value+pid*width*4,actual[i].nbytes)
        m,n=meta['M'],meta['N']
        ids=np.asarray(pids)
        rows=ids[:,None,None]//(8192//n)*m+np.arange(m)[None,:,None]
        cols=(ids[:,None,None]%(8192//n))*n+np.arange(n)[None,None,:]
        values=data.values(rows*8192+cols,self.pattern)
        if meta['output_policy']=='copy':
            expected=values
        elif meta['output_policy']=='centered':
            expected=values-values.max(axis=1,keepdims=True)
        else:
            expected=values.max(axis=1,keepdims=True)+np.arange(m,dtype=np.float32)[None,:,None]
        expected=expected.reshape(len(pids),-1)
        okay=np.array_equal(actual,expected)
        return {'tag':tag,'K':k,'pids':pids,'pattern':self.pattern,'pass':bool(okay),
            'bit_exact':bool(okay),'absolute_tolerance':0.,'elements':actual.size,
            'max_abs_error':float(np.max(np.abs(actual-expected))),
            'actual_SHA256':c.sha(actual.tobytes()),'reference_SHA256':c.sha(expected.tobytes())}


def eligibility(protocol, metadata, root):
    result = {'status':'FAILED', 'checks':[], 'resources':{}, 'formal_timing':False}
    session = None
    try:
        grids = {tag: 2*1024**2//(2*m['M']*m['N']) for tag,m in metadata.items()}
        session = Session(root, metadata, maxgrid=max(grids.values())+1, maxk=1)
        result['environment'] = session.environment()
        for pattern in protocol['correctness']['patterns']:
            session.set_pattern(pattern)
            for tag,meta in metadata.items():
                grid = grids[tag]
                kernel = session.kernels[tag]
                n = grid*c.stride(meta,1)
                session.driver.call('cuMemsetD32_v2', [C.c_uint64,C.c_uint,C.c_size_t],
                                    session.pointers[1], 0x7fc00000, n+16)
                owners, params, _ = session.params(tag, 1)
                kernel.guard()
                kernel.launch(grid, params, session.stream)
                session.driver.call('cuStreamSynchronize', [C.c_void_p], session.stream)
                check = session.check(tag, 1, list(range(grid)))
                tail = (C.c_uint32*16)()
                session.driver.call('cuMemcpyDtoH_v2', [C.c_void_p,C.c_uint64,C.c_size_t],
                    C.cast(tail,C.c_void_p), session.pointers[1].value+n*4, 64)
                check['tail_canary_pass'] = all(x == 0x7fc00000 for x in tail)
                check['pass'] = check['pass'] and check['tail_canary_pass']
                check['complete_output_footprint_bytes'] = 2*1024**2
                result['checks'].append(check)
            print('M4 complete-output correctness',pattern,len(result['checks']),flush=True)
        result['resources'] = {tag:k.binding() for tag,k in session.kernels.items()}
        result['driver_return_codes'] = sorted(session.driver.codes)
        result['Triton_imported'] = any(n == 'triton' or n.startswith('triton.') for n in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes'] == [0]
        result['status'] = 'OK' if all(r['pass'] for r in result['checks']) else 'CORRECTNESS_FAILED'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'],flush=True)
    finally:
        if session:session.close()
    return result


def cells_for(protocol, gate):
    cells, seen = [], {}
    for tag,meta in gate['metadata'].items():
        if meta.get('profile_only'):continue
        for q in protocol['timing']['q_conditions']:
            reservation = gate['resources'][tag]['reservations'][q]
            if q != 'natural' and not reservation.get('feasible') and not reservation.get('alias_of'):
                continue
            actual_q = reservation.get('alias_of',q)
            for factor in protocol['timing']['full_grid_factors']:
                grid = (8192//meta['M'])*(8192//meta['N'])*factor
                key = (c.binary_execution_identity(meta),gate['resources'][tag]['reservations'][actual_q]['dynamic_bytes'],grid)
                identity = f'{tag}|q{q}|G{grid}|K1'
                if key in seen:
                    seen[key]['aliases'].append(identity)
                    continue
                row = {'tag':tag,'q':actual_q,'grid':grid,'K':1,'cell_id':identity,'aliases':[]}
                seen[key]=row
                cells.append(row)
    return cells


def execute(invocation, protocol, gate, payload, call_id):
    root = Path('/tmp')/('m4-timing-'+call_id)
    c.extract(payload,root)
    result = {'status':'FAILED','invocation':invocation,'call_id':call_id,
        'formal_timing':True,'profiler_attached':False,'cells':[],
        'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload)}
    session = None
    try:
        cells = cells_for(protocol,gate)
        max_output = max(row['grid']*c.stride(gate['metadata'][row['tag']]) for row in cells)
        session = Session(root,gate['metadata'],maxgrid=1,maxk=1)
        session.driver.call('cuMemFree_v2',[C.c_uint64],session.pointers[1])
        session.pointers[1] = allocate(session.driver,max_output*4)
        result['environment'] = session.environment()
        result['output_allocation_bytes'] = max_output*4
        rows = [{**cell,'samples_us':[],'visits':[]} for cell in cells]
        result['cells'] = rows
        order = list(range(len(cells)))
        rng = random.Random(protocol['timing']['order_seed']+invocation)
        session.driver.calls.clear()
        for visit in range(protocol['timing']['visits']):
            rng.shuffle(order)
            for index in order:
                cell = cells[index]
                kernel = session.kernels[cell['tag']]
                owners,params,serialized = session.params(cell['tag'],1)
                graph = Graph(kernel,cell['grid'],params,session.stream,cell['q'])
                for _ in range(protocol['timing']['warmups']):graph.run()
                samples = [graph.run() for _ in range(protocol['timing']['samples_per_visit'])]
                rows[index]['samples_us'] += samples
                rows[index]['visits'].append({'visit':visit,'samples_us':samples,'node_types':graph.node_types})
                graph.close()
                if visit == 0:
                    check = session.check(cell['tag'],1,sorted({0,cell['grid']//2,cell['grid']-1}))
                    assert check['pass'],check
                    rows[index]['correctness']=check
                    rows[index]['parameters']=serialized
                session.driver.calls.clear()
            print('M4 timing',invocation,'visit',visit+1,'cells',len(cells),flush=True)
            c.write_once(Path('/evidence/m4/timing')/call_id/f'checkpoint_{visit+1}.json',
                {'invocation':invocation,'completed_visits':visit+1,'protocol_SHA256':result['protocol_SHA256'],'cells':rows})
        result['bindings']={tag:k.binding() for tag,k in session.kernels.items()}
        result['driver_return_codes']=sorted(session.driver.codes)
        result['driver_call_counts']=dict(session.driver.counts)
        result['Triton_imported']=any(n=='triton' or n.startswith('triton.') for n in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes']==[0]
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc()
        print(result['error'],flush=True)
    finally:
        if session:session.close()
    return result
