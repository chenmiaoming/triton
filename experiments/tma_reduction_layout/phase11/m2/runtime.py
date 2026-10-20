"""Frozen-binary timing and full-output correctness; no compiler imports."""
import ctypes as C
from pathlib import Path
import random
import sys
import traceback
from experiments.tma_reduction_layout.phase11.m2 import common as c
from experiments.tma_reduction_layout.phase11.d1.runtime import Session
from experiments.tma_reduction_layout.phase11.d1.driver import allocate, Graph


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
                n = grid*meta['N']
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
            print('M2 complete-output correctness',pattern,len(result['checks']),flush=True)
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
        for q in protocol['timing']['q_conditions']:
            reservation = gate['resources'][tag]['reservations'][q]
            if q != 'natural' and not reservation.get('feasible') and not reservation.get('alias_of'):
                continue
            actual_q = reservation.get('alias_of',q)
            for factor in protocol['timing']['full_grid_factors']:
                grid = (8192//meta['M'])*(8192//meta['N'])*factor
                key = (meta['cubin_SHA256'],gate['resources'][tag]['reservations'][actual_q]['dynamic_bytes'],grid)
                identity = f'{tag}|q{q}|G{grid}|K1'
                if key in seen:
                    seen[key]['aliases'].append(identity)
                    continue
                row = {'tag':tag,'q':actual_q,'grid':grid,'K':1,'cell_id':identity,'aliases':[]}
                seen[key]=row
                cells.append(row)
    return cells


def execute(invocation, protocol, gate, payload, call_id):
    root = Path('/tmp')/('m2-timing-'+call_id)
    c.extract(payload,root)
    result = {'status':'FAILED','invocation':invocation,'call_id':call_id,
        'formal_timing':True,'profiler_attached':False,'cells':[],
        'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload)}
    session = None
    try:
        cells = cells_for(protocol,gate)
        max_output = max(row['grid']*gate['metadata'][row['tag']]['N'] for row in cells)
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
            print('M2 timing',invocation,'visit',visit+1,'cells',len(cells),flush=True)
            c.write_once(Path('/evidence/stage_m2/timing')/call_id/f'checkpoint_{visit+1}.json',
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
