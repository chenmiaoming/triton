"""Local SM86 execution of archived programs; no compilation or Triton import."""
import ctypes as C
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device as PointerDevice
from experiments.tma_reduction_layout.phase8.launch import Driver
from experiments.tma_reduction_layout.local_a10.experiment import Loaded as OldLoaded
from experiments.layout_coordination.clock_control.telemetry import NVML, Sampler, collect_env_readonly
from .engine import Session
from .a10_prepare import UUID, cpu_gate


class Device(PointerDevice):
    """Real A10 initialization; does not execute or bypass the H100 initializer."""
    def __init__(self):
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
        self.driver = Driver()
        d = self.driver
        count = C.c_int()
        d.call('cuDeviceGetCount', [C.POINTER(C.c_int)], C.byref(count))
        assert count.value == 1
        name = C.create_string_buffer(256)
        d.call('cuDeviceGetName', [C.c_void_p,C.c_int,C.c_int], name,256,d.device)
        self.environment = {**d.environment(), 'name':name.value.decode(),
            'cc':[d.attr(75),d.attr(76)], 'SM_count':d.attr(16),
            'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
            'physical_device_index':0, 'PID':os.getpid()}
        assert self.environment['gpu_uuid'] == UUID
        assert self.environment['cc'] == [8,6] and self.environment['name'] == 'NVIDIA A10'
        assert self.environment['SM_count'] > 0
        self.context,self.stream = C.c_void_p(),C.c_void_p()
        try:
            d.call('cuCtxCreate_v2',[C.POINTER(C.c_void_p),C.c_uint,C.c_int], C.byref(self.context),0,d.device)
            d.call('cuStreamCreate',[C.POINTER(C.c_void_p),C.c_uint],C.byref(self.stream),0)
        except Exception:
            if self.stream.value:d.call('cuStreamDestroy_v2',[C.c_void_p],self.stream)
            if self.context.value:d.call('cuCtxDestroy_v2',[C.c_void_p],self.context)
            raise
        self.allocations,self.kernels = [],[]


class A10Session(Session):
    """Reuse frozen sampling logic, with an explicit actual-device SM86 contract."""
    def __init__(self, device, protocol, root, invocation, suite='a10'):
        self.device,self.protocol,self.root,self.invocation,self.suite = device,protocol,Path(root),invocation,suite
        assert device.environment['gpu_uuid'] == protocol['target']['gpu_uuid'] == UUID
        assert device.environment['cc'] == protocol['target']['cc'] == [8,6]
        assert device.environment['name'] == protocol['target']['name'] == 'NVIDIA A10'
        for name,digest in protocol['source_SHA256'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest,name
        self.root.mkdir(parents=True,exist_ok=True)
        self.rng = random.Random(protocol['timing']['seed']+invocation)
        self.last_completion = None
        self.reader = NVML(device.environment['gpu_uuid'])
        try:
            self.sampler = Sampler(self.reader,period_s=protocol['timing']['telemetry_period_s'])
            self.sampler.set_context(phase='setup',suite=suite)
            self.sampler.start()
        except Exception:
            self.reader.close()
            raise
        self.result = dict(status='FAILED',suite=suite,invocation=invocation,
            formal_timing=invocation>0, protocol_SHA256=c.sha(c.encode(protocol)),
            payload_SHA256=protocol['suites'][suite]['payload_SHA256'],
            environment={**device.environment,'readonly':collect_env_readonly(self.reader)},
            clock_control='unlocked', compile_calls=0,native_rebuilds=0,profiler_attached=False,
            shards=[],checks=[],graph_proofs=[],partial_errors=[],process_boundaries=[],
            cohort_scope='Sequential temporal cohorts on one physical A10; not independent hardware.',
            sampling=dict(primes=protocol['timing']['primes'],
                requested_host_gap_seconds=protocol['timing']['host_gap_seconds'],
                target_semantic_iterations=1,
                inter_prime_target_gap='GPU graph dependency; no host submission'))


class Loaded(OldLoaded):
    def __init__(self, device, root, program):
        self.program = program
        super().__init__(device.driver, Path(root)/program['panel'], program['metadata'])
        self.graph_launches = self.captures = 0
        self.attributes = {}
        for key,code in (('n_regs',4),('static_smem_bytes',1),('local_bytes',3)):
            value=C.c_int()
            self.driver.call('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p],
                C.byref(value),code,self.function)
            self.attributes[key]=value.value
        assert self.attributes['n_regs'] == self.meta['num_regs']
        assert self.meta['num_spills'] == self.attributes['local_bytes'] == 0
        assert self.meta['global_scratch_size'] == 0
        capacity=C.c_int()
        self.driver.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',
            [C.POINTER(C.c_int),C.c_void_p,C.c_int,C.c_size_t],C.byref(capacity),
            self.function,self.meta['case']['num_warps']*32,self.meta['dynamic_smem_bytes'])
        self.attributes['occupancy_capacity_q']=capacity.value
        device.kernels.append(self)

    def binding(self):
        return {**super().binding(), 'tag':self.program['tag'], 'role':self.program['role'],
            'panel':self.program['panel'], 'graph_launches':self.graph_launches,
            'captures':self.captures, 'attributes':self.attributes,
            'dynamic_smem_bytes':self.meta['dynamic_smem_bytes'],
            'grid':self.meta['case']['grid'], 'ABI_types':self.program['ABI']}


def quarter_input():
    """All pattern values are exactly representable in BF16; no rounding proxy."""
    period = ((np.arange(97,dtype=np.int32)-48).astype(np.float32)/4)
    bits = (period.view(np.uint32)>>16).astype(np.uint16)
    return np.resize(bits,8192*8192).tobytes()


def expected_output():
    # Phase table covers every possible linear input start. Its construction is
    # independent of the kernel's vec, lane/warp assignment and shared layout.
    phase = np.arange(97,dtype=np.int64)[:,None]
    rows = np.arange(32,dtype=np.int64)[None,:]*8192
    table = (((phase+rows)%97-48)/4).max(axis=1).astype(np.float32)
    start = (np.arange(256,dtype=np.int64)[:,None]*(32*8192)
             +np.arange(8192,dtype=np.int64)[None,:])%97
    return table[start].astype(np.float32)


class Pointer:
    def __init__(self,value):self.value=value
    def data_ptr(self):return self.value


def process_boundary(phase):
    argv=['nvidia-smi','-i',UUID,
          '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']
    query=subprocess.run(argv,capture_output=True,text=True)
    rows=[line.split(',') for line in query.stdout.splitlines() if line.strip()]
    own=os.getpid()
    okay=(query.returncode==0 and all(row[0].strip()==UUID and int(row[1])==own for row in rows))
    return dict(phase=phase,argv=argv,returncode=query.returncode,stdout=query.stdout,
        stderr=query.stderr,monotonic_ns=time.monotonic_ns(),PID=own,**{'pass':okay},
        scope='Boundary process query only; not continuous exclusive-device proof.')


def run(protocol,payload,root,invocation):
    """One fresh CUDA process; old raw artifacts are never written."""
    root=Path(root)
    cpu_gate(protocol,payload)
    c.extract(payload,root/'program_files')
    device=session=None
    loaded=[];error=None
    try:
        device=Device()
        session=A10Session(device,protocol,root,invocation)
        input_blob=quarter_input(); expected=expected_output()
        input_digest=c.sha(input_blob)
        guard=protocol['correctness']['byte_guards']
        prefix=b'\xa5'*guard;suffix=b'\x5a'*guard
        output_bytes=expected.nbytes
        xbase=device.allocate(len(input_blob)+2*guard)
        ybase=device.allocate(output_bytes+2*guard)
        x,y=Pointer(xbase+guard),Pointer(ybase+guard)
        device.upload(xbase,prefix+input_blob+suffix)
        device.upload(ybase,prefix+b'\xcd'*output_bytes+suffix)

        def read_check():
            out=np.frombuffer(device.download(y.value,output_bytes),dtype=np.float32)
            target=expected.reshape(-1)
            finite=bool(np.isfinite(out).all())
            equal=bool(np.array_equal(out,target))
            canary=all(device.download(base,guard)==prefix and
                device.download(base+size+guard,guard)==suffix
                for base,size in ((xbase,len(input_blob)),(ybase,output_bytes)))
            return dict(**{'pass':finite and equal and canary},
                all_output_elements=len(out), finite=finite, exact_equal=equal,
                canaries_pass=canary, output_SHA256=c.sha(out.tobytes()),
                expected_SHA256=c.sha(expected.tobytes()),
                max_abs_error=float(np.max(np.abs(out-target))) if finite else None)

        programs=protocol['suites']['a10']['programs']
        for panel in ('layout','versions'):
            for geometry in sorted({x['geometry'] for x in programs}):
                selected=[x for x in programs if x['panel']==panel and x['geometry']==geometry]
                items=[]
                for program in selected:
                    kernel=Loaded(device,root/'program_files',program);loaded.append(kernel)
                    owners,params=kernel.parameters(x,y)
                    def launch(k=kernel,p=params,keep=owners):k.launch(p,device.stream)
                    def check(launch=launch):
                        device.upload(y.value,np.full(expected.size,np.nan,np.float32).tobytes())
                        launch();device.synchronize()
                        return read_check()
                    items.append(dict(tag=program['tag'],role=program['role'],aliases=[],kernels=[kernel],
                        launch=launch,prepare=None,check=check,check_graph=read_check,
                        expected_nodes=dict(kernels=1,memsets=0,copies=0)))
                case=selected[0]['metadata']['case']
                context=dict(id=panel+':'+geometry,panel=panel,geometry=geometry,
                    M=case['BM'],N=case['BN'],W=case['num_warps'],grid=case['grid'],
                    semantics='column_max',input_bytes=len(input_blob),output_elements=expected.size,
                    original_programs=len(items),input_pattern='quarter',TMA=False)
                boundary=process_boundary(context['id']+':before')
                session.result['process_boundaries'].append(boundary)
                assert boundary['pass'],boundary
                session.measure(context,items)
                boundary=process_boundary(context['id']+':after')
                session.result['process_boundaries'].append(boundary)
                assert boundary['pass'],boundary
                readonly=(c.sha(device.download(x.value,len(input_blob)))==input_digest)
                session.result['checks'].append(dict(context=context,phase='readonly_input_after',
                    tag='input',**{'pass':readonly},input_SHA256=input_digest,bytes=len(input_blob)))
                assert readonly
                print('A10 completed',invocation,panel,geometry,flush=True)
    except Exception:
        error=traceback.format_exc()
        print(error,flush=True)
    finally:
        if session is not None:
            result=session.finish([k.binding() for k in loaded],error=error)
        else:
            cleanup=[]
            if device is not None:
                try:device.close()
                except Exception:cleanup.append(traceback.format_exc())
            result=dict(status='FAILED',invocation=invocation,error=error,
                phase='device_or_session_initialization',cleanup_errors=cleanup,
                protocol_SHA256=c.sha(c.encode(protocol)),GPU_timing_started=False)
            c.write_once(root/'result.json',result)
    return result
