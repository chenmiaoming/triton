"""Frozen CUBIN loader, launch-only shared reservation and one-kernel graphs."""
import ctypes as C
import re
from experiments.tma_reduction_layout.phase8.launch import Driver
from experiments.tma_reduction_layout.phase11.launch import Loaded as Descriptor
from experiments.tma_reduction_layout.phase11.d1 import common as c


class Pointer:
    def __init__(self,value):
        self.value=value
    def data_ptr(self):
        return self.value


def allocate(driver,size):
    pointer=C.c_uint64()
    driver.call('cuMemAlloc_v2',[C.POINTER(C.c_uint64),C.c_size_t],C.byref(pointer),max(1,size))
    return pointer


def upload(driver,pointer,data):
    host=C.create_string_buffer(data)
    driver.call('cuMemcpyHtoD_v2',[C.c_uint64,C.c_void_p,C.c_size_t],pointer,C.cast(host,C.c_void_p),len(data))


class Loaded:
    def descriptor(self, x):
        owner = C.create_string_buffer(128+63)
        address = (C.addressof(owner)+63)&~63
        spec = self.metadata['tensordesc_meta']
        assert isinstance(spec,list) and len(spec)==1, spec
        spec = spec[0]
        swizzle = spec['swizzle']
        assert swizzle in (1,2,3)
        elem_bytes = 2 if self.metadata['dtype']=='bfloat16' else 4
        m,n = self.metadata['M'],self.metadata['N']
        shape = (C.c_uint64*3)(8192,self.metadata.get('global_rows',8192),1)
        strides = (C.c_uint64*2)(8192*elem_bytes,8192*self.metadata.get('global_rows',8192)*elem_bytes)
        box = (C.c_uint32*3)(min(n,{1:32,2:64,3:128}[swizzle]//elem_bytes),m,1)
        element_strides = (C.c_uint32*3)(1,1,1)
        dtype_enum = 9 if elem_bytes==2 else 7
        self.driver.call('cuTensorMapEncodeTiled', [C.c_void_p,C.c_int,C.c_uint32,C.c_void_p,
            C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.POINTER(C.c_uint32),C.POINTER(C.c_uint32),
            C.c_int,C.c_int,C.c_int,C.c_int],C.c_void_p(address),dtype_enum,3,C.c_void_p(x.data_ptr()),
            shape,strides,box,element_strides,0,swizzle,2,0)
        return owner,address


    def __init__(self,driver,root,meta):
        self.driver,self.metadata=driver,meta
        self.path=root/meta['tag'].replace(':','/')/'kernel.cubin'
        blob=self.path.read_bytes()
        self.digest=c.sha(blob)
        assert blob.startswith(b'\x7fELF') and self.digest==meta['cubin_SHA256']
        self.buffer=C.create_string_buffer(blob)
        self.module,self.function=C.c_void_p(),C.c_void_p()
        driver.call('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p],C.byref(self.module),C.cast(self.buffer,C.c_void_p))
        driver.call('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p],C.byref(self.function),self.module,meta['function_name'].encode())
        text=self.path.with_suffix('.ptx').read_text()
        entry=re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid',text,re.S)
        assert entry
        self.abi=['descriptor128' if '.b8' in p and '[128]' in p else 'u32' if '.u32' in p else 'u64' if '.u64' in p else 'UNKNOWN'
                  for p in entry[1].splitlines() if '.param' in p]
        expected=({'descriptor_K':['descriptor128']+['u32']*3+['u64']*4+['u32']+['u64']*2,
                   'descriptor':['descriptor128']+['u32']*3+['u64']*6,
                   'pointer':['u64']*4,'pointer_K':['u64','u64','u32','u64','u64']})[meta['ABI_kind']]
        assert self.abi==expected,(meta['tag'],self.abi,expected)
        self.guards=self.captures=self.graph_launches=self.direct_launches=0
        self.attributes={}
        for key,code in (('num_regs',4),('static_smem_bytes',1),('local_bytes',3)):
            value=C.c_int()
            driver.call('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p],C.byref(value),code,self.function)
            self.attributes[key]=value.value
        optin=driver.attr(97)-self.attributes['static_smem_bytes']
        driver.call('cuFuncSetAttribute',[C.c_void_p,C.c_int,C.c_int],self.function,8,optin)
        driver.call('cuFuncSetAttribute',[C.c_void_p,C.c_int,C.c_int],self.function,9,100)
        for key,code in (('max_dynamic_smem_bytes',8),('preferred_shared_carveout',9)):
            value=C.c_int()
            driver.call('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_void_p],C.byref(value),code,self.function)
            self.attributes[key]=value.value
        assert self.attributes['preferred_shared_carveout']==100
        natural=meta['dynamic_smem_bytes']
        cap=self.capacity(natural)
        self.reservations={'natural':{'dynamic_bytes':natural,'capacity':cap,'target':None}}
        for q in (1,2,4,8):
            if cap<q:
                self.reservations[str(q)]={'feasible':False,'reason':'natural capacity below target','target':q}
                continue
            if cap==q:
                self.reservations[str(q)]={'feasible':False,'alias_of':'natural','target':q,'capacity':cap}
                continue
            lo=(natural+255)//256
            hi=optin//256
            while lo<hi:
                mid=(lo+hi)//2
                if self.capacity(mid*256)<=q:
                    hi=mid
                else:
                    lo=mid+1
            size=lo*256
            actual=self.capacity(size)
            self.reservations[str(q)]=({'feasible':True,'dynamic_bytes':size,'capacity':actual,'target':q}
                                      if actual==q else {'feasible':False,'target':q,'reason':'capacity granularity','capacity':actual})

    def capacity(self,dynamic):
        value=C.c_int()
        self.driver.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',[C.POINTER(C.c_int),C.c_void_p,C.c_int,C.c_size_t],
                         C.byref(value),self.function,self.metadata['num_warps']*32,dynamic)
        return value.value

    def parameters(self,x,out,scratch,k):
        meta=self.metadata
        if meta['ABI_kind'].startswith('descriptor'):
            owner,address=self.descriptor(x)
            values=[C.c_uint32(1),C.c_uint32(self.metadata.get('global_rows',8192)),C.c_uint32(8192),C.c_uint64(self.metadata.get('global_rows',8192)*8192),C.c_uint64(8192),C.c_uint64(1),C.c_uint64(out.data_ptr())]
            if meta['ABI_kind']=='descriptor_K':
                values.append(C.c_uint32(k))
            values += [C.c_uint64(scratch.data_ptr()),C.c_uint64(0)]
            addresses=[address]+[C.addressof(v) for v in values]
            owners=(owner,values)
            serialized=[address]+[v.value for v in values]
        else:
            values=[C.c_uint64(x.data_ptr()),C.c_uint64(out.data_ptr())]
            if meta['ABI_kind']=='pointer_K':
                values.append(C.c_uint32(k))
            values += [C.c_uint64(scratch.data_ptr()),C.c_uint64(0)]
            addresses=[C.addressof(v) for v in values]
            owners=values
            serialized=[v.value for v in values]
        return owners,(C.c_void_p*len(addresses))(*addresses),serialized

    def guard(self):
        assert c.sha(self.path.read_bytes())==self.digest==c.sha(self.buffer.raw[:-1])
        self.guards+=1

    def launch(self,grid,params,stream,q='natural'):
        reservation=self.reservations[q]
        assert q=='natural' or reservation.get('feasible'),reservation
        self.driver.call('cuLaunchKernel',[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p],
                         self.function,grid,1,1,self.metadata['num_warps']*32,1,1,reservation['dynamic_bytes'],stream,params,None)
        self.direct_launches+=1

    def binding(self):
        return {'tag':self.metadata['tag'],'CUBIN_SHA256':self.digest,'loaded_CUBIN_SHA256':c.sha(self.buffer.raw[:-1]),
                'ABI':self.abi,'attributes':self.attributes,'reservations':self.reservations,'guards':self.guards,
                'captures':self.captures,'graph_launches':self.graph_launches,'direct_launches':self.direct_launches}


class Graph:
    def __init__(self,kernel,grid,params,stream,q='natural'):
        self.kernel,self.driver,self.stream=kernel,kernel.driver,stream
        d=self.driver
        self.first,self.last=d.event(),d.event()
        self.graph,self.executable=C.c_void_p(),C.c_void_p()
        d.call('cuStreamBeginCapture',[C.c_void_p,C.c_int],stream,0)
        d.call('cuEventRecordWithFlags',[C.c_void_p,C.c_void_p,C.c_uint],self.first,stream,1)
        kernel.guard(); kernel.launch(grid,params,stream,q)
        kernel.captures+=1
        d.call('cuEventRecordWithFlags',[C.c_void_p,C.c_void_p,C.c_uint],self.last,stream,1)
        d.call('cuStreamEndCapture',[C.c_void_p,C.POINTER(C.c_void_p)],stream,C.byref(self.graph))
        d.call('cuGraphInstantiateWithFlags',[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint64],C.byref(self.executable),self.graph,0)
        count=C.c_size_t()
        d.call('cuGraphGetNodes',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],self.graph,None,C.byref(count))
        nodes=(C.c_void_p*count.value)()
        d.call('cuGraphGetNodes',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)],self.graph,nodes,C.byref(count))
        types=[]
        for node in nodes:
            value=C.c_int()
            d.call('cuGraphNodeGetType',[C.c_void_p,C.POINTER(C.c_int)],node,C.byref(value))
            types.append(value.value)
        assert sorted(types)==[0,7,7],types
        self.node_types=types

    def run(self):
        self.kernel.guard()
        self.driver.call('cuGraphLaunch',[C.c_void_p,C.c_void_p],self.executable,self.stream)
        self.kernel.graph_launches+=1
        self.driver.call('cuStreamSynchronize',[C.c_void_p],self.stream)
        return self.driver.elapsed_us(self.first,self.last)

    def close(self):
        self.driver.call('cuGraphExecDestroy',[C.c_void_p],self.executable)
        self.driver.call('cuGraphDestroy',[C.c_void_p],self.graph)
        for event in (self.first,self.last):
            self.driver.call('cuEventDestroy_v2',[C.c_void_p],event)
