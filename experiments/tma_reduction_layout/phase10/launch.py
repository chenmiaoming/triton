"""Checked launches of Stage A archived ELF bytes, with explicit issue ABI."""
import ctypes as C
import re
from pathlib import Path
from experiments.tma_reduction_layout.phase8.launch import Driver
from experiments.tma_reduction_layout.phase10.common import sha


class Loaded:
    def __init__(self,driver,root,tag,metadata):
        self.driver,self.tag,self.metadata=driver,tag,metadata
        self.path=Path(root)/tag.replace(':','/')/'kernel.cubin'
        data=self.path.read_bytes()
        self.digest=sha(data)
        assert data.startswith(b'\x7fELF') and self.digest==metadata['cubin_SHA256']
        self.buffer=C.create_string_buffer(data)
        ptx=self.path.with_suffix('.ptx').read_text()
        entry=re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid',ptx,re.S)
        assert entry
        self.abi=['descriptor128' if '.b8' in p and '[128]' in p else 'u32' if '.u32' in p else 'u64' if '.u64' in p else 'UNKNOWN' for p in entry[1].splitlines() if '.param' in p]
        expected=['descriptor128']+['u32']*3+['u64']*6 if metadata['path']=='host' else ['u64']*4
        assert self.abi==expected,(tag,self.abi,expected)
        self.module,self.function=C.c_void_p(),C.c_void_p()
        driver.call('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p],C.byref(self.module),C.cast(self.buffer,C.c_void_p))
        driver.call('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p],C.byref(self.function),self.module,metadata['function_name'].encode())
        dynamic=metadata['dynamic_smem_bytes']
        if dynamic>49152:driver.call('cuFuncSetAttribute',[C.c_void_p,C.c_int,C.c_int],self.function,8,dynamic)
        self.guards=self.launches=0

    def descriptor(self,x):
        owner=C.create_string_buffer(128+63)
        address=(C.addressof(owner)+63)&~63
        meta=self.metadata['tensordesc_meta']
        assert isinstance(meta,list) and len(meta)==1,meta
        spec=meta[0]
        swizzle=spec['swizzle']
        # Triton records the enum value, not a byte width.
        assert swizzle in (1,2,3),spec
        m,n=self.metadata['M'],self.metadata['N']
        shape=(C.c_uint64*3)(8192,8192,1)
        strides=(C.c_uint64*2)(8192*2,8192*8192*2)
        box=(C.c_uint32*3)(min(n,{1:16,2:32,3:64}[swizzle]),m,1)
        element_strides=(C.c_uint32*3)(1,1,1)
        self.driver.call('cuTensorMapEncodeTiled',[C.c_void_p,C.c_int,C.c_uint32,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.POINTER(C.c_uint32),C.POINTER(C.c_uint32),C.c_int,C.c_int,C.c_int,C.c_int],
            C.c_void_p(address),9,3,C.c_void_p(x.data_ptr()),shape,strides,box,element_strides,0,swizzle,2,0)
        return owner,address

    def parameters(self,x,out,scratch):
        if self.metadata['path']=='host':
            owner,address=self.descriptor(x)
            values=[C.c_uint32(1),C.c_uint32(8192),C.c_uint32(8192),C.c_uint64(8192*8192),C.c_uint64(8192),C.c_uint64(1),C.c_uint64(out.data_ptr()),C.c_uint64(scratch.data_ptr()),C.c_uint64(0)]
            addresses=[address]+[C.addressof(v) for v in values]
            return (owner,values),(C.c_void_p*len(addresses))(*addresses),[address]+[v.value for v in values]
        values=[C.c_uint64(x.data_ptr()),C.c_uint64(out.data_ptr()),C.c_uint64(scratch.data_ptr()),C.c_uint64(0)]
        return values,(C.c_void_p*len(values))(*[C.addressof(v) for v in values]),[v.value for v in values]

    def guard(self):
        assert sha(self.path.read_bytes())==self.digest==sha(self.buffer.raw[:-1])
        self.guards+=1
        return self.digest

    def launch(self,params,stream):
        self.driver.call('cuLaunchKernel',[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p],
            self.function,self.metadata['grid'],1,1,self.metadata['num_warps']*32,1,1,self.metadata['dynamic_smem_bytes'],stream,params,None)
        self.launches+=1

    def record(self):
        return {'tag':self.tag,'cubin_SHA256':self.digest,'module':hex(self.module.value),'function':hex(self.function.value),'launches':self.launches,'guards':self.guards,'ABI':self.abi}
