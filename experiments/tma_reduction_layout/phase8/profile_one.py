"""One archived launch inside an explicitly delimited diagnostic profiler region."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase8.launch import Driver,Loaded


def main():
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--bundle-root');parser.add_argument('--binding-file');parser.add_argument('--key');parser.add_argument('--mode',type=int);args=parser.parse_args()
    binding=json.loads(Path(args.binding_file).read_text())[args.key];meta=binding['metadata']
    _,harness,_=args.key.split(':');_,m,n=meta['logical_shape']
    torch.manual_seed(42);x=torch.randn((65536,m,n),dtype=torch.bfloat16,device='cuda')
    out=torch.empty((65536,n),dtype=torch.float32,device='cuda');scratch=torch.empty(65536*128,dtype=torch.uint8,device='cuda')
    torch.cuda.synchronize();driver=Driver();root=Path(args.bundle_root)
    ptx=(root/binding['archive_path']).with_suffix('.ptx').read_text();kernel=Loaded(driver,root,binding,ptx)
    descriptor=driver.tensor_map(x.data_ptr(),m,n,binding['shared_layout']['swizzlingByteWidth']) if harness.startswith('host_') or harness=='switch' else None
    values,params=kernel.parameters((x,out,scratch),args.mode,descriptor);stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
    # Initialization, warmup and SHA guard are outside the profiler region.
    kernel.guard();kernel.launch(16384,params,stream);driver.call('cuStreamSynchronize',[C.c_void_p],stream)
    digest=kernel.guard();driver.call('cuProfilerStart',[]);kernel.launch(16384,params,stream)
    driver.call('cuStreamSynchronize',[C.c_void_p],stream);driver.call('cuProfilerStop',[])
    error=(out[:4]-x[:4].float().amax(dim=1)).abs().max().item()
    if error!=0:raise RuntimeError('Profiled archived output incorrect')
    driver.call('cuModuleUnload',[C.c_void_p],kernel.module)
    if any(name=='triton' or name.startswith('triton.') for name in sys.modules):raise RuntimeError('Profiler script imported Triton')
    print('PROFILE_ARCHIVE_BINDING '+json.dumps({'key':args.key,'mode':args.mode,'archive_sha256':digest,'loaded_cubin_sha256':kernel.digest,
          'launch_count':kernel.launches,'sha_guard_count':kernel.guards,'B_DESC':65536,'B_RUN':16384,'correctness_prefix4_max_abs_diff':error,
          'triton_imported':False,'compile_calls':0},sort_keys=True))


if __name__=='__main__':main()
