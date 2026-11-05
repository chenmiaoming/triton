"""Validate the frozen CPU ABI against actual CUDA tensor JIT call sites."""
from pathlib import Path
from . import common as c
from .layout_parsing import instantiate


def run(protocol,compiled):
    import torch
    from triton.experimental.gluon import language as gl
    from .g3b_triton import _layer_norm_bwd_dwdb as original,_layer_norm_bwd_dx_fused as dx
    from .g3b_gluon import merge
    out=[]
    for case in protocol['cases']:
        p,n=case['P'],case['N'];plan=compiled['plans'][case['id']]
        arrays={k:torch.empty((p,n),dtype=torch.bfloat16,device='cuda') for k in ('DW','DB')}
        arrays.update({k:torch.empty(n,dtype=torch.bfloat16,device='cuda') for k in ('FINAL_DW','FINAL_DB','W')})
        arrays.update({k:torch.empty((128,n),dtype=torch.bfloat16,device='cuda') for k in ('X','DY','DX')})
        arrays.update({k:torch.empty(128,dtype=torch.float32,device='cuda') for k in ('Mean','Rstd')})
        arrays['Lock']=torch.zeros(2*p,dtype=torch.int32,device='cuda')
        for row in compiled['records']:
            if row['geometry']!=case['id']:continue
            if row['role']=='dx':fn=dx;constants={'GROUP_SIZE_M':p,'BLOCK_SIZE_N':n}
            elif row['tag'].endswith(':upstream'):fn=original;constants={'BLOCK_SIZE_M':32,'BLOCK_SIZE_N':128}
            else:
                fn=merge;constants={'BLOCK_SIZE_M':32,'BLOCK_SIZE_N':128,'L':instantiate(row['intervention_layout'],gl),
                                   'LOUT':instantiate(plan['output'],gl),'NATIVE_OUT':row['native_output']}
            vals={'M':p,'N':n,'stride':n}
            args={key:arrays[key] if key in arrays else constants[key] if key in constants else vals[key] for key in fn.arg_names}
            code=fn.warmup(**args,grid=(128,) if row['role']=='dx' else ((n+127)//128,),num_warps=row['num_warps'],num_stages=3)
            digest=c.sha(code.asm['cubin'])
            out.append({'tag':row['tag'],'CPU_CUBIN':row['cubin_SHA256'],'real_JIT_CUBIN':digest,'match':digest==row['cubin_SHA256']})
        del arrays
    assert all(x['match'] for x in out),[x for x in out if not x['match']]
    return {'status':'OK','records':out,'actual_cuda_tensor_JIT_match':True}
