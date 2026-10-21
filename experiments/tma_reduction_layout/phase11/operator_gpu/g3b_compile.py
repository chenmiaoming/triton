"""CPU-only compilation with explicit typed/aligned ABI; no native rebuild."""
from pathlib import Path
import subprocess,traceback
from . import common as c
from .layout_parsing import encodings,tensor_encoding,instantiate
from .production_compile import export


def run(protocol,root):
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from .g3b_triton import _layer_norm_bwd_dwdb as original,_layer_norm_bwd_dx_fused as dx
    from .g3b_gluon import merge
    c.verify_sources(protocol)
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    result={'status':'FAILED','records':[],'plans':{},'GPU_allocated':False,'native_rebuilds':0,
            'ccache_before':subprocess.check_output(['ccache','--print-stats'],text=True),
            'native_SHA256':c.NATIVE,'protocol_SHA256':c.sha(c.encode(protocol))}
    def compile_fn(fn,constants,pointers,case,tag,warps=4):
        sig={name:'constexpr' if name in constants else pointers.get(name,'i32') for name in fn.arg_names}
        attrs={(i,):[['tt.divisibility',16]] for i,name in enumerate(fn.arg_names) if name not in constants}
        cls=gluon.GluonASTSource if fn is merge else triton.compiler.ASTSource
        code=triton.compile(cls(fn,sig,constants,attrs),target=GPUTarget('cuda',90,32),options={'num_warps':warps,'num_stages':3})
        metadata=export(code,fn,{**case,'W':warps},tag,root,constants)
        metadata['role']='dx' if fn is dx else 'merge'
        result['records'].append(metadata)
        print('G3b CPU',tag,'PASS',flush=True)
        return code,metadata
    try:
        ptr={n:'*bf16' for n in ['DW','DB','FINAL_DW','FINAL_DB']}
        for case in protocol['cases']:
            constants={'BLOCK_SIZE_M':32,'BLOCK_SIZE_N':128}
            code,meta=compile_fn(original,constants,ptr,case,case['id']+':upstream')
            ir=code.asm['ttgir'];defs=encodings(ir)
            load=next(s for s in ir.splitlines() if 'tt.load' in s and 'tensor<32x128x' in s)
            store=next(s for s in ir.splitlines() if 'tt.store' in s and 'tensor<128x' in s)
            plan={'input':tensor_encoding(load,[32,128],defs),'output':tensor_encoding(store,[128],defs)}
            result['plans'][case['id']]=plan
            variants=[{'name':'gluon_anchor','L':plan['input']},
                      *[{'name':x['name'],'L':{'kind':'blocked','fields':x['layout']}} for x in case['layouts']]]
            variants += [{'name':f'vec{v}:n_first:native','L':next(x['L'] for x in variants if x['name']==f'vec{v}:n_first'),'native':True} for v in (1,8)]
            for variant in variants:
                const={**constants,'L':instantiate(variant['L'],gl),'LOUT':instantiate(plan['output'],gl),'NATIVE_OUT':variant.get('native',False)}
                code,meta=compile_fn(merge,const,ptr,case,case['id']+':'+variant['name'])
                meta['intervention_layout']=variant['L'];meta['native_output']=variant.get('native',False)
            pointers={n:'*bf16' for n in ['DX','DY','DW','DB','X','W']}
            pointers.update(Mean='*fp32',Rstd='*fp32',Lock='*i32')
            compile_fn(dx,{'GROUP_SIZE_M':case['P'],'BLOCK_SIZE_N':case['N']},pointers,case,
                       case['id']+':dx',warps=min(max(case['N']//256,1),8))
        result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
        assert result['ccache_before']==result['ccache_after']
        assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    c.write_once(Path(root)/'result.json',result)
    return result
