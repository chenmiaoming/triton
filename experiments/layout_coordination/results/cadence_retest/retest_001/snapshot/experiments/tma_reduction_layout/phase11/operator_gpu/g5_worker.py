"""CPU pass traces and isolated Python pipeline interventions; native stays pinned."""
from pathlib import Path
import argparse, ast, gzip, inspect, os, subprocess, textwrap, traceback
from . import common as c


def function_and_constants(item, variant, gl):
    from .production_compile import controls
    from .production_cases import configuration
    from . import sglang_triton,sglang_controlled,vllm_triton,vllm_controlled,g3b_triton,g3b_gluon
    from .layout_parsing import instantiate
    case=item['case']; is_u=variant.startswith('U')
    meta=next(r for r in item['records'] if r['tag'].endswith(':upstream' if is_u else ':gluon_anchor'))
    if case['family']=='C01':
        fn=g3b_triton._layer_norm_bwd_dwdb if is_u else g3b_gluon.merge
        constants={'BLOCK_SIZE_M':32,'BLOCK_SIZE_N':128}
        if not is_u: constants.update(L=instantiate(item['plan']['input'],gl),LOUT=instantiate(item['plan']['output'],gl),NATIVE_OUT=False)
    else:
        fn=(vllm_triton.reduce_segments if is_u else vllm_controlled.reduce_segments) if case['family']=='C06' else (sglang_triton._fwd_kernel_stage1 if is_u else sglang_controlled._fwd_kernel_stage1)
        constants=configuration(case)[0]
        if not is_u: constants.update(controls(case,item['plan'],item['plan']['V'],gl))
    # Copy the actual recorded JIT ABI rather than guessing specializations.
    constants={name:value for name,value in constants.items() if meta['JIT_signature'][name]=='constexpr'}
    constants.update(meta['specialized_runtime_values'])
    attrs={ast.literal_eval(key):value for key,value in meta['JIT_attributes'].items()}
    return fn,constants,attrs,meta


def install_intervention(variant, root):
    from triton.backends.nvidia.compiler import CUDABackend
    from triton._C.libtriton import ir,passes
    if variant=='U_no_rmlc':
        source=textwrap.dedent(inspect.getsource(CUDABackend.make_ttgir))
        tree=ast.parse(source)
        class Remove(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='add_remove_layout_conversions':return None
                return self.generic_visit(node)
        tree=Remove().visit(tree);ast.fix_missing_locations(tree)
        function=tree.body[0];function.decorator_list=[]
        namespace=dict(CUDABackend.make_ttgir.__globals__)
        exec(compile(tree,'<g5-U-no-rmlc>','exec'),namespace)
        CUDABackend.make_ttgir=staticmethod(namespace['make_ttgir'])
        c.write_once(root/'pipeline_intervention.py',ast.unparse(tree).encode());return
    if variant in ('U','G0','U_restore_cache','G0_restore_cache'):return
    original=CUDABackend.gluon_to_ttgir
    def patched(self,src,metadata,options,capability):
        mod=original(self,src,metadata,options,capability)
        sequence=[('canonicalize',passes.common.add_canonicalizer),('cse',passes.common.add_cse)]
        if variant in ('G0_rmlc','G0_fullsuffix'):
            sequence += [('remove_layout_conversions',passes.ttgpuir.add_remove_layout_conversions),
                         ('canonicalize',passes.common.add_canonicalizer),('cse',passes.common.add_cse)]
        if variant=='G0_fullsuffix':
            sequence += [('triton_licm',passes.ttir.add_triton_licm),('canonicalize',passes.common.add_canonicalizer),
                         ('loop_aware_cse',passes.ttir.add_loop_aware_cse),
                         ('remove_layout_conversions',passes.ttgpuir.add_remove_layout_conversions),
                         ('reorder_instructions',passes.ttgpuir.add_reorder_instructions),
                         ('sccp',passes.common.add_sccp),('cse',passes.common.add_cse),('canonicalize',passes.common.add_canonicalizer)]
        for i,(name,add) in enumerate(sequence):
            c.write_once(root/'intervention_steps'/f'{i:02}_{name}.before.mlir',str(mod).encode())
            pm=ir.pass_manager(mod.context);add(pm);pm.run(mod,'g5_'+name)
            c.write_once(root/'intervention_steps'/f'{i:02}_{name}.after.mlir',str(mod).encode())
        return mod
    CUDABackend.gluon_to_ttgir=patched


def child(job, root):
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from .production_compile import export
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    item=job['item'];variant=job['variant'];root=Path(root)
    result={'status':'FAILED','variant':variant,'case':item['case']['id'],'GPU_allocated':False,'native_SHA256':c.NATIVE}
    try:
        fn,constants,attrs,old=function_and_constants(item,variant,gl)
        install_intervention(variant,root)
        cls=triton.compiler.ASTSource if variant.startswith('U') else gluon.GluonASTSource
        compiled=triton.compile(cls(fn,dict(old['JIT_signature']),constants,attrs),target=GPUTarget('cuda',90,32),
             options={'num_warps':old['num_warps'],'num_stages':item['case']['num_stages']})
        meta=export(compiled,fn,item['case'],item['case']['id']+':'+variant,root,constants)
        result.update(status='OK',metadata=meta,original_CUBIN_SHA256=old['cubin_SHA256'],
                      matches_original_CUBIN=meta['cubin_SHA256']==old['cubin_SHA256'],
                      kernel_cache_directory=os.environ['TRITON_CACHE_DIR'],
                      original_compile_key=old['compile_key'])
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    c.write_once(root/'result.json',result)


def run(protocol, root, pilot):
    c.verify_sources(protocol);root=Path(root)
    result={'status':'FAILED','GPU_allocated':False,'native_rebuilds':0,'programs':[],
            'protocol_SHA256':c.sha(c.encode(protocol)),
            'ccache_before':subprocess.check_output(['ccache','--print-stats'],text=True)}
    for item in protocol['cases']:
        is_pilot=item['case']['id'] in protocol['pilot']
        if is_pilot != pilot:continue
        for variant in protocol['pilot_variants'] if pilot else protocol['matrix_variants']:
            folder=root/item['case']['id']/variant
            job={'item':item,'variant':variant};c.write_once(folder/'job.json',job)
            env=dict(os.environ,TRITON_ALWAYS_COMPILE='1')
            trace=folder/'trace.native.mlir'
            if variant in ('U','G0'):
                env.update(MLIR_ENABLE_DUMP='1',MLIR_DUMP_PATH=str(trace))
            else:
                env.pop('MLIR_ENABLE_DUMP',None);env.pop('MLIR_DUMP_PATH',None)
            if variant not in ('U','G0','U_restore_cache','G0_restore_cache'):
                env['TRITON_CACHE_DIR']=env['TRITON_CACHE_DIR']+'-g5-diagnostics/'+variant
            argv=[os.sys.executable,'-m','experiments.tma_reduction_layout.phase11.operator_gpu.g5_worker',
                  '--job',str(folder/'job.json'),'--root',str(folder)]
            proc=subprocess.run(argv,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
            c.write_once(folder/'command.json',{'argv':argv,'output':proc.stdout,'returncode':proc.returncode})
            if trace.exists():
                blob=trace.read_bytes();c.write_once(folder/'passes.mlir.gz',gzip.compress(blob,mtime=0))
                c.write_once(folder/'trace_integrity.json',{'uncompressed_SHA256':c.sha(blob),'uncompressed_bytes':len(blob)})
                trace.unlink()
            row=c.read(folder/'result.json') if (folder/'result.json').exists() else {'status':'FAILED','error':proc.stdout}
            result['programs'].append({'case':item['case']['id'],'variant':variant,'path':str(folder.relative_to(root)),**row})
            print('G5 CPU',item['case']['id'],variant,row['status'],'exact-old',row.get('matches_original_CUBIN'),flush=True)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert result['ccache_before']==result['ccache_after']
    result['status']='OK' if all(r['status']=='OK' for r in result['programs']) else 'FAILED'
    c.write_once(root/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--root',required=True)
    args=parser.parse_args();child(c.read(args.job),args.root)
