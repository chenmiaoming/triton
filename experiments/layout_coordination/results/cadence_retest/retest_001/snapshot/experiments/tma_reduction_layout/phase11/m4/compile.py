"""CPU-only compilation of legal shared/consumer interventions using pinned native."""
from pathlib import Path
import subprocess
import traceback
from . import common as c


def run(protocol, root, reuse_blob):
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from experiments.tma_reduction_layout.phase11.m2.kernels import full_output as m2_full
    from experiments.tma_reduction_layout.phase11.m3.kernels import full_output as m3_full
    from experiments.tma_reduction_layout.phase11.c16.kernels import transfer
    from experiments.tma_reduction_layout.phase11.c2c.kernels import one_pass
    from .kernels import dense_output
    c.verify_sources(protocol)
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    c.extract(reuse_blob,root)
    def command(*argv):
        return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT)
    result={'status':'FAILED','GPU_allocated':False,'metadata':{},'rejected':[],
            'new_compiles':0,'reused':0,'native_rebuilds':0,
            'native_SHA256':c.sha(Path(native.__file__).read_bytes()),
            'ccache_before':command('ccache','--print-stats'),
            'protocol_SHA256':c.sha(c.encode(protocol))}
    assert result['native_SHA256']==protocol['baseline']['native_SHA256']
    for case in protocol['cases']:
        folder=root/case['tag'].replace(':','/')
        try:
            if case.get('reuse_source'):
                assert c.sha((folder/'kernel.cubin').read_bytes())==case['cubin_SHA256']
                meta=case
                result['reused']+=1
            else:
                shared=gl.NVMMASharedLayout(**case['shared_layout'])
                constants={'BM':case['M'],'BN':case['N'],'W':case['num_warps']}
                kind=case['compile_source']
                if kind=='c16_transfer':
                    fn=transfer
                    constants['shared_layout']=shared
                else:
                    default={'size_per_thread':[1,1,1],'threads_per_warp':[1,1,32],
                             'warps_per_cta':[1,1,case['num_warps']],'order':[2,1,0]}
                    constants.update(L=gl.BlockedLayout(**case.get('layout',default)),S=shared)
                    if kind=='c2c_transfer':
                        fn=one_pass
                        constants['REDUCE']=False
                    else:
                        fn={'m2_full_output':m2_full,'m3_full_output':m3_full,'dense_output':dense_output}[kind]
                        constants['OUTPUT']=case['output_policy']
                signature={'desc':f'tensordesc<bf16[1,{case["M"]},{case["N"]}],{shared}>',
                           'out':'*fp32',**{key:'constexpr' for key in constants}}
                source=gluon.GluonASTSource(fn,signature,constants,{(1,):[['tt.divisibility',16]]})
                compiled=triton.compile(source,target=GPUTarget('cuda',90,32),options={'num_warps':case['num_warps']})
                for ext in ('glir','ttgir','llir','ptx','cubin'):
                    blob=compiled.asm[ext]
                    c.write_once(folder/('kernel.'+ext),blob if isinstance(blob,bytes) else blob.encode())
                for ext,flag in (('sass','-sass'),('resource.txt','-res-usage')):
                    c.write_once(folder/('kernel.'+ext),command('cuobjdump',flag,str(folder/'kernel.cubin')).encode())
                c.write_once(folder/'kernel.lines.sass',command('nvdisasm','-g','-c',str(folder/'kernel.cubin')).encode())
                meta={**case,'cubin_SHA256':c.sha(compiled.asm['cubin']),'function_name':compiled.name,
                      'dynamic_smem_bytes':compiled.metadata.shared,'global_scratch_size':compiled.metadata.global_scratch_size,
                      'tensordesc_meta':compiled.metadata.tensordesc_meta,'compile_cache_key':compiled.hash}
                result['new_compiles']+=1
            c.write_once(folder/'m4_metadata.json',meta)
            result['metadata'][case['tag']]=meta
            print(c.STAGE,'CPU',case['tag'],'REUSED' if case.get('reuse_source') else 'COMPILED',flush=True)
        except Exception:
            error=traceback.format_exc()
            c.write_once(folder/'compile_rejection.txt',error.encode())
            result['rejected'].append({'tag':case['tag'],'error':error})
            print(c.STAGE,'CPU REJECTED',case['tag'],error[-1000:],flush=True)
    result['native_SHA256_after']=c.sha(Path(native.__file__).read_bytes())
    result['ccache_after']=command('ccache','--print-stats')
    assert result['native_SHA256']==result['native_SHA256_after'] and result['ccache_before']==result['ccache_after']
    mandatory=[row['tag'] for row in protocol['cases'] if row.get('reuse_source')]
    assert all(tag in result['metadata'] for tag in mandatory)
    result['status']='OK'
    c.write_once(root/'result.json',result)
    return result
