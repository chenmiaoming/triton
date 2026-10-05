"""Separate clean-current-main and historical compiler artifact workers."""
from pathlib import Path
import io
import os
import zipfile
import modal
from experiments.tma_reduction_layout.phase10 import common as c

app = modal.App('triton-phase10-current-main-audit')
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase10-evidence',create_if_missing=True)
# Reuse the verified native/tool image without resolving dynamic dependency
# definitions from old scripts. Source and experiment layers remain separate.
cached = modal.Image.from_id('im-1IDQxHY3NeVW9O4TqDJZQc')
SCRIPT_SNAPSHOT = Path(os.environ.get('TMA_PHASE10_SCRIPTS','/tmp/tma-phase10-stage-a-scripts'))
main_core = (cached.add_local_file(c.OUT/'stage_a/upstream_source.tar','/opt/phase10-upstream.tar',copy=True)
    .add_local_file(Path(__file__).with_name('build_native.py'),'/opt/phase10-build.py',copy=True)
    .run_commands('python3 /opt/phase10-build.py',volumes={'/cache':cache}))
def experiments(image):
    return image.env({'PYTHONPATH':'/opt/triton-src','PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'}).add_local_dir(
        SCRIPT_SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
        ignore=['**/results/**','**/__pycache__/**','**/*.pyc'])
main_image = experiments(main_core)
research_image = experiments(cached)


def collect_impl(target, compiler, protocol):
    import json
    import subprocess
    import traceback
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.tools.tensor_descriptor import TensorDescriptor
    from experiments.tma_reduction_layout.phase10.kernels import host_kernel,device_kernel
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    from experiments.tma_reduction_layout.phase4.artifact_gate import ttgir_contract
    stage = Path('/evidence/stage_a/artifacts')/target/compiler/modal.current_function_call_id()
    stage.mkdir(parents=True)
    records=[]
    def command(*args):
        return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
    torch.cuda.init()
    name,cc=torch.cuda.get_device_name(0),list(torch.cuda.get_device_capability(0))
    spec=protocol['targets'][target]
    assert spec['name'] in name and cc==spec['cc'],(name,cc)
    cuda=ExactCUDA()
    env={**cuda.environment(),'target':target,'gpu_name':name,'compute_capability':cc,
        'compiler':compiler,'source_HEAD':protocol['upstream_HEAD'] if compiler=='main' else protocol['baseline_experiment_HEAD'],
        'native_extension_SHA256':c.sha(Path(native.__file__).read_bytes()),'native_path':native.__file__,
        'triton_import':triton.__file__,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'image_id':os.environ.get('MODAL_IMAGE_ID'),'profile':os.environ.get('MODAL_PROFILE', 'chenmiaoming'),
        'kernel_source_SHA256':c.sha(Path(__file__).with_name('kernels.py').read_bytes())}
    ptxas=Path(get_ptxas(cc[0]*10+cc[1]).path)
    env['ptxas']={'path':str(ptxas),'sha256':c.sha(ptxas.read_bytes()),'version':command(str(ptxas),'--version')}
    env['ccache_start']=command('ccache','--print-stats')
    if compiler=='main':
        build=Path('/opt/phase10-build-evidence')
        for p in build.iterdir():
            if p.is_file(): (stage/'build'/p.name).parent.mkdir(exist_ok=True); (stage/'build'/p.name).write_bytes(p.read_bytes())
        env['source_archive_SHA256']=(build/'source_archive_sha256.txt').read_text().strip()
        assert env['source_archive_SHA256']==protocol['source_tar_SHA256']
        assert 'TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT' not in Path('/opt/triton-src/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp').read_text()
    (stage/'environment.json').write_bytes(c.encode(env))
    triton.set_allocator(lambda size,align,stream:torch.empty(size,dtype=torch.int8,device='cuda'))
    torch.manual_seed(42)
    x=torch.randn((1,8192,8192),dtype=torch.bfloat16,device='cuda')
    for case in protocol['cases']:
        m,n,w=case['M'],case['N'],case['num_warps']
        out=torch.empty((8192//m,8192),dtype=torch.float32,device='cuda')
        reference=x.reshape(1,8192//m,m,8192).float().amax(dim=2).reshape_as(out)
        for path in protocol['paths']:
            variants=['upstream_main'] if compiler=='main' else ['research_default','research_vector4']
            for variant in variants:
                tag=f'M{m}_N{n}_w{w}:{path}:{variant}'
                directory=stage/tag.replace(':','/')
                directory.mkdir(parents=True)
                record={'tag':tag,'target':target,'M':m,'N':n,'num_warps':w,'path':path,'variant':variant,'compile_calls':1,'status':'FAILED'}
                try:
                    os.environ.pop('TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT',None)
                    if variant=='research_vector4':os.environ['TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT']='4'
                    os.environ['TRITON_CACHE_DIR']='/tmp/phase10_compile/'+tag.replace(':','/')
                    # This experimental environment knob is not part of the
                    # JIT's in-memory key. A fresh object is required for every
                    # condition, in addition to a distinct disk cache directory.
                    kernel=triton.jit((host_kernel if path=='host' else device_kernel).fn)
                    compile_events=[]
                    triton.knobs.runtime.jit_post_compile_hook=lambda **event: compile_events.append(event['key'])
                    desc=TensorDescriptor(x,list(x.shape),list(x.stride()),[1,m,n])
                    args=(desc,out) if path=='host' else (x,out)
                    compiled=kernel.warmup(*args,BM=m,BN=n,num_warps=w,grid=(1,))
                    assert len(compile_events)==1,(tag,len(compile_events))
                    record['actual_jit_compile_events']=len(compile_events)
                    for ext in ['ttir','ttgir','llir','ptx','cubin']:
                        value=compiled.asm[ext]
                        (directory/('kernel.'+ext)).write_bytes(value if isinstance(value,bytes) else value.encode())
                    observed=ttgir_contract(compiled.asm['ttgir'],[1,m,n],w)
                    record['observed_layout']=observed
                    expected_vector=4 if variant=='research_vector4' else 8
                    assert observed['blocked']['sizePerThread']==[1,1,expected_vector],(tag,observed)
                    for ext,option in [('sass','-sass'),('resource.txt','-res-usage')]:
                        (directory/('kernel.'+ext)).write_text(command('cuobjdump',option,str(directory/'kernel.cubin')))
                    metadata={**record,'function_name':compiled.name,'dynamic_smem_bytes':compiled.metadata.shared,
                        'global_scratch_size':compiled.metadata.global_scratch_size,'global_scratch_align':compiled.metadata.global_scratch_align,
                        'grid':(8192//m)*(8192//n),'cubin_SHA256':c.sha(compiled.asm['cubin']),
                        'tensordesc_meta':compiled.metadata.tensordesc_meta if hasattr(compiled.metadata,'tensordesc_meta') else None}
                    (directory/'metadata.json').write_bytes(c.encode(metadata))
                    occupancy=cuda.occupancy(directory/'kernel.cubin',compiled.name,compiled.metadata.shared,w,c.sha((directory/'kernel.resource.txt').read_bytes()))
                    (directory/'occupancy.json').write_bytes(c.encode(occupancy))
                    out.fill_(float('nan'))
                    compiled[(metadata['grid'],1,1)](*args,m,n)
                    torch.cuda.synchronize()
                    error=(out-reference).abs().max().item()
                    assert error==0,error
                    record.update(status='OK',correctness_max_abs_diff=error,cubin_SHA256=metadata['cubin_SHA256'])
                except Exception:
                    record['error']=traceback.format_exc()
                    print(record['error'],flush=True)
                    (directory/'error.txt').write_text(record['error'])
                record['files']={p.name:c.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()}
                (directory/'attempt.json').write_bytes(c.encode(record))
                records.append(record)
    env['ccache_end']=command('ccache','--print-stats')
    assert env['ccache_start']==env['ccache_end'],'Unexpected native rebuild in artifact worker'
    (stage/'environment.json').write_bytes(c.encode(env))
    (stage/'attempts.json').write_bytes(c.encode(records))
    volume.commit()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(stage)))
    return {'zip':stream.getvalue(),'call_id':modal.current_function_call_id(),'environment':env,'records':records}


@app.function(image=main_image,gpu='H100!:1',timeout=3600,volumes={'/cache':cache,'/evidence':volume},single_use_containers=True,include_source=False)
def main_sm90(protocol): return collect_impl('sm90','main',protocol)
@app.function(image=main_image,gpu='B200:1',timeout=3600,volumes={'/cache':cache,'/evidence':volume},single_use_containers=True,include_source=False)
def main_sm100(protocol): return collect_impl('sm100','main',protocol)
@app.function(image=main_image,gpu='RTX-PRO-6000:1',timeout=3600,volumes={'/cache':cache,'/evidence':volume},single_use_containers=True,include_source=False)
def main_sm120(protocol): return collect_impl('sm120','main',protocol)
@app.function(image=research_image,gpu='H100!:1',timeout=3600,volumes={'/cache':cache,'/evidence':volume},single_use_containers=True,include_source=False)
def research_sm90(protocol): return collect_impl('sm90','research',protocol)
@app.function(image=research_image,gpu='B200:1',timeout=3600,volumes={'/cache':cache,'/evidence':volume},single_use_containers=True,include_source=False)
def research_sm100(protocol): return collect_impl('sm100','research',protocol)
@app.function(image=research_image,gpu='RTX-PRO-6000:1',timeout=3600,volumes={'/cache':cache,'/evidence':volume},single_use_containers=True,include_source=False)
def research_sm120(protocol): return collect_impl('sm120','research',protocol)


@app.local_entrypoint()
def main():
    protocol=c.read(c.OUT/'stage_a/protocol.json')
    for target in protocol['targets']:
        for compiler in ['main','research']:
            root=c.OUT/'stage_a/artifacts'/target/compiler
            if (root/'raw_export.zip').exists():continue
            function=globals()[compiler+'_'+target]
            call=function.spawn(protocol)
            c.write_once(root/f'dispatch_{call.object_id}.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),'compiler':compiler,'target':target})
            result=call.get()
            c.write_once(root/'raw_export.zip',result['zip'])
            c.write_once(root/'return.json',{k:v for k,v in result.items() if k!='zip'})
            with zipfile.ZipFile(io.BytesIO(result['zip'])) as z:z.extractall(root/'export')
            assert all(r['status']=='OK' for r in result['records']),f'Artifact failures preserved: {target}/{compiler}'
            print('Archived',target,compiler,flush=True)
