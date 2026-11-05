"""H100 explicit-layout artifacts and full-output correctness; no timing."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11 import common as c

app = modal.App('triton-phase11-gluon-artifacts')
SNAPSHOT = Path(os.environ.get('TMA_PHASE11_SNAPSHOT','/tmp/tma-phase11-stage-b-snapshot'))
image = (modal.Image.from_id('im-qzTUZumWu2JXjWic8K82cZ')
         .env({'PYTHONPATH':'/opt/triton-src','PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
               'TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache'})
         .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                        ignore=['**/__pycache__/**','**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, gpu='H100!:1', timeout=3600, volumes={'/cache':cache,'/evidence':volume},
              single_use_containers=True, include_source=False)
def collect(protocol):
    import re
    import subprocess
    import traceback
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.tools.tensor_descriptor import TensorDescriptor as TLDescriptor
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GLDescriptor
    from experiments.tma_reduction_layout.phase11 import kernels, data
    from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    c.verify_sources(protocol)
    root = Path('/evidence/stage_b/artifacts')/modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    def command(*args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
    torch.cuda.init()
    name, cc = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
    assert 'H100' in name and cc == [9,0], (name,cc)
    assert c.sha(Path(native.__file__).read_bytes()) == protocol['baseline']['native_SHA256']
    for source, digest in protocol['baseline']['source'].items():
        assert c.sha((Path('/opt/triton-src')/source).read_bytes()) == digest, source
    cuda = ExactCUDA()
    environment = {**cuda.environment(), 'call_id':modal.current_function_call_id(), 'name':name,'cc':cc,
                   'native_SHA256':c.sha(Path(native.__file__).read_bytes()), 'base_image':protocol['baseline']['image'],
                   'image_id':os.environ.get('MODAL_IMAGE_ID'),'torch':torch.__version__,
                   'torch_cuda':torch.version.cuda,'triton':triton.__version__,
                   'kernel_cache':os.environ['TRITON_CACHE_DIR'], 'GPU_timing':False,
                   'protocol_SHA256':c.sha(c.encode(protocol)),'worker_source_SHA256':protocol['worker_source_SHA256'],
                   'ccache_before':command('ccache','--print-stats')}
    ptxas = Path(get_ptxas(90).path)
    environment['ptxas'] = {'path':str(ptxas),'SHA256':c.sha(ptxas.read_bytes()),'version':command(str(ptxas),'--version')}
    triton.set_allocator(lambda size,align,stream: torch.empty(size,dtype=torch.int8,device='cuda'))
    inputs, references, input_hashes = {}, {}, {}
    for dtype in ('bfloat16','float32'):
        for pattern in ('quarter','random','boundary'):
            x = data.input_tensor(dtype,pattern)
            inputs[dtype,pattern] = x
            references[dtype,pattern] = data.reference(x)
            input_hashes[dtype+':'+pattern] = c.sha(x.view(torch.uint8).cpu().numpy().tobytes())
    environment['input_SHA256'] = input_hashes
    records = []
    layout_keys = ('sizePerThread','threadsPerWarp','warpsPerCTA','order')
    def compile_one(case, variant, register_spec=None, shared_spec=None):
        tag = c.metadata_tag(case,variant)
        directory = root/tag.replace(':','/')
        directory.mkdir(parents=True, exist_ok=False)
        is_tl = variant == 'upstream_default'
        record = {**case,'variant':variant,'tag':tag,'pipeline':'triton' if is_tl else 'gluon','status':'FAILED'}
        try:
            m,n,w = case['M'],case['N'],case['num_warps']
            x = inputs[case['dtype'],'quarter']
            out = torch.empty((256,8192),dtype=torch.float32,device='cuda')
            if is_tl:
                template = triton.jit(kernels.canonical.fn)
                args = (TLDescriptor.from_tensor(x,[1,m,n]) if case['path']=='host' else x, out)
                kwargs = {'BM':m,'BN':n,'HOST':case['path']=='host'}
            else:
                layout = gl.BlockedLayout(**register_spec)
                shared = gl.NVMMASharedLayout(swizzle_byte_width=shared_spec['swizzlingByteWidth'],
                    element_bitwidth=shared_spec['elementBitWidth'],rank=3,transposed=shared_spec['transposed'])
                template = gluon.jit(kernels.explicit.fn)
                args = (GLDescriptor.from_tensor(x,[1,m,n],shared) if case['path']=='host' else x, out)
                kwargs = {'BM':m,'BN':n,'W':w,'HOST':case['path']=='host',
                          'DTYPE':gl.bfloat16 if case['dtype']=='bfloat16' else gl.float32,
                          'register_layout':layout,'shared_layout':shared}
                record['requested_layout'] = register_spec
                record['requested_shared_layout'] = shared_spec
            os.environ.pop('TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT',None)
            events = []
            # The legacy JIT hook JSON-serializes constexprs, which cannot
            # represent Gluon BlockedLayout. The compiler listener also reports
            # disk-cache hits and does not serialize layout objects.
            triton.knobs.runtime.jit_post_compile_hook = None
            triton.knobs.compilation.listener = lambda **event: events.append({
                'cache_hit':event['cache_hit'],'compile_hash':event['metadata']['hash']})
            compiled = template.warmup(*args,**kwargs,num_warps=w,grid=(1,))
            record['compiler_cache_events'] = events
            for ext in ('ttir' if is_tl else 'glir','ttgir','llir','ptx','cubin'):
                value = compiled.asm[ext]
                c.write_once(directory/('kernel.'+ext),value if isinstance(value,bytes) else value.encode())
            for ext, flag in [('sass','-sass'),('resource.txt','-res-usage')]:
                c.write_once(directory/('kernel.'+ext),command('cuobjdump',flag,str(directory/'kernel.cubin')).encode())
            info = analyze_ttgir(compiled.asm['ttgir'])
            record['observed_layout'] = {k:info['blocked_encodings'][info['local_load_dest_layout']][k] for k in layout_keys}
            record['observed_shared_layout'] = info['shared_encodings'][info['local_load_src_shared_layout']]
            axes = [int(a) for a in re.findall(r'"tt.reduce".*?<\{axis = (\d+)',compiled.asm['ttgir'])]
            assert axes == [1], axes
            if not is_tl:
                expected = dict(zip(layout_keys,[register_spec[k] for k in
                    ('size_per_thread','threads_per_warp','warps_per_cta','order')]))
                assert record['observed_layout'] == expected, (expected,record['observed_layout'])
            metadata = {**case,'variant':variant,'tag':tag,'pipeline':record['pipeline'],'function_name':compiled.name,
                        'grid':case['full_grid'],'dynamic_smem_bytes':compiled.metadata.shared,
                        'global_scratch_size':compiled.metadata.global_scratch_size,
                        'global_scratch_align':compiled.metadata.global_scratch_align,
                        'tensordesc_meta':compiled.metadata.tensordesc_meta,
                        'cubin_SHA256':c.sha(compiled.asm['cubin'])}
            c.write_once(directory/'metadata.json',metadata)
            occupancy = cuda.occupancy(directory/'kernel.cubin',compiled.name,compiled.metadata.shared,w,
                                      c.sha((directory/'kernel.resource.txt').read_bytes()))
            c.write_once(directory/'occupancy.json',occupancy)
            assert occupancy['blocks_per_sm_actual_dynamic_smem'] > 0
            correctness = {}
            record['correctness'] = correctness
            for pattern in ('quarter','random','boundary'):
                x_test = inputs[case['dtype'],pattern]
                if is_tl:
                    arg = TLDescriptor.from_tensor(x_test,[1,m,n]) if case['path']=='host' else x_test
                else:
                    arg = GLDescriptor.from_tensor(x_test,[1,m,n],shared) if case['path']=='host' else x_test
                out.fill_(float('nan'))
                compiled[(case['full_grid'],1,1)](arg,out,*kwargs.values())
                torch.cuda.synchronize()
                checked = data.compare(out,references[case['dtype'],pattern])
                checked['input_SHA256'] = input_hashes[case['dtype']+':'+pattern]
                if pattern == 'boundary':
                    checked['first_tile_columns_0_to_5_f32_bits'] = out[0,:6].view(torch.int32).cpu().tolist()
                correctness[pattern] = checked
                c.write_once(directory/('correctness_'+pattern+'.json'),checked)
                assert checked['pass'], checked
            record['correctness'] = correctness
            c.write_once(directory/'correctness.json',correctness)
            record['status'] = 'OK'
        except Exception:
            record['error'] = traceback.format_exc()
            c.write_once(directory/'error.txt',record['error'].encode())
            print(record['error'],flush=True)
        record['files'] = {p.name:c.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()}
        c.write_once(directory/'attempt.json',record)
        records.append(record)
        print(tag,record['status'],flush=True)
        return record
    for case in protocol['cases']:
        anchor = compile_one(case,'upstream_default')
        if anchor['status'] != 'OK':
            continue
        shared = anchor['observed_shared_layout']
        matching = None
        for candidate in case['candidates']:
            record = compile_one(case,candidate['variant'],candidate['layout'],shared)
            if record.get('observed_layout') == anchor['observed_layout']:
                matching = candidate['variant']
        if matching is None:
            observed = anchor['observed_layout']
            reg = dict(zip(('size_per_thread','threads_per_warp','warps_per_cta','order'),
                           [observed[k] for k in layout_keys]))
            compile_one(case,'canonical_layout_reproduction',reg,shared)
    environment['ccache_after'] = command('ccache','--print-stats')
    assert environment['ccache_before'] == environment['ccache_after'], 'Unexpected native rebuild'
    c.write_once(root/'environment.json',environment)
    c.write_once(root/'attempts.json',records)
    cache.commit()
    volume.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(attempt: int = 1):
    from experiments.tma_reduction_layout.phase11.protocol import load_b
    protocol,_ = load_b()
    c.protect_prior()
    root = c.OUT/'stage_b/artifacts'/f'attempt_{attempt}'
    assert not root.exists(),root
    call = collect.spawn(protocol)
    c.write_once(root/'dispatch.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),
                                     'image_id':image.object_id})
    try:
        payload = call.get()
    except Exception as e:
        c.write_once(root/'retrieval_failure.json',{'error':repr(e),'call_id':call.object_id})
        raise
    c.write_once(root/'raw_export.zip',payload)
    c.extract(payload,root/'export')
    print('Artifacts returned',len(c.read(root/'export/attempts.json')),flush=True)
