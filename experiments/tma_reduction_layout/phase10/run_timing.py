"""Same-worker current-main/reference timing, with no Triton/compiler import."""
from pathlib import Path
import io
import os
import zipfile
import modal
from experiments.tma_reduction_layout.phase10 import common as c
app=modal.App('triton-phase10-current-main-frozen-timing')
volume=modal.Volume.from_name('triton-phase10-evidence')
image=(modal.Image.from_id('im-1IDQxHY3NeVW9O4TqDJZQc').env({'PYTHONPATH':'/opt/triton-src'})
    .add_local_dir(Path(os.environ.get('TMA_PHASE10_SCRIPTS','/tmp/tma-phase10-stage-a-scripts'))/'experiments','/opt/triton-src/experiments',copy=True,
        ignore=['**/results/**','**/__pycache__/**','**/*.pyc']))


def schedule(tags,invocation):
    return [tags[offset:]+tags[:offset] for r in range(10) for offset in [((invocation-1)*17+r*13)%len(tags)]]


def timing_impl(target,invocation,protocol,metadata,payload):
    import ctypes as C
    import math
    import socket
    import sys
    import traceback
    import torch
    from experiments.tma_reduction_layout.phase10.launch import Driver,Loaded
    raw={'target':target,'invocation':invocation,'call_id':modal.current_function_call_id(),
        'status':'FAILED','payload_SHA256':c.sha(payload),'protocol_SHA256':c.sha(c.encode(protocol)),
        'process_identity':f'{socket.gethostname()}:{os.getpid()}','warmups':[],'visits':[],
        'compilation':{'triton_imported':False,'subprocess_calls':0,'compiler_calls':0}}
    def audit(event,args):
        if event=='subprocess.Popen':
            raw['compilation']['subprocess_calls']+=1
            raise RuntimeError('No subprocess allowed in exact-binary timing worker')
    sys.addaudithook(audit)
    driver=None
    loaded={}
    try:
        root=Path('/tmp/phase10-frozen')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in z.namelist())
            z.extractall(root)
        torch.cuda.init()
        name,cc=torch.cuda.get_device_name(0),list(torch.cuda.get_device_capability(0))
        spec=protocol['targets'][target]
        assert spec['name'] in name and cc==spec['cc'],(name,cc)
        driver=Driver()
        raw['environment']={**driver.environment(),'gpu_name':name,'compute_capability':cc,
            'torch':torch.__version__,'torch_cuda':torch.version.cuda,'image_id':os.environ.get('MODAL_IMAGE_ID')}
        torch.manual_seed(42)
        x=torch.randn((1,8192,8192),dtype=torch.bfloat16,device='cuda')
        out=torch.empty((256,8192),dtype=torch.float32,device='cuda')
        reference=x.reshape(1,256,32,8192).float().amax(dim=2).reshape_as(out)
        scratch=torch.empty(max(1,max(v['global_scratch_size']*v['grid'] for v in metadata.values())),dtype=torch.uint8,device='cuda')
        raw['allocations']={'input_shape':list(x.shape),'input_pointer':x.data_ptr(),'output_pointer':out.data_ptr(),'scratch_pointer':scratch.data_ptr(),'scratch_bytes':scratch.numel()}
        stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
        parameters={}
        for tag,meta in metadata.items():
            k=Loaded(driver,root,tag,meta)
            loaded[tag]=k
            parameters[tag]=k.parameters(x,out,scratch)
        tags=sorted(metadata)
        for tag in schedule(tags,invocation)[0]:
            k=loaded[tag]
            out.fill_(float('nan'))
            owner,params,values=parameters[tag]
            for _ in range(3):k.guard();k.launch(params,stream)
            driver.call('cuStreamSynchronize',[C.c_void_p],stream)
            error=(out-reference).abs().max().item()
            raw['warmups'].append({'tag':tag,'count':3,'correctness_max_abs_diff':error,'actual_parameters':values,'binary':k.record(),'gpu_uuid':raw['environment']['gpu_uuid']})
            assert error==0,(tag,error)
        first,last=driver.event(),driver.event()
        for r,order in enumerate(schedule(tags,invocation),1):
            for tag in order:
                k=loaded[tag]
                owner,params,values=parameters[tag]
                visit={'round':r,'tag':tag,'samples_us':[],'launch_SHA256':[],'actual_parameters':values,'binary':k.record(),'gpu_uuid':raw['environment']['gpu_uuid']}
                raw['visits'].append(visit)
                for _ in range(5):
                    digest=k.guard()
                    driver.record(first,stream);k.launch(params,stream);driver.record(last,stream)
                    driver.synchronize_event(last)
                    value=driver.elapsed_us(first,last)
                    assert math.isfinite(value) and value>0,value
                    visit['samples_us'].append(value);visit['launch_SHA256'].append(digest)
            print(f'{target} invocation{invocation} round{r}/10',flush=True)
        driver.call('cuEventDestroy_v2',[C.c_void_p],first)
        driver.call('cuEventDestroy_v2',[C.c_void_p],last)
        raw['compilation']['triton_imported']=any(m=='triton' or m.startswith('triton.') for m in sys.modules)
        assert not raw['compilation']['triton_imported']
        raw['status']='OK'
    except Exception:
        raw['error']=traceback.format_exc();print(raw['error'],flush=True)
    finally:
        raw['binaries']={tag:k.record() for tag,k in loaded.items()}
        if driver:
            for k in loaded.values():driver.call('cuModuleUnload',[C.c_void_p],k.module)
            raw['cuda_call_counts']=dict(driver.counts);raw['cuda_return_codes']=sorted(driver.codes)
    destination=Path('/evidence/stage_a/timing')/target/(raw['call_id']+'.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(c.encode(raw));volume.commit()
    return raw


@app.function(image=image,gpu='H100!:1',timeout=1800,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def timing_sm90(*args):return timing_impl('sm90',*args)
@app.function(image=image,gpu='B200:1',timeout=1800,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def timing_sm100(*args):return timing_impl('sm100',*args)
@app.function(image=image,gpu='RTX-PRO-6000:1',timeout=1800,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def timing_sm120(*args):return timing_impl('sm120',*args)


@app.local_entrypoint()
def main():
    protocol=c.read(c.OUT/'stage_a/protocol.json')
    for target in protocol['targets']:
        metadata={};stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for compiler in ['main','research']:
                root=c.OUT/'stage_a/artifacts'/target/compiler/'export'
                for p in sorted(root.rglob('metadata.json')):
                    meta=c.read(p);tag=meta['tag'];metadata[tag]=meta
                    assert c.read(p.with_name('attempt.json'))['status']=='OK'
                    for ext in ['cubin','ptx']:z.write(p.with_name('kernel.'+ext),tag.replace(':','/')+'/kernel.'+ext)
        assert len(metadata)==18
        payload=stream.getvalue();root=c.OUT/'stage_a/timing'/target
        if not (root/'payload.zip').exists():
            c.write_once(root/'payload.zip',payload);c.write_once(root/'metadata.json',metadata)
        else:assert (root/'payload.zip').read_bytes()==payload
        for invocation in range(1,4):
            destination=root/f'raw_invocation_{invocation}.json'
            if destination.exists():assert c.read(destination)['status']=='OK';continue
            call=globals()['timing_'+target].spawn(invocation,protocol,metadata,payload)
            c.write_once(root/f'dispatch_{invocation}.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),'image_id':image.object_id})
            raw=call.get();c.write_once(destination,raw)
            assert raw['status']=='OK',raw.get('error')
            print('Accepted',target,invocation,flush=True)
