"""Seven frozen-binary H100 invocations, each sample exactly one kernel."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11 import common as c

app = modal.App('triton-phase11-exact-gluon-timing')
SNAPSHOT = Path(os.environ.get('TMA_PHASE11_SNAPSHOT','/tmp/tma-phase11-stage-b-snapshot'))
image = (modal.Image.from_id('im-qzTUZumWu2JXjWic8K82cZ').env({'PYTHONPATH':'/opt/triton-src'})
         .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                        ignore=['**/__pycache__/**','**/*.pyc']))
volume = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',timeout=1800,volumes={'/evidence':volume},
              single_use_containers=True,include_source=False)
def timing(invocation, protocol, gate, payload):
    import ctypes as C
    import math
    import sys
    import traceback
    import torch
    from experiments.tma_reduction_layout.phase11.launch import Driver, Loaded
    from experiments.tma_reduction_layout.phase11.data import input_tensor, reference, compare
    from experiments.tma_reduction_layout.phase11.schedule import schedule
    c.verify_sources(protocol)
    raw = {'invocation':invocation,'call_id':modal.current_function_call_id(),'status':'FAILED',
           'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload),
           'gate_SHA256':c.sha(c.encode(gate)),'visits':[],'warmups':[],
           'compilation':{'Triton_imported':False,'subprocess_calls':0,'compile_calls':0}}
    def audit(event,args):
        if event=='subprocess.Popen':
            raw['compilation']['subprocess_calls'] += 1
            raise RuntimeError('No subprocess/JIT/compilation in formal timing')
    sys.addaudithook(audit)
    driver, loaded = None, {}
    try:
        root = Path('/tmp/phase11-frozen')
        c.extract(payload,root)
        torch.cuda.init()
        name, cc = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
        assert 'H100' in name and cc==[9,0], (name,cc)
        driver = Driver()
        raw['environment'] = {**driver.environment(),'name':name,'cc':cc,'image_id':os.environ.get('MODAL_IMAGE_ID'),
                              'torch':torch.__version__,'torch_cuda':torch.version.cuda}
        inputs = {dtype:input_tensor(dtype) for dtype in ('bfloat16','float32')}
        refs = {dtype:reference(x) for dtype,x in inputs.items()}
        out = torch.empty((256,8192),dtype=torch.float32,device='cuda')
        scratch_size = max(1,max(m['global_scratch_size']*m['full_grid'] for m in gate['metadata'].values()))
        scratch = torch.empty(scratch_size,dtype=torch.uint8,device='cuda')
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        parameters = {}
        for tag,meta in gate['metadata'].items():
            k = Loaded(driver,root,meta)
            loaded[tag] = k
            parameters[tag] = k.parameters(inputs[meta['dtype']],out,scratch)
        for tag,k in loaded.items():
            out.fill_(float('nan'))
            owner,params,values = parameters[tag]
            for _ in range(3):
                k.guard()
                k.launch(k.metadata['full_grid'],params,stream)
            driver.call('cuStreamSynchronize',[C.c_void_p],stream)
            checked = compare(out,refs[k.metadata['dtype']])
            assert checked['pass'], (tag,checked)
            raw['warmups'].append({'tag':tag,'count':3,'correctness':checked,'parameters':values,'binding':k.binding()})
        first,last = driver.event(),driver.event()
        for cohort,groups in [('gluon_layout',gate['gluon_groups']),('canonical_bridge',gate['bridge_groups'])]:
            plan = schedule(groups,invocation)
            raw[cohort+'_schedule_SHA256'] = c.sha(c.encode(plan))
            for slot in plan:
                k = loaded[slot['tag']]
                owner,params,values = parameters[slot['tag']]
                grid = k.metadata['full_grid']//slot['fraction']
                visit = {**slot,'cohort':cohort,'grid':grid,'samples_us':[],'launch_SHA256':[],
                         'parameters':values,'binding':k.binding(),'gpu_uuid':raw['environment']['gpu_uuid']}
                raw['visits'].append(visit)
                for _ in range(protocol['timing']['samples_per_visit']):
                    digest = k.guard()
                    driver.record(first,stream)
                    k.launch(grid,params,stream)
                    driver.record(last,stream)
                    driver.synchronize_event(last)
                    value = driver.elapsed_us(first,last)
                    assert math.isfinite(value) and value>0
                    visit['samples_us'].append(value)
                    visit['launch_SHA256'].append(digest)
            print('H100',invocation,cohort,len(plan),'visits',flush=True)
        driver.call('cuEventDestroy_v2',[C.c_void_p],first)
        driver.call('cuEventDestroy_v2',[C.c_void_p],last)
        raw['compilation']['Triton_imported'] = any(m=='triton' or m.startswith('triton.') for m in sys.modules)
        assert not raw['compilation']['Triton_imported']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'],flush=True)
    finally:
        raw['binaries'] = {tag:k.binding() for tag,k in loaded.items()}
        if driver:
            for k in loaded.values():
                driver.call('cuModuleUnload',[C.c_void_p],k.module)
            raw['CUDA_call_counts'] = dict(driver.counts)
            raw['CUDA_return_codes'] = sorted(driver.codes)
    root = Path('/evidence/stage_b/timing')/raw['call_id']
    c.write_once(root/'raw.json',raw)
    volume.commit()
    return raw


@app.local_entrypoint()
def main():
    from experiments.tma_reduction_layout.phase11.protocol import load_b
    protocol,_ = load_b()
    gate = c.read(c.OUT/'stage_b/artifact_gate.json')
    payload = (c.OUT/'stage_b/frozen_payload.zip').read_bytes()
    c.protect_prior()
    for invocation in range(1,8):
        root = c.OUT/'stage_b/timing'/f'invocation_{invocation}'
        assert not root.exists(), root
        call = timing.spawn(invocation,protocol,gate,payload)
        c.write_once(root/'dispatch.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),
                                         'image_id':image.object_id})
        try:
            raw = call.get()
        except Exception as e:
            c.write_once(root/'retrieval_failure.json',{'error':repr(e),'call_id':call.object_id})
            raise
        c.write_once(root/'raw.json',raw)
        assert raw['status']=='OK',raw.get('error')
        print('Accepted invocation',invocation,flush=True)
