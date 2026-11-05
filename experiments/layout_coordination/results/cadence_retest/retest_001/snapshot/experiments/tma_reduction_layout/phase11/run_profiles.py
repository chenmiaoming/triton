"""Separate archived-kernel NCU evidence; durations never enter timing."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11 import common as c

app = modal.App('triton-phase11-frozen-profiles')
SNAPSHOT = Path(os.environ.get('TMA_PHASE11_SNAPSHOT','/tmp/tma-phase11-profile-snapshot'))
image = (modal.Image.from_id('im-gCqknrtnwsoW0fVOiG5XF9').env({'PYTHONPATH':'/opt/triton-src'})
         .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                        ignore=['**/__pycache__/**','**/*.pyc']))
volume = modal.Volume.from_name('triton-phase11-evidence')


def collect_impl(target,stage,request,payload):
    import ctypes as C
    import subprocess
    import sys
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    c.verify_sources(request)
    root = Path('/evidence')/('stage_'+stage)/'archived_profiles'/target/modal.current_function_call_id()
    root.mkdir(parents=True,exist_ok=False)
    archives = root/'archives'
    c.extract(payload,archives)
    c.write_once(root/'metadata.json',request['metadata'][target])
    cuda = ExactCUDA()
    env = cuda.environment()
    assert [cuda.attr(75),cuda.attr(76)]==request['targets'][target]['cc']
    records = []
    def run(path,argv):
        try:
            p = subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
            result = {'argv':argv,'returncode':p.returncode,'output':p.stdout}
        except (subprocess.TimeoutExpired,OSError) as e:
            result = {'argv':argv,'returncode':'EXECUTION_FAILURE','output':str(e)}
        c.write_once(path,result)
        return result
    version = run(root/'ncu_version.json',['ncu','--version'])
    for tag,meta in request['metadata'][target].items():
        for repeat in range(1,request['repeats']+1):
            for group,flags in [('hardware',['--metrics',','.join(request['metrics'][target])]),
                                ('software',['--section','InstructionStats'])]:
                folder = root/tag.replace(':','_')/f'repeat_{repeat}'/group
                folder.mkdir(parents=True,exist_ok=False)
                report = folder/'report'
                exe = [sys.executable,'-m','experiments.tma_reduction_layout.phase11.profile_one',
                       '--root',str(archives),'--metadata',str(root/'metadata.json'),'--tag',tag]
                result = run(folder/'collect.json',['ncu','--replay-mode','kernel','--cache-control','none',
                    '--clock-control','none','--profile-from-start','off','--launch-count','1',*flags,
                    '--export',str(report),*exe])
                if report.with_suffix('.ncu-rep').exists():
                    run(folder/'import.json',['ncu','--import',str(report.with_suffix('.ncu-rep')),
                        '--page','raw','--csv','--print-units','base','--print-fp'])
                    run(folder/'source.json',['ncu','--import',str(report.with_suffix('.ncu-rep')),
                        '--page','source','--print-source','sass','--csv'])
                records.append({'tag':tag,'repeat':repeat,'group':group,'returncode':result['returncode']})
    c.write_once(root/'result.json',{'records':records,'environment':env,'call_id':modal.current_function_call_id(),
        'request_SHA256':c.sha(c.encode(request)),'payload_SHA256':c.sha(payload),'formal_timing_sample':False,
        'NCU_version':version['output'],'status':'OK' if all(r['returncode']==0 for r in records) else 'PARTIAL'})
    volume.commit()
    return c.archive(root)


@app.function(image=image,gpu='H100!:1',timeout=3600,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def sm90(*args):
    return collect_impl('sm90',*args)


@app.function(image=image,gpu='B200:1',timeout=3600,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def sm100(*args):
    return collect_impl('sm100',*args)


@app.function(image=image,gpu='RTX-PRO-6000:1',timeout=3600,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def sm120(*args):
    return collect_impl('sm120',*args)


@app.local_entrypoint()
def main(stage: str = 'a', targets: str = 'sm90,sm100,sm120'):
    request = c.read(c.OUT/('stage_'+stage)/'archived_profiles/request.json')
    for target in targets.split(','):
        root = c.OUT/('stage_'+stage)/'archived_profiles'/target
        assert not root.exists(),root
        payload = (c.OUT/('stage_'+stage)/'archived_profiles'/('payload_'+target+'.zip')).read_bytes()
        call = globals()[target].spawn(stage,request,payload)
        c.write_once(root/'dispatch.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),
                                         'image_id':image.object_id})
        try:
            data = call.get()
        except Exception as e:
            c.write_once(root/'retrieval_failure.json',{'error':repr(e),'call_id':call.object_id})
            raise
        c.write_once(root/'raw_export.zip',data)
        c.extract(data,root/'export')
        print(stage,target,c.read(root/'export/result.json')['status'],flush=True)
