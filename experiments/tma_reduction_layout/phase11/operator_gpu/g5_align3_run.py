"""CPU-only, separately frozen literal-vLLM bridge repair and stepwise IR audit."""
from pathlib import Path
import modal
from . import common as c
from .g5_prepare import OUT,SNAPSHOT

app=modal.App('triton-g5-literal3-vllm-bridge')
image=(modal.Image.from_id(c.BASE_IMAGE)
       .env({'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache-g5-literal',
             'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
       .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc']))
cache=modal.Volume.from_name('triton-build-cache');evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,cpu=8,memory=32768,timeout=1800,include_source=False,
              volumes={'/cache':cache,'/evidence':evidence})
def run(p):
    import os,subprocess
    c.verify_sources(p)
    root=Path('/evidence/operator_gpu/g5_literal')/modal.current_function_call_id()
    result={'status':'OK','programs':[],'GPU_allocated':False,'native_rebuilds':0,
            'protocol_SHA256':c.sha(c.encode(p)),
            'ccache_before':subprocess.check_output(['ccache','--print-stats'],text=True)}
    for item in p['cases']:
        if item['case']['family']!='C06':continue
        folder=root/item['case']['id'];job={'item':item,'variant':'G0_literal3'}
        c.write_once(folder/'job.json',job)
        argv=[os.sys.executable,'-m','experiments.tma_reduction_layout.phase11.operator_gpu.g5_align3_run','--job',str(folder/'job.json'),'--root',str(folder)]
        child=subprocess.run(argv,env=dict(os.environ,TRITON_ALWAYS_COMPILE='1'),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
        c.write_once(folder/'command.json',{'argv':argv,'returncode':child.returncode,'output':child.stdout})
        row=c.read(folder/'result.json') if (folder/'result.json').exists() else {'status':'FAILED','error':child.stdout}
        result['programs'].append({'path':item['case']['id'],**row})
        print('G5 literal',item['case']['id'],row['status'],flush=True)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert result['ccache_before']==result['ccache_after']
    result['status']='OK' if all(x['status']=='OK' for x in result['programs']) else 'FAILED'
    c.write_once(root/'result.json',result);cache.commit();evidence.commit();return c.archive(root)


@app.local_entrypoint()
def main(attempt:int=1):
    import os
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(OUT/'preregistration/literal3_protocol.json');c.verify_sources(p)
    folder=OUT/'literal3'/f'attempt_{attempt:03}';assert not folder.exists()
    call=run.spawn(p);c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':'chenmiaoming','GPU_allocated':False,'protocol_SHA256':c.sha(c.encode(p))})
    blob=call.get();c.write_once(folder/'raw_export.zip',blob);c.extract(blob,folder/'export')
    row=c.read(folder/'export/result.json');print('G5 literal',row['status']);assert row['status']=='OK'


def child_main(job,root):
    from . import g5_worker
    from .g5_vllm_literal3 import reduce_segments
    original=g5_worker.function_and_constants
    def replace(item,variant,gl):
        fn,constants,attrs,old=original(item,variant,gl)
        return reduce_segments,constants,attrs,old
    g5_worker.function_and_constants=replace
    install=g5_worker.install_intervention
    g5_worker.install_intervention=lambda variant,path:install('G0_fullsuffix',path)
    g5_worker.child(job,root)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--root',required=True)
    args=parser.parse_args();child_main(c.read(args.job),args.root)
