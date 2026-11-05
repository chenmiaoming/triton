"""Training NCU collection separate from all formal timing workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal
from . import common as c
from .g3b_prepare import OUT
from .g3b_profile_extension_prepare import SNAPSHOT

app=modal.App('triton-operator-layout-g3b-training-ncu-extension')
image=(modal.Image.from_id(c.TOOLS_IMAGE).env({'PYTHONPATH':'/opt/triton-src'})
       .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc']))
evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,gpu='H100!:1',cpu=8,memory=16384,timeout=3600,single_use_containers=True,
              include_source=False,volumes={'/evidence':evidence})
def collect(protocol,payload,index):
    import subprocess,sys,traceback
    c.verify_sources(protocol)
    call=modal.current_function_call_id();root=Path('/evidence/operator_gpu/g3b_profiles_extension')/call
    artifacts=Path('/tmp')/('g3b-profile-artifacts-'+call);c.extract(payload,artifacts)
    group=protocol['groups'][index]
    result={'status':'FAILED','geometry':group['case']['id'],'reports':[],'formal_timing':False,
            'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload),'call_id':call}
    def cmd(path,argv):
        p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
        c.write_once(path,{'argv':argv,'returncode':p.returncode,'output':p.stdout})
        assert p.returncode==0,p.stdout[-3000:]
    try:
        cmd(root/'version.json',['ncu','--version'])
        for selected in group['records']:
            meta=selected['metadata']
            for mode,settings in protocol['modes'].items():
                folder=root/meta['tag'].replace(':','_')/mode
                c.write_once(folder/'job.json',{'case':group['case'],'mode':mode,'metadata':[meta]})
                flags=['--section','InstructionStats'] if mode=='software' else ['--metrics',','.join(protocol['metrics'])]
                cmd(folder/'collect.json',['ncu','--replay-mode',settings['replay'],'--cache-control',settings['cache'],
                    '--clock-control','none','--profile-from-start','off',*flags,'--export',str(folder/'report'),sys.executable,
                    '-m','experiments.tma_reduction_layout.phase11.operator_gpu.g3b_profile_one','--root',str(artifacts),'--job',str(folder/'job.json')])
                report=folder/'report.ncu-rep';assert report.exists()
                cmd(folder/'import.json',['ncu','--import',str(report),'--page','raw','--csv','--print-units','base','--print-fp'])
                if mode=='software':cmd(folder/'source.json',['ncu','--import',str(report),'--page','source','--print-source','sass','--csv'])
                result['reports'].append({'tag':meta['tag'],'aliases':selected['aliases'],'mode':mode,'path':str(report.relative_to(root)),'SHA256':c.sha(report.read_bytes())})
                print('G3b NCU',meta['tag'],mode,'PASS',flush=True)
        result['status']='OK'
    except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
    c.write_once(root/'result.json',result);evidence.commit();return c.archive(root)


@app.local_entrypoint()
def main(attempt:int=1):
    import os
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(OUT/'preregistration/profile_extension_protocol.json');c.verify_sources(p)
    assert c.read(OUT/'correctness/attempt_001/raw.json')['status']=='OK'
    payload=(OUT/'compiled_payload.zip').read_bytes();calls=[]
    for index,g in enumerate(p['groups']):
        folder=OUT/'profiles_extension'/g['case']['id']/f'attempt_{attempt:03}'
        assert not folder.exists(),folder
        call=collect.spawn(p,payload,index);c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':'chenmiaoming'});calls.append((folder,call))
    def save(item):
        folder,call=item;blob=call.get();c.write_once(folder/'raw_export.zip',blob);c.extract(blob,folder/'export')
        value=c.read(folder/'export/result.json');print('G3b NCU return',value['geometry'],value['status'],flush=True);return value['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:states=list(pool.map(save,calls))
    assert all(x=='OK' for x in states),states
