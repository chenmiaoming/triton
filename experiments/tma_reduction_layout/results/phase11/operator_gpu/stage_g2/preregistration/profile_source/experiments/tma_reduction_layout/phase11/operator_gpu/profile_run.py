"""Predeclared NCU representatives on independent H100 workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c

SNAPSHOT=Path('/tmp/operator-gpu-profiles-snapshot-v1')
app=modal.App('triton-operator-production-layout-ncu')
image=(modal.Image.from_id(c.TOOLS_IMAGE).env({'PYTHONPATH':'/opt/triton-src'})
       .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                      ignore=['**/__pycache__/**','**/*.pyc']))
evidence=modal.Volume.from_name('triton-phase11-evidence')


def select(case,compiled,protocol):
    result=[]
    aliases={alias:row['name'] for row in case['layouts'] for alias in row['aliases']}
    bytag={row['tag']:row for row in compiled['records']}
    seen=set()
    for requested in protocol['profiles']['variants']:
        name=aliases.get(requested,requested)
        tag=case['id']+':'+name
        row=bytag[tag]
        if row['cubin_SHA256'] in seen:
            continue
        seen.add(row['cubin_SHA256'])
        result.append(row)
    return result


@app.function(image=image,gpu='H100!:1',cpu=8,memory=32768,timeout=3600,
              single_use_containers=True,include_source=False,volumes={'/evidence':evidence})
def profiles(protocol,profile_protocol,compiled,payload,index):
    import subprocess
    import sys
    import traceback
    c.verify_sources(profile_protocol)
    root=Path('/evidence/operator_gpu/profiles')/modal.current_function_call_id()
    artifacts=Path('/tmp')/('operator-profile-artifacts-'+modal.current_function_call_id())
    c.extract(payload,artifacts)
    case=protocol['geometries'][index]
    records=select(case,compiled,protocol)
    result={'status':'FAILED','index':index,'geometry':case['id'],'reports':[],
            'call_id':modal.current_function_call_id(),'formal_timing':False,
            'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload)}
    def command(path,argv):
        try:
            process=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
            value={'argv':argv,'returncode':process.returncode,'output':process.stdout}
        except Exception:
            value={'argv':argv,'returncode':'EXECUTION_FAILURE','output':traceback.format_exc()}
        c.write_once(path,value)
        return value
    try:
        command(root/'ncu_version.json',['ncu','--version'])
        query=command(root/'metric_query.json',['ncu','--query-metrics','--query-metrics-mode','suffix'])
        assert query['returncode']==0,query['output'][-2000:]
        for metric in protocol['profiles']['metrics']:
            assert metric in query['output'],('Metric not advertised on worker',metric)
        for mode in ('software','hardware'):
            folder=root/mode
            c.write_once(folder/'job.json',{'case':case,'mode':mode,'metadata':records})
            flags=['--section','InstructionStats'] if mode=='software' else ['--metrics',','.join(protocol['profiles']['metrics'])]
            value=command(folder/'collect.json',['ncu','--replay-mode','kernel','--cache-control','none',
                '--clock-control','none','--profile-from-start','off',*flags,'--export',str(folder/'report'),
                sys.executable,'-m','experiments.tma_reduction_layout.phase11.operator_gpu.profile_one',
                '--root',str(artifacts),'--job',str(folder/'job.json')])
            assert value['returncode']==0,value['output'][-5000:]
            assert (folder/'report.ncu-rep').exists()
            value=command(folder/'import.json',['ncu','--import',str(folder/'report.ncu-rep'),'--page','raw',
                          '--csv','--print-units','base','--print-fp'])
            assert value['returncode']==0,value['output'][-2000:]
            if mode=='software':
                value=command(folder/'source.json',['ncu','--import',str(folder/'report.ncu-rep'),
                              '--page','source','--print-source','sass','--csv'])
                assert value['returncode']==0,value['output'][-2000:]
            result['reports'].append({'mode':mode,'launches':len(records),
                'report_SHA256':c.sha((folder/'report.ncu-rep').read_bytes())})
            print('Operator NCU',case['id'],mode,'PASS',len(records),'launches',flush=True)
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc()
        print(result['error'],flush=True)
    c.write_once(root/'result.json',result)
    evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(attempt:int=1):
    import os
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    protocol=c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    profile_protocol=c.read(c.OUT/'stage_g2/preregistration/profile_protocol.json')
    c.verify_sources(profile_protocol)
    assert c.read(c.OUT/'stage_g2/artifact_gate.json')['status']=='OK'
    compiled=c.read(c.OUT/'stage_g2/compile_gate.json')
    payload=(c.OUT/'stage_g2/compiled_payload.zip').read_bytes()
    calls=[]
    for index,case in enumerate(protocol['geometries']):
        folder=c.OUT/'stage_g2/profiles'/case['id']/f'attempt_{attempt:03}'
        assert not folder.exists(),folder
        call=profiles.spawn(protocol,profile_protocol,compiled,payload,index)
        c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':'chenmiaoming',
                      'protocol_SHA256':c.sha(c.encode(protocol))})
        calls.append((folder,call))
    def save(pair):
        folder,call=pair
        payload=call.get()
        c.write_once(folder/'raw_export.zip',payload)
        c.extract(payload,folder/'export')
        result=c.read(folder/'export/result.json')
        print('NCU return',folder.parent.name,result['status'],flush=True)
        return result['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        statuses=list(pool.map(save,calls))
    assert all(status=='OK' for status in statuses),statuses
