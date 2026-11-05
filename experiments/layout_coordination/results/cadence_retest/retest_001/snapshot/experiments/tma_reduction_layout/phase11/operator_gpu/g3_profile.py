"""New write-once NCU evidence; application replay controls cache priming."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c
from .g3_prepare import OUT,SNAPSHOT

app=modal.App('triton-operator-layout-g3a-mechanisms')
image=(modal.Image.from_id(c.TOOLS_IMAGE).env({'PYTHONPATH':'/opt/triton-src'})
       .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                      ignore=['**/__pycache__/**','**/*.pyc']))
evidence=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,cpu=2,memory=4096,timeout=300,include_source=False)
def check_cli(protocol):
    import subprocess
    c.verify_sources(protocol)
    metrics=protocol['mandatory_metrics']+protocol['optional_metrics']
    bases=sorted({x.split('.')[0] for x in metrics})
    argv=['ncu','--query-metrics','--chip','gh100','--query-metrics-mode','suffix','--metrics',','.join(bases)]
    p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    assert p.returncode==0,p.stdout[-2000:]
    missing=[x for x in protocol['mandatory_metrics'] if x not in p.stdout]
    assert not missing,missing
    return {'status':'OK','CPU_only':True,'argv':argv,'query_output':p.stdout,
            'selected_metrics':[x for x in metrics if x in p.stdout],
            'unavailable_optional_metrics':[x for x in protocol['optional_metrics'] if x not in p.stdout],
            'ncu_version':subprocess.check_output(['ncu','--version'],text=True)}


@app.function(image=image,gpu='H100!:1',cpu=8,memory=16384,timeout=5400,
              single_use_containers=True,include_source=False,volumes={'/evidence':evidence})
def collect(protocol,cli,payload,index):
    import subprocess,sys,traceback
    c.verify_sources(protocol)
    assert c.sha(payload)==protocol['G2_payload_SHA256']
    call=modal.current_function_call_id()
    root=Path('/evidence/operator_gpu/g3a')/call
    artifacts=Path('/tmp')/('g3a-artifacts-'+call)
    c.extract(payload,artifacts)
    group=protocol['groups'][index]
    result={'status':'FAILED','index':index,'geometry':group['case']['id'],'call_id':call,
            'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload),
            'formal_timing':False,'reports':[]}
    def command(path,argv):
        p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
        row={'argv':argv,'returncode':p.returncode,'output':p.stdout}
        c.write_once(path,row)
        assert p.returncode==0,p.stdout[-4000:]
        return row
    try:
        command(root/'ncu_version.json',['ncu','--version'])
        for selected in group['records']:
            meta=selected['metadata']
            for mode,settings in protocol['modes'].items():
                folder=root/meta['tag'].replace(':','_')/mode
                job={'case':group['case'],'mode':mode,'metadata':[meta],'warmups':protocol['warmups']}
                c.write_once(folder/'job.json',job)
                flags=['--section','InstructionStats'] if mode=='software' else ['--metrics',','.join(cli['selected_metrics'])]
                command(folder/'collect.json',['ncu','--replay-mode',settings['replay'],
                    '--cache-control',settings['cache'],'--clock-control','none','--profile-from-start','off',
                    *flags,'--export',str(folder/'report'),sys.executable,'-m',
                    'experiments.tma_reduction_layout.phase11.operator_gpu.g3_profile_one',
                    '--root',str(artifacts),'--job',str(folder/'job.json')])
                report=folder/'report.ncu-rep'
                assert report.is_file()
                command(folder/'import.json',['ncu','--import',str(report),'--page','raw','--csv','--print-units','base','--print-fp'])
                if mode=='software':
                    command(folder/'source.json',['ncu','--import',str(report),'--page','source','--print-source','sass','--csv'])
                result['reports'].append({'tag':meta['tag'],'aliases':selected['aliases'],'mode':mode,
                    'path':str(report.relative_to(root)),'SHA256':c.sha(report.read_bytes())})
                print('G3a',meta['tag'],mode,'PASS',flush=True)
        result['status']='OK'
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    c.write_once(root/'result.json',result)
    evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(action:str='check',attempt:int=1):
    import os
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    p=c.read(OUT/'preregistration/protocol.json')
    c.verify_sources(p)
    if action=='check':
        row=check_cli.remote(p)
        c.write_once(OUT/'preregistration/ncu_cli_check.json',row)
        print('G3a CPU-only CLI:',row['status'],len(row['selected_metrics']),'metrics;',row['unavailable_optional_metrics'])
        return
    assert action=='collect'
    cli=c.read(OUT/'preregistration/ncu_cli_check.json')
    payload=(c.OUT/'stage_g2/compiled_payload.zip').read_bytes()
    calls=[]
    for index,g in enumerate(p['groups']):
        folder=OUT/'profiles'/g['case']['id']/f'attempt_{attempt:03}'
        assert not folder.exists(),folder
        call=collect.spawn(p,cli,payload,index)
        c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':'chenmiaoming','protocol_SHA256':c.sha(c.encode(p))})
        calls.append((folder,call))
    def save(item):
        folder,call=item
        blob=call.get();c.write_once(folder/'raw_export.zip',blob);c.extract(blob,folder/'export')
        result=c.read(folder/'export/result.json')
        print('G3a return',result['geometry'],result['status'],flush=True)
        return result['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:statuses=list(pool.map(save,calls))
    assert all(x=='OK' for x in statuses),statuses
