"""CPU preflight and bounded H100 frozen-binary jobs using existing caches."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11.m4 import common as c

SNAPSHOT = Path('/tmp/tma-phase11-' + c.STAGE + '-snapshot')
app = modal.App('triton-phase11-' + c.STAGE)


def image_for(identity):
    environment = {'PYTHONPATH': '/opt/triton-src',
                   'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache', 'TMA_M4_STAGE': c.STAGE}
    if identity == c.BASE_IMAGE:
        environment['PATH'] = '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
    return modal.Image.from_id(identity).env(environment).add_local_dir(
        SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
        ignore=['**/__pycache__/**', '**/*.pyc'])


image = image_for(c.BASE_IMAGE)
tools_image = image_for(c.TOOLS_IMAGE)
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=tools_image,cpu=2,memory=4096,timeout=300,include_source=False)
def capabilities(protocol):
    import subprocess
    c.verify_sources(protocol)
    mandatory=protocol['profiles']['mandatory_metrics']
    optional=protocol['profiles']['optional_metrics']
    bases=sorted({name.split('.')[0] for name in mandatory+optional})
    argv=['ncu','--query-metrics','--chip','gh100','--query-metrics-mode','suffix','--metrics',','.join(bases)]
    process=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    advertised=process.stdout
    selected=[name for name in mandatory+optional if name in advertised]
    missing=[name for name in mandatory if name not in selected]
    return {'status':'OK' if process.returncode==0 and not missing else 'FAILED',
            'CPU_only':True,'argv':argv,'returncode':process.returncode,'output':advertised,
            'missing_mandatory':missing,'missing_optional':[name for name in optional if name not in selected],
            'selected_metrics':selected,'version':subprocess.check_output(['ncu','--version'],text=True),
            'protocol_SHA256':c.sha(c.encode(protocol))}


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={'/cache': cache, '/evidence': evidence}, include_source=False)
def preflight(protocol, reuse_blob):
    from .compile import run
    root = Path('/evidence/m4/cpu_preflight')/modal.current_function_call_id()
    result = run(protocol,root,reuse_blob)
    cache.commit()
    evidence.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', timeout=2400,
              volumes={'/evidence': evidence}, single_use_containers=True, include_source=False)
def collect(protocol, metadata, payload):
    from experiments.tma_reduction_layout.phase11.m4.runtime import eligibility
    c.verify_sources(protocol)
    root = Path('/tmp')/modal.current_function_call_id()
    c.extract(payload, root)
    result = eligibility(protocol, metadata, root)
    result.update(call_id=modal.current_function_call_id(), protocol_SHA256=c.sha(c.encode(protocol)),
                  payload_SHA256=c.sha(payload))
    c.write_once(Path('/evidence/m4/eligibility')/modal.current_function_call_id()/'raw.json', result)
    evidence.commit()
    return result


@app.function(image=image, gpu='H100!:1', timeout=5400,
              volumes={'/evidence': evidence}, single_use_containers=True, include_source=False)
def timing(invocation, protocol, gate, payload):
    from experiments.tma_reduction_layout.phase11.m4.runtime import execute
    c.verify_sources(protocol)
    result = execute(invocation, protocol, gate, payload, modal.current_function_call_id())
    c.write_once(Path('/evidence/m4/timing')/modal.current_function_call_id()/'raw.json', result)
    evidence.commit()
    return result


@app.function(image=tools_image, gpu='H100!:1', timeout=3000,
              volumes={'/evidence': evidence}, single_use_containers=True, include_source=False)
def profiles(protocol, gate, payload, group_index):
    import subprocess
    import sys
    import traceback
    c.verify_sources(protocol)
    root = Path('/evidence/m4/profiles')/modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root/'artifacts'
    c.extract(payload, artifacts)
    group = gate['profile_groups'][group_index]
    result = {'status': 'FAILED', 'group': group_index, 'records': [],
              'call_id': modal.current_function_call_id(), 'protocol_SHA256': c.sha(c.encode(protocol)),
              'formal_timing': False}

    def command(path, argv):
        try:
            process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
            value = {'argv': argv, 'returncode': process.returncode, 'output': process.stdout}
        except Exception as error:
            value = {'argv': argv, 'returncode': 'EXECUTION_FAILURE', 'output': repr(error)}
        c.write_once(path, value)
        return value

    try:
        command(root/'ncu_version.json', ['ncu', '--version'])
        for mode in ('software', 'hardware'):
            cells = []
            for tag in group:
                meta = gate['metadata'][tag]
                if mode == 'software':
                    cells.append({'tag':tag,'K':1,'grid':1,'q':'natural'})
                else:
                    for q in protocol['profiles']['q_conditions']:
                        reservation = gate['resources'][tag]['reservations'][q]
                        if q != 'natural' and not reservation.get('feasible'):
                            continue
                        grid = 1 if meta.get('profile_only') else (8192//meta['M'])*(8192//meta['N'])
                        cells.append({'tag':tag,'K':1,'grid':grid,'q':q})
            folder = root/mode
            folder.mkdir()
            job = {'cells': cells, 'metadata': {tag: gate['metadata'][tag] for tag in group}}
            c.write_once(folder/'job.json', job)
            flags = ['--section', 'InstructionStats'] if mode == 'software' else ['--metrics', ','.join(gate['profile_metrics'])]
            value = command(folder/'collect.json', ['ncu', '--replay-mode', 'kernel', '--cache-control', 'none',
                '--clock-control', 'none', '--profile-from-start', 'off', *flags,
                '--export', str(folder/'report'), sys.executable, '-m',
                'experiments.tma_reduction_layout.phase11.m4.profile_one', '--root', str(artifacts), '--job', str(folder/'job.json')])
            assert value['returncode'] == 0, value['output'][-6000:]
            assert (folder/'report.ncu-rep').exists()
            value = command(folder/'import.json', ['ncu', '--import', str(folder/'report.ncu-rep'), '--page', 'raw',
                             '--csv', '--print-units', 'base', '--print-fp'])
            assert value['returncode'] == 0, value['output'][-3000:]
            if mode == 'software':
                value = command(folder/'source.json', ['ncu', '--import', str(folder/'report.ncu-rep'),
                                 '--page', 'source', '--print-source', 'sass', '--csv'])
                assert value['returncode'] == 0, value['output'][-3000:]
            result['records'].append({'mode': mode, 'launches': len(cells),
                                      'report_SHA256': c.sha((folder/'report.ncu-rep').read_bytes())})
            print('M4 NCU group', group_index, mode, 'PASS', len(cells), flush=True)
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root/'result.json', result)
    # CUBIN payload is already preserved in CPU preflight and its archive.
    import shutil
    shutil.rmtree(artifacts)
    evidence.commit()
    return c.archive(root)


def save_return(folder, call):
    try:
        result=call.get()
    except Exception as error:
        c.write_once(folder/'retrieval_failure.json',{'call_id':call.object_id,'error':repr(error)})
        raise
    if isinstance(result,bytes):
        c.write_once(folder/'raw_export.zip',result)
        c.extract(result,folder/'export')
        status=c.read(folder/'export/result.json')['status']
    else:
        c.write_once(folder/'raw.json',result)
        status=result['status']
    print('M4 return',folder.name,status,flush=True)
    return status


@app.local_entrypoint()
def main(action: str='preflight',attempt: int=1,invocation: int=1,group: int=0):
    from concurrent.futures import ThreadPoolExecutor
    protocol=c.protocol()
    c.protect_prior()
    c.verify_sources(protocol)
    assert 1<=attempt<=3
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    jobs=[]
    if action=='capabilities':
        jobs=[(c.OUT/'capabilities'/f'attempt_{attempt}',capabilities,(protocol,))]
    elif action=='preflight':
        jobs=[(c.OUT/'cpu_preflight'/f'attempt_{attempt}',preflight,
               (protocol,(c.OUT/'preregistration/reuse.zip').read_bytes()))]
    else:
        compiled=c.read(c.OUT/'compile_gate.json')
        payload=(c.OUT/'compiled_payload.zip').read_bytes()
        if action=='collect':
            jobs=[(c.OUT/'eligibility'/f'attempt_{attempt}',collect,(protocol,compiled['metadata'],payload))]
        else:
            gate=c.read(c.OUT/'artifact_gate.json');assert gate['status']=='OK'
            if action in ('profiles','profiles_all'):
                indices=range(len(gate['profile_groups'])) if action=='profiles_all' else [group]
                jobs=[(c.OUT/'profiles'/f'group_{g}'/f'attempt_{attempt}',profiles,
                       (protocol,gate,payload,g)) for g in indices]
            else:
                assert action in ('timing','timing_all')
                indices=range(1,protocol['timing']['workers']+1) if action=='timing_all' else [invocation]
                jobs=[(c.timing_root(protocol)/f'invocation_{i}',timing,(i,protocol,gate,payload)) for i in indices]
    calls=[]
    for folder,function,args in jobs:
        assert not folder.exists(),folder
        call=function.spawn(*args)
        c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':os.environ['MODAL_PROFILE'],
            'action':action,'protocol_SHA256':c.sha(c.encode(protocol))})
        calls.append((folder,call))
    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        statuses=list(executor.map(lambda pair:save_return(*pair),calls))
    assert all(s=='OK' for s in statuses),statuses


if __name__=='__main__':
    raise SystemExit('Use modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.m4.run')
