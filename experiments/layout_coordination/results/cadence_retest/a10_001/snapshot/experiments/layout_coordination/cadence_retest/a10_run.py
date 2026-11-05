"""Launch local GPU0 eligibility, then three sequential one-device cohorts."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from .a10_prepare import DEFAULT,UUID,cpu_gate


def idle_query():
    records=[]
    commands=[['nvidia-smi','-i',UUID,'--query-gpu=index,uuid,name,compute_cap,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits'],
              ['nvidia-smi','-i',UUID,'--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
               '--format=csv,noheader,nounits']]
    for argv in commands:
        p=subprocess.run(argv,text=True,capture_output=True)
        records.append(dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
            UTC=datetime.now(timezone.utc).isoformat(),monotonic_ns=time.monotonic_ns()))
    assert all(x['returncode']==0 for x in records),records
    fields=[x.strip() for x in records[0]['stdout'].strip().split(',')]
    assert fields[:4]==['0',UUID,'NVIDIA A10','8.6'],fields
    assert int(fields[4])<64 and int(fields[5])<=1,('GPU0 must be idle',fields)
    assert not records[1]['stdout'].strip(),('GPU0 has another compute process',records[1])
    return dict(status='PASS',PID=os.getpid(),queries=records,
        physical_device_index=0,gpu_uuid=UUID,setting_writes=False)


def worker(root,invocation):
    p=c.read(root/'preregistration/protocol.json')
    payload=(root/'preregistration/payload.zip').read_bytes()
    cpu_gate(p,payload,root/'snapshot')
    if invocation:
        eligibility=c.read(root/'check/eligibility_gate.json')
        assert eligibility['status']=='PASS' and eligibility['protocol_SHA256']==c.sha(c.encode(p))
    output=root/('check/invocation_0' if invocation==0 else f'timing/invocation_{invocation}')
    assert not output.exists(),'No overwrites or result-triggered retries; use a new protocol revision.'
    query=idle_query()
    c.write_once(output/'idle_before_context.json',query)
    os.environ['CUDA_VISIBLE_DEVICES']=UUID
    from .a10_runtime import run
    result=run(p,payload,output,invocation)
    if invocation:
        c.write_once(output/'eligibility_binding.json',dict(
            eligibility_gate_SHA256=c.sha((root/'check/eligibility_gate.json').read_bytes()),
            protocol_SHA256=c.sha(c.encode(p))))
    return 0 if result['status']=='OK' else 1


def launch(root,action):
    p=c.read(root/'preregistration/protocol.json')
    cpu_gate(p,(root/'preregistration/payload.zip').read_bytes(),root/'snapshot')
    invocations=[0] if action=='eligibility' else [1,2,3]
    if action=='formal':
        gate=c.read(root/'check/eligibility_gate.json')
        assert gate['status']=='PASS' and gate['protocol_SHA256']==c.sha(c.encode(p))
    for invocation in invocations:
        # A fixed interval lets the prior process's utilization window expire;
        # no result-dependent waiting, retries or frequency-setting mutation.
        time.sleep(p['timing']['fixed_pre_process_settle_seconds'])
        target=root/('check' if invocation==0 else 'timing')
        target.mkdir(parents=True,exist_ok=True)
        log=target/f'process_{invocation}.log'
        argv=[sys.executable,'-m','experiments.layout_coordination.cadence_retest.a10_run',
              '--root',str(root),'--action','worker','--invocation',str(invocation)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONUNBUFFERED='1')
        with log.open('x') as stream:
            result=subprocess.run(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=c.ROOT)
        c.write_once(target/f'process_{invocation}.json',dict(argv=argv,returncode=result.returncode,
            log=log.name,log_SHA256=c.sha(log.read_bytes()),physical_device_index=0,gpu_uuid=UUID))
        assert result.returncode==0,log
        print('Local A10 process completed',invocation,flush=True)
    if action=='eligibility':
        from .a10_report import validate
        gate=validate(root,eligibility=True)
        c.write_once(root/'check/eligibility_gate.json',gate)
        assert gate['status']=='PASS',gate


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--action',choices=['eligibility','formal','worker'],required=True)
    parser.add_argument('--invocation',type=int)
    args=parser.parse_args()
    if args.action=='worker':sys.exit(worker(args.root,args.invocation))
    launch(args.root,args.action)
