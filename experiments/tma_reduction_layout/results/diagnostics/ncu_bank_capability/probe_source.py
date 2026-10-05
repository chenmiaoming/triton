"""Bounded tool capability probe on one previously frozen ELF, no new kernel compilation."""
import io
import os
from pathlib import Path
import zipfile
import modal
from experiments.tma_reduction_layout.phase7 import common as c, timing_contract as tc
from experiments.tma_reduction_layout.phase6.run_fresh_gate import image, core_image
from experiments.tma_reduction_layout.source_provenance import generate_provenance,verify_remote_source_manifest

app=modal.App('triton-ncu-bank-capability-probe')
volume=modal.Volume.from_name('triton-phase7-evidence',create_if_missing=True)
PROV=generate_provenance() if modal.is_local() else None
KEY='M128_N32_w16:canonical:4'
METRICS=['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
         'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
         'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
         'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
         'l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum',
         'l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum']
OUT=c.BASE/'results/diagnostics/ncu_bank_capability'


@app.function(image=image,gpu='H100!:1',single_use_containers=True,timeout=360,volumes={'/evidence':volume})
def probe(prov,bindings,payload):
    import ctypes as C
    import subprocess
    import sys
    import torch
    from experiments.tma_reduction_layout.phase7.launch import Driver
    verified=verify_remote_source_manifest(prov)
    torch.cuda.init()
    if 'H100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0)!=(9,0):raise RuntimeError('H100 requirement')
    remote=Path('/evidence/diagnostics/ncu_bank')/modal.current_function_call_id();remote.mkdir(parents=True,exist_ok=False)
    root=Path('/tmp/ncu_bank_archives')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unsafe archive path')
        z.extractall(root)
    binding_file=Path('/tmp/ncu_bank_bindings.json');binding_file.write_bytes(c.encode(bindings))
    driver=Driver();env=driver.environment()
    result={'role':'TOOL_CAPABILITY_ONLY','formal_timing':False,'new_kernel_compilation':False,
            'key':KEY,'requested_metrics':METRICS,'source_provenance':prov,'source_verification':verified,
            'GPU_name':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),
            'driver_environment':env,'uid':os.getuid(),'payload_SHA256':c.sha(payload),
            'dispatch_id':modal.current_function_call_id(),'remote_evidence':str(remote),'commands':[]}
    for name,path,words in [('process_capabilities','/proc/self/status',['CapEff:','CapPrm:']),
                            ('driver_counter_policy','/proc/driver/nvidia/params',['RmProfilingAdminOnly'])]:
        try:result[name]=[line for line in Path(path).read_text().splitlines() if any(word in line for word in words)]
        except Exception as exc:result[name]={'unavailable':str(exc)}
    def execute(label,command,timeout=40):
        record={'label':label,'command':command}
        try:
            run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
            record.update(return_code=run.returncode,output=run.stdout)
        except Exception as exc:record.update(return_code='FAILED_TO_EXECUTE',output=repr(exc))
        (remote/(label+'.json')).write_bytes(c.encode(record));result['commands'].append(record);volume.commit()
        return record
    execute('version',['ncu','--version'])
    query=execute('query_metrics',['ncu','--query-metrics','--query-metrics-mode','all'])
    result['queried_metric_name_present']={name:name in query['output'] for name in METRICS}
    command=['ncu','--replay-mode','kernel','--cache-control','all','--clock-control','none','--profile-from-start','off',
             '--metrics',','.join(METRICS),'--csv','--page','raw','--export',str(remote/'bank_report'),
             sys.executable,'-m','experiments.tma_reduction_layout.phase7.profile_one','--bundle-root',str(root),
             '--binding-file',str(binding_file),'--key',KEY]
    execute('collection',command,timeout=120)
    (remote/'result.json').write_bytes(c.encode(result));volume.commit()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(remote.rglob('*')):
            if path.is_file():z.writestr(path.relative_to(remote).as_posix(),path.read_bytes())
    return stream.getvalue()


@app.local_entrypoint()
def main():
    from experiments.tma_reduction_layout.phase6 import common as prior
    baseline=PROV['git_head_sha'];protected=prior.inventory(baseline)
    _,binaries,_=tc.inputs('stage_c');bindings={KEY:binaries[KEY]}
    OUT.mkdir(parents=True,exist_ok=True)
    c.require(not (OUT/'original_export.zip').exists(),'Never overwrite capability evidence')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for b in bindings.values():
            for name in (b['archive_path'],str(Path(b['archive_path']).with_suffix('.ptx'))):z.writestr(name,(c.ROOT/name).read_bytes())
    prov=PROV;prov['modal_profile']=os.environ.get('MODAL_PROFILE','UNSPECIFIED')
    c.write(OUT/'request.json',{'key':KEY,'metrics':METRICS,'bindings':bindings,'provenance':prov,
            'payload_SHA256':c.sha(stream.getvalue()),'protected_HEAD':baseline,'protected_files':len(protected)})
    (OUT/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    call=probe.spawn(prov,bindings,stream.getvalue())
    c.write(OUT/'dispatch.json',{'call_id':call.object_id,'profile':prov['modal_profile'],'core_image_id':core_image.object_id,'resolved_image_id':image.object_id})
    try:returned=call.get()
    except Exception as exc:
        c.write(OUT/'retrieval_failure.json',{'error':repr(exc),'call_id':call.object_id});raise
    with (OUT/'original_export.zip').open('xb') as f:f.write(returned)
    with zipfile.ZipFile(io.BytesIO(returned)) as z:z.extractall(OUT)
    c.write(OUT/'export_binding.json',{'original_export_SHA256':c.sha(returned)})
    c.require(prior.inventory(baseline)==protected,'All earlier files including Phase7 unchanged')
    result=c.read(OUT/'result.json');print('Original bank-counter capability return preserved; collection return code:',result['commands'][-1]['return_code'])


if __name__=='__main__':main()
