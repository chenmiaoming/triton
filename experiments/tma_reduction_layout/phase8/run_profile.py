"""Separate Nsight counter evidence; its duration is never a formal timing sample."""
import io
import os
from pathlib import Path
import zipfile
import modal
from experiments.tma_reduction_layout.phase8 import common as c,contracts as ct,timing_contract as tc
from experiments.tma_reduction_layout.phase6.run_fresh_gate import image
from experiments.tma_reduction_layout.source_provenance import generate_provenance,verify_remote_source_manifest

app=modal.App('triton-phase8-separate-counters')
volume=modal.Volume.from_name('triton-phase8-evidence',create_if_missing=True)
PROV=generate_provenance() if modal.is_local() else None
SETTINGS={'cases':['M128_N32_w16','M2048_N32_w16'],'B_DESC':65536,'B_RUN':16384,'warmup_launches_outside_region':1,
          'metrics':['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
                     'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
                     'dram__bytes_read.sum','dram__bytes_write.sum','smsp__inst_executed.sum'],
          'replay_mode':'kernel','cache_control':'all','clock_control':'none','profile_from_start':'off',
          'formal_timing_sample':False,'interpretation':'Same-CUBIN runtime mode contrast; descriptive aggregates only.'}


@app.function(image=image,gpu='H100!:1',single_use_containers=True,timeout=1200,volumes={'/evidence':volume})
def collect(prov,bindings,payload):
    import json
    import subprocess
    import sys
    import torch
    verification=verify_remote_source_manifest(prov)
    torch.cuda.init()
    if 'H100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0)!=(9,0):raise RuntimeError('Strict H100 profiling identity')
    root=Path('/tmp/profile_archives')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)
    binding_file=Path('/tmp/profile_bindings.json');binding_file.write_bytes(c.encode(bindings))
    remote=Path('/evidence/phase8_profiles')/modal.current_function_call_id();remote.mkdir(parents=True,exist_ok=False)
    version=subprocess.run(['ncu','--version'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    result={'settings':SETTINGS,'source_provenance':prov,'source_verification':verification,'payload_SHA256':c.sha(payload),
            'version_command':['ncu','--version'],'version_return_code':version.returncode,'version_output':version.stdout,
            'GPU_name':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),
            'remote_evidence':str(remote),'dispatch_id':modal.current_function_call_id(),'commands':[],'counters_enter_formal_timing':False}
    for key,binding in sorted(bindings.items()):
        for mode in (0,1):
            tag=key+':mode'+str(mode)
            target=remote/tag.replace(':','_');target.mkdir()
            command=['ncu','--replay-mode','kernel','--cache-control','all','--clock-control','none','--profile-from-start','off',
                     '--metrics',','.join(SETTINGS['metrics']),'--csv','--page','raw','--export',str(target/'report'),
                     sys.executable,'-m','experiments.tma_reduction_layout.phase8.profile_one','--bundle-root',str(root),
                     '--binding-file',str(binding_file),'--key',key,'--mode',str(mode)]
            record={'key':key,'tag':tag,'mode':mode,'command':command,'archive_sha256':binding['archive_sha256']}
            try:
                run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                record.update(return_code=run.returncode,output=run.stdout)
            except subprocess.TimeoutExpired as exc:record.update(return_code='TIMEOUT',output=str(exc))
            (target/'command.json').write_bytes(c.encode(record));result['commands'].append(record)
            volume.commit()
    result['status']='COUNTERS_COLLECTED' if all(r['return_code']==0 and 'PROFILE_ARCHIVE_BINDING ' in r['output'] for r in result['commands']) else 'COUNTERS_UNAVAILABLE_OR_PARTIAL'
    (remote/'status.json').write_bytes(c.encode(result));volume.commit()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(remote.rglob('*')):
            if path.is_file():z.writestr(path.relative_to(remote).as_posix(),path.read_bytes())
    return stream.getvalue()


@app.local_entrypoint()
def main():
    c.protect();cases,binaries,plan=tc.inputs('stage_c');c.require(set(SETTINGS['cases'])<=set(cases),'Fixed admitted profile diagnostics')
    bindings={key:b for key,b in binaries.items() if key.split(':')[0] in SETTINGS['cases'] and key.split(':')[1]=='switch'}
    root=c.OUT/'stage_c/profiling';root.mkdir(parents=True,exist_ok=True)
    c.require(not (root/'original_export.zip').exists(),'Never overwrite profiler evidence')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for b in bindings.values():
            for name in (b['archive_path'],str(Path(b['archive_path']).with_suffix('.ptx'))):z.writestr(name,(c.ROOT/name).read_bytes())
    prov=PROV;prov['modal_profile']=os.environ.get('MODAL_PROFILE','UNSPECIFIED')
    with zipfile.ZipFile(root/'uploaded_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in prov['source_manifest']:z.write(c.ROOT/name,name)
    (root/'archived_profile_payload.zip').write_bytes(stream.getvalue())
    c.write(root/'request.json',{'settings':SETTINGS,'provenance':prov,'bindings':bindings,'payload_SHA256':c.sha(stream.getvalue())})
    call=collect.spawn(prov,bindings,stream.getvalue());c.write(root/'dispatch.json',{'call_id':call.object_id,'profile':prov['modal_profile']})
    try:returned=call.get()
    except Exception as exc:
        c.write(root/'retrieval_failure.json',{'call_id':call.object_id,'error':repr(exc)});raise
    with (root/'original_export.zip').open('xb') as f:f.write(returned)
    with zipfile.ZipFile(io.BytesIO(returned)) as z:z.extractall(root)
    c.write(root/'export_binding.json',{'original_export_SHA256':c.sha(returned)})
    print(c.read(root/'status.json')['status'],flush=True)


if __name__=='__main__':main()
