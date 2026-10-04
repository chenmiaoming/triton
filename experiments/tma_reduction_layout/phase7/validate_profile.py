"""Offline closure of separate profiler commands, original reports and CUBIN bindings."""
import ast
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c,timing_contract as tc


def validate():
    root=c.OUT/'stage_c/profiling';request=c.read(root/'request.json');status=c.read(root/'status.json')
    source=(c.BASE/'phase7/run_profile.py').read_bytes();tree=ast.parse(source)
    settings=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets))
    c.require(settings==request['settings']==status['settings'],'Frozen profiler settings')
    binding=c.read(root/'export_binding.json');c.require(c.sha((root/'original_export.zip').read_bytes())==binding['original_export_SHA256'],'Original profiler export SHA')
    with zipfile.ZipFile(root/'original_export.zip') as z:
        for name in z.namelist():c.require(z.read(name)==(root/name).read_bytes(),'Original profiler bytes')
    cases,binaries,plan=tc.inputs('stage_c');expected={key:b for key,b in binaries.items() if key.split(':')[0]==settings['case']}
    c.require(request['bindings']==expected and len(expected)==10,'Fixed entire profile case')
    c.require(status['payload_SHA256']==request['payload_SHA256'] and status['counters_enter_formal_timing'] is False,'Separate profile payload/duration')
    c.require('H100' in status['GPU_name'] and status['compute_capability']==[9,0],'H100 profiler')
    c.require(status['source_verification']['remote_source_subset_sha256']==request['provenance']['source_manifest_sha256'],'Verified profiler source')
    c.require(set(r['key'] for r in status['commands'])==set(expected),'Every profile attempt accounted for')
    completed=0
    for r in status['commands']:
        cmd=r['command'];key=r['key'];directory=root/key.replace(':','_')
        c.require(c.read(directory/'command.json')==r and r['archive_sha256']==expected[key]['archive_sha256'],'Exact original command/CUBIN')
        for flag,value in [('replay-mode','kernel'),('cache-control','all'),('clock-control','none'),('profile-from-start','off'),('metrics',','.join(settings['metrics']))]:
            c.require(cmd[cmd.index('--'+flag)+1]==value,'Actual profiler '+flag)
        if r['return_code']==0 and 'PROFILE_ARCHIVE_BINDING ' in r['output']:
            text=next(line.partition('PROFILE_ARCHIVE_BINDING ')[2] for line in r['output'].splitlines() if 'PROFILE_ARCHIVE_BINDING ' in line)
            record=json.loads(text);digest=expected[key]['archive_sha256']
            c.require(record['archive_sha256']==record['loaded_cubin_sha256']==digest and record['key']==key and record['compile_calls']==0
                      and record['triton_imported'] is False and record['correctness_prefix4_max_abs_diff']==0 and record['sha_guard_count']==record['launch_count']==2,'Profiled exact archive launch closure')
            c.require((directory/'report.ncu-rep').is_file() and (directory/'report.ncu-rep').stat().st_size>0,'Actual binary profiler report')
            for metric in settings['metrics']:c.require(metric in r['output'],'Actual metric output '+metric)
            completed+=1
    wanted='COUNTERS_COLLECTED' if completed==10 else 'COUNTERS_UNAVAILABLE_OR_PARTIAL'
    c.require(status['status']==wanted,'Counter availability classification')
    return {'status':'PASS','completed_counter_reports':completed,'attempts':10,'formal_duration_use':False,'counter_status':wanted}


if __name__=='__main__':print(validate())
