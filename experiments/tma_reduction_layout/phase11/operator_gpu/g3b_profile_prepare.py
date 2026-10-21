"""Freeze the already preregistered training profile subset and new driver-only entry point."""
from pathlib import Path
from . import common as c
from .g3b_prepare import OUT,PROTOCOL_FILE
SNAPSHOT=Path('/tmp/operator-gpu-g3b-profile-snapshot-v1')


def main():
    original=c.read(OUT/'preregistration'/PROTOCOL_FILE)
    compiled=c.read(OUT/'cpu/attempt_002/export/result.json')
    groups=[]
    for case in original['cases']:
        if case['id'] not in original['profile_geometries']:continue
        records=[];seen={}
        for name in original['profile_variants']:
            r=next(x for x in compiled['records'] if x['tag']==case['id']+':'+name)
            if r['cubin_SHA256'] in seen:seen[r['cubin_SHA256']]['aliases'].append(r['tag']);continue
            record={'metadata':r,'aliases':[]};records.append(record);seen[r['cubin_SHA256']]=record
        groups.append({'case':case,'records':records})
    names=list(original['worker_source_SHA256'])
    names += [p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.glob('g3b_profile*.py')]
    hashes=c.snapshot(names,SNAPSHOT)
    cli=c.read(c.OUT/'stage_g3a/preregistration/ncu_cli_check.json')
    protocol={'stage':'G3b_profiles','training_protocol_SHA256':c.sha(c.encode(original)),
              'groups':groups,'metrics':cli['selected_metrics'],'worker_source_SHA256':hashes,
              'CPU_metric_check_reused':'same tools image, same gh100 target, stage_g3a ncu_cli_check.json',
              'modes':{'software':{'replay':'kernel','cache':'all'},'warm_application':{'replay':'application','cache':'none'}},
              'selection':'subset explicitly fixed in training protocol before CPU/GPU training runs'}
    c.write_once(OUT/'preregistration/profile_protocol.json',protocol)
    c.write_once(OUT/'preregistration/profile_source_snapshot.zip',c.archive(SNAPSHOT))
    print('G3b profiles frozen',sum(len(x['records']) for x in groups),'binaries,',2*sum(len(x['records']) for x in groups),'reports')


if __name__=='__main__':main()
