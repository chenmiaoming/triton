"""Sequential N5 hardware-only profiling using the unchanged N3 collector."""
from pathlib import Path
import os

from . import common as c


SNAPSHOT = Path(os.environ.get('NONREDUCE_PROFILE_SNAPSHOT', '/tmp/nonreduction-n5-profile-worker-snapshot'))
os.environ['NONREDUCE_PROFILE_SNAPSHOT'] = str(SNAPSHOT)
from .profile_run import app, collect, modes


@app.local_entrypoint()
def n5_profiles(attempt: int=1):
    assert os.environ.get('MODAL_PROFILE')=='chenmiaoming'
    root = c.OUT/'stage_n5_profiles'
    protocol = c.read(root/'preregistration/protocol.json')
    assert Path(protocol['snapshot'])==SNAPSHOT
    c.verify_snapshot(protocol,SNAPSHOT)
    payload = (root/'compiled_payload.zip').read_bytes()
    assert c.sha(payload)==protocol['payload_SHA256']
    assert protocol['maximum_concurrent_GPU_groups']==1 and protocol['expected_reports']==6
    assert len(protocol['profile_groups'])==2
    for index,group in enumerate(protocol['profile_groups']):
        assert modes(group)==[('warm_application',dict(replay='application',cache='none'))]
        assert len(group['records'])==3
        folder = root/'profiles'/group['id']/f'attempt_{attempt:03}'
        assert not folder.exists(), folder
        # Intentionally await this call before spawning another group.
        call = collect.spawn(protocol,payload,index)
        c.write_once(folder/'dispatch.json',dict(call_id=call.object_id,
            profile='chenmiaoming',group=group['id'],
            protocol_SHA256=c.sha(c.encode(protocol)),payload_SHA256=c.sha(payload),
            host_max_concurrent_groups=1,orchestration='sequential spawn/get'))
        blob = call.get()
        c.write_once(folder/'raw_export.zip',blob)
        c.extract(blob,folder/'export')
        result = c.read(folder/'export/result.json')
        print('N5 NCU return',group['id'],result['status'],flush=True)
        assert result['status']=='OK' and len(result['reports'])==3, result.get('error')
        assert all(report['mode']=='warm_application' for report in result['reports'])


if __name__=='__main__':
    raise SystemExit('Use modal run -m ...n5_profile_run::n5_profiles after freezing the protocol.')
