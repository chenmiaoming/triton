"""Probe clock-control permissions on assigned Modal H100s; no kernel timing."""
from pathlib import Path
import csv
import datetime
import hashlib
import io
import json
import os
import subprocess
import traceback

import modal

BASE_IMAGE = 'im-qzTUZumWu2JXjWic8K82cZ'
SOURCE = Path(__file__).resolve()
REPO = SOURCE.parents[3]
OUT = REPO/'experiments/layout_coordination/results/clock_control'
app = modal.App('layout-coordination-clock-permission-probe')
image = (modal.Image.from_id(BASE_IMAGE)
    .env({'PYTHONPATH': '/opt/triton-src'})
    .add_local_file(SOURCE, '/opt/triton-src/experiments/layout_coordination/clock_control/probe_run.py', copy=True))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


@app.function(image=image, gpu='H100!:1', cpu=2, memory=4096, timeout=180,
              single_use_containers=True, include_source=False)
def probe(protocol, invocation):
    result = dict(status='FAILED', invocation=invocation, commands=[],
        source_SHA256=sha(Path(__file__).read_bytes()), euid=os.geteuid(),
        formal_timing=False, kernel_launches=0, clock_request_accepted=False,
        effective_under_load_verified=False, memory_clock_lock_attempted=False,
        memory_clock_policy='Hopper deferred locks/reinitialization are outside this probe; no host reset, persistence or power changes.')
    assert result['source_SHA256'] == protocol['source_SHA256']
    uuid = None
    changed = False

    def command(args, role):
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            process = subprocess.run(['nvidia-smi', *args], capture_output=True,
                                     text=True, timeout=30)
            row = dict(role=role, argv=['nvidia-smi', *args], returncode=process.returncode,
                       stdout=process.stdout, stderr=process.stderr, started_at_UTC=started)
        except Exception:
            row = dict(role=role, argv=['nvidia-smi', *args], returncode=None,
                       error=traceback.format_exc(), started_at_UTC=started)
        result['commands'].append(row)
        return row

    try:
        identity = command(['--query-gpu=uuid,name,driver_version',
                            '--format=csv,noheader'], 'identity')
        assert identity['returncode'] == 0, identity
        rows = list(csv.reader(io.StringIO(identity['stdout'])))
        assert len(rows) == 1 and len(rows[0]) == 3, rows
        uuid, name, driver = [value.strip() for value in rows[0]]
        assert uuid.startswith('GPU-') and 'H100' in name and 'H200' not in name
        result['environment'] = dict(gpu_uuid=uuid, name=name, driver_version=driver)
        command(['--help'], 'help')
        before = command(['-i', uuid, '-q', '-x'], 'before')
        assert before['returncode'] == 0, before
        supported = command(['-i', uuid, '--query-supported-clocks=memory,graphics',
                             '--format=csv,noheader,nounits'], 'supported_clocks')
        assert supported['returncode'] == 0, supported
        pairs = []
        for row in csv.reader(io.StringIO(supported['stdout'])):
            if len(row) == 2 and all(value.strip().isdigit() for value in row):
                pairs.append(tuple(map(lambda value: int(value.strip()), row)))
        assert pairs, supported
        clocks = sorted({graphics for _, graphics in pairs})
        frequency = min(clocks, key=lambda value: (abs(value - protocol['target_MHz']), value))
        result['requested_graphics_MHz'] = frequency
        result['supported_clock_pairs'] = pairs
        attempt = command(['-i', uuid, '--lock-gpu-clocks='+str(frequency)+','+str(frequency)], 'lock_gpu_clocks')
        changed = attempt['returncode'] == 0
        result['clock_request_accepted'] = changed
        command(['-i', uuid, '-q', '-x'], 'after_attempt')
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        if changed:
            reset = command(['-i', uuid, '--reset-gpu-clocks'], 'restore_gpu_clocks')
            result['restore_passed'] = reset['returncode'] == 0
            if not result['restore_passed']:
                result['status'] = 'FAILED'
        else:
            result['restore_passed'] = None
            result['restore_note'] = 'No successful write; no reset issued.'
        if uuid:
            command(['-i', uuid, '-q', '-x'], 'final_state')
    return result


@app.local_entrypoint()
def main(attempt: int=1, workers: int=3):
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    assert 1 <= workers <= 3
    root = OUT/f'probe_{attempt:03}'
    assert not root.exists(), root
    protocol = dict(stage='clock_permission_probe', profile='chenmiaoming', GPU='H100!',
        workers=workers, target_MHz=1500, source_SHA256=sha(SOURCE.read_bytes()),
        base_image=BASE_IMAGE, parent_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        created_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Setting permission only; accepted request does not prove clocks under load. No deferred memory locks, host reset, persistence/power-limit change or GPU timing.')
    write(root/'preregistration/protocol.json', protocol)
    (root/'preregistration/source.py').write_bytes(SOURCE.read_bytes())
    calls = []
    for invocation in range(workers):
        call = probe.spawn(protocol, invocation)
        write(root/f'invocation_{invocation:03}'/'dispatch.json', dict(call_id=call.object_id,
            protocol_SHA256=sha(json.dumps(protocol, sort_keys=True).encode()), profile='chenmiaoming'))
        calls.append((invocation, call))
    for invocation, call in calls:
        result = call.get()
        write(root/f'invocation_{invocation:03}'/'raw.json', result)
        print('H100 clock probe', invocation, result['status'], 'accepted', result['clock_request_accepted'], flush=True)
    assert all(json.loads((root/f'invocation_{i:03}'/'raw.json').read_text())['status']=='OK' for i in range(workers))
