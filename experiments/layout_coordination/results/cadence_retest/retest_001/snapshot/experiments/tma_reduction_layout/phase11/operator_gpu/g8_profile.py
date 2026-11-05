"""New G8 NCU reports, scheduled in bounded batches after formal timing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c

OUT = c.OUT/'stage_g8'
SNAPSHOT = Path('/tmp/operator-gpu-g8-worker-snapshot')
MAX_CONCURRENT_GROUPS = 3
SETTINGS = {
    'software': dict(replay='kernel', cache='all'),
    'warm_application': dict(replay='application', cache='none'),
    'cold_isolated': dict(replay='kernel', cache='all'),
}

app = modal.App('triton-operator-layout-g8-ncu-remeasurement')
image = (modal.Image.from_id(c.TOOLS_IMAGE)
    .env({'PYTHONPATH': '/opt/triton-src'})
    .add_local_dir(SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
                   ignore=['**/__pycache__/**', '**/*.pyc']))
evidence = modal.Volume.from_name('triton-phase11-evidence')


def profile_protocol_path():
    path = OUT/'profile_protocol_v2.json'
    return path if path.exists() else OUT/'profile_protocol.json'


def main_protocol_path():
    path = OUT/'preregistration/protocol_v2.json'
    return path if path.exists() else OUT/'preregistration/protocol.json'


def modes(group):
    selected = group['modes']
    values = selected.items() if isinstance(selected, dict) else ((name, SETTINGS[name]) for name in selected)
    result = []
    for name, settings in values:
        assert name in SETTINGS and settings == SETTINGS[name], (name, settings)
        result.append((name, settings))
    assert result and len({name for name, _ in result}) == len(result)
    return result


@app.function(image=image, gpu='H100!:1', cpu=8, memory=16384, timeout=10800,
              single_use_containers=True, include_source=False, volumes={'/evidence': evidence})
def collect(protocol, payload, index):
    import re
    import json
    import subprocess
    import sys
    import traceback
    from .g8_profile_one import validate_fixture
    c.verify_sources(protocol)
    expected_payload = protocol.get('payload_SHA256', protocol.get('compiled_payload_SHA256'))
    if expected_payload is not None:
        assert c.sha(payload) == expected_payload
    call = modal.current_function_call_id()
    root = Path('/evidence/operator_gpu/g8_profiles')/call
    artifacts = Path('/tmp')/('g8-profile-artifacts-'+call)
    c.extract(payload, artifacts)
    group = protocol['profile_groups'][index]
    result = dict(status='FAILED', group=group['id'], index=index,
                  geometry=group['case']['id'], call_id=call, formal_timing=False,
                  reports=[], protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload))

    def command(path, argv):
        process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        value = dict(argv=argv, returncode=process.returncode, output=process.stdout)
        c.write_once(path, value)
        assert process.returncode == 0, process.stdout[-4000:]
        return value

    try:
        command(root/'version.json', ['ncu', '--version'])
        metrics = protocol['profile_metrics']; assert metrics and len(set(metrics)) == len(metrics)
        for selected in group['records']:
            meta = selected['metadata']
            for mode, settings in modes(group):
                folder = root/meta['tag'].replace(':', '_')/mode
                job = dict(case=group['case'], mode=mode, metadata=[meta], warmups=8)
                validate_fixture(job)
                c.write_once(folder/'job.json', job)
                flags = ['--section', 'InstructionStats'] if mode == 'software' else ['--metrics', ','.join(metrics)]
                collected = command(folder/'collect.json', [
                    'ncu', '--replay-mode', settings['replay'], '--cache-control', settings['cache'],
                    '--clock-control', 'none', '--profile-from-start', 'off', *flags,
                    '--export', str(folder/'report'), sys.executable, '-m',
                    'experiments.tma_reduction_layout.phase11.operator_gpu.g8_profile_one',
                    '--root', str(artifacts), '--job', str(folder/'job.json')])
                bindings = re.findall(r'PROFILE_BINDING (\{[^\n]+\})', collected['output'])
                assert bindings, meta['tag']
                binding = json.loads(bindings[-1])
                assert not binding['Triton_imported'] and binding['compile_calls'] == 0
                assert binding['driver_return_codes'] == [0] and len(binding['records']) == 1
                actual = binding['records'][0]
                assert actual['tag'] == meta['tag'] and actual['correctness']['pass']
                assert actual['binding']['cubin_SHA256'] == actual['binding']['loaded_SHA256'] == meta['cubin_SHA256']
                c.write_once(folder/'binding.json', binding)
                report = folder/'report.ncu-rep'; assert report.is_file()
                command(folder/'import.json', ['ncu', '--import', str(report), '--page', 'raw',
                    '--csv', '--print-units', 'base', '--print-fp'])
                if mode == 'software':
                    command(folder/'source.json', ['ncu', '--import', str(report), '--page', 'source',
                                                  '--print-source', 'sass', '--csv'])
                result['reports'].append(dict(tag=meta['tag'], aliases=selected.get('aliases', []),
                    mode=mode, path=str(report.relative_to(root)), SHA256=c.sha(report.read_bytes()),
                    binding_path=str((folder/'binding.json').relative_to(root)),
                    binding_SHA256=c.sha((folder/'binding.json').read_bytes())))
                print('G8 NCU', group['id'], meta['tag'], mode, 'PASS', flush=True)
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc(); print(result['error'], flush=True)
    c.write_once(root/'result.json', result); evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(action: str='collect', attempt: int=1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    assert action == 'collect'
    original = c.read(main_protocol_path())
    protocol = c.read(profile_protocol_path()); c.verify_sources(protocol)
    gate = c.read(OUT/'compile_gate.json'); assert gate['status'] == 'OK'
    payload = (OUT/'compiled_payload.zip').read_bytes()
    assert gate['payload_SHA256'] == c.sha(payload)
    if 'compile_gate_SHA256' in protocol:
        assert protocol['compile_gate_SHA256'] == c.sha((OUT/'compile_gate.json').read_bytes())
    if 'parent_protocol_SHA256' in protocol:
        assert protocol['parent_protocol_SHA256'] == c.sha(c.encode(original))
    correctness = c.read(OUT/'correctness/attempt_001/raw.json')
    assert correctness['status'] == 'OK' and correctness['payload_SHA256'] == c.sha(payload)
    for i in range(1, 4):
        timing = c.read(OUT/f'timing/invocation_{i}/raw.json')
        assert timing['status'] == 'OK' and timing['payload_SHA256'] == c.sha(payload)
    groups = protocol['profile_groups']
    assert len({g['id'] for g in groups}) == len(groups)
    bytag = {r['tag']: r for r in gate['records']}
    for group in groups:
        modes(group)
        for selected in group['records']:
            assert selected['metadata'] == bytag[selected['metadata']['tag']]

    def save(item):
        folder, call = item
        blob = call.get(); c.write_once(folder/'raw_export.zip', blob); c.extract(blob, folder/'export')
        value = c.read(folder/'export/result.json')
        print('G8 NCU return', value['group'], value['status'], flush=True)
        return value['status']

    # Spawn at most three GPU functions, await every result, then release the
    # next batch.  Do not enqueue every group as a simultaneous GPU request.
    for start in range(0, len(groups), MAX_CONCURRENT_GROUPS):
        calls = []
        for index in range(start, min(start+MAX_CONCURRENT_GROUPS, len(groups))):
            group = groups[index]; folder = OUT/'profiles'/group['id']/f'attempt_{attempt:03}'
            assert not folder.exists(), folder
            call = collect.spawn(protocol, payload, index)
            c.write_once(folder/'dispatch.json', dict(call_id=call.object_id, profile='chenmiaoming',
                protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
                group=group['id'], max_concurrent_groups=MAX_CONCURRENT_GROUPS))
            calls.append((folder, call))
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GROUPS) as pool:
            statuses = list(pool.map(save, calls))
        assert all(state == 'OK' for state in statuses), statuses
