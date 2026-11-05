"""Bounded NCU orchestration for separately frozen non-reduction binaries.

Importing this module or preparing a protocol launches nothing.  The Modal
entrypoint dispatches at most three GPU groups at a time when explicitly run.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

import modal

from . import common as c


MAX_CONCURRENT_GROUPS = 3
SETTINGS = {
    'software': dict(replay='kernel', cache='all'),
    'warm_application': dict(replay='application', cache='none'),
    'cold_isolated': dict(replay='kernel', cache='all'),
}
SNAPSHOT = Path(os.environ.get('NONREDUCE_PROFILE_SNAPSHOT', '/tmp/nonreduction-n3-worker-snapshot'))
DEFAULT_ROOT = c.OUT/'stage_n3'
app = modal.App('triton-nonreduction-layout-ncu')
image = (modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH': '/opt/triton-src',
          'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
                   ignore=['**/__pycache__/**', '**/*.pyc']))
evidence = modal.Volume.from_name('triton-phase11-evidence')
cache = modal.Volume.from_name('triton-build-cache')


def modes(group):
    selected = group['modes']
    choices = selected.items() if isinstance(selected, dict) else ((name, SETTINGS[name]) for name in selected)
    result = []
    for name, value in choices:
        assert name in SETTINGS and value == SETTINGS[name], (name, value)
        result.append((name, value))
    assert result and len({name for name, _ in result}) == len(result)
    return result


def selected_metadata(record):
    """Accept a plain program metadata row or the established alias wrapper."""
    return record['metadata'] if 'metadata' in record else record


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=10800,
              single_use_containers=True, include_source=False,
              volumes={'/evidence': evidence, '/cache': cache})
def collect(protocol, payload, index):
    import json
    import re
    import subprocess
    import sys
    import traceback
    from .profile_one import validate_fixture

    call = modal.current_function_call_id()
    root = Path('/evidence/nonreduction/n3_profiles')/call
    artifacts = Path('/tmp')/('nonreduction-ncu-artifacts-'+call)
    group = protocol['profile_groups'][index]
    result = dict(status='FAILED', group=group['id'], index=index,
        geometry=group['case']['id'], call_id=call, formal_timing=False,
        reports=[], protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
        max_concurrent_groups=MAX_CONCURRENT_GROUPS, native_rebuilds=0)

    def command(path, argv):
        try:
            process = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=1800)
            value = dict(argv=argv, returncode=process.returncode, output=process.stdout, timed_out=False)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            value = dict(argv=argv, returncode=None, output=output, timed_out=True)
        c.write_once(path, value)
        assert value['returncode'] == 0, value['output'][-4000:]
        return value

    try:
        c.verify_sources(protocol)
        assert c.sha(payload) == protocol['payload_SHA256']
        c.extract(payload, artifacts)
        ncu = '/usr/local/cuda/bin/ncu'
        command(root/'version.json', [ncu, '--version'])
        metrics = protocol['metrics']
        assert metrics and len(metrics) == len(set(metrics))
        assert len({selected_metadata(row)['tag'] for row in group['records']}) == len(group['records'])
        for selected in group['records']:
            meta = selected_metadata(selected)
            for mode, settings in modes(group):
                folder = root/meta['tag'].replace(':', '_')/mode
                job = dict(case=group['case'], mode=mode, metadata=[meta], warmups=8)
                validate_fixture(job)
                c.write_once(folder/'job.json', job)
                c.write_once(folder/'settings.json', dict(mode=mode, **settings,
                    warmups=8, one_binary_per_report=True,
                    grid='one CTA, unchanged full arguments' if mode=='software' else 'original full grid',
                    role='InstructionStats only; not hardware timing' if mode=='software'
                        else 'Profiled hardware counters; not formal unprofiled timing'))
                flags = ['--section', 'InstructionStats'] if mode=='software' else ['--metrics', ','.join(metrics)]
                collected = command(folder/'collect.json', [ncu,
                    '--replay-mode', settings['replay'], '--cache-control', settings['cache'],
                    '--clock-control', 'none', '--profile-from-start', 'off', *flags,
                    '--export', str(folder/'report'), sys.executable, '-m',
                    'experiments.tma_reduction_layout.phase11.nonreduction_survey.profile_one',
                    '--root', str(artifacts), '--job', str(folder/'job.json')])
                lines = re.findall(r'PROFILE_BINDING (\{[^\n]+\})', collected['output'])
                assert lines, ('No executable/oracle binding', meta['tag'], mode)
                bindings = [json.loads(line) for line in lines]
                # Application replay can start the executable more than once.
                # Check every emitted binding, retaining all of them as evidence.
                for binding in bindings:
                    assert binding['mode'] == mode and binding['warmups'] == 8
                    assert not binding['Triton_imported'] and binding['compile_calls'] == 0
                    assert binding['driver_return_codes'] == [0] and len(binding['records']) == 1
                    actual = binding['records'][0]
                    assert actual['tag'] == meta['tag'] and actual['correctness']['pass']
                    assert all(actual['correctness']['guards'])
                    assert actual['runtime_arguments_unchanged_from_full_fixture']
                    assert actual['inplace_restored_before_each_launch']
                    assert actual['binding']['cubin_SHA256'] == actual['binding']['loaded_SHA256'] == meta['cubin_SHA256']
                    if mode == 'software':
                        assert actual['grid'] == [1, 1, 1] and actual['software_active_masks']
                    else:
                        assert actual['grid'] == actual['original_grid']
                c.write_once(folder/'binding.json', bindings[-1])
                c.write_once(folder/'all_replay_bindings.json', bindings)
                report = folder/'report.ncu-rep'
                assert report.is_file()
                command(folder/'import.json', [ncu, '--import', str(report), '--page', 'raw',
                    '--csv', '--print-units', 'base', '--print-fp'])
                if mode == 'software':
                    command(folder/'source.json', [ncu, '--import', str(report), '--page', 'source',
                        '--print-source', 'sass', '--csv'])
                result['reports'].append(dict(tag=meta['tag'], aliases=selected.get('aliases', []),
                    mode=mode, path=str(report.relative_to(root)), SHA256=c.sha(report.read_bytes()),
                    binding_path=str((folder/'binding.json').relative_to(root)),
                    binding_SHA256=c.sha((folder/'binding.json').read_bytes()),
                    emitted_replay_bindings=len(bindings)))
                print('Nonreduce NCU', group['id'], meta['tag'], mode, 'PASS', flush=True)
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root/'result.json', result)
    evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(protocol: str='', payload: str='', output: str='', attempt: int=1):
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol_path = Path(protocol) if protocol else DEFAULT_ROOT/'preregistration/protocol.json'
    payload_path = Path(payload) if payload else DEFAULT_ROOT/'compiled_payload.zip'
    output_root = Path(output) if output else DEFAULT_ROOT/'profiles'
    frozen = c.read(protocol_path)
    c.verify_snapshot(frozen, SNAPSHOT)
    blob = payload_path.read_bytes()
    assert c.sha(blob) == frozen['payload_SHA256']
    groups = frozen['profile_groups']
    assert groups and len({group['id'] for group in groups}) == len(groups)
    for group in groups:
        modes(group)
        assert group['records']

    def save(item):
        folder, call = item
        result_blob = call.get()
        c.write_once(folder/'raw_export.zip', result_blob)
        c.extract(result_blob, folder/'export')
        value = c.read(folder/'export/result.json')
        print('Nonreduce NCU return', value['group'], value['status'], flush=True)
        return value['status']

    for start in range(0, len(groups), MAX_CONCURRENT_GROUPS):
        calls = []
        for index in range(start, min(start+MAX_CONCURRENT_GROUPS, len(groups))):
            group = groups[index]
            folder = output_root/group['id']/f'attempt_{attempt:03}'
            assert not folder.exists(), folder
            call = collect.spawn(frozen, blob, index)
            c.write_once(folder/'dispatch.json', dict(call_id=call.object_id, profile='chenmiaoming',
                protocol_SHA256=c.sha(c.encode(frozen)), payload_SHA256=c.sha(blob), group=group['id'],
                max_concurrent_groups=MAX_CONCURRENT_GROUPS))
            calls.append((folder, call))
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GROUPS) as pool:
            statuses = list(pool.map(save, calls))
        assert all(status=='OK' for status in statuses), statuses


if __name__ == '__main__':
    # Modal's CLI invokes the decorated local_entrypoint.  Direct Python
    # execution intentionally does not dispatch a cloud job.
    raise SystemExit('Use modal run with an explicitly frozen N3 protocol and payload.')
