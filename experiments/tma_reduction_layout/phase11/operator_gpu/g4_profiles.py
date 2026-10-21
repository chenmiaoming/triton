"""Predeclared software and warm-application NCU factor matrix, with CPU CLI gate."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c
from .g4_prepare import out, protocol_path, gate_path, payload_path

app = modal.App('triton-operator-layout-g4-ncu')
SNAPSHOT = Path('/tmp/operator-gpu-g4-worker-snapshot-v1')
image = (modal.Image.from_id(c.TOOLS_IMAGE).env({'PYTHONPATH': '/opt/triton-src'})
    .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                   ignore=['**/__pycache__/**', '**/*.pyc']))
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, cpu=2, memory=4096, timeout=300, include_source=False)
def check(protocol):
    import subprocess
    c.verify_sources(protocol); metrics = protocol['mandatory_metrics'] + protocol['optional_metrics']
    argv = ['ncu', '--query-metrics', '--chip', 'gh100', '--query-metrics-mode', 'suffix',
            '--metrics', ','.join(sorted({m.split('.')[0] for m in metrics}))]
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    assert p.returncode == 0 and all(m in p.stdout for m in protocol['mandatory_metrics'])
    return {'status': 'OK', 'CPU_only': True, 'argv': argv, 'query_output': p.stdout,
            'selected_metrics': [m for m in metrics if m in p.stdout],
            'unavailable_optional_metrics': [m for m in protocol['optional_metrics'] if m not in p.stdout],
            'ncu_version': subprocess.check_output(['ncu', '--version'], text=True)}


@app.function(image=image, gpu='H100!:1', cpu=8, memory=16384, timeout=3600,
              single_use_containers=True, include_source=False, volumes={'/evidence': evidence})
def collect(p, gate, payload, cli, index):
    import subprocess, sys, traceback
    c.verify_sources(p); assert c.sha(payload) == gate['payload_SHA256']
    group = p['groups'][index]; call = modal.current_function_call_id()
    root = Path('/evidence/operator_gpu') / p['stage'] / ('profiles-' + call)
    artifacts = Path('/tmp') / ('g4-profile-artifacts-' + call); c.extract(payload, artifacts)
    result = {'status': 'FAILED', 'group': group['id'], 'protocol_SHA256': c.sha(c.encode(p)),
              'payload_SHA256': c.sha(payload), 'reports': [], 'call_id': call}
    def command(path, argv):
        process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
        c.write_once(path, {'argv': argv, 'returncode': process.returncode, 'output': process.stdout})
        assert process.returncode == 0, process.stdout[-4000:]
    try:
        seen = set()
        from experiments.tma_reduction_layout.phase11.operator_gpu.g4_runtime import identity
        for meta in gate['records']:
            if meta['group'] != group['id'] or meta['stage2']: continue
            key = identity(meta)
            if key in seen: continue
            seen.add(key)
            for mode in p['profile_modes']:
                folder = root / meta['tag'].replace(':', '_') / mode
                job = {'case': group['case'], 'metadata': meta, 'mode': mode,
                       'batch': p['profile_batch'], 'length': p['profile_length'], 'warmups': p['timing']['warmups']}
                c.write_once(folder / 'job.json', job)
                flags = ['--section', 'InstructionStats'] if mode == 'software' else ['--metrics', ','.join(cli['selected_metrics'])]
                command(folder / 'collect.json', ['ncu', '--replay-mode', 'kernel' if mode == 'software' else 'application',
                    '--cache-control', 'all' if mode == 'software' else 'none', '--clock-control', 'none',
                    '--profile-from-start', 'off', *flags, '--export', str(folder / 'report'), sys.executable, '-m',
                    'experiments.tma_reduction_layout.phase11.operator_gpu.g4_profile_one', '--root', str(artifacts), '--job', str(folder / 'job.json')])
                report = folder / 'report.ncu-rep'; assert report.is_file()
                command(folder / 'import.json', ['ncu', '--import', str(report), '--page', 'raw', '--csv', '--print-units', 'base', '--print-fp'])
                if mode == 'software':
                    command(folder / 'source.json', ['ncu', '--import', str(report), '--page', 'source', '--print-source', 'sass', '--csv'])
                result['reports'].append({'tag': meta['tag'], 'mode': mode, 'path': str(report.relative_to(root)), 'SHA256': c.sha(report.read_bytes())})
                print(p['stage'], meta['tag'], mode, 'PASS', flush=True)
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc(); print(result['error'], flush=True)
    c.write_once(root / 'result.json', result); evidence.commit(); return c.archive(root)


@app.local_entrypoint()
def main(stage: str = 'g4a', action: str = 'check', attempt: int = 1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    target = out(stage); p = c.read(protocol_path(stage)); c.verify_sources(p)
    if action == 'check':
        row = check.remote(p); c.write_once(target / 'preregistration/ncu_cli_check_v2.json', row)
        print(stage, 'CPU NCU CLI PASS', len(row['selected_metrics'])); return
    assert action == 'collect' and stage == 'g4a'
    assert c.read(target / 'correctness/attempt_001/raw.json')['status'] == 'OK'
    gate = c.read(gate_path(stage)); payload = payload_path(stage).read_bytes()
    cli = c.read(target / 'preregistration/ncu_cli_check_v2.json'); calls = []
    for i, group in enumerate(p['groups']):
        folder = target / 'profiles' / group['id'] / f'attempt_{attempt:03}'; assert not folder.exists()
        call = collect.spawn(p, gate, payload, cli, i)
        c.write_once(folder / 'dispatch.json', {'call_id': call.object_id, 'protocol_SHA256': c.sha(c.encode(p)), 'profile': 'chenmiaoming'})
        calls.append((folder, call))
    def save(item):
        folder, call = item; blob = call.get(); c.write_once(folder / 'raw_export.zip', blob); c.extract(blob, folder / 'export')
        result = c.read(folder / 'export/result.json'); print(stage, result['group'], result['status'], flush=True); return result['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool: statuses = list(pool.map(save, calls))
    assert all(x == 'OK' for x in statuses), statuses
