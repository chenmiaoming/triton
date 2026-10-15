"""Sequential local GPU-0 subprocesses with isolated compiler imports and caches."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from experiments.tma_reduction_layout.local_a10.experiment import GPU_UUID, read, write


def now():
    return datetime.now(timezone.utc).isoformat()


def run(root, mode):
    repo = Path(__file__).resolve().parents[3]
    toolchains = read(root / 'toolchains.json')
    identities = ('3.5.1', '3.6.0', '3.7.1') if mode == 'collect' else range(7)
    records = []
    for identity in identities:
        env = {k: v for k, v in os.environ.items() if not k.startswith('TRITON_')}
        version = identity if mode == 'collect' else '3.7.1'
        env.update({
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': GPU_UUID,
            'CUDA_CACHE_PATH': '/tmp/triton-exp-a10-driver-cache', 'CUDA_MODULE_LOADING': 'EAGER',
            'TRITON_CACHE_DIR': f'/tmp/triton-exp-a10-default-version-{version}-kernel-cache',
            'TRITON_PTXAS_PATH': toolchains['common_ptxas']['path'],
            'PYTHONPATH': toolchains['versions'][version]['python_prefix'] + ':' + str(repo),
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1',
        })
        command = [toolchains['python'], '-B', '-m',
                   'experiments.tma_reduction_layout.local_a10_versions.experiment', mode, '--root', str(root)]
        command += ['--version', str(identity)] if mode == 'collect' else ['--invocation', str(identity)]
        record = {'identity': identity, 'start_UTC': now(), 'command': command,
                  'controlled_environment': {k: env[k] for k in (
                      'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'CUDA_CACHE_PATH', 'CUDA_MODULE_LOADING',
                      'TRITON_CACHE_DIR', 'TRITON_PTXAS_PATH', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                      'PYTHONNOUSERSITE', 'OMP_NUM_THREADS')},
                  'gpu0_before': subprocess.check_output([
                      'nvidia-smi', '-i', '0', '--query-gpu=index,name,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu',
                      '--format=csv,noheader'], text=True)}
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True)
        prefix = root / 'dispatch' / f'{mode}_{identity}'
        write(prefix.with_suffix(prefix.suffix + '.stdout'), result.stdout)
        write(prefix.with_suffix(prefix.suffix + '.stderr'), result.stderr)
        record.update({'end_UTC': now(), 'returncode': result.returncode,
                       'stdout': str(prefix.with_suffix(prefix.suffix + '.stdout').relative_to(root)),
                       'stderr': str(prefix.with_suffix(prefix.suffix + '.stderr').relative_to(root))})
        records.append(record)
        print(mode, identity, 'return code', result.returncode, result.stdout.decode().strip(), flush=True)
        if result.returncode:
            print(result.stderr.decode(), flush=True)
            break
    write(root / 'dispatch' / (mode + '.json'), records)
    if len(records) != len(identities) or any(row['returncode'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'timing'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.mode)
