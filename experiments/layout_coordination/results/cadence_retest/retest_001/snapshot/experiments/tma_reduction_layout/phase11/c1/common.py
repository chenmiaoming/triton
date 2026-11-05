"""C1 scope, source snapshots and write-once evidence utilities."""
from pathlib import Path
import subprocess

from experiments.tma_reduction_layout.phase11 import common as base

ROOT = base.ROOT
SOURCE = Path(__file__).resolve().parent
OUT = base.OUT / 'stage_c1'
read, sha, encode = base.read, base.sha, base.encode
write_once, archive, extract, inventory = base.write_once, base.archive, base.extract, base.inventory


def snapshot(destination):
    names = [(SOURCE / name).relative_to(ROOT).as_posix() for name in
             ('common.py', 'data.py', 'driver.py', 'kernels.py', 'profile_one.py', 'run.py', 'timing.py')]
    names += ['experiments/tma_reduction_layout/phase11/common.py',
              'experiments/tma_reduction_layout/phase11/launch.py',
              'experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    hashes = {}
    for name in sorted(names):
        value = (ROOT / name).read_bytes()
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        hashes[name] = sha(value)
    return hashes


def verify_sources(protocol):
    for name, digest in protocol['worker_source_SHA256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def protect_prior():
    for name, digest in read(OUT / 'preregistration/prior_manifest.json')['files'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def freeze_prior():
    names = subprocess.check_output(['git', 'ls-files', '-z',
        'experiments/tma_reduction_layout/results'], cwd=ROOT).split(b'\0')
    return {name.decode(): sha((ROOT / name.decode()).read_bytes()) for name in names if name}


def load_protocol():
    revisions = list((OUT / 'preregistration/revisions').glob('*/protocol.json'))
    if revisions:
        return read(max(revisions, key=lambda p: int(p.parent.name)))
    return read(OUT / 'preregistration/protocol.json')
