"""Isolated C1.5 evidence; C1 and all original artifacts stay unchanged."""
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c1 import common as prior

ROOT, base = prior.ROOT, prior.base
SOURCE = Path(__file__).resolve().parent
OUT = base.OUT / 'stage_c15'
read, sha, encode = prior.read, prior.sha, prior.encode
write_once, archive, extract, inventory = prior.write_once, prior.archive, prior.extract, prior.inventory
freeze_prior = prior.freeze_prior


def snapshot(destination):
    names = [(SOURCE / name).relative_to(ROOT).as_posix() for name in
             ('common.py', 'data.py', 'driver.py', 'kernels.py', 'profile_one.py', 'run.py', 'timing.py')]
    names += ['experiments/tma_reduction_layout/phase11/c1/common.py',
              'experiments/tma_reduction_layout/phase11/c1/data.py',
              'experiments/tma_reduction_layout/phase11/c1/driver.py',
              'experiments/tma_reduction_layout/phase11/common.py',
              'experiments/tma_reduction_layout/phase11/launch.py',
              'experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    hashes = {}
    for name in names:
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


def load_protocol():
    revisions = list((OUT / 'preregistration/revisions').glob('*/protocol.json'))
    return read(max(revisions, key=lambda p: int(p.parent.name))) if revisions else read(OUT / 'preregistration/protocol.json')
