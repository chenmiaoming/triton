"""C1.6: isolated validation of the proposed overlap model."""
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c15 import common as previous

ROOT, base = previous.ROOT, previous.base
SOURCE = Path(__file__).resolve().parent
OUT = base.OUT / 'stage_c16'
SNAPSHOT = Path('/tmp/tma-phase11-c16-snapshot')
read, sha, encode = previous.read, previous.sha, previous.encode
write_once, archive, extract = previous.write_once, previous.archive, previous.extract


def load_protocol():
    paths = list((OUT / 'preregistration/revisions').glob('*/protocol.json'))
    return read(max(paths, key=lambda p: int(p.parent.name))) if paths else read(OUT / 'preregistration/protocol.json')


def protect_prior():
    for name, digest in read(OUT / 'preregistration/prior_manifest.json')['files'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def verify_sources(protocol):
    for name, digest in protocol['worker_source_SHA256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def snapshot():
    names = [(SOURCE / name).relative_to(ROOT).as_posix() for name in
             ('common.py', 'kernels.py', 'runtime.py', 'run.py', 'profile_one.py')]
    names += ['experiments/tma_reduction_layout/phase11/common.py',
              'experiments/tma_reduction_layout/phase11/launch.py',
              'experiments/tma_reduction_layout/phase11/c1/common.py',
              'experiments/tma_reduction_layout/phase11/c1/driver.py',
              'experiments/tma_reduction_layout/phase11/c1/data.py',
              'experiments/tma_reduction_layout/phase11/c15/common.py',
              'experiments/tma_reduction_layout/phase11/c15/data.py',
              'experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    result = {}
    for name in sorted(names):
        data = (ROOT / name).read_bytes()
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        result[name] = sha(data)
    return result
