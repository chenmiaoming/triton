"""Isolated C2 batches; all previous raw evidence remains immutable."""
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c16 import common as prior

ROOT, base = prior.ROOT, prior.base
SOURCE = Path(__file__).resolve().parent
read, sha, encode = prior.read, prior.sha, prior.encode
write_once, archive, extract = prior.write_once, prior.archive, prior.extract
BASE_IMAGE='im-qzTUZumWu2JXjWic8K82cZ'
TOOLS_IMAGE='im-gCqknrtnwsoW0fVOiG5XF9'


def out(batch):
    assert batch in ('a','b')
    return base.OUT/('stage_c2'+batch)


def load_protocol(batch):
    root=out(batch)/'preregistration'
    revisions=list((root/'revisions').glob('*/protocol.json'))
    return read(max(revisions,key=lambda p:int(p.parent.name))) if revisions else read(root/'protocol.json')


def protect_prior(batch):
    for name,digest in read(out(batch)/'preregistration/prior_manifest.json')['files'].items():
        assert sha((ROOT/name).read_bytes())==digest,name


def verify_sources(protocol):
    for name,digest in protocol['worker_source_SHA256'].items():
        assert sha((ROOT/name).read_bytes())==digest,name


def snapshot(batch):
    target=Path('/tmp/tma-phase11-c2'+batch+'-snapshot')
    names=[(SOURCE/name).relative_to(ROOT).as_posix() for name in
           ('common.py','kernels.py','data.py','driver.py','profile_one.py','timing.py','run.py')]
    names += ['experiments/tma_reduction_layout/phase11/common.py',
              'experiments/tma_reduction_layout/phase11/launch.py',
              'experiments/tma_reduction_layout/phase11/c1/common.py',
              'experiments/tma_reduction_layout/phase11/c15/common.py',
              'experiments/tma_reduction_layout/phase11/c16/common.py',
              'experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    hashes={}
    for name in sorted(names):
        value=(ROOT/name).read_bytes()
        dest=target/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(value)
        hashes[name]=sha(value)
    return target,hashes


def stride(meta,k):
    if meta['family']=='MC':
        return k*meta['N']
    if meta['family'] in ('ML','E'):
        return meta['D']*meta['M']*meta['N']
    return meta['D']*meta['N']
