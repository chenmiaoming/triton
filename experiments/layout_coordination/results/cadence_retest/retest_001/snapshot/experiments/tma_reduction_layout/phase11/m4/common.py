"""New M4 stage paths; stage is fixed by the host launch and remote image env."""
import os
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c2d import common as old

ROOT, BASE, PIN = old.ROOT, old.BASE, old.PIN
SOURCE = Path(__file__).resolve().parent
STAGE = os.environ.get('TMA_M4_STAGE', 'm4a')
assert STAGE in ('m4a', 'm4b'), STAGE
OUT = BASE / 'results/phase11' / ('stage_' + STAGE)
BASE_IMAGE, TOOLS_IMAGE = old.BASE_IMAGE, old.TOOLS_IMAGE
read, encode, sha = old.read, old.encode, old.sha
write_once, archive, extract = old.write_once, old.archive, old.extract


def stride(meta, k=1):
    return meta['M']*meta['N'] if meta.get('output_policy') in ('broadcast', 'centered', 'copy') else meta['N']


def derived(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    blob = value if isinstance(value, bytes) else encode(value)
    if (OUT/'freeze.json').exists() and path.name != 'validation.json':
        assert path.read_bytes() == blob, ('Frozen M4 file', path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def protocol():
    revisions=sorted((OUT/'preregistration/revisions').glob('*/protocol.json'),key=lambda p:int(p.parent.name))
    return read(revisions[-1] if revisions else OUT/'preregistration/protocol.json')


def protect_prior():
    for name, digest in read(BASE/'results/phase11/stage_m4_preparation/prior_manifest.json')['files'].items():
        assert sha((ROOT/name).read_bytes()) == digest, name
    frozen = read(BASE/'results/phase11/stage_m4_preparation/freeze.json')
    for name, digest in frozen['files'].items():
        assert sha((ROOT/name).read_bytes()) == digest, name


def verify_sources(p):
    assert p['stage'].lower() == STAGE
    for name, digest in p['worker_source_SHA256'].items():
        assert sha((ROOT/name).read_bytes()) == digest, name


def snapshot():
    target = Path('/tmp/tma-phase11-' + STAGE + '-snapshot')
    parent = read(BASE/'results/phase11/stage_m3/preregistration/protocol.json')
    names = set(parent['worker_source_SHA256'])
    names.add('experiments/tma_reduction_layout/phase11/c16/kernels.py')
    names.update(str((SOURCE/name).relative_to(ROOT)) for name in
                 ('common.py','driver.py','kernels.py','compile.py','runtime.py','run.py','profile_one.py','static.py'))
    hashes = {}
    for name in sorted(names):
        blob = (ROOT/name).read_bytes()
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        hashes[name] = sha(blob)
    return target, hashes


def artifact_folder(meta, gate=None):
    gate = gate or read(OUT/'compile_gate.json')
    return ROOT/gate['artifact_directory']/meta['tag'].replace(':','/')


def binary_execution_identity(meta):
    return (meta['cubin_SHA256'],sha(encode(meta['tensordesc_meta'])),
            meta.get('global_rows',8192),meta['dtype'],meta['M'],meta['N'])


def timing_root(p=None):
    p=p or protocol()
    return OUT/'timing/revisions/3' if STAGE=='m4a' and p.get('revision',1)>=3 else OUT/'timing'
