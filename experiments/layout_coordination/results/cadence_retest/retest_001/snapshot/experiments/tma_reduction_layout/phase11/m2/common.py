"""M2-only evidence paths; prior sources and artifacts remain immutable."""
from pathlib import Path
import subprocess
from experiments.tma_reduction_layout.phase11.c2d import common as old

ROOT, BASE = old.ROOT, old.BASE
SOURCE = Path(__file__).resolve().parent
OUT = BASE/'results/phase11/stage_m2'
PIN, BASE_IMAGE, TOOLS_IMAGE = old.PIN, old.BASE_IMAGE, old.TOOLS_IMAGE
read, encode, sha = old.read, old.encode, old.sha
write_once, archive, extract, stride = old.write_once, old.archive, old.extract, old.stride


def derived(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    blob = value if isinstance(value, bytes) else encode(value)
    if (OUT/'freeze.json').exists() and path.name != 'validation.json':
        assert path.read_bytes() == blob, ('Frozen M2 file', path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def protocol():
    revisions = sorted((OUT/'preregistration/revisions').glob('*/protocol.json'),
                       key=lambda p: int(p.parent.name))
    return read(revisions[-1] if revisions else OUT/'preregistration/protocol.json')


def protect_prior():
    for name, digest in read(OUT/'preregistration/prior_manifest.json')['files'].items():
        assert sha((ROOT/name).read_bytes()) == digest, name


def verify_sources(p):
    for name, digest in p['worker_source_SHA256'].items():
        assert sha((ROOT/name).read_bytes()) == digest, name


def snapshot():
    target = Path('/tmp/tma-phase11-m2-snapshot')
    names = set(read(BASE/'results/phase11/stage_d1/preregistration/protocol.json')['worker_source_SHA256'])
    names.update(str((SOURCE/name).relative_to(ROOT)) for name in
                 ('common.py', 'kernels.py', 'runtime.py', 'run.py', 'profile_one.py'))
    hashes = {}
    for name in sorted(names):
        blob = (ROOT/name).read_bytes()
        dest = target/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        hashes[name] = sha(blob)
    return target, hashes


def artifact_folder(meta, gate=None):
    gate = gate or read(OUT/'compile_gate.json')
    return ROOT/gate['artifact_directory']/meta['tag'].replace(':', '/')


def git_blob(name):
    return subprocess.check_output(['git', 'show', PIN+':'+name], cwd=ROOT)
