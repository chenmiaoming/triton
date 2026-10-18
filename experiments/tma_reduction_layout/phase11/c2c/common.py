"""C2c evidence utilities. Older stages and raw exports are write-once."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/tma_reduction_layout'
OUT = BASE / 'results/phase11/stage_c2c'
SOURCE = Path(__file__).resolve().parent
BASE_IMAGE = 'im-qzTUZumWu2JXjWic8K82cZ'
TOOLS_IMAGE = 'im-gCqknrtnwsoW0fVOiG5XF9'
PIN = '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'


def read(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else encode(value)
    with path.open('xb') as stream:
        stream.write(data)


def derived(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else encode(value))


def archive(root):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(Path(root).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                output.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


def extract(blob, root):
    with zipfile.ZipFile(io.BytesIO(blob)) as source:
        for name in source.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        source.extractall(root)


def prior_manifest():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', 'experiments/tma_reduction_layout'], cwd=ROOT
    ).decode().split('\0')
    return {name: sha((ROOT / name).read_bytes()) for name in names if name and
            '/phase11/c2c/' not in name and '/stage_c2c/' not in name and
            '/phase11/c32/' not in name and '/stage_c32/' not in name}


def protect_prior():
    for name, digest in read(OUT / 'preregistration/prior_manifest.json')['files'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def protocol():
    revisions = sorted((OUT / 'preregistration/revisions').glob('*/protocol.json'),
                       key=lambda path: int(path.parent.name))
    return read(revisions[-1] if revisions else OUT / 'preregistration/protocol.json')


def verify_sources(value):
    for name, digest in value['worker_source_SHA256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def snapshot():
    target = Path('/tmp/tma-phase11-c2c-snapshot')
    names = [(SOURCE/name).relative_to(ROOT).as_posix() for name in
             ('common.py', 'kernels.py', 'runtime.py', 'run.py', 'profile_one.py')]
    names += [
        'experiments/tma_reduction_layout/phase11/c2/common.py',
        'experiments/tma_reduction_layout/phase11/c2/driver.py',
        'experiments/tma_reduction_layout/phase11/c2/data.py',
        'experiments/tma_reduction_layout/phase11/c2/kernels.py',
        'experiments/tma_reduction_layout/phase11/common.py',
        'experiments/tma_reduction_layout/phase11/launch.py',
        'experiments/tma_reduction_layout/phase11/c1/common.py',
        'experiments/tma_reduction_layout/phase11/c15/common.py',
        'experiments/tma_reduction_layout/phase11/c16/common.py',
        'experiments/tma_reduction_layout/phase8/launch.py',
        'experiments/tma_reduction_layout/phase4/exact_cuda.py',
    ]
    hashes = {}
    for name in sorted(set(names)):
        value = (ROOT / name).read_bytes()
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        hashes[name] = sha(value)
    return target, hashes


def stride(meta, k):
    if meta['kind'] == 'local':
        # Same observable result size in all four matched variants: one FP32
        # checksum per logical thread, followed by every final BF16 state as FP32.
        return 128 * (1 + meta['R'])
    return k * meta['N']
