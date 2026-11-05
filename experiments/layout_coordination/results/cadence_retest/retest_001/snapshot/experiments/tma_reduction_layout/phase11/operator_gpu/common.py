"""New operator GPU evidence only; existing stages are read-only."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[3]
OUT = ROOT / 'experiments/tma_reduction_layout/results/phase11/operator_gpu'
PREP = ROOT / 'experiments/tma_reduction_layout/results/phase11/operator_survey_preparation/20261021'
BASE_IMAGE = 'im-qzTUZumWu2JXjWic8K82cZ'
TOOLS_IMAGE = 'im-FUYtAVgmAyw1ZiFjfu6cmg'
NATIVE = 'fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value if isinstance(value, bytes) else encode(value))


def archive(root):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as target:
        for path in sorted(Path(root).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                target.write(path, path.relative_to(root))
    return stream.getvalue()


def extract(payload, root):
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        for name in source.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        source.extractall(root)


def verify_sources(protocol):
    for name, digest in protocol['worker_source_SHA256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def snapshot(names, target):
    target = Path(target)
    hashes = {}
    for name in sorted(set(names)):
        blob = (ROOT / name).read_bytes()
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        hashes[name] = sha(blob)
    return hashes
