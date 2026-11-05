"""Phase 11 evidence utilities. Original evidence is always written once."""
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/tma_reduction_layout/results/phase11'
BASELINE = '19f9b0fad0624fd809eecfc552228af953b0a588'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(value if isinstance(value, bytes) else encode(value))


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def archive(root):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(root).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p, p.relative_to(root))
    return buffer.getvalue()


def extract(data, root):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in z.namelist())
        assert not any((Path(root) / p).exists() for p in z.namelist())
        z.extractall(root)


def prior_inventory():
    paths = subprocess.check_output(['git', 'ls-files', '-z', 'experiments/tma_reduction_layout'], cwd=ROOT).split(b'\0')
    return {p.decode(): sha((ROOT / p.decode()).read_bytes()) for p in paths if p}


def protect_prior():
    saved = read(OUT / 'stage_a/preregistration/prior_manifest.json')
    for name, digest in saved['files'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def source_snapshot(destination):
    # Upload only exact worker dependencies, never mutable raw/result trees.
    names = [p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')]
    names += ['experiments/tma_reduction_layout/analyze_ir.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py',
              'experiments/tma_reduction_layout/phase8/launch.py']
    target = Path(destination)
    for name in sorted(names):
        p = target / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    return {name: sha((ROOT / name).read_bytes()) for name in sorted(names)}


def verify_sources(protocol):
    for name, digest in protocol['worker_source_SHA256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name


def metadata_tag(case, variant):
    return case['case_id'] + ':' + variant
