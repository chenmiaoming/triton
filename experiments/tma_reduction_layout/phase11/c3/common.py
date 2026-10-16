"""C3 is CPU-only and uses a separate derived-artifact directory."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/tma_reduction_layout'
OUT = BASE / 'results/phase11/stage_c3'
PIN = '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def prior_manifest():
    # All tracked prior experiment evidence, not just the inputs selected for fit.
    import subprocess
    names = subprocess.check_output(['git', 'ls-files', '-z',
        'experiments/tma_reduction_layout/results'], cwd=ROOT).decode().split('\0')
    return {n: digest((ROOT/n).read_bytes()) for n in names if n and
            '/stage_c3/' not in n}
