from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/tma_reduction_layout/results/phase10'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def write_once(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data = value if isinstance(value,bytes) else encode(value)
    with path.open('xb') as f:
        f.write(data)


def read(path):
    return json.loads(Path(path).read_text())
