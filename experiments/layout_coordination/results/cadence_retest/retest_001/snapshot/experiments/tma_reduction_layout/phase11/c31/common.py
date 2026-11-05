"""Isolated C3.1 derived artifacts; all older evidence stays immutable."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
BASE=ROOT/'experiments/tma_reduction_layout'
OUT=BASE/'results/phase11/stage_c31'
V1=OUT/'c3_v1'
V0=BASE/'results/phase11/stage_c3'
PARENT='6a850c0060611dd598466a086c2cf6ec926e33d7'


def read(path):return json.loads(Path(path).read_text())
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(data):return hashlib.sha256(data).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encode(value))


def prior_manifest():
    import subprocess
    names=subprocess.check_output(['git','ls-files','-z','experiments/tma_reduction_layout'],cwd=ROOT).decode().split('\0')
    return {n:sha((ROOT/n).read_bytes()) for n in names if n and
            '/phase11/c31/' not in n and '/stage_c31/' not in n}
