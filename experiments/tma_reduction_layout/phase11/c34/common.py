"""C3.4 CPU-only paths and write-once method/input binding."""
from pathlib import Path
import subprocess
from experiments.tma_reduction_layout.phase11.c33 import common as prior

ROOT, BASE = prior.ROOT, prior.BASE
SOURCE = Path(__file__).resolve().parent
OUT = BASE / 'results/phase11/stage_c34'
read, encode, sha = prior.read, prior.encode, prior.sha
write_once = prior.write_once


def derived(path,value):
    path=Path(path)
    assert path.resolve().is_relative_to(OUT.resolve()),path
    blob=value if isinstance(value,bytes) else encode(value)
    if (OUT/'freeze.json').exists() and path.name!='validation.json':
        assert path.is_file() and path.read_bytes()==blob, f'Frozen C3.4 output: {path}'
        return
    prior.derived(path,blob)


def protocol():
    return read(OUT / 'protocol.json')


def protect_prior():
    for name, digest in read(OUT / 'prior_manifest.json')['files'].items():
        assert sha((ROOT/name).read_bytes()) == digest, name


def preregister():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', 'experiments/tma_reduction_layout'], cwd=ROOT
    ).decode().split('\0')
    files = {n:sha((ROOT/n).read_bytes()) for n in names if n}
    write_once(OUT/'prior_manifest.json', {'files':files})
    p = prior.protocol()
    value = {**p, 'stage':'C3.4', 'date':'2026-10-18',
        'plan_SHA256':sha((SOURCE/'PLAN.md').read_bytes()),
        'C33_commit':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'C33_theta_SHA256':sha((prior.OUT/'theta_H100.json').read_bytes()),
        'resource_margin_candidates':[0.,1.,2.,4.,8.,12.,16.,24.,32.,48.,64.],
        'resource_calibration_target_coverage':1.,
        'eta_q_slopes':[-.25,.25], 'GPU_allocations':0, 'native_rebuilds':0,
        'resource_calibration_uses_timing':False}
    value.pop('resource_margin_registers')
    write_once(OUT/'protocol.json', value)
    print('C3.4 method/input binding frozen before fits; CPU only', flush=True)


if __name__ == '__main__':
    preregister()
