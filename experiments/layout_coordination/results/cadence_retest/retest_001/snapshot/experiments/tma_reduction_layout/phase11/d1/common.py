"""D1-only output paths; every earlier source/evidence file is immutable."""
from pathlib import Path
import subprocess
from experiments.tma_reduction_layout.phase11.c2d import common as prior

ROOT,BASE=prior.ROOT,prior.BASE
SOURCE=Path(__file__).resolve().parent
OUT=BASE/'results/phase11/stage_d1'
PIN,BASE_IMAGE,TOOLS_IMAGE=prior.PIN,prior.BASE_IMAGE,prior.TOOLS_IMAGE
read,encode,sha=prior.read,prior.encode,prior.sha
write_once,archive,extract,stride=prior.write_once,prior.archive,prior.extract,prior.stride


def derived(path,value):
    path=Path(path)
    assert path.resolve().is_relative_to(OUT.resolve()),path
    blob=value if isinstance(value,bytes) else encode(value)
    if (OUT/'freeze.json').exists() and path.name!='validation.json':
        assert path.is_file() and path.read_bytes()==blob, f'Frozen D1 file: {path}'
        return
    prior.derived(path,blob)


def protocol():return read(OUT/'preregistration/protocol.json')


def execution_protocol():
    revisions=sorted((OUT/'preregistration/execution_revisions').glob('*/protocol.json'),
                     key=lambda x:int(x.parent.name))
    return read(revisions[-1]) if revisions else protocol()


def protect_prior():
    for name,digest in read(OUT/'preregistration/prior_manifest.json')['files'].items():
        assert sha((ROOT/name).read_bytes())==digest,name


def verify_sources(value):
    allowed={tuple(v['geometry']) for v in value['D1_cases']}
    assert len(allowed)==10 and (16,64,8) not in allowed
    assert len(value['cases'])==69 and all((r['M'],r['N'],r['num_warps']) in allowed for r in value['cases'])
    for name,digest in value['worker_source_SHA256'].items():
        assert sha((ROOT/name).read_bytes())==digest,name


def snapshot():
    target=Path('/tmp/tma-phase11-d1-snapshot')
    names=set(prior.protocol()['worker_source_SHA256'])
    names.update(str((SOURCE/name).relative_to(ROOT)) for name in
                 ('common.py','driver.py','runtime.py','run.py','profile_one.py'))
    hashes={}
    for name in sorted(names):
        blob=(ROOT/name).read_bytes();dest=target/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
        hashes[name]=sha(blob)
    return target,hashes
