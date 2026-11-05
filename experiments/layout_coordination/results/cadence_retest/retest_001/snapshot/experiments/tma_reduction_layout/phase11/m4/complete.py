"""M4b final audit also verifies every prior M4 preparation/M4a frozen byte."""
from . import common as c
from .validate import main as validate_stage


def main():
    assert c.STAGE=='m4b'
    checked=[]
    for stage in ('stage_m4_preparation','stage_m4a'):
        path=c.BASE/'results/phase11'/stage/'freeze.json'
        frozen=c.read(path)
        for name,digest in frozen['files'].items():
            assert c.sha((c.ROOT/name).read_bytes())==digest,name
        checked.append({'stage':stage,'freeze_SHA256':c.sha(path.read_bytes()),
                        'files_unchanged':len(frozen['files'])})
    c.derived(c.OUT/'prior_M4_chain_validation.json',{'status':'OK','CPU_only':True,'stages':checked})
    validate_stage()


if __name__=='__main__':main()
