"""Write-once non-reduction evidence; no changes to previous experiments."""
from pathlib import Path
import importlib
from ..operator_gpu import common as old

ROOT=old.ROOT
SOURCE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tma_reduction_layout/results/phase11/nonreduction'
NATIVE=old.NATIVE
BASE_IMAGE=old.BASE_IMAGE
TOOLS_IMAGE=old.TOOLS_IMAGE
sha,read,encode,write_once,archive,extract=old.sha,old.read,old.encode,old.write_once,old.archive,old.extract
ADAPTERS={'N01':'quant_adapter','N02':'quant_adapter','N03':'rotary_adapter',
          'N04':'pack_gate_adapter','N05':'pack_gate_adapter','N06':'cache_adapter'}


def adapter(case):
    return importlib.import_module('.'+ADAPTERS[case['family']],__package__)


def verify_sources(protocol):
    for name,digest in protocol['source_SHA256'].items():
        assert sha((ROOT/name).read_bytes())==digest,name


def verify_snapshot(protocol,snapshot):
    for name,digest in protocol['source_SHA256'].items():
        assert sha((Path(snapshot)/name).read_bytes())==digest,name


def cases(families=None):
    result=[]
    for name in dict.fromkeys(ADAPTERS.values()):
        path=SOURCE/(name+'.py')
        if not path.exists():continue
        module=importlib.import_module('.'+name,__package__)
        result.extend(x for x in module.cases() if families is None or x['family'] in families)
    assert len({x['id'] for x in result})==len(result)
    return result
