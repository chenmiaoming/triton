"""Offline artifact and correctness gates for the fixed production workload matrix."""
from collections import Counter
from pathlib import Path
import re
from . import common as c
from .layout_parsing import plan
from .production_compile import variants
from .production_cases import configuration


def compile_gate():
    protocol = c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    root = c.OUT/'stage_g2/bind/attempt_001/export'
    raw = c.read(root/'result.json')
    assert raw['status']=='OK' and raw['native_SHA256']==raw['native_SHA256_after']==c.NATIVE
    assert len(raw['records'])==56
    cases = {case['id']:case for case in protocol['geometries']}
    expected = set()
    for case in cases.values():
        expected.add(case['id']+':upstream')
        expected.update(case['id']+':'+row['name'] for row in variants(case,raw['plans'][case['id']]))
        if case['family']=='C02':
            expected.add(case['id']+':stage2')
    assert expected=={row['tag'] for row in raw['records']}
    static = {}
    for row in raw['records']:
        folder=root/row['artifact_directory']
        for name,digest in row['artifact_SHA256'].items():
            assert c.sha((folder/name).read_bytes())==digest,(row['tag'],name)
        ptx=(folder/'kernel.ptx').read_text()
        entry=re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid',ptx,re.S)
        abi=re.findall(r'\.param\s+\.(\w+)',entry[1])
        assert len(abi)==len(row['runtime_argument_names'])+2,(row['tag'],abi,row['runtime_argument_names'])
        # Independently check that every frozen ABI name can be bound before
        # allocating another GPU. JSON object order is never used as ABI order.
        const,ptr,values,_=configuration(cases[row['geometry']],row['stage2'])
        allowed=set(const)|set(ptr)|set(values)
        assert set(row['runtime_argument_names'])<=allowed,(row['tag'],set(row['runtime_argument_names'])-allowed)
        sass=(folder/'kernel.sass').read_text()
        resource=(folder/'kernel.resource.txt').read_text()
        audit={'opcode_sites':dict(Counter(re.findall(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',sass))),
               'n_regs':list(map(int,re.findall(r'REG:(\d+)',resource))),
               'local_bytes':list(map(int,re.findall(r'LOCAL:(\d+)',resource))),'ABI':abi}
        if not row['stage2']:
            audit['encodings']=plan((folder/'kernel.ttgir').read_text(),cases[row['geometry']])
        static[row['tag']]=audit
    unique=len({row['cubin_SHA256'] for row in raw['records']})
    assert unique<=protocol['max_unique_binaries']
    gate={'status':'OK','records':raw['records'],'plans':raw['plans'],'static_audit':static,
          'unique_CUBINs':unique,'real_JIT_specialization':True,'native_rebuilds':0,
          'artifact_directory':root.relative_to(c.ROOT).as_posix(),
          'protocol_SHA256':c.sha(c.encode(protocol))}
    c.write_once(c.OUT/'stage_g2/compile_gate.json',gate)
    payload=c.archive(root/'artifacts')
    # Preserve the artifacts/ path that is present in every metadata record.
    import io,zipfile
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as target:
        for path in sorted((root/'artifacts').rglob('*')):
            if path.is_file():
                target.write(path,path.relative_to(root))
    c.write_once(c.OUT/'stage_g2/compiled_payload.zip',stream.getvalue())
    print('COMPILE GATE OK:',len(raw['records']),'programs,',unique,'different CUBINs; actual ABI names verified')


def artifact_gate():
    protocol=c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    gate=c.read(c.OUT/'stage_g2/compile_gate.json')
    raw=c.read(c.OUT/'stage_g2/correctness/attempt_001/raw.json')
    assert raw['status']=='OK' and all(row['pass'] for row in raw['checks'])
    assert not raw['Triton_imported'] and raw['driver_return_codes']==[0]
    assert raw['protocol_SHA256']==c.sha(c.encode(protocol))
    assert raw['payload_SHA256']==c.sha((c.OUT/'stage_g2/compiled_payload.zip').read_bytes())
    expected={(row['tag'],pattern) for row in gate['records'] if not row['stage2'] for pattern in protocol['correctness']['patterns']}
    assert expected=={(row['tag'],row['pattern']) for row in raw['checks']}
    bytag={row['tag']:row for row in gate['records']}
    for tag,binding in raw['bindings'].items():
        assert binding['cubin_SHA256']==binding['loaded_SHA256']==bytag[tag]['cubin_SHA256']
    result={'status':'OK','checks':len(raw['checks']),'bindings':raw['bindings'],'environment':raw['environment'],
            'protocol_SHA256':raw['protocol_SHA256'],'payload_SHA256':raw['payload_SHA256'],
            'source':'correctness/attempt_001/raw.json','all_inputs_canaries_and_complete_outputs':True}
    c.write_once(c.OUT/'stage_g2/artifact_gate.json',result)
    print('ARTIFACT GATE OK:',len(raw['checks']),'complete production reproduction checks')


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['compile','artifact'])
    args=parser.parse_args()
    (compile_gate if args.action=='compile' else artifact_gate)()


if __name__=='__main__':
    main()
