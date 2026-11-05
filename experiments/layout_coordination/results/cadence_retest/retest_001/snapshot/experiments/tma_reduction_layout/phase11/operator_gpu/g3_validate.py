"""Final CPU validator for immutable G3 evidence; does not rerun GPU experiments."""
import argparse
from pathlib import Path
import ast
from zipfile import ZipFile
from . import common as c
from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows,source_blocks


def previous_unchanged():
    p=c.read(c.OUT/'stage_g3a/preregistration/protocol.json')
    for name,digest in p['old_evidence_SHA256'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    return len(p['old_evidence_SHA256'])


def snapshot_matches(folder,protocol_name,snapshot_name):
    p=c.read(folder/'preregistration'/protocol_name)
    with ZipFile(folder/'preregistration'/snapshot_name) as z:
        for name,digest in p['worker_source_SHA256'].items():assert c.sha(z.read(name))==digest,(snapshot_name,name)


def a():
    out=c.OUT/'stage_g3a';p=c.read(out/'preregistration/protocol.json');c.verify_sources(p)
    snapshot_matches(out,'protocol.json','source_snapshot.zip')
    data=c.read(out/'counter_analysis.json');assert len(data['reports'])==len(data['offline_imports'])==60
    expected={(r['metadata']['tag'],mode) for g in p['groups'] for r in g['records'] for mode in p['modes']}
    assert {(r['tag'],r['mode']) for r in data['rows']}==expected
    for row,report,local in zip(data['rows'],data['reports'],data['offline_imports']):
        path=c.ROOT/report['path'];assert c.sha(path.read_bytes())==report['SHA256']
        actual=hardware_rows(c.read(path.parent/'import.json')['output']);assert actual==[row['metrics']]
        imported=c.read(c.ROOT/local['path']);assert hardware_rows(imported['commands'][0]['output'])==actual
        assert row['correctness']['pass']
        if row['mode']=='software':assert source_blocks(imported['commands'][1]['output'])==[row['per_PC']]
    assert (out/'REPORT.md').exists() and (out/'FINDINGS.md').exists()
    return {'status':'OK','reports':60,'CPU_reimports':60,'new_compiles':0,'new_timing_samples':0}


def b():
    from .g3b_prepare import PROTOCOL_FILE
    out=c.OUT/'stage_g3b';p=c.read(out/'preregistration'/PROTOCOL_FILE);c.verify_sources(p)
    for pn,sn in [('protocol.json','source_snapshot.zip'),('protocol_v2.json','source_snapshot_v2.zip'),
                  ('profile_protocol.json','profile_source_snapshot.zip'),
                  ('profile_extension_protocol.json','profile_extension_source_snapshot.zip')]:snapshot_matches(out,pn,sn)
    original=ast.parse((out/'preregistration/upstream_05-layer-norm.py').read_text())
    derivative=ast.parse((c.SOURCE/'g3b_triton.py').read_text())
    for name in ('_layer_norm_bwd_dx_fused','_layer_norm_bwd_dwdb'):
        lhs=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        rhs=next(n for n in derivative.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(lhs,include_attributes=False)==ast.dump(rhs,include_attributes=False),name
    root=out/'cpu/attempt_002/export';compiled=c.read(root/'result.json');assert compiled['status']=='OK'
    assert compiled['native_SHA256']==c.NATIVE and compiled['ccache_before']==compiled['ccache_after']
    assert compiled['protocol_SHA256']==c.sha(c.encode(p))
    assert len(compiled['records'])==36
    for r in compiled['records']:
        for name,digest in r['artifact_SHA256'].items():assert c.sha((root/r['artifact_directory']/name).read_bytes())==digest
    bound=c.read(out/'binding/attempt_001/raw.json');assert bound['status']=='OK' and len(bound['records'])==36 and all(x['match'] for x in bound['records'])
    check=c.read(out/'correctness/attempt_001/raw.json');assert check['status']=='OK' and len(check['checks'])==270 and all(x['pass'] for x in check['checks'])
    assert not check['Triton_imported'] and check['driver_return_codes']==[0]
    expected={(case['id'],mode,m,pattern,tag['tag']) for case in p['cases']
              for mode,m in [('merge',128)]+[('pipeline',m) for m in case['pipeline_M']]
              for pattern in p['correctness']['patterns'] for tag in compiled['records']
              if tag['geometry']==case['id'] and tag['role']=='merge'}
    observed={(r['tag'].split(':')[0],r['mode'],r['M'],r['pattern'],tag)
              for r in check['checks'] for tag in [r['tag']]+r['aliases']}
    assert expected==observed
    timing=c.read(out/'timing_analysis.json');assert timing['raw_samples']==41850 and len(timing['rows'])==99
    for source in timing['sources']:assert c.sha((c.ROOT/source['path']).read_bytes())==source['SHA256']
    data=c.read(out/'counter_analysis.json');assert len(data['reports'])==len(data['offline_imports'])==60
    # A predeclared G0 and later vec8 tag can have the exact same CUBIN/ABI.
    # Verify physical execution identity rather than demand duplicate reports.
    def execution_key(meta):
        return (meta['geometry'],meta['cubin_SHA256'],tuple(meta['runtime_argument_names']),
                tuple(sorted(meta['specialized_runtime_values'].items())),meta['num_warps'],meta['dynamic_smem_bytes'])
    bytag={x['tag']:x for x in compiled['records']}
    expanded={(x['geometry'],meta['tag'],x['mode']) for x in data['rows'] for meta in compiled['records']
              if meta['role']=='merge' and execution_key(meta)==execution_key(bytag[x['tag']])}
    wanted={(x['geometry'],x['tag'],mode) for x in compiled['records'] if x['role']=='merge' for mode in ('software','warm_application')}
    assert expanded==wanted,(wanted-expanded,expanded-wanted)
    for report,local in zip(data['reports'],data['offline_imports']):
        path=c.ROOT/report['path'];assert c.sha(path.read_bytes())==report['SHA256']
        imported=c.read(c.ROOT/local['path'])
        assert imported['report_SHA256']==report['SHA256']
        assert hardware_rows(imported['commands'][0]['output'])==hardware_rows(c.read(path.parent/'import.json')['output'])
        if report['mode']=='software':assert source_blocks(imported['commands'][1]['output'])==source_blocks(c.read(path.parent/'source.json')['output'])
    for name in ['REPORT.md','all_timings.csv']:assert (out/name).is_file()
    return {'status':'OK','original_AST_parity':True,'real_JIT_matches':36,'correctness_checks':270,
            'raw_timing_samples':41850,'NCU_reports':60,'CPU_reimports':60,'native_rebuilds':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['a','b','all'],default='all')
    p.add_argument('--write',action='store_true');args=p.parse_args()
    result={'status':'OK','previous_G2_files_unchanged':previous_unchanged()}
    if args.stage in ('a','all'):result['G3a']=a()
    if args.stage in ('b','all'):result['G3b']=b()
    if args.write:
        for key,folder in [('G3a','stage_g3a'),('G3b','stage_g3b')]:
            if key not in result:continue
            dest=c.OUT/folder/'final_validation.json'
            record={**result[key],'previous_G2_files_unchanged':result['previous_G2_files_unchanged'],
                    'validator_SHA256':c.sha(Path(__file__).read_bytes()),
                    'REPORT_SHA256':c.sha((dest.parent/'REPORT.md').read_bytes())}
            if dest.exists():assert c.read(dest)==record
            else:c.write_once(dest,record)
    print(c.encode(result).decode())


if __name__=='__main__':main()
