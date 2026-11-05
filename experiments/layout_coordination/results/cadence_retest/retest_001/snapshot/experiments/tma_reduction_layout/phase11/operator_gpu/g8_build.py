"""Verify effective layouts and package exact binaries before GPU execution."""
from pathlib import Path
import io
import zipfile
from . import common as c
from .g5_analyze import elf
from .g8_prepare import OUT
from .layout_parsing import encodings,tensor_encoding,plan


def main():
    p=c.read(OUT/'preregistration/protocol.json');c.verify_sources(p)
    root=OUT/'cpu/attempt_001/export';raw=c.read(root/'result.json')
    assert raw['status']=='OK' and not raw['GPU_allocated'] and not raw['compiler_intervention']
    assert raw['native_SHA256']==c.NATIVE and raw['ccache_before']==raw['ccache_after']
    records=raw['records'];by={r['tag']:r for r in records};assert len(by)==len(records)
    expected={g['id']+':'+v['name'] for g in p['inference_groups'] for v in g['variants']}
    expected|={g['id']+':stage2' for g in p['inference_groups'] if g['case']['family']=='C02'}
    expected|={g['id']+':'+v['name'] for g in p['training_cases'] for v in g['variants']}
    expected|={g['id']+':dx' for g in p['training_cases']};assert set(by)==expected
    cases={g['id']:g['case'] for g in p['inference_groups']};cases.update({g['id']:g for g in p['training_cases']})
    audit=[];paths=set()
    for meta in records:
        folder=root/meta['artifact_directory'];paths.add(meta['artifact_directory'])
        for name,digest in meta['artifact_SHA256'].items():assert c.sha((folder/name).read_bytes())==digest
        if meta['role'] in ['dx','stage2']:continue
        case=cases[meta['group']];ir=(folder/'kernel.ttgir').read_text();defs=encodings(ir)
        if case['family']=='C02':
            actual=plan(ir,case)
            okay=actual['V']==meta['requested_input'] and actual['K']==meta['requested_K'] and actual['output']==meta['requested_output']
        else:
            load=next(s for s in ir.splitlines() if 'tt.load ' in s and f'tensor<{case["BM"]}x{case["BN"]}x' in s)
            store=next(s for s in ir.splitlines() if 'tt.store ' in s and f'tensor<{case["BN"]}x' in s)
            actual=dict(input=tensor_encoding(load,[case['BM'],case['BN']],defs),
                        output=tensor_encoding(store,[case['BN']],defs))
            output={'kind':'slice','dim':0,'parent':meta['requested_input']} if meta['native_output'] else meta['requested_output']
            okay=actual['input']==meta['requested_input'] and actual['output']==output
        audit.append(dict(tag=meta['tag'],effective_plan=actual,requested_plan_preserved=okay))
        assert okay,(meta['tag'],actual,meta['requested_input'])
        if meta['variant']['name'].split(':')[-1]=='gluon_anchor':
            prefix=meta['variant']['name'].removesuffix('gluon_anchor')
            upstream=by[meta['group']+':'+prefix+'upstream']
            assert elf((folder/'kernel.cubin').read_bytes())==elf((root/upstream['artifact_directory']/'kernel.cubin').read_bytes()),meta['tag']
            assert all(meta[k]==upstream[k] for k in ['runtime_argument_names','num_warps','dynamic_smem_bytes','global_scratch_size'])
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for dirname in sorted(paths):
            for f in sorted((root/dirname).iterdir()):
                if f.is_file():archive.write(f,f.relative_to(root))
    payload=stream.getvalue();c.write_once(OUT/'compiled_payload.zip',payload)
    gate=dict(status='OK',records=records,protocol_SHA256=c.sha(c.encode(p)),payload_SHA256=c.sha(payload),
        CPU_export_directory=str(root.relative_to(c.ROOT)),native_rebuilds=0,
        new_unique_Gluon_programs=raw['new_unique_Gluon_programs'],effective_layout_audit=audit)
    c.write_once(OUT/'compile_gate.json',gate)
    groups={g['id']:g for g in p['inference_groups']};groups.update({g['id']:dict(case=g,variants=g['variants']) for g in p['training_cases']})
    profile_groups=[];profile_seen={}
    for selected in p['profile_selectors']:
        g=groups[selected['group']];items=[];seen={}
        for name in selected['names']:
            tag=selected['group']+':'+name
            if tag not in by:
                canonical=next((r['name'] for r in g['case']['layouts'] if name in r.get('aliases',[])),None)
                assert canonical,(selected['group'],name)
                tag=selected['group']+':'+canonical
            meta=by[tag];key=(meta['cubin_SHA256'],tuple(meta['runtime_argument_names']))
            if key in seen:seen[key]['aliases'].append(selected['group']+':'+name);continue
            globalkey=(g['case']['id'],*key)
            prior=profile_seen.get(globalkey)
            if prior is not None and set(selected['modes'])<=prior['modes']:
                prior['item']['aliases'].append(selected['group']+':'+name)
                continue
            item=dict(metadata=meta,aliases=[]);items.append(item);seen[key]=item
            profile_seen[globalkey]=dict(item=item,modes=set(selected['modes']))
        if items:profile_groups.append(dict(id=selected['group'],case=g['case'],records=items,modes=selected['modes']))
    pp={**p,'profile_groups':profile_groups,'payload_SHA256':c.sha(payload),'compile_gate_SHA256':c.sha(c.encode(gate)),
        'parent_protocol_SHA256':c.sha(c.encode(p)),
        'profile_alias_policy':'Identical CUBIN/ABI and same geometry may reuse an existing report only when all requested replay modes are covered.'}
    c.write_once(OUT/'profile_protocol.json',pp)
    print('G8 compile/layout gate OK:',len(records),'logical programs;',raw['new_unique_Gluon_programs'],'new unique Gluon programs;',
          sum(len(g['records'])*len(g['modes']) for g in profile_groups),'NCU reports planned')


if __name__=='__main__':main()
