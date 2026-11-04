"""Combine immutable original-U references, including U-only selected KDA configs."""
from pathlib import Path
import copy
from . import common as c


def main():
    target=c.OUT/'stage_n1/references';assert not target.exists()
    package=target/'export';records=[];cases=[];lineage=[]
    origins=[c.OUT/'stage_n1/revision_002/binding/attempt_001/export',
             c.OUT/'stage_n1/revision_003/autotune/attempt_001/export']
    for origin in origins:
        result=c.read(origin/'result.json');assert result['status']=='OK'
        if 'selected_records' in result:
            metas=result['selected_records'];case_map={x['id']:x for x in result['selected_cases']}
        else:
            metas=[x['metadata'] for x in result['records'] if x['status']=='OK'];case_map={m['geometry']:m['case'] for m in metas}
        for meta in metas:
            row=copy.deepcopy(meta);case=case_map[meta['geometry']];cases.append(case)
            row['artifact_directory']='artifacts/'+case['id']+'_U'
            for f in sorted((origin/meta['artifact_directory']).iterdir()):
                if f.is_file():c.write_once(package/row['artifact_directory']/f.name,f.read_bytes())
            for name,digest in row['artifact_SHA256'].items():
                assert c.sha((package/row['artifact_directory']/name).read_bytes())==digest
            row['case']=case;records.append(dict(status='OK',metadata=row))
            lineage.append(dict(case=case['id'],original_metadata=meta,
                result_path=str((origin/'result.json').relative_to(c.ROOT)),
                result_SHA256=c.sha((origin/'result.json').read_bytes())))
    assert len(cases)==len(records)==14 and len({x['id'] for x in cases})==14
    c.write_once(package/'result.json',dict(status='OK',records=records,selected_cases=cases,lineage=lineage))
    c.write_once(target/'upstream_payload.zip',c.archive(package))
    print('Combined original U references:',len(records),'programs; six families, RoPE subtypes retained')


if __name__=='__main__':main()
