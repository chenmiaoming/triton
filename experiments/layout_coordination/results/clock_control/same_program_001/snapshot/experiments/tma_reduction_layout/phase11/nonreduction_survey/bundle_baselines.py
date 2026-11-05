"""Assemble a derived payload from accepted actual-Tensor U/G0 evidence."""
from pathlib import Path
import shutil
from . import common as c


def run(revision=9):
    source=c.OUT/'stage_n1'/f'revision_{revision:03}'/'g0bind/attempt_001/export'
    raw=c.read(source/'result.json');assert raw['status']=='OK'
    destination=c.OUT/'stage_n1'/'baseline_payload';assert not destination.exists()
    records=[];programs=[]
    for row in raw['records']:
        assert row['status']=='OK' and row['allocated_sections_equal'] and row['ABI_and_launch_equal']
        case=row['metadata']['case'];pair={}
        for variant,key,base in [('U','upstream',source/'reference/reference'),('G0','metadata',source)]:
            meta=dict(row[key]);target=Path('artifacts')/(case['id']+'_'+variant)
            shutil.copytree(base/meta['artifact_directory'],destination/target)
            meta.update(tag=case['id']+':'+variant,geometry=case['id'],variant=variant,
                        artifact_directory=str(target),case=case)
            pair[variant]=meta;programs.append(meta)
        records.append(dict(case=case['id'],status='OK',metadata=pair['G0'],upstream=pair['U'],
            lineage=dict(result_path=str((source/'result.json').relative_to(c.ROOT)),
                         result_SHA256=c.sha((source/'result.json').read_bytes()))))
    result=dict(status='OK',records=records,programs=programs,
        cases=[r['metadata']['case'] for r in records],
        kind='Derived payload; original compiler artifacts copied byte-for-byte; paths rewritten only in this index.')
    c.write_once(destination/'result.json',result)
    path=c.OUT/'stage_n1'/'baseline_payload.zip';c.write_once(path,c.archive(destination))
    print(path,len(programs))


if __name__=='__main__':run()
