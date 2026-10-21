"""Exploratory extension: complete all missing training candidates, without retiming."""
from datetime import datetime,timezone
from pathlib import Path
from . import common as c
from .g3b_prepare import OUT,PROTOCOL_FILE
SNAPSHOT=Path('/tmp/operator-gpu-g3b-profile-extension-snapshot-v1')


def main():
    original=c.read(OUT/'preregistration/profile_protocol.json')
    training=c.read(OUT/'preregistration'/PROTOCOL_FILE)
    compiled=c.read(OUT/'cpu/attempt_002/export/result.json')
    groups=[]
    for case in training['cases']:
        previous={r['metadata']['cubin_SHA256'] for g in original['groups'] if g['case']['id']==case['id'] for r in g['records']}
        rows=[];seen={}
        for meta in compiled['records']:
            if meta['geometry']!=case['id'] or meta['role']!='merge' or meta['cubin_SHA256'] in previous:continue
            digest=meta['cubin_SHA256']
            if digest in seen:seen[digest]['aliases'].append(meta['tag']);continue
            row={'metadata':meta,'aliases':[]};rows.append(row);seen[digest]=row
        if rows:groups.append({'case':case,'records':rows})
    names=list(training['worker_source_SHA256'])
    names += [p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.glob('g3b_profile*.py')]
    hashes=c.snapshot(names,SNAPSHOT)
    protocol={**original,'stage':'G3b_profiles_extension','groups':groups,'worker_source_SHA256':hashes,
              'frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
              'selection':'All previously unprofiled unique merge candidates at all three geometries, including controls; zero new timing',
              'selection_after_initial_timing':True,'confirmatory_claim':False,
              'initial_timing_analysis_SHA256':c.sha((OUT/'timing_analysis.json').read_bytes()),
              'reason':'Complete NCU coverage rather than attribute vec2/vec4 gains from only vec1/vec8 counters.'}
    c.write_once(OUT/'preregistration/profile_extension_protocol.json',protocol)
    c.write_once(OUT/'preregistration/profile_extension_source_snapshot.zip',c.archive(SNAPSHOT))
    print('Exploratory complete-matrix NCU extension:',sum(len(g['records']) for g in groups),'binaries; no new timing')


if __name__=='__main__':main()
