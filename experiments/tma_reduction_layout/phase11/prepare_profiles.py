"""Freeze archive profile bindings and source snapshots, without GPU calls."""
import argparse
import io
import zipfile
from experiments.tma_reduction_layout.phase11 import common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('a','b'),required=True)
    args = parser.parse_args()
    if args.stage=='a':
        from experiments.tma_reduction_layout.phase11.validate_a import derive
        coverage = derive()
        targets = c.read(c.OUT/'stage_a/preregistration/revisions/3/protocol.json')['targets']
        bindings = {}
        for target in targets:
            bindings[target] = []
            for variant,label in [('baseline','archived_ordinary_load'),('patch','archived_matrix_load')]:
                root = c.ROOT/'experiments/tma_reduction_layout/results/phase10/stage_c/artifacts'/target/variant/'export/M32_N128_w4/bfloat16/max/reduce/host'/variant
                meta = c.read(root/'metadata.json')
                meta['tag'] = label
                meta['full_grid'] = meta['grid']
                bindings[target].append((meta,root))
        purpose = 'Instrument coverage on archived initial LDS/LDSM kernels; not old-model evaluation'
        repeats = 1
    else:
        coverage = c.read(c.OUT/'stage_a/analysis.json')
        targets = {'sm90':{'cc':[9,0],'name':'H100','gpu':'H100!:1'}}
        gate = c.read(c.OUT/'stage_b/artifact_gate.json')
        bindings = {'sm90':[]}
        artifact_root = c.OUT/'stage_b/artifacts'/gate['accepted_attempt']/'export'
        for tag in sorted({tag for tags in gate['bridge_groups'].values() for tag in tags}):
            bindings['sm90'].append((gate['metadata'][tag],artifact_root/tag.replace(':','/')))
        purpose = 'Canonical Triton versus matching-layout Gluon reproduction only; no C1/C2 mechanism intervention'
        repeats = 3
    snapshot = '/tmp/tma-phase11-profile-'+args.stage+'-snapshot'
    sources = c.source_snapshot(snapshot)
    request = {'stage':args.stage.upper(),'purpose':purpose,'targets':targets,'worker_source_SHA256':sources,
               'metadata':{},'metrics':{},'repeats':repeats,'performance_timing':False,
               'outcome_based_repeats':False,'original_sources':{}}
    destination = c.OUT/('stage_'+args.stage)/'archived_profiles'
    for target,pairs in bindings.items():
        request['metadata'][target] = {}
        request['metrics'][target] = coverage['targets'][target]['metric_mapping']['available']
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for meta,root in pairs:
                request['metadata'][target][meta['tag']] = meta
                request['original_sources'][target+':'+meta['tag']] = {
                    'metadata_path':(root/'metadata.json').relative_to(c.ROOT).as_posix(),
                    'metadata_SHA256':c.sha((root/'metadata.json').read_bytes()),'CUBIN_SHA256':meta['cubin_SHA256']}
                for ext in ('ptx','cubin'):
                    z.writestr(meta['tag'].replace(':','/')+'/kernel.'+ext,(root/('kernel.'+ext)).read_bytes())
        c.write_once(destination/('payload_'+target+'.zip'),stream.getvalue())
    c.write_once(destination/'request.json',request)
    c.write_once(destination/'source_snapshot.zip',c.archive(snapshot))
    print('Profile request frozen',args.stage,len(request['original_sources']), 'binaries')


if __name__=='__main__':
    main()
