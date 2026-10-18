"""Recompute FULL/MC slopes and frozen evidence; never launch a GPU."""
import argparse
from collections import Counter
import statistics as st
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c2d.analyze import fit
from experiments.tma_reduction_layout.phase11.c2d.runtime import cells_for
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def run(committed=False):
    c.protect_prior();p=c.protocol();c.verify_sources(p)
    g=c.read(c.OUT/'artifact_gate.json');cg=c.read(c.OUT/'compile_gate.json')
    a=c.read(c.OUT/'analysis.json');counter=c.read(c.OUT/'counter_gate.json')
    matrix=c.read(c.OUT/'matrix.json')
    assert all(v['status']=='OK' for v in (g,cg,a,counter,matrix))
    assert len(g['metadata'])==76 and cg['new_compiles']==31 and cg['reused']==45
    split=c.read(c.OUT/'preregistration/D1_split.json')
    assert split['moved_to_development']==[16,64,8] and len(split['cases'])==10
    assert Counter(x['stratum'] for x in split['cases'])=={'empirically_unseen_core':5,'legacy_boundary':5}
    assert all(x['history_category']=='strict_zero_mention' for x in split['cases'])
    assert c.sha((c.OUT/'preregistration/C33_METHOD.md').read_bytes())==p['C33_method_SHA256']
    assert c.sha((c.SOURCE/'PLAN.md').read_bytes())==p['plan_SHA256']
    protected={tuple(row) for row in p['reserved_D1']}
    rawc=c.read(c.ROOT/cg['artifact_directory']/'result.json')
    assert rawc['ccache_before']==rawc['ccache_after'] and not rawc['GPU_allocated']
    assert rawc['native_SHA256']==p['baseline']['native_SHA256']
    for tag,m in g['metadata'].items():
        assert (m['M'],m['N'],m['num_warps']) not in protected
        assert m['path']=='host' and m['global_rows']==24576
        folder=c.ROOT/cg['artifact_directory']/tag.replace(':','/')
        assert c.sha((folder/'kernel.cubin').read_bytes())==m['cubin_SHA256']==g['resources'][tag]['loaded_CUBIN_SHA256']
        assert not g['resources'][tag]['attributes']['local_bytes']
        assert cg['static_audit'][tag]['local_load_store_SASS_sites']==0
        if m.get('reuse_source'):
            for f in folder.glob('kernel.*'):
                assert f.read_bytes()==(c.ROOT/m['reuse_source']/f.name).read_bytes()
        if m.get('original_once'):
            assert m['ABI_kind']=='descriptor'
            assert 'scf.for' not in (folder/'kernel.glir').read_text()
            assert 'scf.while' not in (folder/'kernel.glir').read_text()
    expected={case['id'] for geo in p['calibration_geometries'] for case in candidates(*geo)}
    assert len(expected)==61 and {x['layout'] for x in matrix['matrix']}==expected
    sources={};samples=0;uuids=[]
    expected_cells=cells_for(p,g)
    for i in (1,2,3):
        raw=c.read(c.OUT/'timing'/f'invocation_{i}'/'raw.json')
        assert raw['status']=='OK' and not raw['Triton_imported'] and not raw['profiler_attached']
        assert raw['protocol_SHA256']==c.sha(c.encode(p))
        assert raw['payload_SHA256']==c.sha((c.OUT/'compiled_payload.zip').read_bytes())
        assert raw['driver_return_codes']==[0]
        assert len(raw['cells'])==len(expected_cells)==1248
        uuids.append(raw['environment']['gpu_uuid']);values={}
        assert [r['cell_id'] for r in raw['cells']]==[r['cell_id'] for r in expected_cells]
        for row in raw['cells']:
            assert len(row['samples_us'])==12 and row['correctness']['pass']
            assert all(sorted(v['node_types'])==[0,7,7] for v in row['visits'])
            assert row['correctness']['pids']==sorted({0,row['grid']//2,row['grid']-1})
            samples+=len(row['samples_us'])
            for identity in [row['cell_id']]+row['aliases']:
                tag,q,grid,k=identity.split('|')
                values[tag,q[1:],int(grid[1:]),int(k[1:])]=st.median(row['samples_us'])
        sources[i]=values
    assert samples==44928==a['event_samples']
    for row in a['oneshot']:
        values=sources[row['invocation']];gs=[row['full_grid']*v for v in (1,2,3)]
        f=fit([v/132 for v in gs],[values[row['tag'],row['q_condition'],v,1] for v in gs])
        assert abs(1000*f['slope']-row['tile_slope_ns'])<1e-8
        assert abs(values[row['tag'],row['q_condition'],row['full_grid'],1]-row['full_us'])<1e-10
    for row in a['services']:
        values=sources[row['invocation']];grids=[32,64,96]
        slopes=[fit([16,64,256],[values[row['tag'],row['q_condition'],v*132,k] for k in (16,64,256)])['slope'] for v in grids]
        assert abs(1000*fit(grids,slopes)['slope']-row['service_ns'])<1e-8
    assert len(counter['reports'])==14 and len(counter['dynamic'])==82 and len(counter['hardware'])==152
    imported=c.read(c.OUT/'offline_import_validation.json')
    assert imported['status']=='OK' and len(imported['reports'])==14
    for row in counter['reports']: assert c.sha((c.ROOT/row['path']).read_bytes())==row['SHA256']
    for row in imported['reports']:
        assert row['all_metrics_and_source_match']
        assert c.sha((c.ROOT/row['local_import']).read_bytes())==row['local_import_SHA256']
    for path in list((c.OUT/'cpu_preflight').glob('*/raw_export.zip'))+list((c.OUT/'profiles').glob('*/*/raw_export.zip')):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist(): assert (path.parent/'export'/name).read_bytes()==z.read(name)
    if (c.OUT/'freeze.json').exists():
        for name,digest in c.read(c.OUT/'freeze.json')['files'].items():
            assert c.sha((c.ROOT/name).read_bytes())==digest,name
    if committed:
        assert not subprocess.check_output(['git','diff','HEAD','--',str(c.SOURCE.relative_to(c.ROOT)),str(c.OUT.relative_to(c.ROOT))],cwd=c.ROOT)
    result={'stage':'C2d','status':'OK','programs':76,'FULL_candidates':61,'missing_FULL_candidates':0,
        'new_compiles':31,'reused_exact_binaries':45,'eligibility_checks':g['checks'],
        'event_samples':samples,'timing_workers':3,'physical_GPU_UUIDs':sorted(set(uuids)),
        'NCU_reports':14,'NCU_reports_independently_reimported':14,
        'prior_files_hash_checked':len(c.read(c.OUT/'preregistration/prior_manifest.json')['files']),
        'native_rebuilds':0,'protected_D1':10,'D1_compiled':False,'D1_timed':False,
        'historical_26_6_used':False,'runtime_grid_selector_feature':False}
    c.derived(c.OUT/'validation.json',result)
    print('C2d validator OK; 61/61 FULL;',samples,'events; 14 NCU reports; D1 untouched')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--committed',action='store_true')
    run(parser.parse_args().committed)
