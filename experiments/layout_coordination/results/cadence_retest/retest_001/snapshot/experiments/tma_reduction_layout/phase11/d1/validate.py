"""Final CPU validator: bindings, raw-derived recomputation, then write-once freeze."""
import argparse
from collections import Counter
import io
import math
import statistics as st
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.d1 import common as c
from experiments.tma_reduction_layout.phase11.d1.runtime import cells_for, reference
from experiments.tma_reduction_layout.phase11.d1 import analyze, evaluate, report, counter_summary
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def same(a,b):assert math.isclose(a,b,rel_tol=2e-11,abs_tol=1e-8),(a,b)


def run(freeze=False,committed=False):
    p=c.protocol();c.protect_prior();c.verify_sources(p)
    assert p['pinned_upstream']==c.PIN and p['profile']=='chenmiaoming'
    assert not p['D1_compiled_at_freeze'] and not p['D1_timed_at_freeze']
    assert p['models']['C34_resource_only']['role']=='primary_outer_CV_selected'
    assert p['models']['C34_full_auto']['role']=='automated_inner_CV_comparator'
    assert len(p['policies'])==11 and all(len(v)==10 for v in p['policies'].values())
    assert Counter(v['stratum'] for v in p['D1_cases'])=={'empirically_unseen_core':5,'legacy_boundary':5}
    assert all(v['history_category']=='strict_zero_mention' for v in p['D1_cases'])
    assert c.sha((c.SOURCE/'PLAN.md').read_bytes())==p['plan_SHA256']
    assert c.sha((c.OUT/'preregistration/prior_manifest.json').read_bytes())==p['prior_manifest_SHA256']
    for label,model in p['models'].items():
        source=c.ROOT/model['original_path'];bound=c.OUT/'preregistration'/model['bound_copy']
        assert c.sha(source.read_bytes())==c.sha(bound.read_bytes())==model['SHA256']
        rows=c.read(bound)['cases']
        for suffix,key in (('conservative','selected'),('nominal','nominal_choice')):
            assert p['policies'][label+'_'+suffix]==[{'geometry':r['geometry'],'layout':r['prediction'][key]} for r in rows]
    with zipfile.ZipFile(c.OUT/'preregistration/source_snapshot.zip') as z:
        for name,digest in p['worker_source_SHA256'].items():assert c.sha(z.read(name))==digest
    cg=c.read(c.OUT/'compile_gate.json');gate=c.read(c.OUT/'artifact_gate.json')
    assert len(gate['metadata'])==69 and cg['new_compiles']==69 and cg['reused']==0
    expected={r['id'] for definition in p['D1_cases'] for r in candidates(*definition['geometry'])}
    assert len(expected)==59 and {m['layout_id'] for m in gate['metadata'].values() if m['family']=='ONE_FULL'}==expected
    compiler=c.read(c.ROOT/cg['artifact_directory']/'result.json')
    assert compiler['status']=='OK' and not compiler['GPU_allocated']
    assert compiler['native_SHA256']==p['baseline']['native_SHA256']
    assert compiler['ccache_before']==compiler['ccache_after']
    assert compiler['protocol_SHA256']==c.sha(c.encode(p))
    with zipfile.ZipFile(c.OUT/'compiled_payload.zip') as z:
        for tag,meta in gate['metadata'].items():
            folder=c.ROOT/cg['artifact_directory']/tag.replace(':','/')
            assert meta['path']=='host' and meta['one_pass'] and meta['ABI_kind']=='descriptor'
            assert meta['global_rows']==24576
            assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256']==gate['resources'][tag]['loaded_CUBIN_SHA256']
            assert cg['static_audit'][tag]['physical_registers_from_cuobjdump']==[gate['resources'][tag]['attributes']['num_regs']]
            for path in folder.glob('kernel.*'):assert z.read(path.relative_to(c.ROOT/cg['artifact_directory']).as_posix())==path.read_bytes()
            ir=(folder/'kernel.glir').read_text()
            assert 'scf.for' not in ir and 'scf.while' not in ir
    eligibility=c.read(c.ROOT/gate['eligibility_path']);assert eligibility['status']=='OK'
    assert len(eligibility['checks'])==207 and not eligibility['Triton_imported']
    assert eligibility['driver_return_codes']==[0]
    reference_cache={}
    def check_oracle(check,meta):
        key=(meta['M'],meta['N'],meta['family'],check['pattern'],tuple(check['pids']))
        if key not in reference_cache:
            reference_cache[key]=c.sha(reference(meta,1,check['pids'],check['pattern']).astype('<f4').tobytes())
        assert check['pass'] and check['bit_exact'] and check['actual_SHA256']==check['reference_SHA256']==reference_cache[key]
    for check in eligibility['checks']:check_oracle(check,gate['metadata'][check['tag']])
    cells=cells_for(p,gate);sources=[];events=0;grids={};uuids=[]
    for invocation in (1,2,3):
        path=c.OUT/'timing'/f'invocation_{invocation}/raw.json';raw=c.read(path)
        assert raw['status']=='OK' and raw['driver_return_codes']==[0]
        assert not raw['Triton_imported'] and not raw['profiler_attached']
        assert raw['protocol_SHA256']==c.sha(c.encode(p))
        assert raw['payload_SHA256']==c.sha((c.OUT/'compiled_payload.zip').read_bytes())
        assert raw['environment']['cc']==[9,0] and raw['environment']['SM_count']==132
        uuids.append(raw['environment']['gpu_uuid'])
        assert len(raw['cells'])==len(cells)
        expected_cells={v['cell_id']:v for v in cells}
        for row in raw['cells']:
            expected_cell=expected_cells[row['cell_id']]
            for key in ('tag','q','grid','K','aliases'):assert row[key]==expected_cell[key]
            assert len(row['samples_us'])==12 and len(row['visits'])==3
            assert all(v>0 for v in row['samples_us'])
            assert [v for visit in row['visits'] for v in visit['samples_us']]==row['samples_us']
            assert all(sorted(v['node_types'])==[0,7,7] and len(v['samples_us'])==4 for v in row['visits'])
            check_oracle(row['correctness'],gate['metadata'][row['tag']])
            events+=12
            for identity in [row['cell_id']]+row['aliases']:
                tag,q,g,k=identity.split('|');assert k=='K1'
                grids[tag,q[1:],invocation,int(g[1:])]=st.median(row['samples_us'])
        for tag,binding in raw['bindings'].items():
            assert binding['CUBIN_SHA256']==binding['loaded_CUBIN_SHA256']==gate['metadata'][tag]['cubin_SHA256']
            assert binding['guards']>=binding['graph_launches']
        sources.append({'path':str(path.relative_to(c.ROOT)),'SHA256':c.sha(path.read_bytes())})
    assert len(set(uuids))==3, 'Three fresh GPU workers required'
    for row in c.read(c.OUT/'analysis.json')['oneshot']:
        full=row['full_grid'];i=row['invocation'];tag=row['tag'];q=row['q_condition']
        ys=[grids[tag,q,i,full*f] for f in (1,2,3)]
        same(row['full_us'],ys[0])
        # Independent closed-form OLS slope for equally spaced three-grid points.
        slope=(ys[2]-ys[0])/(2*full/132)*1000
        same(row['tile_slope_ns'],slope)
    assert events==c.read(c.OUT/'analysis.json')['event_samples']
    # Recreate every derived table from retained raw values / immutable decisions.
    names=('analysis.json','truth.json','scorecards.json','coverage.json','counter_gate.json','counter_summary.json','REPORT.md')
    before={name:(c.OUT/name).read_bytes() for name in names}
    analyze.services();analyze.counters();counter_summary.run();evaluate.run();report.run()
    assert all((c.OUT/name).read_bytes()==blob for name,blob in before.items()), 'Derived table is not reproducible'
    offline=c.read(c.OUT/'offline_import_validation.json');assert offline['status']=='OK'
    counters=c.read(c.OUT/'counter_gate.json')
    assert len(offline['reports'])==len(counters['reports'])==12
    assert len(counters['single_pass'])==69 and not counters['affine']
    assert {r['tag'] for r in counters['single_pass']}==set(gate['metadata'])
    for row in offline['reports']:
        record=c.read(c.ROOT/row['local_import']);reportpath=c.ROOT/row['report']
        assert record['report_SHA256']==c.sha(reportpath.read_bytes())
        assert c.sha((c.ROOT/row['local_import']).read_bytes())==row['local_import_SHA256']
        assert row['all_metrics_and_source_match']
    from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows,source_blocks
    for record in counters['reports']:
        assert c.sha((c.ROOT/record['path']).read_bytes())==record['SHA256']
    for row in offline['reports']:
        local=c.read(c.ROOT/row['local_import']);remote=(c.ROOT/row['report']).parent
        assert hardware_rows(local['commands'][0]['output'])==hardware_rows(c.read(remote/'import.json')['output'])
        if row['mode']=='software':assert source_blocks(local['commands'][1]['output'])==source_blocks(c.read(remote/'source.json')['output'])
    # Every dispatch is protocol-bound; raw archive bytes equal extracted files.
    dispatches=list(c.OUT.rglob('dispatch.json'))
    for path in dispatches:
        d=c.read(path);assert d['protocol_SHA256']==c.sha(c.encode(p)) and d['profile']=='chenmiaoming'
        folder=path.parent
        if (folder/'raw_export.zip').exists():
            with zipfile.ZipFile(folder/'raw_export.zip') as z:
                assert all(z.read(n)==(folder/'export'/n).read_bytes() for n in z.namelist())
    if freeze and not (c.OUT/'freeze.json').exists():
        files={str(path.relative_to(c.ROOT)):c.sha(path.read_bytes())
               for root in (c.SOURCE,c.OUT) for path in sorted(root.rglob('*'))
               if path.is_file() and '__pycache__' not in path.parts and path.name not in ('freeze.json','validation.json')}
        c.write_once(c.OUT/'freeze.json',{'stage':'D1','files':files,
            'protocol_SHA256':c.sha(c.encode(p)),'no_next_stage':True})
    frozen=c.read(c.OUT/'freeze.json') if (c.OUT/'freeze.json').exists() else None
    if frozen:
        for name,digest in frozen['files'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    if committed:
        assert frozen
        tracked=set(subprocess.check_output(['git','ls-files'],cwd=c.ROOT,text=True).splitlines())
        assert set(frozen['files'])<=tracked
        subprocess.run(['git','diff','--exit-code','HEAD','--',str(c.SOURCE),str(c.OUT)],cwd=c.ROOT,check=True)
    result={'status':'PASS','stage':'D1','CPU_only_validator':True,
        'protocol_SHA256':c.sha(c.encode(p)),'FULL_candidates':59,'controls':10,
        'correctness_checks':207,'independent_GPU_workers':3,'timing_cells_per_worker':len(cells),
        'event_samples':events,'NCU_reports':12,'NCU_single_pass_instruction_launches':69,
        'prior_SHA_preserved_files':len(c.read(c.OUT/'preregistration/prior_manifest.json')['files']),
        'frozen_files':len(frozen['files']) if frozen else 0,'policy_count':11,'strata_sizes':[5,5,10],
        'no_refit':True,'no_old_evidence_changed':True,'next_stage_started':False,
        'raw_timing_sources':sources}
    c.derived(c.OUT/'validation.json',result)
    print('D1 VALIDATION PASS',len(cells),'cells/worker;',events,'events; frozen',result['frozen_files'],'files')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--committed',action='store_true')
    args=parser.parse_args();run(args.freeze,args.committed)
