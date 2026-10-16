"""Steady-K and multi-grid normalization; no per-layout fitted selector."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c2 import common as c
from experiments.tma_reduction_layout.phase11.c2.timing import cells_for
from experiments.tma_reduction_layout.phase11.c2.analyze import derived


def fit(xs,ys):
    xs,ys=np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)
    b,a=np.polyfit(xs,ys,1);res=ys-(a+b*xs)
    return {'slope':float(b),'intercept':float(a),'max_abs_residual':float(np.max(np.abs(res))),
            'adjacent_slopes':[float(v) for v in np.diff(ys)/np.diff(xs)],'xs':xs.tolist(),'ys':ys.tolist()}


def analyze(batch):
    protocol=c.load_protocol(batch);gate=c.read(c.out(batch)/'artifact_gate.json')
    counter=c.read(c.out(batch)/'counter_gate.json');assert counter['status']=='OK'
    cells=cells_for(protocol,gate)
    expected={x['cell_id'] for x in cells};workers=[];curves=[];summaries=[];anchors=[]
    aliases={cell['cell_id']:cell['aliases'] for cell in cells}
    for invocation in range(1,4):
        path=c.out(batch)/'timing'/f'invocation_{invocation}'/'raw.json';raw=c.read(path)
        assert raw['status']=='OK' and raw['driver_return_codes']==[0] and not raw['Triton_imported']
        assert raw['payload_SHA256']==c.sha((c.out(batch)/'frozen_payload.zip').read_bytes())
        assert {x['cell_id'] for x in raw['cells']}==expected
        workers.append({'invocation':invocation,'GPU_UUID':raw['environment']['gpu_uuid'],'cell_count':len(raw['cells']),
                        'samples':sum(len(x['samples_us']) for x in raw['cells'])})
        grouped=defaultdict(dict)
        for cell in raw['cells']:
            assert len(cell['samples_us'])==20 and len(cell['visits'])==4 and cell['correctness']['pass']
            assert all(sorted(v['node_types'])==[0,7,7] and len(v['samples_us'])==5 for v in cell['visits'])
            meta=gate['metadata'][cell['tag']]
            median=st.median(cell['samples_us'])
            if meta['family']=='TMA':
                anchors.append({'invocation':invocation,'tag':cell['tag'],'N':meta['N'],'median_us':median,
                                'grid':cell['grid'],'HBM_tile_SM_us':median*132/cell['grid']})
                continue
            for identity in [cell['cell_id']]+cell['aliases']:
                tag=identity.split('|')[0]
                grouped[tag,cell['q'],cell['grid']][cell['K']]=median
        slopes=defaultdict(list)
        for (tag,q,grid),byk in sorted(grouped.items()):
            meta=gate['metadata'][tag]
            if meta['family']=='MO':
                slopes[tag,q].append((grid,byk[1]));continue
            assert {16,64,256}.issubset(byk),(tag,q,grid,byk)
            curve={'invocation':invocation,'tag':tag,'q':q,'grid':grid,
                   'steady':fit([16,64,256],[byk[k] for k in (16,64,256)]),'K1_us':byk.get(1)}
            if 1 in byk:curve['K1_minus_steady_slope_us']=byk[1]-curve['steady']['slope']
            curves.append(curve);slopes[tag,q].append((grid,curve['steady']['slope']))
        for (tag,q),points in sorted(slopes.items()):
            meta=gate['metadata'][tag];bulk=sorted((g,b) for g,b in points if g>1)
            assert len(bulk)==3,(tag,q,bulk)
            fitted=fit([g/132 for g,b in bulk],[b for g,b in bulk]);d=meta['D']
            service=fitted['slope']/d
            summaries.append({'invocation':invocation,'tag':tag,'q':q,'D':d,'family':meta['family'],
                'M':meta['M'],'N':meta['N'],'service_us_per_tile_SM':service,
                'service_ns_per_tile_SM':1000*service,'grid_fit':fitted,
                'grid1_effective_chain_us':next((b for g,b in points if g==1),None),
                'normalization':'T_grid slope for MO; dT/dK grid slope for repeated subgraphs; divided by D',
                'predictor_features':{key:meta[key] for key in protocol['analysis']['selector_features']}})
    if batch=='b':
        prior=c.read(c.out('a')/'analysis.json')
        prior_uuids={w['invocation']:w['GPU_UUID'] for w in prior['workers']}
        current_uuids={w['invocation']:w['GPU_UUID'] for w in workers}
        summaries=[{**r,'cohort':'C2a','GPU_UUID':prior_uuids[r['invocation']]} for r in prior['services']]+[
            {**r,'cohort':'C2b','GPU_UUID':current_uuids[r['invocation']]} for r in summaries]
        curves=[{**r,'cohort':'C2a','GPU_UUID':prior_uuids[r['invocation']]} for r in prior['steady_K_curves']]+[
            {**r,'cohort':'C2b','GPU_UUID':current_uuids[r['invocation']]} for r in curves]
        anchors=prior['HBM_anchors']
    by=defaultdict(list)
    for row in summaries:by[row['tag'],row['q']].append(row['service_ns_per_tile_SM'])
    aggregate=[{'tag':tag,'q':q,'median_ns_per_tile_SM':st.median(v),'worker_min_ns':min(v),'worker_max_ns':max(v)} for (tag,q),v in sorted(by.items())]
    grouped_curves=defaultdict(list)
    for row in curves:
        if row['grid']>1:grouped_curves[row['invocation'],row['tag'],row['q']].append(row)
    setup=[]
    for (invocation,tag,q),rows in sorted(grouped_curves.items()):
        rows=sorted(rows,key=lambda r:r['grid']);assert len(rows)==3
        xs=[r['grid']/132 for r in rows]
        intercept=fit(xs,[r['steady']['intercept'] for r in rows])
        single=fit(xs,[r['K1_us'] for r in rows]) if all(r['K1_us'] is not None for r in rows) else None
        setup.append({'invocation':invocation,'tag':tag,'q':q,'setup_grid_fit':intercept,
                      'setup_ns_per_CTA_SM':intercept['slope']*1000,
                      'K1_grid_fit':single,'K1_ns_per_CTA_SM':single['slope']*1000 if single else None,
                      'interpretation':'Measured fixture/setup service, including transfer and final output; not pure TMA latency. MC K1 uses one transfer, steady-K uses two.'})
    combos=[]
    for layout in sorted({m.get('layout_id') for m in gate['metadata'].values()}-{None}):
        for invocation in (1,2,3):
            series={r['tag']:r for r in summaries if r['invocation']==invocation and r['q']=='natural'}
            tags={f:f'{layout}:{f}:D1' for f in ('ML','MR','MO','MC')}
            if not all(t in series for t in tags.values()):continue
            values={f:series[t]['service_ns_per_tile_SM'] for f,t in tags.items()}
            total=sum(values[f] for f in ('ML','MR','MO'));n=gate['metadata'][tags['MC']]['N']
            hbm=st.median(r['HBM_tile_SM_us'] for r in anchors if r['N']==n)*1000
            combos.append({'layout_id':layout,'invocation':invocation,'service_ns_per_tile_SM':values,
                           'ML_MR_MO_sum_ns':total,'MC_minus_sum_ns':values['MC']-total,
                           'HBM_tile_SM_ns':hbm,'rho_MC_service_proxy':values['MC']/hbm,
                           'rho_is_full_kernel_cost':False,
                           'residual_identifies_pure_overlap':False,
                           'reason':'ML checksum and MR private BF16 update/accumulation differ from MC; MO has input load; MC stores per iteration and double-buffer resources'})
    value={'status':'OK','workers':workers,'steady_K_curves':curves,'services':summaries,'aggregate':aggregate,
           'HBM_anchors':anchors,'combination_checks':combos,'empirical_offset_26_6_used':False,
           'timing_samples_new':sum(r['samples'] for r in workers),'cells_per_worker':len(cells),
           'selector_feature_schema':protocol['analysis']['selector_features'],
           'units':'ns per tile per SM steady service interval; not single-CTA naked hardware latency',
           'three_workers_not_three_independent_devices':True,
           'fit_quality':{'max_steady_K_residual_percent':max(100*r['steady']['max_abs_residual']/max(r['steady']['ys']) for r in curves),
                          'max_grid_residual_percent':max(100*r['grid_fit']['max_abs_residual']/max(r['grid_fit']['ys']) for r in summaries)},
           'identifiability':'Subgraph combinations measured; fixture residual is retained. No claim of independently identified pure load/max/convert latencies.'}
    if batch=='b':
        value['setup_and_K1_services']=setup
        value['reused_C2a_workers']=prior['workers']
        value['q_to_reused_natural_comparisons_unpaired']=True
        value['setup_interpretation_checks']={'negative_setup_fits_retained':sum(r['setup_ns_per_CTA_SM']<0 for r in setup),
            'K1_max_grid_residual_percent':max(100*r['K1_grid_fit']['max_abs_residual']/max(r['K1_grid_fit']['ys']) for r in setup if r['K1_grid_fit']),
            'not_identified_as_positive_primitive_latency':True}
    derived(c.out(batch)/'analysis.json',value)
    write_report(batch,protocol,gate,counter,value)
    return value


def write_report(batch,protocol,gate,counter,value):
    lines=[f'# Stage C2{batch} — H100 calibration', '',
        f'90 core programs plus {40 if batch=="b" else 0} parameter programs; {protocol["new_programs"]} new logical compilations in this batch.',
        f'New graph event samples: {value["timing_samples_new"]:,}; {value["cells_per_worker"]} physical cells per worker; three worker allocations.',
        f'New NCU reports: {len(counter["new_reports"])}; C1/C1.5 original 128 reports and CUBINs remain unchanged.',
        f'Resource boundaries excluded from regular timing/fit: {", ".join(gate["spill_boundaries"]) or "none"}. Correctness and artifacts are retained for these programs.',
        '', '## Measurement and units', '',
        'Repeated subgraphs use K16/64/256 OLS, with K1 retained separately. Three grids fit steady service per SM per tile; grid is calibration provenance, absent from static predictor features. MO is one-shot and uses T versus grid. All samples are single-kernel CUDA Graph event intervals. NCU duration is not used.',
        'Independent unchanged TMA controls run at full grids. Their time is normalized in the same tile/SM units. No 26.6 us offset appears in the analysis.',
        'The preferred shared carveout is set and read back as 100% for controlled q comparisons; these fresh curves have their own measurement conditions. Old timing samples are not pooled into them.',
        f'Maximum steady-K fit residual: {value["fit_quality"]["max_steady_K_residual_percent"]:.2f}% of the largest measured point. Maximum grid-fit residual: {value["fit_quality"]["max_grid_residual_percent"]:.2f}%. Adjacent slopes and all worker estimates remain in analysis.json.',
        '', '## Natural service curves', '',
        '| Program | Median ns/tile/SM | Worker range ns |', '|---|---:|---:|']
    for row in value['aggregate']:
        if row['q']=='natural':lines.append(f'| {row["tag"]} | {row["median_ns_per_tile_SM"]:.3f} | {row["worker_min_ns"]:.3f}–{row["worker_max_ns"]:.3f} |')
    lines += ['', '## Counter and fixture boundaries', '',
        'Dynamic software counts and hardware shared-load wavefront/bank-conflict measurements are exported by tag in counter_gate.json. K-invariant setup is separated from K-dependent body counts. Within-warp MR tests zero loop shared-load wavefronts, not zero total initial loads.',
        'MO includes an input load. A compiler-elided conversion does not identify standalone output-conversion latency. MC has no private recurrence or checksum, but repeated output writes and two-buffer resources differ from the original kernel.',
        'Therefore MC − (ML + MR + MO) is a measured composition/fixture residual, not pure overlap. The MC/HBM ratio is a service proxy, not a complete kernel prediction. C3 cannot silently turn this proxy or the residual into an unisolated empirical constant.',
        'One-shot MO includes per-CTA setup and input-load work that repeated MC amortizes across K. Consequently its grid slope cannot be added as though it were an isolated repeated conversion cost.',
        '', 'No compiler change, model selector, C3, architecture transfer or PR was started. Raw evidence, technical failures and source snapshots are preserved.']
    if batch=='a':
        lines += ['', '## Placement intervention: measured D1 counters', '',
            '| Geometry | vec | Placement | ML load WF/K | ML load bank conflicts/K | MR load WF/K | MR BAR.SYNC warp inst/K |',
            '|---|---:|---|---:|---:|---:|---:|']
        ld='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum';bc='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'
        for n,w in ((64,8),(128,4)):
            for vec in (1,2,4,8):
                for placement in ('n_first','within'):
                    tag=f'M32_N{n}_w{w}:vec{vec}:{placement}'
                    if tag+':MR:D1' not in counter['binaries']:continue
                    ml,mr=[counter['binaries'][tag+':'+f+':D1'] for f in ('ML','MR')]
                    a,b=ml['hardware_delta_per_iteration'],mr['hardware_delta_per_iteration']
                    lines.append(f'| 32×{n}/w{w} | {vec} | {placement} | {a[ld]:g} | {a[bc]:g} | {b[ld]:g} | {mr["families"]["barrier"]["loop_per_iteration_warp_instructions"]:g} |')
        lines += ['', '## Preserved technical failures', '',
            'The first CPU preflight rejected a nonconstant mbarrier byte argument. A uniform K branch with constant byte counts repaired the API use.',
            'Initial MR D1 for 32×64/vec8/within failed 12 checks, affecting columns 1 modulo 8. Its PTX packed step construction read an undefined half-register. The mathematically identical step was factored as (13*m)%7 before lowering. Only this one logical program was recompiled; the other 89 core binaries stayed fixed. The repaired binary passed all 12 checks; all 807 total checks pass. Failed binaries, exact mismatch values, source snapshots and the unsupported broadcast helper attempt remain archived.',
            '', '## Review artifacts', '', '[Core service figure](figures/core_services.png), [tidy service table](figures/services.csv), [full derived data](analysis.json), [dynamic counters](counter_gate.json).']
    else:
        lines += ['', '## Parameter probes and q residency controls', '',
            'Unit convention: the service quantity is s = (1/D) d(dT/dK)/d(G/S), where S is the SM count. It is nanoseconds for one tile serviced on one SM. MO uses dT/d(G/S) because it is one-shot. Grid belongs to calibration provenance, not the selector feature schema.',
            f'All 40 new logical probes passed 480 new complete-output checks; 807 C2a checks were reused. Two U1 binaries alias P1 exactly, leaving 38 new physical binaries. {len(counter["q_dynamic_opcode_checks"])} q interventions preserve every target opcode count; {len(counter["q_achieved_occupancy_checks"])} achieved-occupancy observations are within their CUDA capacity bounds. NCU launch IDs, function names, grid/block sizes and each measured source PC opcode were matched to the frozen SASS.',
            'Core natural curves and TMA controls are reused from C2a. Their comparisons to C2b q curves are unpaired across cohorts. The new parameter probes have natural and q measurements in the same C2b workers. Cohort and GPU UUID are retained on every service/curve row.',
            '', '| Probe | Natural D1 ns/tile/SM | Natural D2 ns/tile/SM (per stream) |', '|---|---:|---:|']
        table={(r['tag'],r['q']):r['median_ns_per_tile_SM'] for r in value['aggregate']}
        for family,values in [('P',[1,2,4,8,16,32,64]),('U',[1,2,4,8,16,32]),('W',[1,2,4,8]),('E',[2,8,32])]:
            for parameter in values:
                a,b=[table[f'probe_{family}{parameter}:D{d}','natural'] for d in (1,2)]
                lines.append(f'| {family}{parameter} | {a:.3f} | {b:.3f} |')
        lines += ['', '| D1 program | Natural ns | q1 ns | q2 ns | q4 ns |', '|---|---:|---:|---:|---:|']
        for tag in ['probe_P1:D1','probe_P64:D1','probe_U32:D1','probe_W8:D1','probe_E32:D1',
                    'M32_N64_w8:vec8:n_first:MC:D1','M32_N64_w8:vec8:within:MC:D1']:
            numbers=[]
            for q in ('natural','1','2','4'):
                state=gate['resources'][tag]['reservations'][q]
                numbers.append(f'{table[tag,q]:.3f}' if (tag,q) in table else 'alias natural' if state.get('alias_of') else 'infeasible')
            lines.append('| '+tag+' | '+' | '.join(numbers)+' |')
        lines += ['', '## Aggregate-counter diagnostic', '',
            f'An extra, non-preregistered diagnostic assumed raw shared wavefront/bank-conflict counters would remain identical under q. This hypothesis failed in {counter["aggregate_counter_diagnostic_rejections"]} of {len(counter["q_wavefront_bank_conflict_invariance"])} contrasts. Every contrast preserves the arithmetic quantity wavefronts minus reported conflicts. This difference is not a separately collected request or ideal-wavefront metric.',
            'The official [NVIDIA Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant) explains that aggregate bank counters can include shared/L1 cross-client contention. Our observations are consistent with this scope; the experiment does not causally isolate all extra events. The largest relative raw-wavefront difference is about 8.57%, in a one-shot MO load case. No affected q point was deleted, normalized away, or rerun.',
            'The initial failed extra check is archived in audits/aggregate_invariance_initial_gate.zip, with a counter-scope audit. The final gate follows the preregistered target-instruction invariance and measured occupancy requirements and retains all hardware-counter differences.',
            '', '## Setup and composition identification', '',
            'analysis.json additionally fits the steady-K intercept versus grid in per-CTA/per-SM units and fits K1 versus grid separately. These are measured fixture/setup combinations, including transfer and final-output work. MC K1 uses one transfer while steady K uses two. They are not universal TMA constants.',
            f'The {len(value["setup_and_K1_services"])} setup fits include {value["setup_interpretation_checks"]["negative_setup_fits_retained"]} negative estimate, retained without clamping. K1 grid fits have up to {value["setup_interpretation_checks"]["K1_max_grid_residual_percent"]:.2f}% residual. These bounds prevent interpreting fitted intercepts as independently identified positive hardware latencies.',
            'For 32×64/vec8 D1, the median MC K1 service is about 463.25 ns/tile/SM for N-first versus 363.50 ns for within, whereas steady-body service is 377.21 versus 227.86 ns. The steady-body improvement is therefore not the whole single-pass improvement. K1 is measured over the registered cropped grids, with the recorded cache/resource differences; it is not a new full-grid original-kernel benchmark.',
            'The two retained local-memory boundaries belong to the ML D2 checksum fixture; they do not establish spilling in the corresponding original complete kernel.',
            'This stage identifies effective p/u/w/e and residency-dependent subgraph costs and composition residuals. It does not uniquely separate every naked load/max/convert latency. In particular, one-shot MO preparation and repeated-MC output traffic differ. No residual is silently substituted for 26.6 us, and no static selector was trained or evaluated.',
            '', '## Review artifacts', '', '[Parameter curves](figures/parameters.png), [all service curves](figures/services.csv), [full derived calibration](analysis.json), [counter and q evidence](counter_gate.json).',
            '', 'Final offline checks: `.venv/bin/python -m experiments.tma_reduction_layout.phase11.c2.validate --batch a --committed` and the same command with `--batch b --committed`.']
    (c.out(batch)/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['a','b'],required=True)
    args=p.parse_args();c.protect_prior(args.batch);value=analyze(args.batch)
    print('C2'+args.batch,'analysis OK',value['timing_samples_new'],'samples')


if __name__=='__main__':main()
