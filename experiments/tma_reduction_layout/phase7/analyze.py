"""Uncalibrated factorial gaps and preregistered independent-cohort tests."""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c, contracts as ct, timing_contract as tc
from experiments.tma_reduction_layout.phase6.analyze import decide

NAMES = {'host_native': 'H00', 'host_canonical': 'H01', 'device_native': 'H10', 'device_canonical': 'H11'}


def errors(rows):
    return {p: {'MAE': st.mean(abs(r[p]['mean']-r['G']['mean']) for r in rows),
                'RMSE': math.sqrt(st.mean((r[p]['mean']-r['G']['mean'])**2 for r in rows))}
            for p in (*NAMES.values(), 'zero')} if rows else {}


def derive(stage):
    tc.validate_raw(stage)
    cases, binaries, plan = tc.inputs(stage)
    root = c.OUT/stage
    fits, statistics, metrics = {}, {}, {}
    for i in (1, 2, 3):
        raw = c.read(root/f'raw_invocation_{i}.json')
        statistics[str(i)], fits[str(i)] = c.raw_fits(raw)
        for cfg in cases:
            def gap(h):
                return (fits[str(i)][f'{cfg}:{h}:default:RNone']['slope_ns_per_additional_CTA']-
                        fits[str(i)][f'{cfg}:{h}:4:RNone']['slope_ns_per_additional_CTA'])
            value = {'G': gap('canonical'), **{name: gap(h) for h, name in NAMES.items()}, 'zero': 0.0}
            a,b,d,e = [value[k] for k in ('H00','H01','H10','H11')]
            value.update(store_effect=b-a, descriptor_effect=d-a, interaction=e-d-b+a,
                         store_effect_with_device=e-d, descriptor_effect_with_canonical_store=e-b)
            value.update({f'G_minus_{p}': value['G']-value[p] for p in NAMES.values()})
            metrics.setdefault(cfg, []).append(value)
    rows = {}
    for cfg, case in cases.items():
        row = {**case, 'invocation_metrics': metrics[cfg], **{name: c.sign([v[name] for v in metrics[cfg]]) for name in metrics[cfg][0]}}
        row['prediction_errors'] = {p: {'prediction': row[p]['mean'],
            'error_prediction_minus_G': row[p]['mean']-row['G']['mean'],
            'absolute_error': abs(row[p]['mean']-row['G']['mean']),
            'squared_error': (row[p]['mean']-row['G']['mean'])**2} for p in (*NAMES.values(), 'zero')}
        row['resource_context'] = {key.split(':')[1]+':'+key.split(':')[2]: {
            'num_regs': b['metadata']['resources']['num_regs'], 'dynamic_smem_bytes': b['metadata']['dynamic_smem_bytes'],
            'blocks_per_sm': b['occupancy']['blocks_per_sm_actual_dynamic_smem']} for key,b in binaries.items() if key.split(':')[0]==cfg}
        row['cross_cell_residency_matched'] = len({v['blocks_per_sm'] for v in row['resource_context'].values()})==1
        rows[cfg] = row
    populations = {'ALL_ELIGIBLE': list(rows.values()), 'PRIMARY': [r for r in rows.values() if r['final_class']=='PRIMARY'],
                   'SECONDARY': [r for r in rows.values() if r['final_class']=='SECONDARY']}
    comparison = {name: {'n': len(pop), 'errors': errors(pop)} for name,pop in populations.items()}
    protocol = c.read(ct.DEST/'protocol.json')
    hypotheses = {}
    if stage=='stage_d':
        for name, comp in comparison.items():
            if name=='SECONDARY': continue
            e = comp['errors']; result = {'n': comp['n'], 'case_level_population': name}
            for h, definition in protocol['fresh_hypotheses'].items():
                ref,alt = NAMES[definition['reference']],NAMES[definition['alternative']]
                result[h] = decide(e[ref],e[alt],comp['n']) if e else 'INCONCLUSIVE_BY_COVERAGE'
            hypotheses[name] = result
    result = {'stage': stage, 'role': 'OUTCOME_INFORMED_INTERVENTION_DIAGNOSTIC' if stage=='stage_c' else 'PROSPECTIVE_FIXED_UNSEEN_COHORT',
        'units': 'ns/additional CTA; gaps between default and candidate4; not single-CTA/component latency',
        'condition_statistics': statistics, 'fits': fits, 'cases': rows, 'predictive_comparisons': comparison, 'hypotheses': hypotheses,
        'raw_manifest_SHA256': c.sha((root/'raw_manifest.json').read_bytes()), 'protocol_SHA256': c.sha((ct.DEST/'protocol.json').read_bytes()),
        'limits': ['Every admitted case retained; coefficient1/intercept0, no calibration or outcome-based repeats.',
                   'The four interventions are valid source-level Gluon programs; changing source also changes compiler scheduling/register allocation.',
                   'Full reduction opcode matching is not complete machine dataflow/scheduling equivalence.',
                   'Store/descriptor effects and interaction are differences in layout gaps; they are not pure component costs or causal shares.',
                   'Three processes may share physical GPU UUIDs; invocation sign bands do not prove practical equivalence.',
                   'Nsight counters, if available, are separate diagnostic evidence; replay duration never enters formal timing.',
                   'Held-out results apply only to admitted identities in the fixed large-M/warp domain; resource exclusions limit coverage.',
                   'H2b/H2c remain UNVERIFIED; no compiler/production heuristic change.']}
    return json.loads(c.encode(result))


def report(result):
    def val(r,k):
        v=r[k];return f'{v["mean"]:.12g} ± {v["sample_SD"]:.12g} ({v["category"]})'
    rows = result['cases']
    lines = ['# Phase7 '+result['stage']+' — Exact-binary factorial results','',result['role'],'',result['units'],'',
        c.table(['Case','Class','G','H00 host/native','H01 host/canonical','H10 device/native','H11 device/canonical'],
                [[cfg,r['final_class'],*[val(r,k) for k in ('G','H00','H01','H10','H11')]] for cfg,r in sorted(rows.items())]),'',
        c.table(['Case','Store gap effect','Descriptor gap effect','Interaction','G−H00','G−H01','G−H10','G−H11'],
                [[cfg,*[val(r,k) for k in ('store_effect','descriptor_effect','interaction','G_minus_H00','G_minus_H01','G_minus_H10','G_minus_H11')]] for cfg,r in sorted(rows.items())]),'',
        c.table(['Population','n','Predictor','MAE','RMSE'],[[name,v['n'],p,e['MAE'],e['RMSE']] for name,v in sorted(result['predictive_comparisons'].items()) for p,e in sorted(v['errors'].items())]),'']
    if result['hypotheses']:
        names=list(c.read(ct.DEST/'protocol.json')['fresh_hypotheses'])
        lines += [c.table(['Scope','n',*names],[[scope,h['n'],*[h[name] for name in names]] for scope,h in sorted(result['hypotheses'].items())]),'']
    lines += ['All intercepts, R², residuals, invocation values, resource contexts, errors and uncertainty bands remain in results.json.','']
    lines += ['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'


def validate(stored,stage):
    expected=derive(stage)
    c.require(stored==expected,'Every factorial/predictive field independently rederives from raw')
    for i in (1,2,3):
        raw=c.read(c.OUT/stage/f'raw_invocation_{i}.json'); samples={}
        for visit in raw['visits']:samples.setdefault(visit['condition_tag'],[]).extend(visit['samples_us'])
        for tag,values in samples.items():
            values=sorted(values); median=float((Fraction(values[49])+Fraction(values[50]))/2)
            c.require(median==stored['condition_statistics'][str(i)][tag]['median_us'],'Independent median order statistics')
        for key,fit in stored['fits'][str(i)].items():
            x=list(map(Fraction,c.B_VALUES));y=[Fraction(stored['condition_statistics'][str(i)][key+f':B{b}']['median_us']) for b in c.B_VALUES]
            slope=(3*sum(a*b for a,b in zip(x,y))-sum(x)*sum(y))/(3*sum(a*a for a in x)-sum(x)**2)
            c.require(math.isclose(float(slope*1000),fit['slope_ns_per_additional_CTA'],abs_tol=1e-12),'Independent OLS sum identity')
    c.require((c.OUT/stage/'summary.md').read_text()==report(stored),'Report closure')
    return expected


def main():
    stage=sys.argv[1] if len(sys.argv)>1 else 'stage_c';root=c.OUT/stage
    if '--validate' not in sys.argv:
        result=derive(stage);c.write(root/'results.json',result);(root/'summary.md').write_text(report(result));print('Derived',stage);return
    stored=c.read(root/'results.json');expected=validate(stored,stage);probes=[]
    for name,mutate in (('slope',lambda x:next(iter(x['fits']['1'].values())).update(slope_ns_per_additional_CTA=99)),
                        ('interaction',lambda x:next(iter(x['cases'].values()))['interaction'].update(mean=99)),
                        ('case',lambda x:x['cases'].pop(next(iter(x['cases'])))),
                        ('metric',lambda x:x['predictive_comparisons']['ALL_ELIGIBLE']['errors']['H11'].update(MAE=99))):
        bad=copy.deepcopy(stored);mutate(bad);c.require(bad!=expected,'Analysis tamper rejected');probes.append(name)
    for label,ref,alt,n,outcome in (
        ('support',{'MAE':2.,'RMSE':3.},{'MAE':1.,'RMSE':2.},5,'SUPPORTED'),
        ('equal',{'MAE':2.,'RMSE':3.},{'MAE':2.,'RMSE':3.},5,'FALSIFIED'),
        ('mixed',{'MAE':2.,'RMSE':3.},{'MAE':1.,'RMSE':4.},5,'INCONCLUSIVE'),
        ('coverage',{'MAE':2.,'RMSE':3.},{'MAE':0.,'RMSE':0.},4,'INCONCLUSIVE_BY_COVERAGE')):
        c.require(decide(ref,alt,n)==outcome,'Hypothesis branch '+label);probes.append(label)
    c.write(root/'analysis_validation.json',{'status':'PASS','stage':stage,'raw_recomputed_cases':len(stored['cases']),'corruption_and_branch_probes':probes,'raw_bytes_unchanged':True})
    print('PASS',stage,stored['hypotheses'])


if __name__=='__main__':main()
