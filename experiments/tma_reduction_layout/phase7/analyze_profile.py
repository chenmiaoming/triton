"""Descriptive counters from immutable Nsight CSV; no formal timing inference."""
import copy
import csv
import io
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c
from experiments.tma_reduction_layout.phase7.validate_profile import validate

ROOT=c.OUT/'stage_c/profiling'


def derive():
    valid=validate();status=c.read(ROOT/'status.json');rows={}
    for command in status['commands']:
        if command['return_code']!=0:continue
        output=command['output'];offset=output.index('"ID","Process ID"')
        table=list(csv.reader(io.StringIO(output[offset:])))
        c.require(len(table)==3 and table[2][0]=='0','One profiled kernel in raw CSV')
        header,units,values=table;raw=dict(zip(header,values));unit=dict(zip(header,units))
        metrics={name:{'value':float(raw[name].replace(',','')),'unit':unit[name]} for name in status['settings']['metrics']}
        c.require(all(math.isfinite(v['value']) and v['value']>0 for v in metrics.values()),'Finite positive actual counters')
        rows[command['key']]={'archive_sha256':command['archive_sha256'],'kernel_name':raw['Kernel Name'],
            'metrics':metrics,'registers_per_thread':raw['launch__registers_per_thread'],
            'profiled_grid':raw['Grid Size'],'profiled_block':raw['Block Size'],
            'replay_passes':raw['profiler__replayer_passes']}
    contrasts={}
    for candidate in ('default','4'):
        metrics={h:rows[f'M128_N32_w16:{h}:{candidate}']['metrics'] for h in c.HARNESSES if f'M128_N32_w16:{h}:{candidate}' in rows}
        if set(metrics)!=set(c.HARNESSES):continue
        contrasts[candidate]={}
        for name in status['settings']['metrics']:
            a,b,d,e=[metrics[h][name]['value'] for h in ('host_native','host_canonical','device_native','device_canonical')]
            contrasts[candidate][name]={'unit':metrics['host_native'][name]['unit'],
                'store_difference':b-a,'descriptor_difference':d-a,'interaction':e-d-b+a,
                'canonical_minus_device_canonical':metrics['canonical'][name]['value']-e}
    return {'counter_validation':valid,'case':'M128_N32_w16','role':'SINGLE_CASE_SINGLE_PROFILE_PER_BINARY_DESCRIPTIVE_ONLY',
        'settings':status['settings'],'rows':rows,'within_candidate_source_contrasts':contrasts,
        'formal_event_durations_or_fits_modified':False,
        'limits':['Ten independent executable launches in one separate H100 profiler worker; one report per binary, no variance estimate.',
                  'Kernel replay/cache flush changes context; clocks uncontrolled; profiler duration is diagnostic only.',
                  'Instruction/DRAM counters cover the entire kernel and compiler responses, not pure descriptor/store costs.',
                  'The chosen old case is outcome-informed; counters are not a held-out confirmatory test.']}


def report(r):
    rows=[]
    for key,row in sorted(r['rows'].items()):
        m=row['metrics'];rows.append([key,row['registers_per_thread'],m['smsp__inst_executed.sum']['value'],m['dram__bytes_read.sum']['value'],m['dram__bytes_write.sum']['value'],m['gpu__time_duration.sum']['value'],row['replay_passes']])
    return '\n'.join(['# Separate StageC Nsight diagnostic counters','',r['role'],'',
        c.table(['Binary','Registers/thread','Executed instructions','DRAM read bytes','DRAM write bytes','Replay duration ns (diagnostic)','Passes'],rows),'',
        'All commands, settings, original CSV text and binary .ncu-rep reports remain frozen. Source contrasts are retained in counter_results.json.','',*['- '+s for s in r['limits']],''])


def main():
    r=derive()
    if '--validate' in sys.argv:
        c.require(c.read(ROOT/'counter_results.json')==r,'Independent raw-counter closure')
        c.require((ROOT/'counter_summary.md').read_text()==report(r),'Counter report closure')
        bad=copy.deepcopy(r);next(iter(bad['rows'].values()))['metrics']['dram__bytes_read.sum']['value']=1
        c.require(bad!=r,'Counter tamper rejected');print('Counter analysis PASS',len(r['rows']))
    else:c.write(ROOT/'counter_results.json',r);(ROOT/'counter_summary.md').write_text(report(r));print('Derived descriptive counters')


if __name__=='__main__':main()
