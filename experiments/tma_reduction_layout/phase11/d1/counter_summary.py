"""Normalize retained FULL/control counters per CTA, without changing the model."""
from experiments.tma_reduction_layout.phase11.d1 import common as c

METRICS={
    'shared_load_wavefronts':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'shared_store_wavefronts':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'shared_load_bank_conflicts':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'shared_store_bank_conflicts':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'warp_instructions':'smsp__inst_executed.sum',
    'DRAM_read_bytes':'dram__bytes_read.sum',
    'DRAM_write_bytes':'dram__bytes_write.sum'}


def run():
    raw=c.read(c.OUT/'counter_gate.json');gate=c.read(c.OUT/'artifact_gate.json')
    dynamic={r['tag']:r for r in raw['single_pass']};hardware={}
    for row in raw['hardware']:hardware.setdefault(row['tag'],[]).append(row)
    records=[]
    for tag,meta in gate['metadata'].items():
        origin=raw['profile_aliases'].get(tag,tag);d=dynamic[origin];hs=hardware[origin]
        records.append({'tag':tag,'geometry':[meta['M'],meta['N'],meta['num_warps']],
            'layout':meta.get('layout_id'),'family':meta['family'],'profile_origin':origin,
            'grid1_dynamic_warp_instructions':d['counts'],
            'grid1_total_warp_instructions':sum(d['opcode_counts'].values()),
            'grid1_local_load_store_warp_instructions':sum(v for k,v in d['opcode_counts'].items() if k.split('.')[0] in ('LDL','STL')),
            'hardware':[{'q_condition':r['q'],'grid':r['grid'],
                'per_CTA':{name:r['metrics'][metric]/r['grid'] for name,metric in METRICS.items()},
                'DRAM_throughput_percent':r['metrics']['dram__throughput.avg.pct_of_peak_sustained_elapsed'],
                'warps_active_percent':r['metrics']['sm__warps_active.avg.pct_of_peak_sustained_active']}
                for r in hs]})
    result={'status':'OK','records':records,'NCU_duration_used_for_timing':False,
        'scope':'Whole loop-free FULL/control kernels. No subtraction claiming isolated ML/MR traffic. Software dynamic instruction counts use grid=1; hardware averages use 128 MiB full grid.'}
    c.derived(c.OUT/'counter_summary.json',result)
    print('D1 counter normalization OK',len(records),'programs')
    return result


if __name__=='__main__':run()
