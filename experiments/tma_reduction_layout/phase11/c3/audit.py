"""Independent checks of static predictions against frozen development evidence."""
import re
import subprocess
from collections import Counter
from experiments.tma_reduction_layout.phase11.c3 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, plan, compact

LD='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'
BC='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'


def run():
    gate=c.read(c.BASE/'results/phase11/stage_c2b/artifact_gate.json')
    counters=c.read(c.BASE/'results/phase11/stage_c2b/counter_gate.json')['binaries']
    root=c.ROOT/c.read(c.BASE/'results/phase11/stage_c2a/compile_gate.json')['artifact_directory']
    rows=[];addresses={}
    for m,n,w in ((32,64,8),(32,128,4)):
        for case in candidates(m,n,w):
            p=plan(case)
            addresses[case['id']]=p['ML']
            ml,mr=[counters[case['id']+':'+f+':D1'] for f in ('ML','MR')]
            observed={'ML_WF':ml['hardware_delta_per_iteration'][LD],
                'ML_BC':ml['hardware_delta_per_iteration'][BC],
                'MR_WF':mr['hardware_delta_per_iteration'][LD],
                'MR_SHFL':mr['families']['shuffle']['loop_per_iteration_warp_instructions'],
                'MR_BAR':mr['families']['barrier']['loop_per_iteration_warp_instructions'],
                'MR_MAX':mr['families']['max']['loop_per_iteration_warp_instructions']}
            predicted={'ML_WF':p['ML']['wavefronts'],'ML_BC':p['ML']['bank_conflicts'],
                'MR_WF':p['MR']['shared_load_wavefronts'],
                'MR_SHFL':p['MR']['shuffle_warp_instructions'],
                'MR_BAR':p['MR']['barrier_warp_instructions']}
            checks={k:abs(v-observed[k])<1e-6 for k,v in predicted.items()}
            lo,hi=p['MR']['local_max_dynamic_interval']
            checks['MR_MAX_interval']=lo+predicted['MR_SHFL']<=observed['MR_MAX']<=hi+predicted['MR_SHFL']
            folder=root/case['id'].replace(':','/')/'MR/D1'
            text=(folder/'kernel.ptx').read_text().split('.file')[0]
            ptx=dict(Counter(re.findall(r'^\s*(max\.[\w.]+)\s',text,re.M)))
            checks['BF16_local_PTX_kind_interval']=all(op in ('max.bf16','max.bf16x2','max.f32') for op in ptx)
            checks['nontrivial_local_BF16_tree']=(any(op.startswith('max.bf16') for op in ptx))==(p['p']>1)
            r=gate['resources'][case['id']+':MC:D1']
            regs=r['attributes']['num_regs']
            reglo,reghi=p['resources']['registers_per_thread_interval']
            checks['MC_register_interval']=reglo<=regs<=reghi
            smem=r['reservations']['natural']['dynamic_bytes']
            # MC has two input buffers, rather than the selector's one buffer.
            lo,hi=p['resources']['shared_bytes_interval']
            checks['MC_two_buffer_shared_interval']=lo+2*m*n<=smem<=hi+2*m*n
            rows.append({'case':case,'static':compact(p),'predicted':predicted,
                'observed':observed,'checks':checks,'all_pass':all(checks.values()),
                'PTX_max_sites_diagnostic_only':ptx,
                'MC_measured_resource_diagnostic_only':{'registers':regs,'shared':smem,
                    'capacity':r['reservations']['natural']['capacity']}})
    # Snapshot exact pinned implementation, including the old pass versions.
    refs={}
    for name in ['lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp',
                 'lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp',
                 'lib/Analysis/Utility.cpp','lib/Dialect/TritonGPU/Transforms/Coalesce.cpp',
                 'lib/Conversion/TritonGPUToLLVM/Utility.cpp']:
        data=subprocess.check_output(['git','show',c.PIN+':'+name],cwd=c.ROOT)
        dest=c.OUT/'source_reference'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        refs[name]=c.digest(data)
    value={'rows':rows,'source_reference_commit':c.PIN,'source_reference_SHA256':refs,
           'all_pass':all(r['all_pass'] for r in rows),
           'max_instruction_prediction_is_interval':True,
           'PTX_packed_choice_not_uniquely_identified_from_layout':True,
           'aggregate_q_bank_conflicts_not_predicted_by_address_model':True}
    c.write(c.OUT/'static_audit.json',value)
    c.write(c.OUT/'static_addresses.json',addresses)
    return value


if __name__=='__main__':
    value=run()
    for row in value['rows']:
        print(row['case']['id'],[k for k,v in row['checks'].items() if not v] or 'PASS')
