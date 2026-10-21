"""Freeze new profiling questions; previous G2 artifacts remain read-only."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
from . import common as c

OUT=c.OUT/'stage_g3a'
SNAPSHOT=Path('/tmp/operator-gpu-g3a-snapshot-v1')
QUESTIONS={
 'C02_n64_d128':['upstream','gluon_anchor','vec1:n_first','vec4:n_first','vec8:within'],
 'C02_n64_d64':['upstream','gluon_anchor','vec1:n_first','vec4:n_first','vec2:within'],
 'C06_s16_d128':['upstream','gluon_anchor','vec1:n_first','vec2:within','vec8:within'],
 'C06_s32_d256':['upstream','gluon_anchor','vec1:n_first','vec4:within','vec8:within'],
}


def main():
    old=c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    gate=c.read(c.OUT/'stage_g2/compile_gate.json')
    bytag={x['tag']:x for x in gate['records']}
    groups=[]
    for geom,names in QUESTIONS.items():
        case=next(x for x in old['geometries'] if x['id']==geom)
        rows=[];seen={}
        for name in names:
            row=bytag[geom+':'+name]
            key=(row['cubin_SHA256'],tuple(row['runtime_argument_names']))
            if key in seen:
                seen[key]['aliases'].append(row['tag']);continue
            item={'metadata':row,'aliases':[]}
            rows.append(item);seen[key]=item
        groups.append({'case':case,'records':rows})
    names=list(old['worker_source_SHA256'])
    names += [p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.glob('g3_*.py')]
    hashes=c.snapshot(names,SNAPSHOT)
    c.write_once(OUT/'preregistration/source_snapshot.zip',c.archive(SNAPSHOT))
    mandatory=old['profiles']['metrics']
    optional=['smsp__warp_issue_stalled_'+r+'_per_warp_active.pct' for r in
              ('barrier','short_scoreboard','long_scoreboard','mio_throttle','math_pipe_throttle','not_selected')]
    optional+=['smsp__issue_active.avg.pct_of_peak_sustained_active',
               'smsp__warps_eligible.avg.per_cycle_active','smsp__warps_active.avg.per_cycle_active',
               'sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed',
               'lts__t_sector_hit_rate.pct','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
               'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
               'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
    protocol={'stage':'G3a','frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
      'parent_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'profile':'chenmiaoming','target':'H100!','groups':groups,
      'mandatory_metrics':mandatory,'optional_metrics':optional,'warmups':8,
      'modes':{'software':{'replay':'kernel','cache':'all','section':'InstructionStats'},
               'warm_application':{'replay':'application','cache':'none'},
               'cold_isolated':{'replay':'kernel','cache':'all'}},
      'hypotheses':[
       'C02 d128 vec4 winner needs its own executed-instruction and hardware evidence.',
       'C02 d64 vec1 may reduce shuffle while worsening conversions, shared traffic and residency.',
       'C06 s16d128 vec1 gain should be checked alongside equal global-sector requests and reduced communication.',
       'C06 s32d256 U to G0 lowering differences must remain separate from G0 to GL effects.',
       'Application replay with target-specific priming tests whether warm-cache evidence differs from isolated cold replay.'],
      'formal_timing':'Reuse all frozen G2 raw times; no result-driven retiming or selector fitting.',
      'terminology':'Default-layout optimization opportunity; no claim of version-to-version regression.',
      'worker_source_SHA256':hashes,'native_rebuilds':0,
      'G2_payload_SHA256':c.sha((c.OUT/'stage_g2/compiled_payload.zip').read_bytes()),
      'G2_gate_SHA256':c.sha((c.OUT/'stage_g2/compile_gate.json').read_bytes()),
      'old_evidence_SHA256':{str(p.relative_to(c.ROOT)):c.sha(p.read_bytes()) for p in
          sorted((c.OUT/'stage_g2').rglob('*')) if p.is_file()},
      'official_documentation':[
       'https://docs.nvidia.com/nsight-compute/ProfilingGuide/',
       'https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html',
       'https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/']}
    c.write_once(OUT/'preregistration/protocol.json',protocol)
    print('G3a frozen:',len(groups),'geometries,',sum(len(x['records']) for x in groups),'unique binaries; no new compilation')


if __name__=='__main__':main()
