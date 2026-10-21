"""Offline, independent re-import of every new report and comparison with frozen G2 times."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json,re,subprocess
from pathlib import Path
from . import common as c
from .g3_prepare import OUT
from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows,source_blocks,FAMILIES


def imported(folder,mode):
    raw=c.read(folder/'import.json')
    commands=[['/usr/local/cuda/bin/ncu','--import',str(folder/'report.ncu-rep'),
              '--page','raw','--csv','--print-units','base','--print-fp']]
    if mode=='software':commands.append(['/usr/local/cuda/bin/ncu','--import',str(folder/'report.ncu-rep'),
                                       '--page','source','--print-source','sass','--csv'])
    rel=folder.relative_to(OUT).as_posix().replace('/','_')
    dest=OUT/'offline_imports'/(rel+'.json')
    if dest.exists():value=c.read(dest)
    else:
        outputs=[]
        for cmd in commands:
            p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
            assert p.returncode==0,p.stdout[-2000:]
            outputs.append({'argv':cmd,'returncode':p.returncode,'output':p.stdout})
        value={'CPU_only':True,'report_SHA256':c.sha((folder/'report.ncu-rep').read_bytes()),'commands':outputs}
        c.write_once(dest,value)
    assert value['report_SHA256']==c.sha((folder/'report.ncu-rep').read_bytes())
    assert hardware_rows(value['commands'][0]['output'])==hardware_rows(raw['output'])
    if mode=='software':assert source_blocks(value['commands'][1]['output'])==source_blocks(c.read(folder/'source.json')['output'])
    return {'path':str(dest.relative_to(c.ROOT)),'SHA256':c.sha(dest.read_bytes()),'values_identical':True}


def main():
    protocol=c.read(OUT/'preregistration/protocol.json')
    c.verify_sources(protocol)
    for name,digest in protocol['old_evidence_SHA256'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    rows=[];jobs=[];reports=[]
    for group in protocol['groups']:
        export=OUT/'profiles'/group['case']['id']/'attempt_001/export'
        result=c.read(export/'result.json');assert result['status']=='OK'
        assert result['protocol_SHA256']==c.sha(c.encode(protocol))
        assert result['payload_SHA256']==protocol['G2_payload_SHA256']
        for record in result['reports']:
            folder=export/Path(record['path']).parent
            assert c.sha((folder/'report.ncu-rep').read_bytes())==record['SHA256']
            bind=re.findall(r'PROFILE_BINDING (\{[^\n]+\})',c.read(folder/'collect.json')['output'])
            assert bind,record
            binding=json.loads(bind[-1]);assert not binding['Triton_imported'] and binding['compile_calls']==0
            assert len(binding['records'])==1
            item=binding['records'][0];assert item['correctness']['pass']
            assert item['binding']['cubin_SHA256']==item['binding']['loaded_SHA256']
            metrics=hardware_rows(c.read(folder/'import.json')['output'])
            assert len(metrics)==1
            row={**record,'geometry':group['case']['id'],'resources':item['binding']['attributes'],
                 'grid':item['grid'],'CTAs':int(__import__('math').prod(item['grid'])),
                 'environment':binding['environment'],'metrics':metrics[0],'correctness':item['correctness']}
            if record['mode']=='software':
                pcs=source_blocks(c.read(folder/'source.json')['output']);assert len(pcs)==1
                row['counts']={key:sum(x['count'] for x in pcs[0].values() if key in x['families']) for key in FAMILIES}
                for name,pattern in {'global_load':r'\bLDG(?:\.|\s)','local_load':r'\bLDL(?:\.|\s)',
                                     'local_store':r'\bSTL(?:\.|\s)'}.items():
                    row['counts'][name]=sum(x['count'] for x in pcs[0].values() if re.search(pattern,x['instruction']))
                row['per_PC']=pcs[0]
            rows.append(row);jobs.append((folder,record['mode']))
            reports.append({'path':str((folder/'report.ncu-rep').relative_to(c.ROOT)),
                            'SHA256':record['SHA256'],'mode':record['mode'],'tag':record['tag']})
    assert len(rows)==60
    with ThreadPoolExecutor(max_workers=4) as pool:reimports=list(pool.map(lambda args:imported(*args),jobs))
    data={'status':'OK','reports':reports,'offline_imports':reimports,'rows':rows,
          'new_compile_count':0,'native_rebuilds':0,'new_formal_timing_samples':0,
          'old_G2_files_unchanged':len(protocol['old_evidence_SHA256']),
          'ncu_local_version':subprocess.check_output(['/usr/local/cuda/bin/ncu','--version'],text=True)}
    c.write_once(OUT/'counter_analysis.json',data)
    old=c.read(c.OUT/'stage_g2/timing_analysis.json')['rows']
    lines=['# G3a：非 TMA 推理算子的布局机制补证','',
      '本轮复用固定 G2 二进制与全部正式时间，新增 60 份 NCU 报告。所有报告经本地 CPU 重新导入，数值与远端一致。',
      '定位为默认布局的优化空间；没有进行旧版本比较，不称为性能回归。','',
      '## 采集与口径','',
      '- 4 个几何、20 个原二进制。每项单 CTA InstructionStats；实际 batch8 grid 的 application replay + cache-control none；另以 kernel replay + cache-control all 作冷缓存诊断。',
      '- 8 次相同目标预热发生在 cuProfilerStart 之前，每份报告只含一个目标。缓存策略比较未重新计正式时间。',
      '- 动态指令为单 CTA 的 warp 执行实例；C02 包含完整 QK/softmax/PV/output 与 8 次 KV 循环。WF/sector 为实际 grid 总量 / CTA 数。',
      '- q_cap 是驻留容量，achieved occupancy 是实际计数。stall 为活跃 warp 的时间比例，不是可直接相加的串行延迟。小 grid 的 tail 与 cache 多 pass 数值需谨慎。','',
      '## 正式时间与动态结构','',
      '时间直接来自 G2：C02 为 batch8/context2048 两阶段 decode；C06 为 batch8/context8192 merge。U 与 G0 不保证整体 lowering 相同，纯布局效果使用 G0→GL。','',
      '| geometry / variant | G2 µs | gain vs G0 | regs / q_cap | LDS / STS | SHFL / BAR |',
      '|---|---:|---:|---|---|---|']
    for row in rows:
        if row['mode']!='software':continue
        mode='decode' if row['geometry'].startswith('C02') else 'merge'
        t=next(x for x in old if x['tag']==row['tag'] and x['batch']==8 and x['mode']==mode and x['length']==(2048 if mode=='decode' else 8192))
        res=row['resources'];cnt=row['counts']
        lines.append(f'| {row["tag"]} | {t["mean_visit_median_us"]:.3f} | {t["gain_vs_anchor_percent"]:+.2f}% | {res["n_regs"]} / {res["occupancy_capacity_q"]} | {cnt["shared_load"]} / {cnt["shared_store"]} | {cnt["shuffle"]} / {cnt["barrier"]} |')
    lines+=['','## 预热应用态的硬件计数','',
       '| geometry / variant | load WF / CTA | load BC / CTA | global LD sectors / CTA | issue active % | eligible warp / scheduler | barrier % | short scoreboard % | long scoreboard % | MIO % | achieved occupancy % | L2 hit % |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['mode']!='warm_application':continue
        mt=row['metrics'];g=row['CTAs']
        vals=[mt['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']/g,
              mt['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']/g,
              mt['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']/g,
              mt['smsp__issue_active.avg.pct_of_peak_sustained_active'],mt['smsp__warps_eligible.avg.per_cycle_active']]
        vals += [mt['smsp__warp_issue_stalled_'+s+'_per_warp_active.pct'] for s in ('barrier','short_scoreboard','long_scoreboard','mio_throttle')]
        vals += [mt['sm__warps_active.avg.pct_of_peak_sustained_active'],mt['lts__t_sector_hit_rate.pct']]
        lines.append('| '+row['tag']+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','## 缓存控制对照','',
      '| tag | warm / cold global LD sectors per CTA | warm / cold L2 hit % | warm / cold long scoreboard % |',
      '|---|---|---|---|']
    for warm in rows:
        if warm['mode']!='warm_application':continue
        cold=next(x for x in rows if x['tag']==warm['tag'] and x['mode']=='cold_isolated')
        a,b=warm['metrics'],cold['metrics'];g=warm['CTAs']
        lines.append(f'| {warm["tag"]} | {a["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]/g:.2f} / {b["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]/g:.2f} | {a["lts__t_sector_hit_rate.pct"]:.2f} / {b["lts__t_sector_hit_rate.pct"]:.2f} | {a["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]:.2f} / {b["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]:.2f} |')
    lines+=['','## 官方文档与解释边界','',
      '1. [CUDA Best Practices：合并访存与 shared banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：连续线程访问按所需 sector 合并；逻辑 vec 或静态 load 条数不能替代请求流量。Bank 冲突会拆分服务请求，但减少冲突不是总时间改善的充分条件。',
      '2. [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：共享访问的 instruction、request、wavefront 有不同计量单位；application replay 可保留应用自身预热；默认隔离冷缓存与实际服务场景必须区分。',
      '3. [NVIDIA Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)：先看发射是否受限，再解读 stall；barrier 等待也可能体现其他 warp 尚未完成的工作，不能把 BAR 条数乘固定延迟。',
      '', '以上为本轮查阅的官方版本（2026-10-21）；采集 NCU 与本地重导入 NCU 的实际版本均保存在原始记录中。',
      '计数用于检验解释与排除简化说法，尚不能独立分配每项开销占多少百分比。没有修改编译器、拟合新选择器或扩展 PR。']
    c.write_once(OUT/'REPORT.md','\n'.join(lines).encode())
    c.write_once(OUT/'validation.json',{'status':'OK','reports':60,'local_reimports':len(reimports),
         'all_correctness_pass':True,'old_G2_files_unchanged':data['old_G2_files_unchanged'],'new_formal_timing_samples':0,
         'protocol_SHA256':c.sha(c.encode(protocol)),'counter_analysis_SHA256':c.sha(c.encode(data))})
    print('G3a PASS: 60 reports, 60 local reimports; previous G2 unchanged.')


if __name__=='__main__':main()
