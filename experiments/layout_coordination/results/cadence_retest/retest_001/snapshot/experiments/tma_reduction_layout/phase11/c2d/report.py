"""Tabulate the complete measured FULL matrix and matched control evidence."""
from collections import defaultdict
import statistics as st
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream

WF='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'
BC='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'


def build():
    p=c.protocol();a=c.read(c.OUT/'analysis.json');g=c.read(c.OUT/'artifact_gate.json')
    counters=c.read(c.OUT/'counter_gate.json');compiled=c.read(c.OUT/'compile_gate.json')
    idx=defaultdict(list)
    for row in a['oneshot']: idx[row['tag'],row['q_condition']].append(row)
    hw={(row['tag'],row['q']):row for row in counters['hardware']}
    dyn={row['tag']:row for row in counters['single_pass']}
    matrix=[]
    for geo in p['calibration_geometries']:
        for case in candidates(*geo):
            tag=case['id']+':ONE_FULL';rows=idx[tag,'natural'];assert len(rows)==3
            h=hw[tag,'natural'];d=dyn[tag]
            rs=g['resources'][tag]
            matrix.append({'geometry':geo,'layout':case['id'],'placement':case['placement'],
                'vec':case['register_layout']['size_per_thread'][2],
                'default':case['id']==upstream(*geo),'tag':tag,
                'service_ns':st.median(r['tile_slope_ns'] for r in rows),
                'worker_service_range_ns':[min(r['tile_slope_ns'] for r in rows),max(r['tile_slope_ns'] for r in rows)],
                'full_us':st.median(r['full_us'] for r in rows),
                'q':rs['reservations']['natural']['capacity'],'n_regs':rs['attributes']['num_regs'],
                'local_bytes':rs['attributes']['local_bytes'],
                'shared_load_wavefronts_per_CTA':h['metrics'][WF]/h['grid'],
                'shared_load_bank_conflicts_per_CTA':h['metrics'][BC]/h['grid'],
                'dynamic_warp_instructions_per_CTA':d['counts'],
                'reused_CUBIN':bool(g['metadata'][tag].get('reuse_source'))})
    assert len(matrix)==61
    summary=[]
    for geo in p['calibration_geometries']:
        rows=[r for r in matrix if r['geometry']==geo]
        default=next(r for r in rows if r['default']);best=min(rows,key=lambda r:r['service_ns'])
        summary.append({'geometry':geo,'R':geo[0]*geo[1]//(32*geo[2]),'candidates':len(rows),
            'default':default['layout'],'best':best['layout'],'default_service_ns':default['service_ns'],
            'best_service_ns':best['service_ns'],'oracle_gain_percent':100*(1-best['service_ns']/default['service_ns']),
            'default_full_us':default['full_us'],'best_full_us':best['full_us']})
    controls=[]
    for geo in p['calibration_geometries']:
        tag=f'M{geo[0]}_N{geo[1]}_w{geo[2]}:control:ONE_TRANSFER'
        response=[]
        for q in p['timing']['q_conditions']:
            rows=idx.get((tag,q),[])
            if rows:
                response.append({'q_condition':q,'actual_q':rows[0]['q'],
                    'service_ns':st.median(r['tile_slope_ns'] for r in rows),
                    'full_us':st.median(r['full_us'] for r in rows)})
        controls.append({'geometry':geo,'responses':response})
    quality=[]
    for family in ('ONE_FULL','ONE_TRANSFER'):
        rows=[r for r in a['oneshot'] if r['family']==family]
        errors=[r['grid_fit']['max_abs_residual']/max(r['grid_fit']['ys']) for r in rows]
        natural=[r for r in rows if r['q_condition']=='natural']
        quality.append({'family':family,'median_grid_residual_percent':100*st.median(errors),
            'max_grid_residual_percent':100*max(errors),
            'natural_global_intercept_us_range':[min(r['global_intercept_us'] for r in natural),max(r['global_intercept_us'] for r in natural)]})
    value={'status':'OK','matrix':matrix,'geometries':summary,'controls':controls,
        'fit_quality':quality,'all_candidates_measured':True,'missing_FULL_candidates':0,
        'old_geometries_candidates':55,'new_R4_candidates':6,'D1_remaining':10,
        'event_samples':a['event_samples'],'NCU_reports':len(counters['reports'])}
    c.derived(c.OUT/'matrix.json',value)
    lines=['# Stage C2d — FULL 完整矩阵与低 R 标定','',
        '## 1. 边界、二进制与测量','',
        '在原 8 个开发几何上补齐 18 个 within 候选，得到 55/55；另将历史仅代码出现的 `16×64/w8` 正式转入开发集，补齐其 6 个候选。合计 9 个几何、61/61 个 FULL，另有 9 个 TMA 对照和 6 个新 R4 MC。剩余 D1 严格保留 10 个历史 strict-zero-mention（5 核心、5 边界），本阶段均未编译、计时。',
        '',f'76 个逻辑程序：原样复用 {compiled["reused"]} 个 CUBIN，新编译 {compiled["new_compiles"]} 个。CPU-only 编译使用 pinned `{c.PIN}` 与已有 Triton cache；native SHA 固定，ccache 前后统计相同，native rebuild=0。H100! 132 SM，host descriptor，BF16、128B shared swizzle；未修改 C++。',
        '',f'每个计时 worker 1248 个物理 cell，3 个独立调用，共 {a["event_samples"]} 个原始 CUDA-event 样本（每 cell 3 visits×4 次）；每个 cell 均检查首/中/末 CTA。eligibility 264 组检查全部通过。NCU 单独采集 {len(counters["reports"])} 份报告：{len(counters["dynamic"])} 个软件计数 launch、{len(counters["hardware"])} 个硬件计数 launch；全部在本机 CPU 重导入核对。计时/NCU 仅 CUDA driver 加载校验过的 CUBIN，不导入 Triton/JIT。',
        '', 'FULL 保持无循环单次程序。为避免 quarter/half 图落入缓存容量内，host descriptor 扩为 24576×8192，FULL 三个 grid 分别连续读取 128/256/384 MiB；stride 与旧 binary 不变。报告 ns/tile/SM 来自 `time_us ~ intercept + slope×grid/132`，全局截距保留诊断，不进入静态选择器。128 MiB 的 time_us 另列。与旧 C2c 的 quarter/half/full 拟合数值不得不加区分地混用。MC 保持原 8192 行双槽包装，K>=16 拟合；K1 不参与稳态斜率。',
        '', '镜像构建首次遇到 package mirror timeout，未 dispatch 实验；停止该 app，revision 2 改为复用已完成的 NumPy tools 镜像。完整失败日志在 infrastructure/，不是性能重跑。',
        '', '## 2. 九个几何的完整候选真值','',
        '收益为该轮完整候选的 oracle，对比同 worker/q 后按服务斜率汇总；不是静态选择器的测试成绩。','',
        '| BM×BN/w | R | 候选 | 默认 ns | 最优 ns | oracle 收益 | 最优布局 |','|---|---:|---:|---:|---:|---:|---|']
    for r in summary:
        m,n,w=r['geometry'];lines.append(f'| {m}×{n}/w{w} | {r["R"]} | {r["candidates"]} | {r["default_service_ns"]:.3f} | {r["best_service_ns"]:.3f} | {r["oracle_gain_percent"]:.2f}% | {r["best"].split(":",1)[1]} |')
    lines+=['','## 3. 全部 61 个 FULL（natural q）','',
        'WF/BC 为 NCU 整个 FULL 的 shared-load wavefronts/bank conflicts，除以 grid；包含输入加载、归约通信和输出布局转换，不能全部算成 ML。动态指令按 warp instructions 计，详见 matrix.json 与 counter_gate.json。','',
        '| 几何 | 布局 | 默认 | ns/tile/SM | 128 MiB µs | n_regs | q | WF/CTA | BC/CTA |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in matrix:
        m,n,w=r['geometry'];lines.append(f'| {m}×{n}/w{w} | vec{r["vec"]}:{r["placement"]} | {"是" if r["default"] else ""} | {r["service_ns"]:.3f} | {r["full_us"]:.3f} | {r["n_regs"]} | {r["q"]} | {r["shared_load_wavefronts_per_CTA"]:.3f} | {r["shared_load_bank_conflicts_per_CTA"]:.3f} |')
    lines+=['','## 4. T_control 的受控 q 响应','',
        '这是包含 TMA/wait/首行读取/写回的操作对照，不是纯 TMA latency。q 用额外动态 shared reservation 控制；实际 alias 不作为独立实验重复采样。','',
        '| 几何 | natural（实际 q） | q1 | q2 | q4 | q8 |','|---|---:|---:|---:|---:|---:|']
    for r in controls:
        d={x['q_condition']:x for x in r['responses']};vals=[]
        for label in ('natural','1','2','4','8'):
            x=d.get(label);vals.append('—' if x is None else f'{x["service_ns"]:.2f}'+(f' (q{x["actual_q"]})' if label=='natural' else ''))
        m,n,w=r['geometry'];lines.append(f'| {m}×{n}/w{w} | '+' | '.join(vals)+' |')
    lines+=['','## 5. 质量与方法学边界','']
    for q in quality:
        lines.append(f'- {q["family"]}: grid 拟合相对残差中位 {q["median_grid_residual_percent"]:.3f}%，最大 {q["max_grid_residual_percent"]:.3f}%；natural 全图截距范围 {q["natural_global_intercept_us_range"]} µs。残差保留，不转成已识别的 tile 固定底座。')
    lines+=['','全部 FULL 无 local storage/spill（driver 与 cuobjdump/SASS 记录可核对）。本阶段得到完整 FULL 与控制 q 响应；没有重新宣称 K64 MC 与 K1 FULL 简单相加成立，也没有将 26.6 µs 当作硬件常数。C3.3 使用本阶段 FULL 为主要拟合目标，旧 MC 与 fixed-R 净对照仅作结构协变量，是否有效由外层留一几何结果决定。',
        '', '主要可复核文件：preregistration/protocol.json（及 revision 2）、D1_split.json、prior_manifest.json、compile_gate.json、artifact_gate.json、timing/*/raw.json、profiles/*/*/raw_export.zip、counter_gate.json、offline_import_validation.json、analysis.json、matrix.json、validation.json。']
    c.derived(c.OUT/'REPORT.md',('\n'.join(lines)+'\n').encode())
    print('C2d report OK; 61/61 FULL candidates')
    return value


if __name__=='__main__': build()
