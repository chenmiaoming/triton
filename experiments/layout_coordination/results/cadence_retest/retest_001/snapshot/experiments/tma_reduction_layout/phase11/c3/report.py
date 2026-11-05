"""Generate the C3 audit report from derived, reproducible results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tma-c3-mpl')
from collections import Counter
import statistics as st
from experiments.tma_reduction_layout.phase11.c3 import common as c


def run():
    theta=c.read(c.OUT/'theta_H100.json');dev=c.read(c.OUT/'development.json')
    audit=c.read(c.OUT/'static_audit.json');blind=c.read(c.OUT/'D1_predictions.json')
    history=c.read(c.OUT/'history_audit.json')
    lines=['# Stage C3 — 静态区间选择器原型与 D1 预测冻结','',
        '2026-10-17。仅 CPU 分析；新增 GPU 作业、kernel 编译、NCU 采集、GPU 计时及 native rebuild 均为 0。没有修改 C++，没有启动 D1，也没有创建 PR。','',
        '**结论：静态访问/通信计划通过开发集核对，但时间模型尚不具备可靠改选能力。** '
        '完整保守模型在两个开发几何上均回退默认（0/2 改选），八个 D1 case 也全部回退。'
        '这不能称为模型有效或修复完成。点预测和所有竞争布局的区间仍完整冻结，供后续独立评价。','',
        '## 1. 输入与审计边界','',
        '固定 host-created descriptor、BF16 输入、FP32 max/output、沿 M 规约。基线 upstream 为 `'+c.PIN+'`。'
        '预测器只接收 BM/BN/W/dtype/axis/register/shared layout；没有 runtime grid、最终 CUBIN/SASS、实际寄存器数或测试时间入口。'
        '既有 C2 编译资源只用于开发审计和校准探针 q 容量，不能作为盲测特征。','',
        '`e_vector=vec`；`elements_per_thread=BM*BN/(32W)`；C2 的 E 探针参数是后者，三者没有混用。'
        '服务单位为 ns/tile/SM，即一个 SM 服务一个 tile 的稳态时间间隔；不是 event 时间除以 CTA 后再除 SM。'
        '所有旧样本、compiler artifacts、NCU report 和失败记录的 SHA 在协议中保护。','',
        '## 2. 静态操作计划 G_L','',
        'Blocked register/lane/warp 基向量按 N-first order 展开。对 BF16，128B nvmma core 是 8×64 个元素。'
        '物理 halfword offset 为：','',
        '```text',
        'offset(m,n) = floor(m/8)*8*BN + floor(n/64)*512',
        '            + (m mod 8)*64 + ((n mod 64) XOR (8*(m mod 8)))',
        '```','',
        '这来自固定 upstream 的 LinearLayout 转换源码，未强制 opcode。每 lane 的 2*vec 字节访问按 128B 子事务分组；'
        '每组逐 bank 统计不同 32-bit word，同 word 的读按广播处理。分别求理想波前与冲突附加波前，'
        '没有使用 `32/vec` 作为加载次数。完整地址在 [static_addresses.json](static_addresses.json)，固定版本源码在 [source_reference](source_reference)。','',
        '| 布局 | p/u/w/e | ML ideal / actual WF / BC | MR WF / SHFL / BAR |',
        '|---|---|---|---|']
    for r in audit['rows']:
        p=r['static'];ml=p['ML'];mr=p['MR']
        lines.append(f'| {p["id"]} | {p["p"]}/{p["u"]}/{p["w"]}/{p["e_vector"]} | '
                     f'{ml["ideal_wavefronts"]}/{ml["wavefronts"]}/{ml["bank_conflicts"]} | '
                     f'{mr["shared_load_wavefronts"]}/{mr["shuffle_warp_instructions"]}/{mr["barrier_warp_instructions"]} |')
    lines += ['',
        '15/15 的 ML 波前/地址冲突，以及 MR shared-load 波前、shuffle 和 BAR.SYNC 动态计数完全匹配冻结 NCU 数据。'
        '32×128 within vec2/4/8 正确得到 ideal=64、actual=128、BC=64；其余地址冲突为 0。'
        '这不等于预测所有驻留条件下的 NCU 聚合 bank counter；C2 已记录其中的额外争用事件。','',
        '**BF16 指令只做到区间预测。** p>1 时预测 BF16 本地树，跨线程部分为 FP32 max/shuffle。'
        'PTX 的 `max.bf16` 与 `max.bf16x2` 分组以及 ptxas 的 HMNMX2/VHMNMX 融合没有被唯一静态重建。'
        '本地树动态指令范围保留双 lane 打包和三输入融合的影响；加上跨线程 max 后，15/15 实测计数落入范围。'
        '不能把这称为精确 opcode 预测，也不能拿 PTX site 数冒充硬件执行数。','',
        '输出保留 native SliceLayout 到固定 Blocked 输出的线程归属转换需求；MO 仅提供实际 lowering 诊断，'
        '当前原型没有证明所有新布局上的精确输出转换成本。','',
        '静态 shared 门检查 M≥8、N≥64、block shared 上限及 scratch/lifetime 上界。寄存器由活跃值数给出宽区间，'
        '驻留同时受 shared/register/warp/CTA 约束。15 个 MC binary 的寄存器与双缓冲 shared 均落入估计范围；'
        '原 kernel 按单缓冲估计。存在 spill 可能时回退，未用实际盲测寄存器缩窄区间。'
        '驻留区间是无 spill 条件下的容量估计，不是 achieved occupancy 保证。','',
        'H100 资源上限依据 [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)；'
        '聚合 bank counter 边界参见 [NCU Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant)。','',
        '## 3. θ_H100 与时间区间','',
        '首先尝试了带 q 依赖的非负操作计数线性拟合。它不能同时解释小探针与高扇出、低驻留行为，'
        '尤其留出中性 W1 后出现零预测。因此拒绝将其作为选择器。完整失败参数和训练/留出结果保留于 '
        '[attempts/linear_v0.json](attempts/linear_v0.json)。没有删除异常测点。','',
        f'冻结版本使用 {len(theta["primitive_surfaces"])} 个 P/U/W/E 响应表项和 {len(theta["load_surfaces"])} 个 ML 模板速率表项。'
        '这些按原语结构参数、D 与 q 共享，未按 15 个布局分别拟合系数。已测 q 节点保持原曲线；中间 q 在 1/q 上插值，'
        '原语尺寸在 log-log 空间插值。超出测量支撑域明确标记，不承诺泛化。','',
        '本地树采用 Pp 相对 P1 的每列有效增量；lane 与跨 warp 路径分别使用 Uu/U1、Ww/W1 的每列有效增量。'
        '按静态独立列数和 warp 数换算。P 探针没有逐列匹配的跨列 packed tree，因此相关本地增量下界保留为 0，'
        '点诊断沿用未打包模板；没有按指令减少比例直接缩放时间。'
        'E(elements_per_thread) 约束 fixture consumer 工作的可选上界。中性探针相减仍是有效模板对比，'
        '不能命名为纯 max/shuffle/barrier 延迟。','',
        '依赖图为 load → local tree → lane exchange → warp exchange。下界允许不同 CTA 在服务资源间流水，'
        '上界串行执行依赖节点并保留 consumer fixture 工作。它是宏观依赖/资源包络，**尚未实现逐 warp、逐 cycle 的发射模拟器**。'
        '对所有静态 q 范围、D1/D2 调度敏感性取包络，再由 MC 约束全局无量纲组合范围。','',
        f'全局 MC 组合比例范围为 {theta["composition"]["steady_ratio"][0]:.3f}–{theta["composition"]["steady_ratio"][1]:.3f}；'
        '这反映流水重叠、fixture 差异及尚未辨识的调度效应，范围很宽。'
        '它是公开的经验敏感性边界，不是硬件保证，也不是按布局存储的残差常数。'
        'MC−(ML+MR+MO) 的全部残差仍在参数文件中逐点保留。没有引入历史固定时间偏置。','',
        'K1 单独使用 max(body, MO 的输出/setup 代理) 的全局组合包络；one-shot MO 没有线性加到循环斜率上。'
        'MO 的 input load、setup 未被独立辨识，因此 K1 仍很宽。'
        'single-CTA 项来自 grid1 的有效循环依赖链范围，不含已辨识的纯 TMA 启动延迟；不能冒充完整单 CTA 的精确完成时间。','',
        '独立 full-grid TMA 控制按每 tile 的输入+输出字节归一，进入同单位的传输包络。'
        '目前只实测 BM32、BN64/128；其他几何的 byte scaling 是冻结的外推假设，选择器据此回退。'
        '候选预测中没有 grid 参数。','',
        f'原语 leave-size-out 最坏相对误差为 {100*theta["primitive_leave_size_out_max_error"]:.1f}%，发生在中性/端点外推。'
        '这说明不能把已测表项的精确插值当成结构泛化证据。完整留出结果保留于参数文件。'
        '开发集 body/K1 区间各覆盖 15/15，但组合区间就是用开发 MC 构建的，因此这只是校准覆盖，不能证明预测有效。','',
        '## 4. 基线、消融与开发集结果','',
        '独立选择问题只有两个几何，15 个布局不是 15 个独立 held-out case。'
        '保守规则要求候选在 steady、K1 和单 CTA 有效链三个区间上均较默认稳健改善超过 3%，且无资源/外推风险。'
        '否则保留默认并返回可比集合。点预测只作诊断，不触发改选。','',
        '| 几何 | 方法 | 选择 | MC K1 regret | 旧完整 N-first kernel µs |',
        '|---|---|---|---:|---:|']
    for r in dev['decisions']:
        if r['method'].endswith('_point_diagnostic'):continue
        full=r['original_N_first_FULL_us']
        lines.append(f'| {r["geometry"]} | {r["method"]} | {r["choice"].split(":",1)[1]} | '
                     f'{r["MC_K1_regret_percent"]:.2f}% | {full:.3f} |' if full is not None else
                     f'| {r["geometry"]} | {r["method"]} | {r["choice"]} | {r["MC_K1_regret_percent"]:.2f}% | 未测 |')
    lines += ['',
        'Load-blind 去掉 ML；Comm-blind 去掉 lane/warp 通信；Resource/Envelope-blind 固定理想 q 并去掉传输平台。'
        '它们仍受组合不确定性限制，所以实际改选均为零。各候选的区间变化及点选择在 development.json 中保留；'
        '当前结果不足以声称消融证明了任一项带来选择收益。','',
        'fanout-only 基线在这两个开发几何上选择 vec1，并没有被数据证明失效。'
        '但同 vec 的 N-first/within 贡献线程数完全相同，它不能区分跨 warp 代价与地址冲突。'
        '不能为了显示模型价值，强行宣称该基线在开发集必然选错。','',
        '**实际错失收益：** 旧完整 N-first 32×64/w8 数据为 default vec8 111.120 µs、vec1 60.592 µs，'
        '保守模型未取得约 45.47% 的已知可用改善。32×128/w4 的旧完整 vec1/vec8 为 55.264/56.768 µs，差异约 2.7%，'
        '低于 3% 实用门槛。**MC K1 在 32×128 上的排序与旧完整 kernel 还会反转**，'
        '不能将 MC K1 最优值替代完整 kernel oracle。全 15 布局的原始完整 kernel 尚未测全，within 不虚构 FULL 数据。','',
        '## 5. 历史审计与八个 D1 冻结预测','',
        f'扫描 {len(history["scanned"]):,} 个受版本控制的实验文件，并检查 ZIP 内元数据与 artifact 名称。'
        'Phase 1–11、早期无 phase 标签的 artifacts、普通 load/A10 记录、source pool 和参数探针均包含在审计范围内。'
        '任何已记录几何曝光都排除，包括仅规划、未计时的旧池条目。未记录的外部实验无法证明不存在。','',
        '旧草案要求 C2 前冻结，但当时没有 case split 记录。本次按照最新授权在 C3 拟合前冻结八个 case；'
        '不追认成 C2 前预注册。`preregistration/protocol.json` 保留准确时间关系和历史 hash。','',
        '补充的 Python AST 三整数 tuple/list 审计对八个 case 也均为零曝光；两个 upstream-source TAR 仅为 compiler provenance，'
        '不含本项目实验记录。宽 BN、小 vec 候选在开发时的双射检查发现 N 方向寄存器重复未展开；修正后全部候选覆盖完整 tile。'
        '原始 split 文件保持不变，静态计划修订及原因追加在 `preregistration/static_plan_revision_1.json`，没有编译/计时后改预测。','',
        '| 未见几何 BM×BN / W | 候选数 | 默认 / 保守选择 | 点预测诊断 | 默认 q 区间 |',
        '|---|---:|---|---|---|']
    for row in blind['cases']:
        base=next(p for p in row['predictions'] if p['id']==row['choice'])
        lines.append(f'| {row["geometry"]} | {len(row["predictions"])} | '
                     f'{row["choice"].split(":",1)[1]} | {row["point_choice"].split(":",1)[1]} | '
                     f'{base["static"]["resources"]["q_interval"]} |')
    lines += ['',
        '八个 case 均通过静态 shared 和寄存器必要条件；较大 case 的寄存器上界可能越过 spill 门，'
        '这是保留的资源不确定性，不是已观察到 OOM/spill。'
        '所有默认选择都是 abstention，不是“预测 vec8/vec4 最优”。每个竞争布局的 steady/K1/chain 区间、'
        'q 区间、rho、点预测及回退原因全部冻结于 [D1_predictions.json](D1_predictions.json)。','',
        '## 6. 当前停点与验证','',
        '已冻结代码、θ、历史/输入 SHA 和八个预测；没有编译 D1，也没有执行 D1 GPU 计时。'
        'C3 完成的是可审计的保守原型及其失败边界，**没有完成可用于修复的有效选择模型**。'
        '优先缺口是 fixture 到单次完整 kernel 的调度/输出识别、静态驻留区间的收紧、以及新几何的独立传输控制；'
        '这些不是继续补一个固定常数能解决的。本次不自行进入后续实验。','',
        '可复现：`python -m experiments.tma_reduction_layout.phase11.c3.validate`。'
        '验证静态映射/通信计数、合法性、八个历史零曝光、代码/参数/预测 hash、确定性重算和旧证据不变。'
        '最终检查同时复跑 C2a/C2b/C1.6 的既有离线 validator。','',
        '[静态核对](static_audit.json) · [共享参数/全部曲线](theta_H100.json) · '
        '[开发与消融](development.json) · [区间图](figures/development_intervals.png) · '
        '[历史审计](history_audit.json) · [冻结清单](freeze.json)','']
    (c.OUT/'REPORT.md').write_text('\n'.join(lines))
    figure(dev)


def figure(dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,n in zip(axes,(64,128)):
        rows=[r for r in dev['candidate_predictions'] if r['case']['BN']==n]
        for i,r in enumerate(rows):
            lo,hi=r['prediction']['onchip_steady_ns']
            ax.plot([lo,hi],[i,i],color='0.55',lw=2,label='Model envelope' if i==0 else None)
            ax.plot(r['measured_development']['MC_body_ns'],i,'o',color='#234f75',
                    label='MC D1 calibration' if i==0 else None)
        ax.set_yticks(range(len(rows)),[r['case']['id'].split(':',1)[1] for r in rows])
        ax.set_xscale('log');ax.set_xlabel('ns / tile / SM (log scale)')
        ax.set_title(f'BM32 BN{n}: interval vs development MC')
        ax.grid(axis='x',alpha=.2)
        ax.legend(loc='lower right',fontsize=8)
    path=c.OUT/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/'development_intervals.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    run()
