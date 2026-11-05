"""Report pre-bound choices on independent observations, including nulls."""
from . import common as c


def main():
    root=c.OUT/'stage_n4';a=c.read(root/'analysis/analysis.json')
    p=c.read(root/'preregistration/protocol.json');decision=c.read(root/'selection/decision.json')
    n2=c.read(c.OUT/'stage_n2/analysis/analysis.json')
    assert a['protocol_SHA256']==c.sha(c.encode(p))
    assert len(decision['choices'])==14
    by={r['tag']:r for r in a['rows']};old={r['tag']:r for r in n2['rows']}
    def ci(x):return '['+', '.join(f'{v:+.2f}' for v in x)+']'
    lines=['# Stage N4：冻结候选的独立同配置复测','',
        'N2 是探索性全候选测量。为检查选优偏差，本阶段在获得任何新样本前，为全部 14 个配置各冻结 N2 均值最快的一个 GL；包括没有收益的配置。随后使用新的 H100 worker，同时重新测 U/G0。既有样本不混入新估计量，未改算法、tile、warps、编译器或二进制。','',
        '这是一批相同配置上的独立时间观测，不是未见几何、跨 GPU 或整模型泛化试验。','',
        f'主估计量：3 workers × 10 visits × 31 samples；先取 visit 中位数再取算术平均；逐 round 随机交错。共 {a["audit"]["raw_sample_count"]:,} 个物理原始样本，42 个逻辑程序（U/G0 独立计时；完整 binary/ABI 相同的 GL alias 不重复制造样本）。4096 次配对 worker/visit bootstrap，逐比较 95% 区间，无多重比较校正。','',
        '严格的布局改进口径同时要求相对 U 和 G0 的区间下界均超过 +3%；只超过零或点估计超过 3% 不算通过。','',
        '| Case | 冻结 GL | U / G0 / GL (µs) | N2 vs U (%) | 独立 vs U 收益 / CI (%) | 独立 vs G0 收益 / CI (%) | 两个下界均 >3% |',
        '|---|---|---:|---:|---|---|---|']
    passed=[]
    for choice in decision['choices']:
        tag=choice['candidate'];r=by[tag];case=r['case']
        u=next(x for x in a['rows'] if x['case']==case and x['variant']=='U')
        g=next(x for x in a['rows'] if x['case']==case and x['variant']=='G0')
        okay=r['gain_vs_U_CI95'][0]>3 and r['gain_vs_G0_CI95'][0]>3
        if okay:passed.append(case)
        lines.append(f'| `{case}` | `{r["variant"]}` | {u["time_us"]:.3f} / {g["time_us"]:.3f} / {r["time_us"]:.3f} | {old[tag]["gain_vs_U_percent"]:+.2f} | {r["gain_vs_U_percent"]:+.2f} / {ci(r["gain_vs_U_CI95"])} | {r["gain_vs_G0_percent"]:+.2f} / {ci(r["gain_vs_G0_CI95"])} | {"是" if okay else "否"} |')
    lines += ['',f'严格通过 {len(passed)}/14：'+', '.join('`'+x+'`' for x in passed)+'。',
        '所有 14 个 U/G0 控制的 CI 均在 ±3% 内；全部配置未观察到区间支持的 >3% 候选退化。逐点区间不作全家族错误率保证。','',
        'N01 小配置的独立相对 U 收益为约 2.47%，区间跨越 3% 门槛，不能沿用 N2 的 >3% 结论。N05 D128 相对 U 的收益区间超过 3%，但相对 G0 的下界约为 2.989%；在上述严格双控制口径下保留为边缘结果，不四舍五入后宣称通过。RoPE、MRoPE、FP8 pack 与 KV scatter 没有达到本轮的严格实用收益标准。','',
        '每个配置都保留全部输出/guard/只读区域的独立正确性检查；原地 MRoPE 每次测量前恢复输入，恢复位于计时事件之外。N4 的所选程序已在 N2 的全部三种模式检查中通过；本阶段另重复检查，不引入新 GPU 编译。','',
        '机制解释使用独立 NCU 阶段。不能由上述时间表推导 shared-memory、bank conflict 或特定指令的因果纳秒权重。N04 源自 SM120 准备路径，本轮只有 H100 上独立 pack 的可移植性证据。','',
        '证据：`selection/decision.json` 固定选择和 N2 输入 SHA；`preregistration/protocol.json` 固定协议；`analysis/analysis.json`、`analysis/timing.csv` 保留所有 worker/visit 估计与区间；`check/` 与 `timing/` 保留原始 binding、输出检查、样本和执行顺序。','']
    assert all(x['classification']=='practically_equivalent' for x in a['baseline_equivalence'])
    c.write_once(root/'REPORT.md','\n'.join(lines).encode())
    print('N4 strict successes',passed)


if __name__=='__main__':main()
