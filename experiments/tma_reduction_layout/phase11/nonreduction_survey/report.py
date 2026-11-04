"""Render a bounded standalone-kernel report from frozen CPU analysis JSON."""
import argparse
from collections import Counter
from pathlib import Path

from . import common as c


FAMILIES = {
    'N01': '量化 scale 的 swizzle / 排列转换',
    'N02': 'AWQ 反量化',
    'N03': 'RoPE / MRoPE',
    'N04': '固定外部 scale 的 FP8 QKV pack',
    'N05': 'KDA forward gate',
    'N06': 'KV cache 写入',
}


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def interval(value):
    return f'[{value[0]:+.2f}, {value[1]:+.2f}]'


def text(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render(result, protocol, accepted_manifest=None):
    """Return Markdown; rendering cannot select or alter any timing record."""
    assert result['protocol_SHA256'] == c.sha(c.encode(protocol))
    audit = result['audit']
    candidates = [row for row in result['rows'] if row['variant'] not in ('U', 'G0')]
    stage = 'Stage N2 — Non-reduction layout survey' if candidates else 'Stage N1 — U / G0 baseline alignment'
    lines = [f'# {stage}', '',
        '本轮只评价公开源码中固定配置的独立 kernel。U 是冻结公开 Triton DSL 程序，G0 是按 U 实际编译结果重建的 Gluon 基线；候选仅改变显式布局。', '',
        '## 1. 冻结输入与覆盖', '',
        f'- 协议 SHA256：`{result["protocol_SHA256"]}`。',
        f'- 运行 payload SHA256：`{result["payload_SHA256"]}`。',
        f'- {len(result["cases"])} 个 case / {len(result["families"])} 个算子族；每个逻辑程序均覆盖 {len(audit["workers"])} 个 worker、每 worker {protocol["timing"]["visits"]} 个 visit、每 visit {protocol["timing"]["samples"]} 个原始样本。',
        f'- 逻辑程序 {audit["logical_programs"]} 个；全部 worker 的实际计时 cell 共 {audit["physical_cells_all_workers"]} 个，原始计时样本 {audit["raw_sample_count"]:,} 个；展开 alias 后逻辑等效样本 {audit["logical_equivalent_sample_count"]:,} 个。物理样本与 alias 展开数量分别统计。',
        '- U 与 G0 在每个 worker 都单独计时。与其他候选共用同一实际程序的 alias 继承相同观测，不伪装成额外重复实验。']
    if accepted_manifest is not None:
        path = Path(accepted_manifest)
        lines.append(f'- 已接受编译基线清单：`{path}`；SHA256：`{c.sha(path.read_bytes())}`。编译等价性及 ABI/资源检查以该独立清单和原始编译产物为准。')
    lines += ['', '## 2. 估计量与证据边界', '',
        f'- 主时间：先对每 visit 的 {protocol["timing"]["samples"]} 个样本取中位数，再对 {protocol["timing"]["workers"]}×{protocol["timing"]["visits"]} 个 worker/visit 中位数取算术平均；同一 case 内采用 `{protocol["timing"].get("sample_order", "variant_blocks")}` 顺序。',
        '- 收益：`100 × (mean(reference) − mean(candidate)) / mean(reference)`；正值表示耗时下降。',
        '- 置信区间：4096 次分层 bootstrap，先重采样 worker、再重采样该 worker 内的 visit；同一 case 的 U/G0/全部候选共用同一组抽样索引。不额外重采样 visit 内原始样本。',
        '- `±3%` 是实际差异门槛。95% 区间完整落入该区间才记为 practical equivalence；完整高于 `+3%` / 低于 `−3%` 才记为 practical improvement / regression，其余为 inconclusive。',
        '- 区间为逐比较区间，未作多重比较校正。按本轮结果选出的 best observed 候选是探索性结果，不能当作未见数据上的确认性胜率。',
        '- 不把固定 shape 的独立 kernel 结果外推为框架、完整 LLM 或整条 attention/KDA 流水的性能；也不凭布局改进直接认定存在版本性能回归。', '',
        '## 3. U / G0 同轮计时', '',
        '| Case | U (µs) | G0 (µs) | G0 相对 U 收益 (%) | 95% CI (%) | 分类 |',
        '|---|---:|---:|---:|---|---|']
    for row in result['baseline_equivalence']:
        lines.append(f'| {text(row["case"])} | {number(row["U_time_us"])} | {number(row["G0_time_us"])} | {row["gain_percent"]:+.2f} | {interval(row["CI95"])} | {row["classification"]} |')
    counts = Counter(row['classification'] for row in result['baseline_equivalence'])
    lines += ['', '分类计数：' + '；'.join(f'`{key}` = {counts[key]}' for key in sorted(counts)) + '。计时等价是统计分类，不作为预先保证的结果。']
    if candidates:
        lines += ['', '## 4. 各 case 的候选结果', '',
            '以下按观测均值列出最快候选。所有合法候选及相对 U、G0 的配对区间见 `timing.csv` 与 `analysis.json`；最快候选选择不删除其他结果。', '',
            '| Case | 候选数 | Best observed | 相对 U 收益 (%) | 相对 G0 收益 (%) | vs U 超过 +3% 的候选数 | vs U 低于 −3% 的候选数 |',
            '|---|---:|---|---:|---:|---:|---:|']
        for row in result['cases']:
            lines.append(f'| {text(row["case"])} | {row["candidate_count"]} | {text(row["best_observed_variant"])} | {number(row["best_observed_gain_vs_U_percent"],2)} | {number(row["best_observed_gain_vs_G0_percent"],2)} | {row["practical_improvement_candidates_vs_U"]} | {row["practical_regression_candidates_vs_U"]} |')
        lines += ['', '## 5. 算子族汇总', '',
            '族内收益均值仅为各 case 等权的描述性汇总，不混合不同 kernel 的绝对耗时，也不代表真实模型调用频率加权的加速。', '',
            '| 族 | 独立 kernel 范围 | Case 数 | G0 计时等价数 | 有 practical improvement 候选的 case 数（vs U） | Best observed 平均收益（vs U，%） |',
            '|---|---|---:|---:|---:|---:|']
        for row in result['families']:
            lines.append(f'| {row["family"]} | {FAMILIES.get(row["family"],row["family"])} | {row["cases"]} | {row["baseline_equivalent_cases"]} | {row["cases_with_practical_candidate_improvement"]} | {number(row["best_observed_gain_vs_U_mean_percent"],2)} |')
        lines += ['', '## 6. 全部候选的逐比较结果', '',
            '| Case | Variant | 时间 (µs) | vs U 收益 / CI (%) | vs G0 收益 / CI (%) | vs U 分类 |',
            '|---|---|---:|---|---|---|']
        for row in result['rows']:
            if row['variant'] in ('U','G0'):
                continue
            lines.append(f'| {text(row["case"])} | {text(row["variant"])} | {number(row["time_us"])} | {row["gain_vs_U_percent"]:+.2f} / {interval(row["gain_vs_U_CI95"])} | {row["gain_vs_G0_percent"]:+.2f} / {interval(row["gain_vs_G0_CI95"])} | {row["direction_vs_U"]} |')
    lines += ['', '## 7. 各 case 的适用范围', '']
    for row in result['cases']:
        lines.append(f'- `{row["case"]}`：{row["claim_scope"]}')
    lines += ['', '## 8. 保留证据', '',
        '- 原始计时、编译产物、失败尝试与自适应决策记录保留原样；本报告只新增 derived analysis。',
        '- `analysis.json`：输入文件 SHA、精确覆盖审计、环境、每 worker/visit 中位数、配对 bootstrap 区间与实际物理代表程序。',
        '- `timing.csv`：全部逻辑程序的可复核结果表。',
        '- 硬件机制须结合独立的 NCU/SASS/IR 证据逐项讨论；本时间分析本身不把快慢归因于某一计数器或 pass。', '']
    return '\n'.join(lines)


def run(analysis_path, protocol_path, output, accepted_manifest=None):
    analysis_path, protocol_path, output = map(Path, (analysis_path, protocol_path, output))
    result, protocol = c.read(analysis_path), c.read(protocol_path)
    c.write_once(output, render(result, protocol, accepted_manifest).encode())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--accepted-manifest', type=Path)
    args = parser.parse_args()
    print(run(args.analysis, args.protocol, args.output, args.accepted_manifest))


if __name__ == '__main__':
    main()
