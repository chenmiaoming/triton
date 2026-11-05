"""CPU-only N3b analysis with an explicitly post-timing report boundary."""
import argparse
from pathlib import Path

from . import common as c
from . import profile_analyze as original


def render(result, protocol):
    assert protocol['evidence_role']=='post_timing_mechanism_diagnostic'
    assert protocol['selection_uses_candidate_timing'] and not protocol['confirmatory']
    report=original.render(result,protocol)
    old_title='# Stage N3 — 静态预选布局对照的 NCU 证据'
    assert report.startswith(old_title)
    report=report.replace(old_title,'# Stage N3b — 后验 KDA gate 机制诊断',1)
    old_scope='所有对照由布局机制静态预选，未按候选耗时挑选。'
    assert old_scope in report
    report=report.replace(old_scope,
        '本补充由 N2 的 feature_vec4 结果和 N4 固定选择复测触发，明确属于后验计数器诊断；与原 N3 静态预选分母分开，不作为新增确认性计时。',1)
    report=report.replace('## 5. 静态对照的适用范围','## 5. 后验诊断的预绑定对照范围',1)
    evidence=protocol['selection_evidence']
    rows=evidence['N4_independent_fixed_choice']['rows']
    boundary=['## 6. 保留 N4 原始统计结论','',
        '此次只补计数器，不改写或追加正式计时。严格确认要求相对 U 与 G0 的 95% 区间下界均严格高于 +3%；计数器机制成立也不能将未过门槛的计时升级为确认收益。','',
        '| Case | N4 收益 vs U (%) | N4 CI vs U (%) | N4 收益 vs G0 (%) | N4 CI vs G0 (%) | 两基线严格 +3% 确认 |',
        '|---|---:|---|---:|---|---|']
    for row in rows:
        okay=all(row[key][0]>3 for key in ['gain_vs_U_CI95','gain_vs_G0_CI95'])
        boundary.append(f'| {row["case"]} | {row["gain_vs_U_percent"]:.6f} | {row["gain_vs_U_CI95"]} | {row["gain_vs_G0_percent"]:.6f} | {row["gain_vs_G0_CI95"]} | {okay} |')
    boundary += ['',f'N2 触发分析 SHA256：`{evidence["N2_exploratory"]["SHA256"]}`。',
        f'N4 固定选择复测分析 SHA256：`{evidence["N4_independent_fixed_choice"]["SHA256"]}`。','',
        'D64 保留为同变换的实用无收益对照；D128 保留其严格门槛下的 provisional 状态。寄存器和驻留容量来自真实 binary binding；容量变化不等同于实际占用率或可直接换算的加速比例。','']
    return report+'\n'+'\n'.join(boundary)


def run(protocol_path,profiles,output,payload_path=None,attempts=None):
    protocol_path,output=Path(protocol_path),Path(output)
    protocol=c.read(protocol_path)
    assert protocol['evidence_role']=='post_timing_mechanism_diagnostic'
    payload_path=Path(payload_path) if payload_path else Path(protocol['payload_path'])
    result=original.analyze(protocol,profiles,payload_path,attempts)
    result.update(evidence_role=protocol['evidence_role'],confirmatory=False,
        selection_uses_candidate_timing=True,formal_timing_or_selector_changes=False,
        selection_evidence=protocol['selection_evidence'],
        N4_outcome_boundaries=protocol['N4_outcome_boundaries'],
        scope='Explicitly post-timing mechanism diagnostic; separate from original static N3. No new timing, selector change, counter-to-latency weighting or confirmation upgrade.',
        analysis_sources=[dict(path=str(path),SHA256=c.sha(path.read_bytes()))
                          for path in [Path(original.__file__),Path(__file__)]])
    c.write_once(output/'summary.json',result)
    c.write_once(output/'summary.csv',original.csv_bytes(result,protocol['metrics']))
    c.write_once(output/'metrics.csv',original.metric_csv_bytes(result))
    c.write_once(output/'REPORT.md',render(result,protocol).encode())
    print('N3b CPU analysis',result['report_count'],'reports; post-timing diagnostic only')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,default=c.OUT/'stage_n3b/preregistration/protocol.json')
    parser.add_argument('--profiles',type=Path,default=c.OUT/'stage_n3b/profiles')
    parser.add_argument('--output',type=Path,default=c.OUT/'stage_n3b/analysis')
    parser.add_argument('--payload',type=Path)
    parser.add_argument('--attempt-map',type=Path)
    args=parser.parse_args()
    run(args.protocol,args.profiles,args.output,args.payload,c.read(args.attempt_map) if args.attempt_map else None)


if __name__=='__main__':main()
