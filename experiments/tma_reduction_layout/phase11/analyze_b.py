"""Paired worker-median statistics; no score model or fitted weights."""
import argparse
import itertools
import math
import statistics as s
from experiments.tma_reduction_layout.phase11 import common as c


def t6_quantile(probability):
    # Exact Student df6 CDF, using u=t/sqrt(6+t*t); invert by bisection.
    def cdf(t):
        u = t/math.sqrt(6+t*t)
        return .5+15/16*(u-2*u**3/3+u**5/5)
    lo,hi = 0.0,100.0
    for _ in range(100):
        mid = (lo+hi)/2
        if cdf(mid)<probability:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def contrast(a,b,comparisons):
    assert len(a)==len(b)==7
    ratios = [math.log(x/y) for x,y in zip(a,b)]
    mean = s.mean(ratios)
    radius = t6_quantile(1-.05/(2*comparisons))*s.stdev(ratios)/math.sqrt(7)
    low,high = mean-radius,mean+radius
    return {'paired_log_ratios':ratios,'mean_log_ratio':mean,'log_interval':[low,high],
            'time_reduction_percent':100*(1-math.exp(mean)),
            'time_reduction_interval_percent':[100*(1-math.exp(high)),100*(1-math.exp(low))],
            't_critical':t6_quantile(1-.05/(2*comparisons)),'df':6,'family_comparisons':comparisons}


def derive():
    gate = c.read(c.OUT/'stage_b/artifact_gate.json')
    raw = [c.read(c.OUT/'stage_b/timing'/f'invocation_{i}/raw.json') for i in range(1,8)]
    medians = {}
    for run in raw:
        assert run['status']=='OK'
        collected = {}
        for visit in run['visits']:
            key = (visit['cohort'],visit['group'],visit['fraction'],visit['tag'])
            collected.setdefault(key,[]).extend(visit['samples_us'])
        for key,values in collected.items():
            medians.setdefault(key,[]).append(s.median(values))
    results,bridge = [],[]
    for group,physical in gate['gluon_groups'].items():
        logical = sorted(tag for tag in gate['aliases'] if tag.rsplit(':',1)[0]==group)
        assert len(logical)==4
        for fraction in (1,16):
            times = {tag:medians['gluon_layout',group,fraction,gate['aliases'][tag]] for tag in logical}
            pairs = []
            for a,b in itertools.combinations(logical,2):
                pairs.append({'a':a,'b':b,**contrast(times[a],times[b],6),
                              'same_physical_binary':gate['aliases'][a]==gate['aliases'][b]})
            possible,clear = [],[]
            for a in logical:
                comparisons = [contrast(times[a],times[b],6) for b in logical if a!=b]
                if all(r['time_reduction_interval_percent'][0]>3 for r in comparisons):
                    clear.append(a)
                beaten = any(contrast(times[b],times[a],6)['time_reduction_interval_percent'][0]>3 for b in logical if a!=b)
                if not beaten:
                    possible.append(a)
            ranking = sorted(logical,key=lambda tag:s.median(times[tag]))
            results.append({'case_id':group,'fraction':fraction,'worker_medians_us':times,
                            'median_of_worker_medians_us':{tag:s.median(v) for tag,v in times.items()},
                            'observed_ranking':ranking,'clear_winner':clear,'possible_best':possible,'pairs':pairs})
    for group,tags in gate['bridge_groups'].items():
        anchor,gluon = tags
        for fraction in (1,16):
            a,b = [medians['canonical_bridge',group,fraction,tag] for tag in (gluon,anchor)]
            band = contrast(a,b,1)
            low,high = band['time_reduction_interval_percent']
            bridge.append({'case_id':group,'fraction':fraction,'canonical':anchor,'gluon':gluon,
                           'canonical_worker_medians_us':b,'gluon_worker_medians_us':a,**band,
                           'within_three_percent_band':low>=-3 and high<=3,
                           'code_generation':gate['reproduction'][group],
                           'cross_pipeline_causal_layout_gain_claimed':False})
    return {'stage':'B','target':'H100/SM90','status':'COMPLETE','layout_results':results,'canonical_bridge':bridge,
            'worker_invocations':7,'GPU_UUIDs':[r['environment']['gpu_uuid'] for r in raw],
            'raw_timing_samples':sum(len(v['samples_us']) for r in raw for v in r['visits']),
            'statistical_unit':'7 paired worker invocation medians, not individual launches or 7 independent physical GPUs',
            'assumptions':'t intervals assume independent invocation log ratios with approximately normal distribution; repeated GPU UUIDs do not establish independent devices',
            'family_scope':'6 unordered layout comparisons per condition/grid; no global multiple-family or universal non-regression conclusion',
            'architecture_scope':'H100 only; stage A tool coverage on other targets is not performance replication',
            'NCU_durations_enter_performance_statistics':False,'score_model_used':False,'PR_created':False,
            'C_or_D_executed':False,'stop_after_B':True}


def summary(result):
    text = '# Phase 11 Stage B — H100 Gluon layout sweep\n\n'
    text += '固定upstream、未修改native、同一个Gluon kernel，通过BlockedLayout控制vec1/2/4/8。数学工作、TMA/shared布局和输出布局固定，实际lowering与资源共同变化保留。\n\n'
    text += '| Condition | vec1 μs | vec2 μs | vec4 μs | vec8 μs | 观测最低 | 仍可能最优 |\n|---|---:|---:|---:|---:|---|---|\n'
    for row in result['layout_results']:
        if row['fraction']!=1:
            continue
        values = row['median_of_worker_medians_us']
        times = [values[row['case_id']+':fixed_vec'+str(v)] for v in (1,2,4,8)]
        best = row['observed_ranking'][0].rsplit(':',1)[1]
        possible = ', '.join(tag.rsplit(':',1)[1] for tag in row['possible_best'])
        text += f'| {row["case_id"]} | '+ ' | '.join(f'{v:.3f}' for v in times)+f' | {best} | {possible} |\n'
    text += (f'\n完整保留{result["raw_timing_samples"]:,}个单kernel CUDA-event样本，7次独立worker调用；'
             '物理GPU可重复，不能把launch数或worker数当成独立设备数。表中为worker中位数的中位数。'
             '主口径是full grid，1/16 grid只作规模诊断。六个布局对照使用同条件内Bonferroni/t df6区间，'
             '预声明实用差异3%；不合成成本总分，不拟合权重。\n\n')
    text += '| Canonical↔matching-layout Gluon | Gluon时间减少 % | 95%描述区间 % | 完整过滤指令序列相同 |\n|---|---:|---|---|\n'
    for row in result['canonical_bridge']:
        if row['fraction']!=1:
            continue
        lo,hi = row['time_reduction_interval_percent']
        text += f'| {row["case_id"]} | {row["time_reduction_percent"]:.3f} | [{lo:.3f}, {hi:.3f}] | {row["code_generation"]["filtered_sequence_equal"]} |\n'
    text += ('\nTriton锚点与Gluon使用不同pipeline；相同输入layout不足以证明等价。'
             '对应binary的代码生成、资源与独立NCU报告一并保留。过滤指令签名覆盖完整kernel，包括初始加载与输出，'
             '不称为纯规约耗时或纯规约计数。跨pipeline差异不作为布局因果收益。\n\n'
             '本阶段没有进行C1/C2机制干预、跨架构性能复制或创建PR；在阶段B停止，等待讨论是否调整下一步。\n')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args = parser.parse_args()
    result = derive()
    if args.validate:
        assert c.read(c.OUT/'stage_b/analysis.json')==result
        assert (c.OUT/'stage_b/summary.md').read_text()==summary(result)
    else:
        c.write_once(c.OUT/'stage_b/analysis.json',result)
        c.write_once(c.OUT/'stage_b/summary.md',summary(result).encode())
    print('Stage B analysis',result['raw_timing_samples'],'samples; no score model, no C/D')


if __name__=='__main__':
    main()
