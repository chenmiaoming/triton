"""Derived old/new evidence summary; no GPU execution or model selection.

This postprocessing tool is separate from the frozen measurement controller.
Old and new confidence intervals retain their original estimands and cohorts.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
from pathlib import Path

import numpy as np

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from . import analyze as a


SOURCE = Path(__file__).resolve()
BASE = c.ROOT/'experiments/tma_reduction_layout/results/phase11'
SUITES = ('nonreduction', 'operators', 'legacy', 'anchors')


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), SHA256=c.sha(path.read_bytes()))


def old_gain(row, name):
    return dict(estimate=row['gain_vs_'+name+'_percent'], CI95=row['gain_vs_'+name+'_CI95'])


def dual_pass(u, g):
    return bool(u and g and u.get('CI95') and g.get('CI95') and u['CI95'][0] > 3 and g['CI95'][0] > 3)


def transition(old_supported, new_supported, control):
    if control != 'validated':
        return 'new_controls_inconclusive'
    return ('retained' if old_supported and new_supported else
        'newly_resolved_in_new_context' if new_supported else
        'not_resolved_in_new_context' if old_supported else 'neither_resolved')


def comparison(new, old, old_source):
    u, g = old_gain(old, 'U'), old_gain(old, 'G0')
    old_supported = dual_pass(u, g)
    new_supported = bool(new['strict_practical_improvement'])
    return dict(context_id=new['context_id'], context=new['context'], tag=new['tag'],
        representative_tag=new['representative_tag'], old_source=old_source,
        old_gain_vs_U_percent=u, old_gain_vs_G0_percent=g,
        old_dual_lower_above_3_percent=old_supported,
        new_gain_vs_U_percent=new['gain_vs_U_percent'],
        new_gain_vs_G0_percent=new['gain_vs_G0_percent'],
        new_control_status=new['control_status'], new_strict_dual_improvement=new_supported,
        gain_vs_U_point_change_pp=new['gain_vs_U_percent']['estimate']-u['estimate'],
        gain_vs_G0_point_change_pp=new['gain_vs_G0_percent']['estimate']-g['estimate'],
        support_transition=transition(old_supported,new_supported,new['control_status']),
        interpretation='Threshold-resolution change across separately sampled execution contexts, not a paired causal old/new effect.')


def operator_key(context, tag):
    return (context['domain'],context['group'],context.get('batch'),context.get('length'),
        context['mode'],context.get('M'),tag)


def nonreduction(result):
    old4_path = BASE/'nonreduction/stage_n4/analysis/analysis.json'
    old5_path = BASE/'nonreduction/stage_n5_timing/analysis/analysis.json'
    old4,old5 = c.read(old4_path),c.read(old5_path)
    by4 = {row['tag']:row for row in old4['rows']}
    by5 = {row['tag']:row for row in old5['rows']}
    chosen = [comparison(row,by4[row['tag']],'stage_n4') for row in result['rows']
              if row['predetermined_N4_choice']]
    large = [comparison(row,by5[row['tag']],'stage_n5_timing') for row in result['rows']
             if row['context'].get('source') == 'n5' and row['role'] == 'GL']
    a.require(len(chosen)==14 and len(large)==2,'Incomplete frozen N4/N5 choices')
    return dict(frozen_N4_choices=chosen,N5_frozen_large_shape_choices=large,
        N4_support_transitions=dict(Counter(row['support_transition'] for row in chosen)),
        N4_old_supported=sum(row['old_dual_lower_above_3_percent'] for row in chosen),
        N4_new_supported=sum(row['new_strict_dual_improvement'] for row in chosen),
        sources=[ref(old4_path),ref(old5_path)])


def operators(result):
    path = BASE/'operator_gpu/stage_g8/timing_analysis.json'
    old = c.read(path)
    index = {operator_key(row,row['tag']):row for row in old['rows']}
    comparisons = [comparison(row,index[operator_key(row['context'],row['tag'])],'stage_g8')
        for row in result['rows'] if row['role'] == 'GL']
    a.require(len(comparisons)==361,'Incomplete G8 logical candidate matrix')
    # These match pre-existing G8 report questions; they are not picked using
    # the replacement measurements or a newly fitted selector.
    focal = []
    for row in comparisons:
        ctx,tag = row['context'],row['tag']
        chosen = ((ctx['panel']=='g2' and ctx['geometry'].startswith('C06_')
            and ctx['batch'] in (8,64) and tag.endswith(('vec1:n_first','vec2:within','vec4:within')))
            or (ctx['panel']=='g3b' and ctx['mode'] in ('merge','pipeline')
                and (ctx['mode']=='merge' or ctx['M']==2048)
                and tag.endswith(('vec4:n_first','vec4:within')))
            or (ctx['panel']=='g4a' and ctx['batch']==8 and ctx['length']==2048
                and ctx['mode']=='stage1' and tag.endswith(('pv_only','qk_only','coordinated'))))
        if chosen:
            focal.append(row)
    return dict(all_candidate_comparisons=comparisons,previous_report_questions=focal,
        support_transitions=dict(Counter(row['support_transition'] for row in comparisons)),
        old_supported=sum(row['old_dual_lower_above_3_percent'] for row in comparisons),
        new_supported=sum(row['new_strict_dual_improvement'] for row in comparisons),sources=[ref(path)])


def legacy(result,protocol):
    # Keep all 189 original within-vs-N-first grid/consumer/q comparisons.
    points = []
    for row in result['rows']:
        if row['role']!='candidate' or row['tag'] != row['representative_tag']:
            continue
        previous = [entry for entry in row['old_comparisons'] if entry['stage']=='stage_m3']
        a.require(len(previous)==1,'Missing M3 historical comparison')
        old = previous[0]
        points.append(dict(context_id=row['context_id'],context=row['context'],tag=row['tag'],
            reference_tag=row['reference_tag'],old_raw_reestimated_gain_percent=old.get('old_gain_vs_reference_percent'),
            new_gain_percent=row['gain_vs_reference_percent'],
            control_status=row['control_status'],new_strict_improvement=row['strict_practical_improvement'],
            gain_point_change_pp=old.get('new_minus_old_gain_pp'),
            old_estimator='Mean of old worker/visit medians recomputed for comparability; not necessarily the published flat-sample median.'))
    a.require(len(points)==189,'Incomplete M3 paired matrix')
    # Reproduce the old three-grid metric for mechanism comparisons without
    # altering the frozen per-grid analysis or fitting any model parameters.
    contexts,cells = {},{}
    for row in result['physical_rows']:
        if row['worker_visit_medians_us'] is None:
            continue
        ctx = next(c['context'] for c in result['context_summaries'] if c['context_id']==row['context_id'])
        context = {**ctx,'geometry':f"M{ctx['M']}_N{ctx['N']}_w{ctx['W']}|q{ctx['q']}"}
        contexts[row['context_id']] = context
        role = 'control' if ':control:' in row['tag'] else row['role']
        cells[row['context_id'],row['tag']] = dict(role=role,array=np.asarray(row['worker_visit_medians_us']))
    rng=np.random.default_rng(protocol['timing']['seed'])
    wi=rng.integers(0,3,(8192,3));vi=rng.integers(0,10,(8192,3,10))
    source = BASE/'stage_m3/analysis.json'
    old = c.read(source)
    published = defaultdict(list)
    for row in old['oneshot']:
        published[(row['tag'],str(row['q_condition']))].append(row['tile_slope_ns'])
    services = a._anchor_services(contexts,cells,wi,vi,3,validated=result['status']=='OK')
    for row in services:
        q=row['geometry'].rsplit('|q',1)[1]
        values=published.get((row['tag'],q))
        row['old_published_service_ns_per_tile_per_SM']=float(np.mean(values)) if values else None
        row['old_published_estimator']='Mean across three worker slopes; each old grid point is the median of all 12 old samples.'
    return dict(all_point_comparisons=points,three_grid_service_rows=services,
        sources=[ref(source)],U_G0_scope='No source-aligned U/G0 pair exists; only explicit N-first/within references are compared.')


def anchors(result):
    published = {}
    sources = []
    for stage in ('stage_c2d','stage_d1'):
        path=BASE/stage/'analysis.json';old=c.read(path);sources.append(ref(path))
        grouped=defaultdict(list)
        for row in old['oneshot']:
            if row['q_condition']=='natural' and row['family'] in ('ONE_FULL','ONE_TRANSFER'):
                grouped[row['tag']].append(row['tile_slope_ns'])
        for tag,values in grouped.items():published[tag]=(stage,float(np.mean(values)))
    by_geometry=defaultdict(list)
    for row in result['anchors_service_rows']:
        if row['role']!='default_repeat':by_geometry[row['geometry']].append(row)
    rows=[]
    for geometry,entries in sorted(by_geometry.items()):
        default=next(row for row in entries if row['role']=='default')
        transfer=next(row for row in entries if row['role']=='control')
        candidates=[]
        old_default=published[default['tag']][1]
        for row in entries:
            if row['role'] in ('default','control'):continue
            old_stage,old_service=published[row['tag']]
            old_gain=100*(old_default-old_service)/old_default
            candidates.append(dict(tag=row['tag'],old_stage=old_stage,
                old_published_gain_percent=old_gain,
                old_raw_reestimated_gain_percent=row.get('old_gain_vs_default_percent',{}).get(old_stage),
                new_gain_percent=row.get('gain_vs_default_percent'),
                point_change_from_published_pp=row['gain_vs_default_percent']['estimate']-old_gain if row.get('gain_vs_default_percent') else None,
                new_strict_improvement=row['strict_practical_improvement']))
        rows.append(dict(geometry=geometry,default_tag=default['tag'],
            old_published_default_service_ns=old_default,new_default_service_ns=default['service_ns_per_tile_per_SM'],
            old_published_transfer_service_ns=published[transfer['tag']][1],new_transfer_service_ns=transfer['service_ns_per_tile_per_SM'],
            default_service_sham=default['default_service_sham_equivalence'],candidates=candidates))
    a.require(len(rows)==19,'Incomplete natural FULL/T_control geometry set')
    return dict(geometries=rows,sources=sources,
        old_published_CI_scope='No new old-CI support flag is inferred from published point slopes. Old raw re-estimation and original published slopes are separately named.',
        model_scope='Frozen models/predictions remain unchanged; this is seen-case response remeasurement, not a new D1 blind test.')


def build(root,suites=SUITES):
    root=Path(root).resolve();protocol_path=root/'preregistration/protocol.json';p=c.read(protocol_path)
    inputs=[ref(protocol_path),ref(SOURCE),ref(Path(a.__file__))]
    reports={};overall='OK';raw_samples=0
    for suite in suites:
        path=root/suite/'analysis/analysis.json';result=c.read(path)
        a.require(result['suite']==suite and result['protocol_SHA256']==c.sha(c.encode(p)),'Analysis protocol mismatch')
        a.require(result['status']=='OK','Incomplete suite cannot enter final summary')
        inputs.append(ref(path));raw_samples+=result['audit']['raw_sample_count']
        validations=[]
        for candidate in sorted((root/suite).glob('**/*validation*.json')):
            if 'export' in candidate.parts:continue
            value=c.read(candidate)
            if value.get('analysis_recomputed') is True and value.get('suite')==suite:
                a.require(value['status']=='OK' and value['protocol_SHA256']==result['protocol_SHA256'],'Final validator mismatch')
                validations.append(ref(candidate));inputs.append(ref(candidate))
        if not validations:overall='DERIVED_PENDING_FINAL_VALIDATION'
        details=globals()[suite](result,p) if suite=='legacy' else globals()[suite](result)
        inputs.extend(details.pop('sources',[]))
        reports[suite]=dict(summary=result['summary'],raw_samples=result['audit']['raw_sample_count'],
            contexts=result['audit']['distinct_contexts'],cadence=result['cadence'],
            baseline_and_candidate_visit_drift=result['baseline_and_candidate_visit_drift'],
            final_validators=validations,details=details)
    return dict(status=overall,suites=list(suites),source_inputs=inputs,reports=reports,
        raw_target_samples=raw_samples,
        interpretation=[
            'This is derived postprocessing of frozen measurements, not a newly selected cost model or sampling protocol.',
            'A support transition means a difference in pointwise CI resolution between two execution contexts. It does not prove the old gain was caused by idle or that a new gain would hold under every real workload.',
            'Old N4/G8 CI-supported thresholds use their actual stored confidence intervals. Legacy/anchor published point slopes do not acquire invented old confidence intervals.',
            'All losing, unchanged and control-inconclusive outcomes remain in the source analyses. Aliases do not create additional independent observations.',
            'Each suite has three independently allocated formal GPU workers. Raw sample counts and logical comparison counts are not independent GPU counts.'])


def text(value):return a.cell_text(value)


def render(result):
    lines=['# 固定预热与采样间隙复测：旧证据与新证据汇总','',
        f"状态：**{result['status']}**；{result['raw_target_samples']:,} 个目标计时样本。此文件只汇总已有冻结分析，不调参、不重新选模型、不改变旧数据。",'',
        '| Suite | Contexts | 原始样本 | U/G0 实用等价（逻辑比较） | Sham 实用等价（逻辑比较） |',
        '|---|---:|---:|---:|---:|']
    for suite,data in result['reports'].items():
        s=data['summary'];lines.append(f"| {suite} | {data['contexts']} | {data['raw_samples']:,} | {s['U_G0_practically_equivalent']}/{s['U_G0_pairs']} | {s['sham_practically_equivalent']}/{s['sham_comparisons']} |")
    if 'nonreduction' in result['reports']:
        data=result['reports']['nonreduction']['details']
        lines += ['', '## 原 N4 冻结选择及 N5 大尺寸对照','',
            '这里沿用原来预先冻结的候选，没有从新矩阵挑选替代赢家。旧/新两套 CI 属于各自采样条件，百分点变化不是同卡配对因果量。','',
            '| Case / frozen candidate | 旧相对 U % [CI95] | 新相对 U % [CI95] | 新相对 G0 % [CI95] | 门槛状态 |',
            '|---|---:|---:|---:|---|']
        for row in data['frozen_N4_choices']+data['N5_frozen_large_shape_choices']:
            lines.append(f"| {row['tag']} | {text(row['old_gain_vs_U_percent'])} | {text(row['new_gain_vs_U_percent'])} | {text(row['new_gain_vs_G0_percent'])} | {row['support_transition']} |")
    if 'operators' in result['reports']:
        data=result['reports']['operators']['details']
        lines += ['', '## 完整 G8 候选矩阵','',
            f"361 条逻辑候选比较的双控制 CI 下界均超过 +3%：旧 {data['old_supported']} 条，新 {data['new_supported']} 条。别名仍共享原始物理观测；这些不是独立 GPU 数，也不是家族错误率保证。",'',
            '全部 361 条及未通过的新控制均保留在 summary.json。下列是旧 G8 报告已经讨论的固定问题列表，不根据新结果筛选。','',
            '| Context / candidate | 旧相对 U % [CI95] | 新相对 U % [CI95] | 新相对 G0 % [CI95] | 状态 |',
            '|---|---:|---:|---:|---|']
        for row in data['previous_report_questions']:
            lines.append(f"| {row['context_id']} / {row['tag']} | {text(row['old_gain_vs_U_percent'])} | {text(row['new_gain_vs_U_percent'])} | {text(row['new_gain_vs_G0_percent'])} | {row['support_transition']} |")
    if 'legacy' in result['reports']:
        data=result['reports']['legacy']['details']
        lines += ['', '## M3 配对机制证据','',
            f"全部 {len(data['all_point_comparisons'])} 个相同 vec、consumer、q、grid 的 within/N-first 配对保留。没有把这些显式 Gluon 对照改称 Triton U/G0。",'',
            'summary.json 另给出按 q 分层的三网格服务斜率。旧原始样本按新 visit 估计量重算的点值与旧报告的 flat-sample median 斜率分别命名；没有给旧点值虚构 CI。']
    if 'anchors' in result['reports']:
        data=result['reports']['anchors']['details']
        lines += ['', '## 19 个已见 FULL / T_control 几何','',
            '以下均为 128/256/384 MiB 三网格服务斜率，单位 ns/tile/SM。完整布局列表保留在 summary.json；没有重拟合冻结模型，也没有把 D1 再次称为盲测。','',
            '| Geometry | 旧报告默认值 | 新默认值 [CI95] | 旧报告 transfer | 新 transfer [CI95] | Default slope sham % [CI95] |',
            '|---|---:|---:|---:|---:|---:|']
        for row in data['geometries']:
            lines.append(f"| {row['geometry']} | {row['old_published_default_service_ns']:.3f} | {text(row['new_default_service_ns'])} | {row['old_published_transfer_service_ns']:.3f} | {text(row['new_transfer_service_ns'])} | {text(row['default_service_sham'])} |")
    lines += ['', '## 解释边界','']
    lines.extend('- '+entry for entry in result['interpretation'])
    return '\n'.join(lines)+'\n'


def comparisons_csv(result):
    stream=io.StringIO(newline='')
    names=['suite','context_id','tag','old_gain_vs_U_percent','old_gain_vs_G0_percent',
        'new_gain_vs_U_percent','new_gain_vs_G0_percent','new_control_status','support_transition']
    writer=csv.DictWriter(stream,names);writer.writeheader()
    for suite,data in result['reports'].items():
        if suite=='nonreduction':rows=data['details']['frozen_N4_choices']+data['details']['N5_frozen_large_shape_choices']
        elif suite=='operators':rows=data['details']['all_candidate_comparisons']
        else:continue
        for row in rows:
            value={key:row.get(key) for key in names};value['suite']=suite
            writer.writerow({key:json_value(item) for key,item in value.items()})
    return stream.getvalue().encode()


def json_value(value):
    import json
    return json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suites',nargs='+',choices=SUITES,default=list(SUITES))
    args=parser.parse_args();result=build(args.root,args.suites)
    outputs={'summary.json':c.encode(result),'comparison.csv':comparisons_csv(result),'REPORT.md':render(result).encode()}
    a.require(not any((args.output/name).exists() for name in outputs),'Refusing to overwrite derived summary')
    for name,blob in outputs.items():c.write_once(args.output/name,blob)
    print(result['status'],list(result['reports']),result['raw_target_samples'])


if __name__=='__main__':main()
