"""Pure CPU audit of six N5 warm-application hardware reports only.

No formal timing files are read. InstructionStats was not collected, so the
analysis never synthesizes software instruction counts or U/G0 equality tests.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import uuid
import zipfile

from . import common as c
from .n5_profile_prepare import PAYLOAD_SHA256
from .profile_analyze import hardware_rows, _command, _additive_counter, metric_csv_bytes
from .profile_prepare import _abi, _member


def _flag(argv, name):
    assert argv.count(name)==1, (name,argv)
    index = argv.index(name)
    assert index+1<len(argv), (name,argv)
    return argv[index+1]


def _binding(binding, meta, case, expected_abi):
    assert binding['mode']=='warm_application' and binding['warmups']==8
    assert binding['compile_calls']==0 and not binding['Triton_imported']
    assert binding['profiler_attached'] and not binding['formal_timing']
    assert binding['driver_return_codes']==[0] and len(binding['records'])==1
    environment = binding['environment']
    assert environment['cc']==[9,0] and 'H100' in environment['name']
    assert environment['SM_count']>0
    assert environment['gpu_uuid'].startswith('GPU-')
    uuid.UUID(environment['gpu_uuid'].removeprefix('GPU-'))
    actual = binding['records'][0]
    assert actual['tag']==meta['tag']
    assert actual['correctness']['pass'] and all(actual['correctness']['guards'])
    assert all(row['pass_'] for row in actual['correctness']['details'])
    assert actual['inplace_restored_before_each_launch']
    assert actual['runtime_arguments_unchanged_from_full_fixture']
    expected_grid = ([(case['rows']+127)//128,(case['cols']+3)//4,1]
                     if case['family']=='N01' else
                     [(case['packed_cols']+31)//32,(case['rows']+31)//32,1])
    assert actual['grid']==actual['original_grid']==expected_grid
    assert actual['fixture']==meta['fixture'], 'The full finite oracle fixture changed'
    loaded = actual['binding']
    assert loaded['cubin_SHA256']==loaded['loaded_SHA256']==meta['cubin_SHA256']
    assert loaded['ABI']==expected_abi
    assert loaded['direct_launches']==loaded['guards']==9 and loaded['graph_launches']==0
    resources = loaded['attributes']
    assert resources['n_regs']==meta['resources']['REG'] and resources['local_bytes']==meta['resources']['LOCAL']
    assert isinstance(resources['occupancy_capacity_q'],int) and resources['occupancy_capacity_q']>0
    return actual


def analyze(protocol, profiles, payload_path, attempts=None):
    """Validate raw evidence and return a deterministic analysis; write nothing."""
    profiles, payload_path = Path(profiles), Path(payload_path)
    attempts = attempts or {}
    assert protocol['InstructionStats_collected'] is False
    assert protocol['maximum_concurrent_GPU_groups']==1 and protocol['expected_reports']==6
    assert len(protocol['profile_groups'])==2 and not protocol['alias_proofs']
    assert set(attempts)<={group['id'] for group in protocol['profile_groups']}
    assert c.sha(payload_path.read_bytes())==protocol['payload_SHA256']==PAYLOAD_SHA256
    protocol_sha = c.sha(c.encode(protocol))
    snapshot_path = profiles.parent/'preregistration/source_snapshot.zip'
    with zipfile.ZipFile(snapshot_path) as snapshot:
        for name,digest in protocol['source_SHA256'].items():
            assert c.sha(snapshot.read(name))==digest, name
    metric_origin = c.ROOT/protocol['metric_provenance']['path']
    assert c.sha(metric_origin.read_bytes())==protocol['metric_provenance']['SHA256']
    assert c.read(metric_origin)['profile_metrics']==protocol['metrics']
    assert len(protocol['metrics'])==len(set(protocol['metrics']))==26
    rows, reports, versions, audits = [], [], [], []
    with zipfile.ZipFile(payload_path) as archive:
        accepted = json.loads(archive.read('result.json'))
        assert accepted['status']=='OK' and len(accepted['cases'])==2 and len(accepted['programs'])==6
        by_tag = {meta['tag']:meta for meta in accepted['programs']}
        assert len(by_tag)==6
        for group in protocol['profile_groups']:
            assert group['modes']==['warm_application'] and len(group['records'])==3
            assert group['logical_variants'][0:2]==['U','G0']
            attempt = int(attempts.get(group['id'],1))
            folder = profiles/group['id']/f'attempt_{attempt:03}'
            root = folder/'export'
            dispatch = c.read(folder/'dispatch.json')
            assert dispatch['protocol_SHA256']==protocol_sha and dispatch['payload_SHA256']==PAYLOAD_SHA256
            assert dispatch['host_max_concurrent_groups']==1 and dispatch['orchestration']=='sequential spawn/get'
            result = c.read(root/'result.json')
            assert result['status']=='OK' and result['group']==group['id'] and result['geometry']==group['case']['id']
            assert result['protocol_SHA256']==protocol_sha and result['payload_SHA256']==PAYLOAD_SHA256
            assert result['call_id']==dispatch['call_id'] and not result['formal_timing'] and result['native_rebuilds']==0
            selected = {item['metadata']['tag']:item for item in group['records']}
            assert len(selected)==3
            assert {(item['tag'],item['mode']) for item in result['reports']}=={(tag,'warm_application') for tag in selected}
            assert len(result['reports'])==3
            version_output = _command(root/'version.json')
            versions.append(dict(group=group['id'],output=version_output,
                SHA256=c.sha((root/'version.json').read_bytes())))
            group_uuids = set()
            for report in result['reports']:
                item = selected[report['tag']]
                meta = item['metadata']
                assert not item['aliases'] and item['logical_variants']==[meta['variant']]
                assert not report['aliases'] and report['mode']=='warm_application'
                assert meta==by_tag[meta['tag']], 'Profile metadata differs from the accepted binding payload'
                for name,digest in meta['artifact_SHA256'].items():
                    assert c.sha(archive.read(_member(meta,name)))==digest
                assert c.sha(archive.read(_member(meta,'kernel.cubin')))==meta['cubin_SHA256']
                expected_abi = _abi(archive.read(_member(meta,'kernel.ptx')).decode())
                binary_path = root/report['path']
                assert c.sha(binary_path.read_bytes())==report['SHA256']
                report_folder = binary_path.parent
                assert not (report_folder/'source.json').exists(), 'Unexpected software InstructionStats artifact'
                settings = c.read(report_folder/'settings.json')
                assert settings['mode']=='warm_application' and settings['replay']=='application' and settings['cache']=='none'
                assert settings['warmups']==8 and settings['one_binary_per_report'] and settings['grid']=='original full grid'
                job = c.read(report_folder/'job.json')
                assert job==dict(case=group['case'],mode='warm_application',metadata=[meta],warmups=8)
                collected = c.read(report_folder/'collect.json')
                assert collected['returncode']==0 and not collected['timed_out']
                argv = collected['argv']
                for name,value in [('--replay-mode','application'),('--cache-control','none'),
                                   ('--clock-control','none'),('--profile-from-start','off'),
                                   ('--metrics',','.join(protocol['metrics']))]:
                    assert _flag(argv,name)==value
                assert '--section' not in argv
                expected_suffix = [meta['tag'].replace(':','_'),'warm_application']
                assert list(Path(_flag(argv,'--job')).parts[-3:])==expected_suffix+['job.json']
                assert list(Path(_flag(argv,'--export')).parts[-3:])==expected_suffix+['report']
                replay_path = report_folder/'all_replay_bindings.json'
                replays = c.read(replay_path)
                emitted = [json.loads(line) for line in re.findall(r'PROFILE_BINDING (\{[^\n]+\})',collected['output'])]
                assert replays==emitted and len(replays)==report['emitted_replay_bindings'] and replays
                binding_path = report_folder/'binding.json'
                assert c.sha(binding_path.read_bytes())==report['binding_SHA256']
                assert report['binding_path']==str(binding_path.relative_to(root))
                binding = c.read(binding_path)
                assert binding==replays[-1]
                for replay in replays:
                    _binding(replay,meta,group['case'],expected_abi)
                    group_uuids.add(replay['environment']['gpu_uuid'])
                actual = binding['records'][0]
                imported = c.read(report_folder/'import.json')
                assert imported['returncode']==0 and not imported['timed_out']
                import_argv = imported['argv']
                assert _flag(import_argv,'--page')=='raw' and _flag(import_argv,'--print-units')=='base'
                assert '--csv' in import_argv and '--print-fp' in import_argv
                assert list(Path(_flag(import_argv,'--import')).parts[-3:])==expected_suffix+['report.ncu-rep']
                hardware = hardware_rows(imported['output'])
                assert len(hardware)==1, 'One frozen program per hardware report'
                assert re.search(r'\b'+re.escape(meta['function_name'])+r'\b',hardware[0]['metadata']['Kernel Name'])
                metrics = hardware[0]['metrics']
                assert all(name in metrics and metrics[name]['value'] is not None for name in protocol['metrics'])
                assert all('byte' in metrics[name]['unit'].lower() for name in
                           ['dram__bytes_read.sum','dram__bytes_write.sum'])
                ctas = math.prod(actual['grid'])
                chosen = {name:metrics[name] for name in protocol['metrics']}
                normalized = {name:dict(value=item['value']/ctas,unit=item['unit']+'/CTA')
                    for name,item in chosen.items() if _additive_counter(name)}
                row = dict(group=group['id'],case=group['case']['id'],family=group['case']['family'],
                    tag=meta['tag'],variant=meta['variant'],logical_variants=item['logical_variants'],
                    aliases=[],mode='warm_application',CTAs=ctas,InstructionStats_collected=False,
                    report_path=str(binary_path),report_SHA256=report['SHA256'],
                    import_path=str(report_folder/'import.json'),import_SHA256=c.sha((report_folder/'import.json').read_bytes()),
                    binding_path=str(binding_path),binding_SHA256=report['binding_SHA256'],
                    all_replay_bindings_SHA256=c.sha(replay_path.read_bytes()),replay_binding_count=len(replays),
                    raw_metric_metadata=hardware[0]['metadata'],metrics=metrics,selected_metrics=chosen,
                    selected_additive_counts_per_CTA=normalized,
                    resources=actual['binding']['attributes'],environment=binding['environment'],
                    correctness=actual['correctness'])
                rows.append(row)
                reports.append(dict(group=group['id'],tag=meta['tag'],mode='warm_application',
                    path=str(binary_path),SHA256=report['SHA256']))
            assert len(group_uuids)==1, 'The sequential group changed its allocated GPU'
            audits.append(dict(group=group['id'],attempt=attempt,
                result_SHA256=c.sha((root/'result.json').read_bytes()),
                dispatch_SHA256=c.sha((folder/'dispatch.json').read_bytes()),
                raw_export_SHA256=c.sha((folder/'raw_export.zip').read_bytes()),
                reports=3,all_replay_copies_valid=True,GPU_UUID=next(iter(group_uuids))))
    assert len(rows)==6
    differences = []
    for group in protocol['profile_groups']:
        values = {row['variant']:row for row in rows if row['group']==group['id']}
        candidate = values[group['logical_variants'][2]]
        for base in ['U','G0']:
            comparison = {}
            for name in protocol['metrics']:
                left,right = candidate['selected_metrics'][name],values[base]['selected_metrics'][name]
                assert left['unit']==right['unit']
                comparison[name] = dict(candidate=left['value'],reference=right['value'],
                    difference=left['value']-right['value'],unit=left['unit'],
                    difference_unit='percentage points' if left['unit']=='%' else left['unit'])
            differences.append(dict(group=group['id'],case=group['case']['id'],
                variant=candidate['variant'],reference=base,mode='warm_application',
                selected_metric_differences=comparison))
    return dict(status='OK',CPU_only=True,GPU_allocations=0,compile_calls=0,
        formal_timing_input_read=False,InstructionStats_collected=False,
        baseline_software_crosscheck=dict(status='NOT_COLLECTED',reason='N5 collects only six full-grid warm-application hardware reports; no software equality test is fabricated.'),
        protocol_SHA256=protocol_sha,payload_SHA256=PAYLOAD_SHA256,
        source_snapshot_SHA256=c.sha(snapshot_path.read_bytes()),
        analyzer_source_SHA256=c.sha(Path(__file__).read_bytes()),
        report_count=6,rows=rows,comparisons=differences,reports=reports,
        group_audits=audits,NCU_versions=versions,
        attempts={group['id']:int(attempts.get(group['id'],1)) for group in protocol['profile_groups']},
        normalization='Only selected additive traffic/instruction sums are divided by actual CTA count. Preserve all raw names and units; percentages, ratios and times are never divided.',
        interpretation='Profiled hardware diagnostics only; no formal timing read, no retuning, no inference of cache residency from allocation size alone.')


def csv_bytes(result, metrics):
    fields = ['group','case','variant','mode','CTAs','GPU_UUID','n_regs','local_bytes','occupancy_capacity_q']+list(metrics)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for row in result['rows']:
        value = {name:row[name] for name in fields[:5]}
        value['GPU_UUID'] = row['environment']['gpu_uuid']
        value.update({name:row['resources'][name] for name in fields[6:9]})
        value.update({name:row['selected_metrics'][name]['value'] for name in metrics})
        writer.writerow(value)
    return stream.getvalue().encode()


def render(result, protocol):
    lines = ['# N5 较大 footprint 的硬件计数（warm application）','',
        '本阶段沿用 N5 已冻结的两个大 case、U/G0 与各自唯一候选，只验证流量、请求分组及缓存行为。'
        '共六份报告，每份一个完整 grid 的冻结 CUBIN；不读取正式计时，不重新挑选布局。','',
        '`application replay / cache none / clock none`，每轮八次相同程序 warmup；'
        'warmup/target 前恢复原地输入，恢复位于 profiler 范围之外。全部 replay 的完整 oracle、guard、'
        'UUID、CUBIN SHA、运行参数及 CUDA 驻留容量已验证。','',
        '**未采集 InstructionStats**：没有 CTA0 软件指令表，也不伪造 U/G0 软件计数一致性检验。'
        '`smsp__inst_executed.sum` 仅作为实际硬件指标保留。','',
        f"协议 SHA256：`{result['protocol_SHA256']}`；payload SHA256：`{result['payload_SHA256']}`。",'',
        '| Case | Variant | CTA | REG / CUDA q | L2 hit (%) | DRAM read (MiB) | DRAM write (MiB) | Global load sectors/CTA | Global store sectors/CTA |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in result['rows']:
        metric = lambda key:row['selected_metrics'][key]['value']
        sectors = lambda op:row['selected_additive_counts_per_CTA'][f'l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum']['value']
        lines.append(f"| {row['case']} | {row['variant']} | {row['CTAs']} | {row['resources']['n_regs']} / {row['resources']['occupancy_capacity_q']} | {metric('lts__t_sector_hit_rate.pct'):.4f} | {metric('dram__bytes_read.sum')/2**20:.4f} | {metric('dram__bytes_write.sum')/2**20:.4f} | {sectors('ld'):.4f} | {sectors('st'):.4f} |")
    lines += ['', '表中 MiB 仅为原始 byte counter 的显示换算；全部 26 项指标的原始名称、单位及数值保留在 summary.json/metrics_long.csv。'
        '只有 additive 流量/指令 sum 按实际 CTA 数归一，百分比及比值不归一。', '',
        'L2 命中率是缓存层汇总，不能单凭一个比例断言所有 buffer 均驻留 L2。DRAM 流量、全局请求/sector、'
        '输出写策略与不同 footprint 必须共同解释；这些单次 profiler 数据不替代无 profiler 的正式计时。', '',
        '## 采集与完整性', '', '| Group | 报告数 | GPU UUID | 全部 replay 检查 |',
        '| --- | --- | --- | --- |']
    for audit in result['group_audits']:
        lines.append(f"| {audit['group']} | {audit['reports']} | {audit['GPU_UUID']} | True |")
    return '\n'.join(lines)+'\n'


def run(stage):
    stage = Path(stage)
    protocol = c.read(stage/'preregistration/protocol.json')
    result = analyze(protocol,stage/'profiles',stage/'compiled_payload.zip')
    folder = stage/'analysis'
    c.write_once(folder/'summary.json',result)
    c.write_once(folder/'summary.csv',csv_bytes(result,protocol['metrics']))
    c.write_once(folder/'metrics_long.csv',metric_csv_bytes(result))
    c.write_once(folder/'REPORT.md',render(result,protocol).encode())
    print('N5 hardware-only analysis OK:',result['report_count'],'reports')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,default=c.OUT/'stage_n5_profiles')
    run(parser.parse_args().stage)
