"""CPU-only parsing and audit of saved N3 NCU exports.

No NCU collection, CUDA launch, compiler invocation or performance fitting is
performed here. Raw metric names, units and source instruction counts survive
the derived summaries; no instruction category is assigned a latency weight.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import re
import zipfile

from . import common as c
from .profile_prepare import _abi, _identity, _member


INSTRUCTION_CLASSES = {
    'global_load': r'^LDG(?:\.|$)', 'global_store': r'^STG(?:\.|$)',
    'shared_load': r'^(?:LDS|LDSM)(?:\.|$)',
    'shared_store': r'^(?:STS|STSM)(?:\.|$)',
    'local_load': r'^LDL(?:\.|$)', 'local_store': r'^STL(?:\.|$)',
    'shuffle': r'^SHFL(?:\.|$)', 'barrier': r'^BAR(?:\.|$)',
    'memory_barrier': r'^MEMBAR(?:\.|$)',
    'conversion': r'^(?:F2F|F2FP|F2I|I2F|I2I)(?:\.|$)',
    'special_function': r'^MUFU(?:\.|$)',
    'byte_permute': r'^PRMT(?:\.|$)',
}


def numeric(value):
    text = str(value).strip().replace(',', '')
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def hardware_rows(output):
    """Read both NCU wide raw export and older long metric CSV formats."""
    rows = list(csv.reader(io.StringIO(output)))
    result = []
    for i, header in enumerate(rows):
        if 'Metric Name' in header and 'Metric Value' in header:
            grouped = {}
            for line in rows[i+1:]:
                if len(line) != len(header) or line == header:
                    continue
                row = dict(zip(header, line))
                if not row.get('Metric Name'):
                    continue
                identity = tuple(row.get(key, '') for key in ('ID','Kernel Name','Context','Stream'))
                group = grouped.setdefault(identity, dict(
                    metadata={key:row.get(key,'') for key in ('ID','Kernel Name','Context','Stream','Block Size','Grid Size')},
                    metrics={}))
                name = row['Metric Name']
                assert name not in group['metrics'], ('Duplicate NCU metric', name, identity)
                group['metrics'][name] = dict(name=name, unit=row.get('Metric Unit',''),
                    value_text=row['Metric Value'], value=numeric(row['Metric Value']))
            return list(grouped.values())
        if 'ID' not in header or 'Kernel Name' not in header or not any('__' in value for value in header):
            continue
        assert i+1 < len(rows), 'Missing NCU units row'
        units = rows[i+1]
        assert len(units) == len(header)
        for line in rows[i+2:]:
            if len(line) != len(header) or not re.fullmatch(r'\d+', line[header.index('ID')]):
                continue
            result.append(dict(
                metadata={name:value for name,value in zip(header,line) if '__' not in name},
                metrics={name:dict(name=name,unit=unit,value_text=value,value=numeric(value))
                         for name,unit,value in zip(header,units,line) if '__' in name}))
        return result
    raise AssertionError('No supported NCU raw metric CSV header')


def opcode(instruction):
    tokens = instruction.strip().split()
    assert tokens, instruction
    return (tokens[1] if tokens[0].startswith('@') else tokens[0]).rstrip(';')


def source_rows(output):
    """Retain exact Instructions Executed values, including zero-count sites."""
    blocks = []
    current, header = None, None
    for row in csv.reader(io.StringIO(output)):
        if row and row[0]=='Kernel Name':
            current = dict(kernel=row[1], count_column='Instructions Executed', instructions=[])
            blocks.append(current)
            header = None
        elif row and row[0]=='Address' and 'Source' in row and 'Instructions Executed' in row:
            assert current is not None
            header = row
        elif current is not None and header is not None and row and row[0].startswith('0x'):
            assert len(row) == len(header), (row, header)
            values = dict(zip(header,row))
            count = numeric(values['Instructions Executed'])
            assert count is not None and count>=0 and count.is_integer(), ('Unavailable dynamic instruction count',values)
            current['instructions'].append(dict(address=values['Address'], instruction=values['Source'],
                opcode=opcode(values['Source']), count=int(count),
                count_text=values['Instructions Executed'], raw_columns=values))
    assert blocks and all(block['instructions'] for block in blocks)
    for block in blocks:
        addresses = [item['address'] for item in block['instructions']]
        assert len(addresses) == len(set(addresses)), ('Duplicate SASS source addresses',block['kernel'])
    return blocks


def _additive_counter(name):
    # Explicitly exclude gpu__time_duration.sum and other duration/cycle
    # quantities. Only selected additive traffic/instruction counts normalize.
    return name.endswith('.sum') and (name.startswith('l1tex__')
        or name.startswith('dram__bytes_') or name=='smsp__inst_executed.sum')


def _command(path):
    value = c.read(path)
    assert value['returncode'] == 0, path
    return value['output']


def _validate_aliases(protocol, payload_path):
    payload_path = Path(payload_path)
    assert c.sha(payload_path.read_bytes()) == protocol['payload_SHA256']
    physical = {record['metadata']['tag']:record['metadata']
                for group in protocol['profile_groups'] for record in group['records']}
    proofs = protocol.get('alias_proofs', [])
    with zipfile.ZipFile(payload_path) as archive:
        for proof in proofs:
            left, right = proof['logical_metadata'], physical[proof['representative']]
            assert left['tag'] == proof['tag'] and left['geometry'] == right['geometry']
            left_blob, right_blob = [archive.read(_member(meta, 'kernel.cubin')) for meta in (left,right)]
            assert left_blob == right_blob
            assert c.sha(left_blob) == left['cubin_SHA256'] == right['cubin_SHA256']
            left_abi, right_abi = [_abi(archive.read(_member(meta,'kernel.ptx')).decode()) for meta in (left,right)]
            assert _identity(left,left_abi) == _identity(right,right_abi)
            assert left['variant']!='U' and right['variant']!='U'
    return proofs


def analyze(protocol, profiles, payload_path, attempts=None):
    """Audit every expected physical report; never choose an attempt by outcome."""
    profiles = Path(profiles)
    attempts = attempts or {}
    assert set(attempts) <= {group['id'] for group in protocol['profile_groups']}
    aliases = _validate_aliases(protocol, payload_path)
    protocol_sha = c.sha(c.encode(protocol))
    rows, reports, versions, logical_rows = [], [], [], []
    expected_count = 0
    for group in protocol['profile_groups']:
        attempt = int(attempts.get(group['id'],1))
        root = profiles/group['id']/f'attempt_{attempt:03}'/'export'
        result = c.read(root/'result.json')
        assert result['status']=='OK' and result['group']==group['id']
        assert result['geometry']==group['case']['id']
        assert result['protocol_SHA256']==protocol_sha
        assert result['payload_SHA256']==protocol['payload_SHA256']
        selected = {record['metadata']['tag']:record for record in group['records']}
        targets = {(tag,mode) for tag in selected for mode in group['modes']}
        observed = [(row['tag'],row['mode']) for row in result['reports']]
        assert len(observed)==len(set(observed))==len(targets) and set(observed)==targets
        expected_count += len(targets)
        version_path = root/'version.json'
        versions.append(dict(group=group['id'], attempt=attempt, path=str(version_path),
            SHA256=c.sha(version_path.read_bytes()), output=_command(version_path)))
        for report in result['reports']:
            selection = selected[report['tag']]
            meta = selection['metadata']
            assert report.get('aliases',[]) == selection.get('aliases',[])
            folder = root/Path(report['path']).parent
            binary_report = root/report['path']
            assert c.sha(binary_report.read_bytes())==report['SHA256']
            binding_path = folder/'binding.json'
            assert c.sha(binding_path.read_bytes())==report['binding_SHA256']
            binding = c.read(binding_path)
            assert binding['mode']==report['mode'] and binding['warmups']==8
            assert binding['compile_calls']==0 and not binding['Triton_imported']
            assert binding['driver_return_codes']==[0] and len(binding['records'])==1
            actual = binding['records'][0]
            assert actual['tag']==meta['tag'] and actual['correctness']['pass']
            assert all(actual['correctness']['guards'])
            assert actual['binding']['cubin_SHA256']==actual['binding']['loaded_SHA256']==meta['cubin_SHA256']
            assert actual['inplace_restored_before_each_launch']
            assert actual['runtime_arguments_unchanged_from_full_fixture']
            mode = report['mode']
            if mode=='software':
                assert actual['grid']==[1,1,1] and actual['software_active_masks']
            else:
                assert actual['grid']==actual['original_grid']
            ctas = math.prod(actual['grid'])
            assert ctas>0
            imported = hardware_rows(_command(folder/'import.json'))
            assert len(imported)==1, ('Exactly one kernel per NCU report',folder)
            metrics = imported[0]['metrics']
            required = protocol['metrics'] if mode!='software' else []
            for name in required:
                assert name in metrics and metrics[name]['value'] is not None, (folder,name,metrics.get(name))
            selected_metrics = {name:metrics[name] for name in protocol['metrics'] if name in metrics}
            per_cta = {name:dict(value=item['value']/ctas, unit=item['unit']+'/CTA')
                for name,item in selected_metrics.items() if _additive_counter(name) and item['value'] is not None}
            row = dict(group=group['id'], case=group['case']['id'], family=group['case']['family'],
                tag=meta['tag'], variant=meta['variant'], aliases=selection.get('aliases',[]),
                logical_variants=selection['logical_variants'], mode=mode, CTAs=ctas,
                report_path=str(binary_report), report_SHA256=report['SHA256'],
                import_path=str(folder/'import.json'), import_SHA256=c.sha((folder/'import.json').read_bytes()),
                binding_path=str(binding_path), binding_SHA256=report['binding_SHA256'],
                raw_metric_metadata=imported[0]['metadata'], metrics=metrics,
                selected_metrics=selected_metrics, selected_additive_counts_per_CTA=per_cta,
                resources=actual['binding']['attributes'], environment=binding['environment'],
                correctness=actual['correctness'], software_active_masks=actual.get('software_active_masks',{}))
            if mode=='software':
                blocks = source_rows(_command(folder/'source.json'))
                assert len(blocks)==1
                source = blocks[0]
                assert re.search(r'\b'+re.escape(meta['function_name'])+r'\b',source['kernel'])
                opcodes, classes = Counter(), {name:0 for name in INSTRUCTION_CLASSES}
                for item in source['instructions']:
                    opcodes[item['opcode']] += item['count']
                    for name,pattern in INSTRUCTION_CLASSES.items():
                        if re.search(pattern,item['opcode']):
                            classes[name] += item['count']
                row.update(source=source, source_path=str(folder/'source.json'),
                    source_SHA256=c.sha((folder/'source.json').read_bytes()),
                    opcode_counts=dict(sorted(opcodes.items())), instruction_classes=classes,
                    instruction_classes_per_CTA={name:value/ctas for name,value in classes.items()},
                    total_dynamic_instructions=sum(opcodes.values()),
                    instruction_count_unit='Dynamic warp instruction instances reported by NCU Instructions Executed; not thread instructions or memory transactions.',
                    instruction_class_scope='Categories are descriptive and can overlap; no latency weights are assigned.')
            rows.append(row)
            reports.append(dict(group=group['id'],tag=meta['tag'],mode=mode,
                path=str(binary_report), SHA256=report['SHA256']))
            for variant in selection['logical_variants']:
                logical_rows.append(dict(group=group['id'],case=group['case']['id'],variant=variant,
                    mode=mode,representative_tag=meta['tag'],report_SHA256=report['SHA256']))
    assert len(rows)==expected_count
    crosschecks = []
    for group in protocol['profile_groups']:
        software = {variant:next(row for row in rows if row['group']==group['id'] and row['mode']=='software'
            and variant in row['logical_variants']) for variant in ('U','G0')}
        u,g = software['U'],software['G0']
        crosschecks.append(dict(group=group['id'],case=group['case']['id'],
            independently_profiled=u['tag']!=g['tag'],
            opcode_counts_equal=u['opcode_counts']==g['opcode_counts'],
            instruction_classes_equal=u['instruction_classes']==g['instruction_classes'],
            U_total_dynamic_instructions=u['total_dynamic_instructions'],
            G0_total_dynamic_instructions=g['total_dynamic_instructions']))
    comparisons = []
    for group in protocol['profile_groups']:
        for mode in group['modes']:
            group_rows=[row for row in rows if row['group']==group['id'] and row['mode']==mode]
            by_variant={variant:row for row in group_rows for variant in row['logical_variants']}
            for variant in group['logical_variants']:
                if variant in ('U','G0'):
                    continue
                candidate=by_variant[variant]
                for base in ('U','G0'):
                    reference=by_variant[base]
                    deltas={}
                    for name in protocol['metrics']:
                        a=candidate['selected_metrics'].get(name);b=reference['selected_metrics'].get(name)
                        if a is None or b is None or a['value'] is None or b['value'] is None:
                            continue
                        assert a['unit']==b['unit'], (name,a,b)
                        deltas[name]=dict(candidate=a['value'],reference=b['value'],
                            difference=a['value']-b['value'],unit=a['unit'],
                            difference_unit='percentage points' if a['unit']=='%' else a['unit'])
                    comparisons.append(dict(group=group['id'],case=group['case']['id'],variant=variant,
                        reference=base,mode=mode,shared_physical_report=candidate is reference,
                        selected_metric_differences=deltas,
                        instruction_class_differences={name:candidate['instruction_classes'][name]-reference['instruction_classes'][name]
                            for name in INSTRUCTION_CLASSES} if mode=='software' else {}))
    return dict(status='OK',CPU_only=True,GPU_allocations=0,compile_calls=0,
        protocol_SHA256=protocol_sha,payload_SHA256=protocol['payload_SHA256'],
        report_count=len(rows),logical_report_count=len(logical_rows),rows=rows,
        logical_rows=logical_rows,comparisons=comparisons,baseline_software_crosschecks=crosschecks,
        reports=reports,NCU_versions=versions,alias_audit=aliases,
        attempts={group['id']:int(attempts.get(group['id'],1)) for group in protocol['profile_groups']},
        normalization='Divide selected additive traffic/instruction .sum counters by the actual profiled CTA count only. Preserve raw metric names/units. Never divide durations, percentages or ratios.',
        scope='Static-selected standalone-kernel contrasts; no formal timing read, no counter-to-latency weighting, no causal attribution from counts alone. Software CTA0 and full-grid cache modes remain separate.')


def csv_bytes(result, metrics):
    columns=['group','case','family','tag','variant','aliases','logical_variants','mode','CTAs',
        'n_regs','local_bytes','occupancy_capacity_q','total_dynamic_instructions']
    columns += ['inst_'+name for name in INSTRUCTION_CLASSES]
    columns += metrics
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=columns)
    writer.writeheader()
    for row in result['rows']:
        value={name:row.get(name,'') for name in columns[:9]}
        for name in ('aliases','logical_variants'):
            value[name]=json.dumps(value[name],separators=(',',':'))
        value.update({name:row['resources'].get(name,'') for name in ('n_regs','local_bytes','occupancy_capacity_q')})
        value['total_dynamic_instructions']=row.get('total_dynamic_instructions','')
        value.update({'inst_'+name:row.get('instruction_classes',{}).get(name,'') for name in INSTRUCTION_CLASSES})
        value.update({name:row['selected_metrics'].get(name,{}).get('value','') for name in metrics})
        writer.writerow(value)
    return stream.getvalue().encode()


def metric_csv_bytes(result):
    fields=['group','case','tag','mode','CTAs','metric','unit','value_text','value','value_per_CTA','per_CTA_unit']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in result['rows']:
        for name,item in row['selected_metrics'].items():
            normalized=row['selected_additive_counts_per_CTA'].get(name,{})
            writer.writerow({**{key:row[key] for key in fields[:5]},'metric':name,
                **{key:item[key] for key in ['unit','value_text','value']},
                'value_per_CTA':normalized.get('value',''),'per_CTA_unit':normalized.get('unit','')})
    return stream.getvalue().encode()


def render(result, protocol):
    lines=['# Stage N3 — 静态预选布局对照的 NCU 证据','',
        f'覆盖 {len(protocol["profile_groups"])} 个 kernel 子类型；{result["report_count"]} 份实际报告，alias 展开后 {result["logical_report_count"]} 个逻辑报告。所有对照由布局机制静态预选，未按候选耗时挑选。','',
        f'协议 SHA256：`{result["protocol_SHA256"]}`。Payload SHA256：`{result["payload_SHA256"]}`。','',
        '## 1. 口径','',
        '- `software`：保持原始尺寸、stride、标量及二进制，grid 改为一个 CTA，读取 InstructionStats/SASS 动态执行数；检查该 CTA 的完整输出、其余 inactive bytes、输入和 canaries。',
        '- `warm_application`：完整 grid，application replay / cache none，八次同程序 warmup。',
        '- 如存在 `cold_isolated`：完整 grid，kernel replay / cache all；与 warm_application 分开解释。',
        '- 每次 warmup 和 target 前恢复原地输入，恢复位于 profiler 范围之外；不导入 Triton，不执行 JIT。',
        '- 软件计数的单位是动态 warp instruction instances；一次指令并不等同于一次 shared/global 事务。硬件 additive counter 可按实际 CTA 数归一；百分比、比值及时间不按 CTA 数相除。',
        '- 单次 NCU 采集没有计时置信区间；profiler 下的时间不能替代无 profiler 的正式计时，也不把指令数加权成成本分数。','',
        '## 2. U / G0 动态计数交叉验证','',
        '| Case | 分别采集 | Opcode counts 相同 | 指令类别相同 | U 动态指令 | G0 动态指令 |',
        '|---|---|---|---|---:|---:|']
    for row in result['baseline_software_crosschecks']:
        lines.append(f'| {row["case"]} | {row["independently_profiled"]} | {row["opcode_counts_equal"]} | {row["instruction_classes_equal"]} | {row["U_total_dynamic_instructions"]} | {row["G0_total_dynamic_instructions"]} |')
    lines += ['', '## 3. CTA0 的软件动态指令','',
        '| Case | Variant（含 alias） | 总指令 | LDG | STG | LDS/LDSM | STS/STSM | SHFL | BAR | LDL+STL |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        if row['mode']!='software':continue
        counts=row['instruction_classes']
        lines.append(f'| {row["case"]} | {", ".join(row["logical_variants"])} | {row["total_dynamic_instructions"]} | {counts["global_load"]} | {counts["global_store"]} | {counts["shared_load"]} | {counts["shared_store"]} | {counts["shuffle"]} | {counts["barrier"]} | {counts["local_load"]+counts["local_store"]} |')
    lines += ['', '## 4. 完整 grid 的请求与共享内存服务','',
        '下表仅展示所选 additive counter 的每 CTA 值。完整 raw 名称、单位、数值、stall 百分比、DRAM/L2 利用率及资源数据保留于 `summary.json`、`summary.csv`、`metrics.csv`。这些列不做延迟加权。','',
        '| Case | Variant（含 alias） | Mode | Global LD sectors/CTA | Global ST sectors/CTA | Shared LD WF/CTA | Shared LD conflicts/CTA | n_regs | q |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    names=['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']
    for row in result['rows']:
        if row['mode']=='software':continue
        values=[row['selected_additive_counts_per_CTA'].get(name,{}).get('value') for name in names]
        formatted=['—' if value is None else f'{value:.3f}' for value in values]
        lines.append(f'| {row["case"]} | {", ".join(row["logical_variants"])} | {row["mode"]} | '+
            ' | '.join(formatted)+f' | {row["resources"].get("n_regs","—")} | {row["resources"].get("occupancy_capacity_q","—")} |')
    lines += ['', '## 5. 静态对照的适用范围','']
    for group in protocol['profile_groups']:
        lines.append(f'- `{group["id"]}` / `{group["case"]["id"]}`：{group["static_rationale"]}')
    lines += ['', '原始 `.ncu-rep`、raw/source 导出、NCU 版本、绑定与正确性记录均按 SHA 保留。U/G0 的软件计数一致与否如实展示；任何不一致需要单独调查，不在分析器中改写。对完整框架、LLM 端到端收益、跨设备泛化或版本回归均不作结论。','']
    return '\n'.join(lines)


def run(protocol_path, profiles, output, payload_path=None, attempts=None):
    protocol_path,output=Path(protocol_path),Path(output)
    protocol=c.read(protocol_path)
    payload_path=Path(payload_path) if payload_path else Path(protocol['payload_path'])
    result=analyze(protocol,profiles,payload_path,attempts)
    result['analysis_source']=dict(path=str(Path(__file__)),SHA256=c.sha(Path(__file__).read_bytes()))
    result['protocol_path']=str(protocol_path)
    c.write_once(output/'summary.json',result)
    c.write_once(output/'summary.csv',csv_bytes(result,protocol['metrics']))
    c.write_once(output/'metrics.csv',metric_csv_bytes(result))
    c.write_once(output/'REPORT.md',render(result,protocol).encode())
    print('N3 CPU analysis',result['report_count'],'physical reports; no GPU allocation')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--payload',type=Path)
    parser.add_argument('--attempt-map',type=Path)
    args=parser.parse_args()
    attempts=c.read(args.attempt_map) if args.attempt_map else None
    run(args.protocol,args.profiles,args.output,args.payload,attempts)


if __name__=='__main__':main()
