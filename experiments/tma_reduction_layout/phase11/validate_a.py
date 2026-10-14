"""Offline NCU import/correctness/identity validation, with explicit coverage."""
import argparse
import csv
import io
import json
import math
import re
from experiments.tma_reduction_layout.phase11 import common as c


def csv_metrics(text):
    rows = list(csv.reader(io.StringIO(text)))
    headers = [i for i,r in enumerate(rows) if 'Kernel Name' in r]
    assert headers, 'NCU kernel row absent'
    h = headers[-1]
    head = rows[h]
    data = [dict(zip(head,r)) for r in rows[h+1:] if len(r)==len(head)]
    data = [r for r in data if r.get('Kernel Name')]
    if 'Metric Name' in head:
        return {r['Metric Name']:r['Metric Value'] for r in data}
    assert len(data)==1, len(data)
    return data[0]


def source_loads(text):
    rows = list(csv.reader(io.StringIO(text)))
    h = next(i for i,r in enumerate(rows) if 'Source' in r and 'Instructions Executed' in r)
    head = rows[h]
    found = []
    for row in rows[h+1:]:
        if len(row) != len(head):
            continue
        rec = dict(zip(head,row))
        if re.search(r'\bLDS(?:M)?(?:\.|\s)',rec['Source']):
            found.append({'PC':rec.get('Address'), 'instruction':rec['Source'].strip(),
                          'dynamic_warp_instructions':int(rec['Instructions Executed'].replace(',',''))})
    assert found, 'Dynamic shared-load source count absent'
    return found


def derive():
    c.protect_prior()
    rows = {}
    failure_count = 0
    for target in ('sm90','sm100','sm120'):
        attempts = sorted((c.OUT/'stage_a/probes'/target).glob('attempt_*'))
        failures = []
        accepted = None
        for attempt in attempts:
            raw = c.read(attempt/'export/result.json')
            if raw['status']=='PROBE_RETURNED':
                assert accepted is None, 'Do not select favorable repeated calibration'
                accepted = (attempt,raw)
            else:
                failures.append({'attempt':attempt.name,'call_id':raw['call_id'],'error':raw.get('error')})
        failure_count += len(failures)
        assert accepted is not None, target
        root, raw = accepted
        env = raw['environment']
        assert env['cc']=={'sm90':[9,0],'sm100':[10,0],'sm120':[12,0]}[target]
        assert env['gpu_uuid'].startswith('GPU-') and raw['formal_timing_samples'] is False
        assert raw['native_rebuild'] is False and env['native_compiler_loaded'] is False
        fixtures = {}
        for kind in ('ordinary_contiguous','ordinary_conflict32','matrix_x4'):
            for group in ('hardware','software'):
                collection = raw['commands'][kind+'_'+group+'_collect']
                assert collection['returncode']==0, (target,kind,group,collection['output'][-1500:])
                bindings = [line[len('FIXTURE_BINDING '):] for line in collection['output'].splitlines()
                            if line.startswith('FIXTURE_BINDING ')]
                assert len(bindings)==1
                binding = json.loads(bindings[0])
                assert binding['correctness'] and binding['environment']['gpu_uuid']==env['gpu_uuid']
                assert binding['CUBIN_SHA256']==c.sha((root/'export'/(kind+'.cubin')).read_bytes())
                assert binding['grid']==128 and binding['threads']==32
                assert (root/'export'/(kind+'_'+group+'.ncu-rep')).stat().st_size>0
            metrics = csv_metrics(raw['commands'][kind+'_hardware_import']['output'])
            def count(name):
                if name not in metrics or metrics[name] in ('N/A','nan',''):
                    return None
                value = float(metrics[name].replace(',',''))
                assert math.isfinite(value)
                return value
            loads = source_loads(raw['commands'][kind+'_software_source']['output'])
            dynamic = sum(l['dynamic_warp_instructions'] for l in loads)
            assert dynamic==128, (target,kind,dynamic)
            fixtures[kind] = {
                'dynamic_shared_load_warp_instructions':dynamic,'source_loads':loads,
                'wavefronts':count('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'),
                'bank_conflicts':count('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
                'hardware_requests':count('l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum'),
                'hardware_requests_status':'UNAVAILABLE' if 'l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum' not in metrics else 'AVAILABLE',
                'correctness':True}
        clean,bad,matrix = [fixtures[k] for k in ('ordinary_contiguous','ordinary_conflict32','matrix_x4')]
        assert clean['wavefronts']==128 and clean['bank_conflicts']==0
        assert bad['wavefronts']==4096 and bad['bank_conflicts']==3968
        matrix_valid = matrix['wavefronts'] is not None and matrix['wavefronts']>0
        rows[target] = {'environment':env,'NCU_version':raw['commands']['ncu_version']['output'],
                        'metric_mapping':raw['metric_mapping'],'accepted_attempt':root.name,
                        'retained_failures':failures,'fixtures':fixtures,
                        'ordinary_counter_coverage':'VERIFIED',
                        'matrix_wavefront_coverage':'VERIFIED' if matrix_valid else 'UNVERIFIED',
                        'matrix_bank_conflict_positive_control':'NOT_TESTED',
                        'request_normalization':'Use measured per-PC warp instructions; direct hardware request metric is unavailable when marked so. Do not substitute vec arithmetic.',
                        'NCU_report_UUID_missing_does_not_imply_unavailable':True}
    archived = {}
    request_path = c.OUT/'stage_a/archived_profiles/request.json'
    if request_path.exists():
        request = c.read(request_path)
        for key,source in request['original_sources'].items():
            assert c.sha((c.ROOT/source['metadata_path']).read_bytes())==source['metadata_SHA256'],key
        for target in request['targets']:
            root = c.OUT/'stage_a/archived_profiles'/target/'export'
            original = c.read(root/'result.json')
            assert original['status']=='OK' and original['formal_timing_sample'] is False
            assert original['request_SHA256']==c.sha(c.encode(request))
            actual = {}
            for tag,meta in request['metadata'][target].items():
                for group in ('hardware','software'):
                    folder = root/tag.replace(':','_')/'repeat_1'/group
                    collected = c.read(folder/'collect.json')
                    assert collected['returncode']==0
                    line = next(l for l in collected['output'].splitlines() if l.startswith('PROFILE_BINDING '))
                    binding = json.loads(line[len('PROFILE_BINDING '):])
                    assert binding['CUBIN_SHA256']==meta['cubin_SHA256'] and binding['correctness_prefix128']
                    assert binding['environment']['gpu_uuid']==original['environment']['gpu_uuid']
                    assert binding['compile_calls']==0 and binding['Triton_imported'] is False
                hardware = csv_metrics(c.read(root/tag.replace(':','_')/'repeat_1/hardware/import.json')['output'])
                loads = source_loads(c.read(root/tag.replace(':','_')/'repeat_1/software/source.json')['output'])
                assert sum(v['dynamic_warp_instructions'] for v in loads)>0
                actual[tag] = {'CUBIN_SHA256':meta['cubin_SHA256'],'dynamic_shared_load_warp_instructions':
                              sum(v['dynamic_warp_instructions'] for v in loads),
                              'wavefronts':float(hardware['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'].replace(',','')),
                              'bank_conflicts':float(hardware['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'].replace(',','')),
                              'source_loads':loads,'scope':'whole-kernel shared accesses, not initial-load-only'}
            archived[target] = {'environment':original['environment'],'binaries':actual}
    return {'stage':'A','status':'COMPLETE','targets':rows,'retained_failed_calls':failure_count,
            'archived_kernel_coverage':archived,
            'native_rebuilds':0,'Triton_compiler_modifications':0,'formal_timing_samples':0,
            'next_authorized_stage':'B H100 layout performance; stop after B'}


def summary(result):
    text = '# Phase 11 Stage A — NCU coverage\n\n'
    text += '工具校准完成；没有正式性能样本，也没有重建或加载Triton native。所有早期源码/工具失败保留。\n\n'
    text += '| Target | 普通LDS动态warp指令 | 无冲突wavefront | 32路冲突wavefront / conflict | LDSM动态warp指令 | LDSM wavefront / conflict |\n|---|---:|---:|---|---:|---|\n'
    for target,row in result['targets'].items():
        a,b,m = [row['fixtures'][k] for k in ('ordinary_contiguous','ordinary_conflict32','matrix_x4')]
        text += f'| {target} | {a["dynamic_shared_load_warp_instructions"]} | {a["wavefronts"]} | {b["wavefronts"]} / {b["bank_conflicts"]} | {m["dynamic_shared_load_warp_instructions"]} | {m["wavefronts"]} / {m["bank_conflicts"]} |\n'
    text += ('\n每个fixture固定128个单warp CTA，输出已验证。LDSM的wavefront覆盖单独检查；'
             '无冲突LDSM不能验证有冲突LDSM的bank计数。直接shared Requests硬件指标未提供，'
             '后续使用NCU实际测得的每PC warp指令数并明确口径，不能用32/vec代替。\n\n'
             'NCU 2024.3不识别SM100/SM120；复用已有CUDA13.1 / NCU2025.4工具镜像修复工具覆盖。'
             '每次失败、工具版本、query-metrics、query-sections、Driver UUID、PCI映射、完整命令、'
             'PTX/CUBIN/SASS、原始ncu-rep及独立report-import均归档。硬件计数与software指令插桩分开采集。\n\n'
             '阶段B继续复用固定upstream的baseline native；工具校准不构成布局性能或回归因果结论。\n')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--committed',action='store_true')
    args = parser.parse_args()
    result = derive()
    assert set(result['archived_kernel_coverage'])=={'sm90','sm100','sm120'}
    root = c.OUT/'stage_a'
    manifest = c.read(root/'raw_manifest.json') if args.validate else None
    if args.validate:
        for name,digest in manifest['files'].items():
            assert c.sha((root/name).read_bytes())==digest,name
        assert c.read(root/'analysis.json')==result
        assert (root/'summary.md').read_text()==summary(result)
    else:
        raw = {name:digest for name,digest in c.inventory(root).items()
               if name not in ('analysis.json','summary.md','raw_manifest.json')}
        c.write_once(root/'raw_manifest.json',{'files':raw})
        c.write_once(root/'analysis.json',result)
        c.write_once(root/'summary.md',summary(result).encode())
    if args.committed:
        import subprocess
        for name,digest in manifest['files'].items():
            relative = (root/name).relative_to(c.ROOT).as_posix()
            blob = subprocess.check_output(['git','show','HEAD:'+relative],cwd=c.ROOT)
            assert c.sha(blob)==digest,relative
    print('Stage A PASS: independent CSV/source counts, UUID/SHA/correctness and immutable prior evidence')


if __name__=='__main__':
    main()
