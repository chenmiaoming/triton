"""Offline validation/derived analysis for the local non-TMA diagnostic."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def opcodes(path):
    return Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)', path.read_text()))


def families(ops):
    tests = {'global_load': lambda op: op == 'LDG' or op.startswith('LDG.'),
             'async_global_to_shared': lambda op: op.startswith('LDGSTS'),
             'shared_load': lambda op: op.startswith('LDS'),
             'shared_store': lambda op: op.startswith('STS'),
             'shuffle': lambda op: op.startswith('SHFL'),
             'barrier': lambda op: op.startswith('BAR.SYNC'),
             'max': lambda op: op.startswith(('HMNMX', 'FMNMX', 'FMAX'))}
    return {name: sum(count for op, count in ops.items() if test(op)) for name, test in tests.items()}


def source_counts(path):
    rows = list(csv.reader(path.open()))
    header = next(i for i, row in enumerate(rows) if row and row[0] == 'Address')
    counts, predicated_threads, sites = Counter(), Counter(), []
    for row in rows[header+1:]:
        if len(row) != len(rows[header]):
            continue
        record = dict(zip(rows[header], row))
        match = re.match(r'\s*(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)', record['Source'])
        if not match:
            continue
        count = record['Instructions Executed']
        if count in ('', '-', 'n/a'):
            continue
        op = match[1]
        count = int(float(count.replace(',', '')))
        threads = int(float(record['Predicated-On Thread Instructions Executed'].replace(',', '')))
        counts[op] += count
        predicated_threads[op] += threads
        if any(families({op: 1}).values()):
            sites.append({'PC': record['Address'], 'instruction': record['Source'],
                          'warp_instructions_ignoring_predicates': count, 'predicated_on_thread_instructions': threads})
    return {'dynamic_warp_instructions_ignoring_predicates': families(counts),
            'predicated_on_thread_instructions': families(predicated_threads), 'sites': sites}


def analyze(root):
    protocol = read(root / 'protocol.json')
    collected = read(root / 'collect/1/raw.json')
    assert collected['status'] == 'OK' and len(collected['artifacts']) == 22
    assert len(collected['correctness']) == 44 and all(row['pass'] for row in collected['correctness'])
    assert collected['environment']['gpu_uuid'] == protocol['GPU_UUID']
    assert collected['CUDA_return_codes'] == [0]
    for name, digest in protocol['source_SHA256'].items():
        assert sha(root / 'source' / name) == digest
        assert sha(Path(__file__).parent / name) == digest
    artifacts = {}
    for tag, meta in collected['artifacts'].items():
        folder = root / meta['archive_path']
        assert sha(folder / 'kernel.cubin') == meta['CUBIN_SHA256']
        assert meta['num_spills'] == 0
        ptx = (folder / 'kernel.ptx').read_text()
        assert '.target sm_86' in ptx and 'cp.async.bulk' not in ptx
        ops = opcodes(folder / 'kernel.sass')
        assert ops and not any('TMA' in op for op in ops)
        artifacts[tag] = {**meta, 'static_SASS_sites': families(ops), 'opcodes': dict(ops)}
        if meta['case']['path'] == 'cp_async':
            assert families(ops)['async_global_to_shared'] > 0 and families(ops)['shared_load'] > 0
        else:
            assert families(ops)['global_load'] > 0 and families(ops)['async_global_to_shared'] == 0
    local = artifacts['M32_N128_w4:direct:vec1']['static_SASS_sites']
    assert local['global_load'] > 0 and local['max'] > 0
    assert all(local[key] == 0 for key in ('shared_load', 'shared_store', 'shuffle', 'barrier'))
    for bn, w in ((128, 4), (64, 8)):
        copies = [meta['static_SASS_sites']['async_global_to_shared'] for meta in artifacts.values()
                  if meta['case']['BN'] == bn and meta['case']['path'] == 'cp_async']
        assert len(copies) == 5 and len(set(copies)) == 1
    medians = defaultdict(list)
    total_samples = 0
    for invocation in range(3):
        raw = read(root / 'timing' / str(invocation) / 'raw.json')
        assert raw['status'] == 'OK' and not raw['Triton_imported'] and raw['compile_calls'] == 0
        assert raw['environment']['gpu_uuid'] == protocol['GPU_UUID'] and raw['CUDA_return_codes'] == [0]
        assert raw['protocol_SHA256'] == sha(root / 'protocol.json')
        samples = defaultdict(list)
        assert len(raw['visits']) == 132
        for visit in raw['visits']:
            assert len(visit['samples_us']) == 5
            samples[visit['tag']].extend(visit['samples_us'])
        assert set(samples) == set(artifacts)
        for tag, values in samples.items():
            assert len(values) == 30 and all(value > 0 for value in values)
            medians[tag].append(statistics.median(values))
            total_samples += len(values)
            binding = raw['bindings'][tag]
            assert binding['CUBIN_SHA256'] == binding['loaded_SHA256'] == artifacts[tag]['CUBIN_SHA256']
            assert binding['launches'] == binding['sha_guards'] == 33
        assert raw['CUDA_call_counts']['cuLaunchKernel'] == 726
    assert total_samples == 1980
    timing = {tag: {'worker_medians_us': values, 'median_of_worker_medians_us': statistics.median(values)}
              for tag, values in sorted(medians.items())}
    profiles = {}
    dispatch = read(root / 'ncu/dispatch_v2.json')
    assert len(dispatch) == 8 and all(row['returncode'] == 0 for row in dispatch)
    for row in dispatch:
        folder = root / 'ncu' / row['identity']
        report = root / row['actual_reports'][0]
        assert sha(report) == row['actual_report_SHA256'][report.name]
        worker = read(root / 'profile_workers' / row['identity'] / 'raw.json')
        assert worker['status'] == 'OK' and not worker['Triton_imported']
        assert worker['environment']['gpu_uuid'] == protocol['GPU_UUID']
        binding = worker['bindings'][row['tag']]
        assert binding['CUBIN_SHA256'] == binding['loaded_SHA256'] == artifacts[row['tag']]['CUBIN_SHA256']
        assert binding['launches'] == binding['sha_guards'] == 4
        record = {'tag': row['tag'], 'kind': row['kind'], 'report_SHA256': sha(report)}
        if row['kind'] == 'hardware':
            table = list(csv.reader((folder / 'raw.csv').open()))
            assert len(table) == 3
            values = dict(zip(table[0], table[2]))
            record['metrics'] = {name: float(value.replace(',', '')) for name, value in values.items()
                                 if name.endswith('.sum') and name.startswith(('l1tex__data_', 'smsp__inst_executed'))}
            assert record['metrics']['smsp__inst_executed.sum'] > 0
        else:
            record.update(source_counts(folder / 'source.csv'))
        profiles[row['identity']] = record
    assert profiles['0_hardware']['metrics']['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'] == 0
    assert profiles['0_source']['dynamic_warp_instructions_ignoring_predicates']['shared_load'] == 0
    assert profiles['0_source']['dynamic_warp_instructions_ignoring_predicates']['global_load'] > 0
    assert profiles['0_source']['dynamic_warp_instructions_ignoring_predicates']['max'] > 0
    return {'status': 'PASS', 'GPU_UUID': protocol['GPU_UUID'], 'compiler': collected['compiler'],
            'analyzer_SHA256': sha(Path(__file__)),
            'artifact_count': 22, 'correctness_checks': 44, 'timing_samples': total_samples,
            'timing_scope': 'one A10, three temporal process repeats, descriptive',
            'artifacts': artifacts, 'timing': timing, 'profiles': profiles,
            'direct_column_local_reduction_without_shared_memory': True,
            'native_builds': 0, 'TMA_executed': False, 'Phase11_C_started': False}


def summary(result):
    text = ['# 本地 A10 非 TMA 诊断', '',
            'GPU：物理卡 0，NVIDIA A10 / SM86。复用已安装 Triton 3.7.1；不是 Phase 11 的固定 native。',
            '22 个 artifact，44 项有限输入完整输出验证，3 次独立进程共 1,980 个单 kernel event 样本。',
            '同一张卡的时间重复，不宣称覆盖多个设备或唯一最优；没有做历史 release 回归 A/B。', '',
            '## 数据路径与直接证据', '',
            '普通 global load 的值直接进入寄存器，max 在寄存器上计算。',
            '32×128/w4 的 direct vec1 没有 LDS、STS、shuffle 或 barrier；SASS 和 NCU 均支持。',
            'direct vec8 的 global load 仍直接进入寄存器，后续 shared 访问来自规约/输出转换，不能叫初始 shared-load。',
            'cp.async 使用 LDGSTS 先填充 shared，再由 LDS/LDSM 读入寄存器；它没有 TMA。',
            '相同几何内五种 cp.async 消费者布局使用固定复制布局，LDGSTS 的静态 opcode/数量相同。', '',
            '## 完整 kernel 时间（微秒）', '',
            '表中为三次进程中位数的中位数，完整逐次结果见 analysis.json；不由此做机制归因。', '',
            '| 几何 / 路径 | vec1 | vec2 | vec4 | vec8 | vec8 within-warp |',
            '|---|---:|---:|---:|---:|---:|']
    for bn, w in ((128, 4), (64, 8)):
        for path in ('direct', 'cp_async'):
            values = [result['timing'][f'M32_N{bn}_w{w}:{path}:{variant}']['median_of_worker_medians_us']
                      for variant in ('vec1', 'vec2', 'vec4', 'vec8', 'vec8_within')]
            text.append(f'| 32×{bn}/w{w}, {path} | ' + ' | '.join(f'{value:.3f}' for value in values) + ' |')
    text += ['', '普通 Triton 的两个 canonical anchor 实际使用 vec8 的输入布局；',
             '它们的时间与代码单独保存，不把跨 Triton/Gluon pipeline 的差异当纯布局因果效果。', '',
             '## NCU', '', '四个 32×128/w4 binary 各有硬件与 source 采集，合计 8 份报告。',
             '硬件指标与软件插桩分开，profiler duration 不进入性能表。下列动态指令计数忽略谓词，',
             '另存 predicated-on thread 计数；不能等同为硬件 requests 或阶段时间。', '',
             '| 路径 / 布局 | 动态 global LDG | 动态 shared-load | shared-load wavefronts |',
             '|---|---:|---:|---:|']
    for index in range(4):
        hw, sw = result['profiles'][f'{index}_hardware'], result['profiles'][f'{index}_source']
        counts = sw['dynamic_warp_instructions_ignoring_predicates']
        wf = hw['metrics']['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']
        text.append(f"| {hw['tag']} | {counts['global_load']} | {counts['shared_load']} | {wf:g} |")
    text += ['', '这里证明了无需 TMA 也会出现依赖布局的规约/重排成本；不证明原 issue 的版本回归同样发生于普通 load。',
             'A10 没有执行 TMA，不能替代 H100/B200 上初始 shared-load 的成本校准。', '',
             '## 修正与证据保留', '',
             '先前 20261015_gpu0 采集器因未初始化 CompiledKernel handles、使用一维 launch tuple 失败；',
             '完整错误、源代码和已导出文件保留，未产生正式性能数据。修正仅涉及采集器，kernel SHA 不变。',
             '成功采集前后 cache CUBIN 文件数均为 22，native SHA 未变；没有重建 Triton。',
             '未安装逐次 cache listener，因此不把文件数量当作精确的逐条 cache-hit 统计。',
             'NCU 2026.3 默认生成 .ncu-repz。首次 dispatch 对旧扩展名的判断错误保留；',
             'dispatch_v2 识别并复用已成功的原报告，没有重跑这次 GPU profile。',
             '所有 raw、原始 compiler artifacts、原始 NCU reports 和失败记录保持 write-once。',
             'Phase 11 C1/C2/model 仍未执行；本轮没有修改 native compiler 或创建 PR。', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--committed', action='store_true')
    args = parser.parse_args()
    result = analyze(args.root)
    report = summary(result)
    if args.validate:
        assert read(args.root / 'analysis.json') == result
        assert (args.root / 'summary.md').read_text() == report
        repo = Path(__file__).resolve().parents[3]
        manifest = read(args.root.parent / 'raw_manifest.json')
        for name, expected in manifest['files'].items():
            assert sha(repo / name) == expected, name
        if args.committed:
            archive = subprocess.check_output(['git', 'archive', 'HEAD',
                'experiments/tma_reduction_layout/local_a10',
                'experiments/tma_reduction_layout/results/local_a10'], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for name, expected in manifest['files'].items():
                    assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected, name
                for path in [Path(__file__), args.root / 'analysis.json', args.root / 'summary.md']:
                    name = path.resolve().relative_to(repo).as_posix()
                    assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == sha(path), name
    else:
        (args.root / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        (args.root / 'summary.md').write_text(report)
    print('PASS: 22 binaries, 44 correctness checks, 1980 timing samples, 8 NCU reports; A10 device 0 only')


if __name__ == '__main__':
    main()
