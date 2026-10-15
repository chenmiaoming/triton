"""Offline, source-bound validation and release comparison; no GPU imports."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tarfile

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
VERSIONS = ('3.5.1', '3.6.0', '3.7.1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def layout_evidence(folder, bn):
    text = (folder / 'kernel.ttgir').read_text()
    aliases = {}
    for name, fields in re.findall(r'(#[\w]+) = #ttg.blocked<\{([^}]+)\}>', text):
        aliases[name] = {key: [int(v.strip()) for v in re.search(key + r' = \[([^]]+)\]', fields)[1].split(',')]
                         for key in ('sizePerThread', 'threadsPerWarp', 'warpsPerCTA', 'order')}
    lines = text.splitlines()
    loads = [(i + 1, line) for i, line in enumerate(lines) if re.search(r'= tt.load\s', line)]
    reductions = [(i + 1, '\n'.join(lines[i:i + 12])) for i, line in enumerate(lines) if '= "tt.reduce"(' in line]
    assert len(loads) == len(reductions) == 1
    load_alias = re.search(r'tensor<1x32x' + str(bn) + r'x!tt.ptr<bf16>,\s*(#\w+)>', loads[0][1])[1]
    reduce_alias = re.search(r'tensor<1x32x' + str(bn) + r'xf32,\s*(#\w+)>', reductions[0][1])[1]
    load, reduction = aliases[load_alias], aliases[reduce_alias]
    assert 'axis = 1 : i32' in reductions[0][1]
    contributors = reduction['threadsPerWarp'][1] * reduction['warpsPerCTA'][1]
    base_m = reduction['sizePerThread'][1] * contributors
    return {'global_load_layout': load, 'reduction_input_layout': reduction,
            'load_line': loads[0][0], 'load_operation': loads[0][1],
            'reduce_line': reductions[0][0], 'reduce_operation': reductions[0][1],
            'column_contributor_threads': contributors,
            'column_contributor_warps': reduction['warpsPerCTA'][1],
            'column_values_per_contributor': 32 // contributors, 'base_tile_M': base_m,
            'repeats_in_M': 32 // base_m}


def sass_evidence(folder):
    text = (folder / 'kernel.sass').read_text()
    instructions = []
    for line in text.splitlines():
        match = re.search(r'/\*[0-9a-f]+\*/\s+((?:@!?U?P\d+\s+)?[A-Z][A-Z0-9_.]*\s.*?;)', line)
        if match:
            instructions.append(match[1])
    assert instructions
    opcodes = Counter(re.match(r'(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)', line)[1] for line in instructions)
    assert any(op.startswith('LDG.') for op in opcodes)
    assert not any(op.startswith(('LDGSTS', 'UTMA', 'TMA')) for op in opcodes)
    return {'instruction_text_SHA256': hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),
            'static_sites_only': True, 'opcodes': dict(sorted(opcodes.items()))}


def contrast(old, new):
    assert len(old) == len(new) == 7
    logs = [math.log(b / a) for a, b in zip(old, new)]
    center = statistics.mean(logs)
    half = 2.4469118511449692 * statistics.stdev(logs) / math.sqrt(7)
    interval = [math.exp(center - half), math.exp(center + half)]
    decision = ('detected_above_3_percent' if interval[0] > 1.03 else
                'excludes_3_percent_slowdown_for_this_workload' if interval[1] < 1.03 else 'inconclusive')
    return {'paired_process_ratios': [b / a for a, b in zip(old, new)],
            'geomean_time_ratio': math.exp(center), 'CI95': interval, 'decision': decision,
            'unit': 'temporal process repeats on one A10; not independent GPUs'}


def analyze(root):
    protocol = read(root / 'protocol.json')
    assert protocol['status'] == 'FROZEN_BEFORE_COMPILATION_AND_TIMING'
    assert not protocol['manual_layouts'] and not protocol['TMA'] and not protocol['Phase11_C_started']
    for name, expected in protocol['source_SHA256'].items():
        assert sha(root / 'source' / name) == sha(SOURCE / name) == expected, name
    for name, expected in protocol['dependency_SHA256'].items():
        assert sha(root / 'source' / name) == sha(SOURCE.parent / name) == expected, name
    assert sha(root / 'toolchains.json') == protocol['toolchains_SHA256']
    def canonical_ast(path):
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'canonical')
        return ast.dump(node, include_attributes=False)
    assert canonical_ast(SOURCE / 'kernels.py') == canonical_ast(SOURCE.parent / 'local_a10/kernels.py')
    artifacts, compilers = {}, {}
    correctness = 0
    common_environment = None
    for version in VERSIONS:
        collected = read(root / 'collect' / version / '1/raw.json')
        assert collected['status'] == 'OK' and not collected['errors']
        assert collected['protocol_SHA256'] == sha(root / 'protocol.json')
        assert len(collected['artifacts']) == 2 and len(collected['correctness']) == 4
        assert all(row['pass'] and row['max_abs_error'] == 0 and row['elements'] == 256 * 8192
                   for row in collected['correctness'])
        assert set(collected['correctness_bindings']) == set(collected['artifacts'])
        for tag, binding in collected['correctness_bindings'].items():
            assert binding['CUBIN_SHA256'] == binding['loaded_SHA256'] == collected['artifacts'][tag]['CUBIN_SHA256']
            assert binding['launches'] == binding['sha_guards'] == 2
        assert collected['environment']['gpu_uuid'] == protocol['GPU_UUID']
        assert collected['CUDA_return_codes'] == [0]
        current = {key: collected['environment'][key] for key in ('torch', 'torch_cuda', 'CC', 'gpu_uuid')}
        if common_environment is None:
            common_environment = current
        assert current == common_environment
        compiler = collected['compiler']
        assert compiler['version'] == version and compiler['native_builds'] == 0
        assert compiler['ptxas_SHA256'] == protocol['common_ptxas']['SHA256']
        compilers[version] = compiler
        correctness += 4
        for tag, meta in collected['artifacts'].items():
            folder = root / meta['archive_path']
            assert sha(folder / 'kernel.cubin') == meta['CUBIN_SHA256']
            assert read(folder / 'metadata.json') == meta
            assert meta['num_spills'] == meta['global_scratch_size'] == 0
            ptx = (folder / 'kernel.ptx').read_text()
            assert '.target sm_86' in ptx and 'cp.async' not in ptx
            artifacts[tag] = {**meta, 'layouts': layout_evidence(folder, meta['case']['BN']),
                              'SASS': sass_evidence(folder)}
    assert len(artifacts) == 6 and correctness == 12
    medians = defaultdict(list)
    total_samples = 0
    for invocation in range(7):
        raw = read(root / 'timing' / str(invocation) / 'raw.json')
        assert raw['status'] == 'OK' and not raw['Triton_imported'] and raw['compile_calls'] == 0
        assert raw['protocol_SHA256'] == sha(root / 'protocol.json') and raw['CUDA_return_codes'] == [0]
        assert {key: raw['environment'][key] for key in common_environment} == common_environment
        assert raw['compiler_native_SHA256'] == {v: c['native_SHA256'] for v, c in compilers.items()}
        assert len(raw['correctness']) == 6 and all(row['pass'] for row in raw['correctness'])
        assert len(raw['visits']) == 72
        samples = defaultdict(list)
        selected = sorted(artifacts)
        for round_index in range(12):
            shift = (round_index + invocation) % 6
            order = selected[shift:] + selected[:shift]
            if (round_index + invocation) % 2:
                order = order[::-1]
            visits = raw['visits'][round_index * 6:(round_index + 1) * 6]
            assert [row['tag'] for row in visits] == order
            assert all(row['round'] == round_index for row in visits)
        for visit in raw['visits']:
            assert len(visit['samples_us']) == 5
            assert all(math.isfinite(x) and x > 0 for x in visit['samples_us'])
            samples[visit['tag']].extend(visit['samples_us'])
        assert set(samples) == set(artifacts)
        for tag, values in samples.items():
            assert len(values) == 60
            medians[tag].append(statistics.median(values))
            total_samples += len(values)
            binding = raw['bindings'][tag]
            assert binding['CUBIN_SHA256'] == binding['loaded_SHA256'] == artifacts[tag]['CUBIN_SHA256']
            assert binding['launches'] == binding['sha_guards'] == 66
            assert len(binding['ABI']) in (2, 3, 4)
        assert raw['CUDA_call_counts']['cuLaunchKernel'] == 396
    assert total_samples == 2520
    timing = {tag: {'process_medians_us': values, 'median_of_process_medians_us': statistics.median(values)}
              for tag, values in sorted(medians.items())}
    comparisons = {}
    for bn, w in ((128, 4), (64, 8)):
        shape = f'M32_N{bn}_w{w}'
        old = medians[f'3.5.1:{shape}']
        comparisons[shape] = {version: contrast(old, medians[f'{version}:{shape}']) for version in VERSIONS[1:]}
    return {'status': 'PASS', 'protocol_SHA256': sha(root / 'protocol.json'), 'analyzer_SHA256': sha(Path(__file__)),
            'environment': common_environment, 'artifacts': artifacts, 'compilers': compilers,
            'compile_correctness_checks': correctness, 'timing_correctness_checks': 42,
            'timing_samples': total_samples, 'timing': timing, 'comparisons': comparisons,
            'native_builds': 0, 'manual_layouts': False, 'TMA_executed': False, 'Phase11_C_started': False}


def summary(result):
    lines = ['# A10 普通 Triton 默认布局的版本对比', '',
             '仅物理卡 0，输入 BF16 `[1,8192,8192]`，每个 tile 的 M=32，沿 axis=1 取 max，输出 FP32。',
             '同一普通 `tl.load` kernel，AST 与上一轮 canonical anchor 相同；无 Gluon、手动布局或 descriptor。',
             '3.5.1 / 3.6.0 官方二进制 wheel 与已安装 3.7.1 分别在独立进程编译，',
             '使用同一 Python/Torch 和同一个 ptxas；没有重建 Triton。实际版本及 SHA 见 toolchains.json。', '',
             '## 编译器实际选择', '',
             '去掉大小为 1 的 batch 维，下面数组的次序均为 `[M,N]`；order 为 `[N,M]`。',
             '同时读取最终 TTGIR 中 global-load 和 reduction 输入的 encoding，不由 num_warps 或性能倒推。', '',
             '| tile / warps | Triton | load SPT / TPW / WPC | reduction SPT / TPW / WPC | 每列贡献线程 / warp |',
             '|---|---|---|---|---|']
    def arrays(layout):
        return ' / '.join(str(layout[k][1:]) for k in ('sizePerThread', 'threadsPerWarp', 'warpsPerCTA'))
    for bn, w in ((128, 4), (64, 8)):
        for version in VERSIONS:
            a = result['artifacts'][f'{version}:M32_N{bn}_w{w}']['layouts']
            lines.append(f'| 32×{bn} / {w} | {version} | {arrays(a["global_load_layout"])} | '
                         f'{arrays(a["reduction_input_layout"])} | {a["column_contributor_threads"]} / '
                         f'{a["column_contributor_warps"]} |')
    lines += ['', '这些数字描述元素归属和规约通信需求；sizePerThread 的 N 分量不等于硬件访存次数。',
              '完整 TTGIR、LLVM IR、PTX、CUBIN、SASS 与资源信息均保留，SASS 数量仅是静态指令位置。', '',
              '## 完整 kernel 时间', '',
              '冻结 CUBIN 后，统一 driver-only harness 配对计时：7 个顺序进程、每项 60 个单 kernel event 样本，',
              '共 2,520 个样本。每次只执行一次 kernel，不用 graph 或批次时间除以次数。',
              '下表为 7 个进程中位数的中位数，单位微秒。', '',
              '| tile / warps | 3.5.1 | 3.6.0 | 3.7.1 |', '|---|---:|---:|---:|']
    for bn, w in ((128, 4), (64, 8)):
        values = [result['timing'][f'{v}:M32_N{bn}_w{w}']['median_of_process_medians_us'] for v in VERSIONS]
        lines.append(f'| 32×{bn} / {w} | ' + ' | '.join(f'{x:.3f}' for x in values) + ' |')
    lines += ['', '比值以同一进程内的中位数配对，取 log-ratio 的均值和 95% t 区间（df=6）。',
              '3% 实用回归界限在编译和 timing 前冻结；不是跨设备置信区间，也不证明所有 shape 无回归。', '',
              '| tile / warps | 比较 | 时间比 | 95% CI | 判定 |', '|---|---|---:|---|---|']
    for shape, comparisons in result['comparisons'].items():
        for version, c in comparisons.items():
            lines.append(f'| {shape} | {version} / 3.5.1 | {c["geomean_time_ratio"]:.5f} | '
                         f'[{c["CI95"][0]:.5f}, {c["CI95"][1]:.5f}] | {c["decision"]} |')
    lines += ['', '## 边界与证据', '',
              '12 项编译阶段完整输出检查和 42 项冻结 CUBIN 的完整输出检查通过，覆盖两个有限输入模式。',
              '每次 launch 前核对 CUBIN SHA，所有 timing 进程不导入 Triton、不调用 JIT。',
              '原始样本、源代码快照、编译产物均 write-once；raw_manifest.json 固定原始文件哈希。',
              '此轮直接回答普通 load 的默认布局及版本变化，不比较手工 vec2/vec8 优劣。',
              'A10 没有运行 TMA；此结果不能代替原 issue 的 TMA 路径或其他 GPU 的版本验证。',
              'Phase 11 C/model 仍未启动，未创建 PR。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--committed', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    result = analyze(root)
    report = summary(result)
    if args.validate:
        assert read(root / 'analysis.json') == result
        assert (root / 'summary.md').read_text() == report
        manifest = read(root / 'raw_manifest.json')
        for name, expected in manifest['files'].items():
            assert sha(root / name) == expected, name
        if args.committed:
            names = ['experiments/tma_reduction_layout/local_a10_versions', str(root.parent.relative_to(REPO))]
            archive = subprocess.check_output(['git', 'archive', 'HEAD', *names], cwd=REPO)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for name, expected in manifest['files'].items():
                    member = (root / name).resolve().relative_to(REPO).as_posix()
                    assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected, member
                for path in list(SOURCE.glob('*.py')) + [root / 'analysis.json', root / 'summary.md', root / 'raw_manifest.json']:
                    assert hashlib.sha256(tar.extractfile(path.relative_to(REPO).as_posix()).read()).hexdigest() == sha(path)
    else:
        (root / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        (root / 'summary.md').write_text(report)
    print('PASS: six default-layout binaries, 12+42 correctness checks, 2520 paired event samples; GPU 0 only')


if __name__ == '__main__':
    main()
