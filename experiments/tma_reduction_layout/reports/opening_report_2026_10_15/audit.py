"""Read-only CPU audit of selection rules and existing experimental evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parents[1]
ATTACHMENT = Path('/home/chenmiaoming/.codex/attachments/5644614f-d2ee-420a-ae92-40e91f4e39bc/tma_reduction_layout_opening_report.pptx')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def blocked(shape, size, order, warps):
    # Exact allocation order in BlockedEncodingAttr's shape-based builder,
    # specialized to one CTA and 32 lanes. Not a latency/cost model.
    lanes, cta = [0] * len(shape), [0] * len(shape)
    rem_lanes, rem_threads, rem_warps = 32, warps * 32, warps
    prev_lanes = prev_warps = 1
    for dim in order[:-1]:
        threads = min(max(rem_threads, 1), max(1, shape[dim] // size[dim]))
        lanes[dim] = min(max(threads, 1), rem_lanes)
        cta[dim] = min(max(threads // lanes[dim], 1), rem_warps)
        rem_warps //= cta[dim]
        rem_lanes //= lanes[dim]
        rem_threads //= threads
        prev_lanes *= lanes[dim]
        prev_warps *= cta[dim]
    lanes[order[-1]], cta[order[-1]] = 32 // prev_lanes, warps // prev_warps
    return {'sizePerThread': size, 'threadsPerWarp': lanes, 'warpsPerCTA': cta, 'order': order}


def compute():
    protected = {}
    for meta in ('source_audit/local_sources.json', 'source_audit/observed_main.json'):
        protected[meta] = sha(ROOT / meta)
    local = read(ROOT / 'source_audit/local_sources.json')
    live = read(ROOT / 'source_audit/observed_main.json')
    comparisons = {}
    for path, record in live['files'].items():
        assert sha(ROOT / 'source_audit/observed_main' / path) == record['SHA256']
        comparisons[path] = (ROOT / 'source_audit/observed_main' / path).read_bytes() == (
            ROOT / 'source_audit/experiment_upstream' / path).read_bytes()
    for label, record in local.items():
        for path, expected in record['files'].items():
            assert sha(ROOT / 'source_audit' / label / path) == expected
    # These five files contain the current default-register-selection rules,
    # propagation and the limited thread-locality transformation.
    for name in ('Coalesce.cpp', 'CoalesceUtils.cpp', 'Utility.cpp',
                 'OptimizeThreadLocality.cpp', 'RemoveLayoutConversions.cpp'):
        assert comparisons['lib/Dialect/TritonGPU/Transforms/' + name]
    source = ROOT / 'source_audit/observed_main'
    coalesce = (source / 'lib/Dialect/TritonGPU/Transforms/Coalesce.cpp').read_text()
    assert 'int maxVectorSize = 128 / type.getElementTypeBitWidth();' in coalesce
    assert 'int vectorSize = std::min(numElemsPerThread, maxVectorSize);' in coalesce
    utility = (source / 'lib/Dialect/TritonGPU/Transforms/Utility.cpp').read_text()
    assert 'unsigned alignment = std::min(maxMultiple, maxContig);' in utility
    locality = (source / 'lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp').read_text()
    assert 'if (reduce.getAxis() != rank - 1)' in locality

    input_dir = ROOT / 'input'
    input_dir.mkdir(exist_ok=True)
    draft = input_dir / 'followup_review_draft.md'
    if not draft.exists():
        shutil.copyfile(EXP / 'phase11/FOLLOWUP_PLAN.md', draft)
    files = [
        EXP / 'results/local_a10/20261015_gpu0_r2/analysis.json',
        EXP / 'results/local_a10_versions/20261015_gpu0_r2/analysis.json',
        EXP / 'results/phase11/stage_b/interpretation.md',
        EXP / 'results/phase11/stage_b/summary.md',
        draft,
    ]
    for version in ('baseline', 'patch'):
        files.append(EXP / 'results/phase10/stage_c/artifacts/sm90' / version /
                     'export/M32_N128_w4/bfloat16/max/reduce/host' / version / 'kernel.ptx')
    for bn, warps in ((128, 4), (64, 8)):
        files.append(EXP / 'results/phase11/stage_b/artifacts/attempt_4/export' /
                     f'M32_N{bn}_w{warps}/bfloat16/max/host/upstream_default/kernel.ttgir')
    evidence = {str(path.relative_to(EXP)): sha(path) for path in files}
    a10 = read(files[0])
    releases = read(files[1])
    assert a10['status'] == releases['status'] == 'PASS'
    cases = []
    for bn, warps in ((128, 4), (64, 8)):
        extent = (32 * bn) // (32 * warps)
        tma_vec = min(max(extent, 1), 128 // 16)
        # For this contiguous, pointer-aligned BF16 tensor, the address analysis
        # permits at least eight adjacent elements. No generic-mask claim.
        ordinary_vec = min(min(16 // 2, bn, 128 // 16), max(extent, 1))
        assert tma_vec == ordinary_vec == 8
        layouts = {label: blocked([1, 32, bn], [1, 1, 8], order, warps)
                   for label, order in (('main', [2, 1, 0]), ('release_3_5_3_6', [2, 0, 1]))}
        for version in ('3.5.1', '3.6.0', '3.7.1'):
            actual = releases['artifacts'][f'{version}:M32_N{bn}_w{warps}']['layouts']
            pred = layouts['main' if version == '3.7.1' else 'release_3_5_3_6']
            assert actual['global_load_layout'] == pred
            assert actual['reduction_input_layout'] == pred
        tma_path = next(path for path in files if path.suffix == '.ttgir' and f'M32_N{bn}_w{warps}' in path.parts)
        tma_text = tma_path.read_text()
        alias = re.search(r'#blocked = #ttg.blocked<\{([^}]+)\}>', tma_text)[1]
        tma_actual = {key: [int(x.strip()) for x in re.search(key + r' = \[([^]]+)\]', alias)[1].split(',')]
                      for key in ('sizePerThread', 'threadsPerWarp', 'warpsPerCTA', 'order')}
        assert tma_actual == layouts['main']
        assert f'-> tensor<1x32x{bn}xbf16, #blocked>' in tma_text and 'ttg.local_load' in tma_text
        timing = {suffix: a10['timing'][f'M32_N{bn}_w{warps}:{suffix}']['median_of_worker_medians_us']
                  for suffix in ('canonical', 'direct:vec1', 'direct:vec2', 'direct:vec4', 'direct:vec8')}
        candidates = {key: value for key, value in timing.items() if key.startswith('direct:')}
        cases.append({'BM': 32, 'BN': bn, 'warps': warps, 'logical_elements_per_thread': extent,
                      'TMA_default_vec': tma_vec, 'ordinary_default_vec': ordinary_vec,
                      'layouts': layouts,
                      'H100_TMA_actual_input_layout': tma_actual,
                      'column_contributors': layouts['main']['threadsPerWarp'][1] * layouts['main']['warpsPerCTA'][1],
                      'A10_same_cohort_us': timing,
                      'Gluon_vec1_vs_vec8_time_reduction': 1 - timing['direct:vec1'] / timing['direct:vec8'],
                      'default_vs_fastest_measured_Gluon_time_ratio_descriptive': timing['canonical'] / min(candidates.values())})
    conversion = {}
    for version in ('baseline', 'patch'):
        path = next(p for p in files if p.suffix == '.ptx' and version in p.parts)
        lines = path.read_text().splitlines()
        max_lines = [i + 1 for i, line in enumerate(lines) if 'max.bf16x2' in line]
        cvt_lines = [i + 1 for i, line in enumerate(lines) if 'cvt.f32.bf16' in line]
        assert min(max_lines) < min(cvt_lines)
        assert len(cvt_lines) == (8 if version == 'baseline' else 2)
        conversion[version] = {'packed_max_lines': max_lines, 'conversion_lines': cvt_lines}
    original = ROOT / 'input/original_opening_report.pptx'
    if not original.exists():
        original.parent.mkdir(exist_ok=True)
        shutil.copyfile(ATTACHMENT, original)
    if ATTACHMENT.exists():
        assert sha(original) == sha(ATTACHMENT)
    return {'status': 'PASS', 'input_attachment_SHA256': sha(original),
            'source_versions': {**{label: record['commit'] for label, record in local.items()},
                                'observed_main': live['commit']},
            'main_source_comparison': comparisons, 'protected_source_metadata': protected,
            'existing_evidence_SHA256': evidence, 'cases': cases, 'packed_max_audit': conversion,
            'scope': 'CPU source reconstruction checked against existing final TTGIR; not an instrumented compiler trace',
            'new_GPU_experiments': 0, 'native_builds': 0, 'new_main_performance_tested': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    result = compute()
    path = ROOT / 'audit.json'
    if args.validate:
        assert read(path) == result
    else:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('PASS: archived source hashes, eight default layouts, paired-cohort data and packed-max order')
