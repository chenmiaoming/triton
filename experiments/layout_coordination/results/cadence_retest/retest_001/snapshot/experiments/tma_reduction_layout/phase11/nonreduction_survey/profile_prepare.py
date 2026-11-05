"""Freeze static N3 counter contrasts without reading any candidate timings."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import zipfile

from . import common as c


# These are mechanism contrasts, not measured winners.  Keep their names and
# rationale in the frozen protocol even when two logical targets share code.
SELECTIONS = [
    dict(id='N01_scale_swizzle', case='N01_m4096_n128',
        candidates=['output_fully_coalesced', 'coordinated_fully_coalesced'],
        rationale='Compare output-friendly ownership alone with coordinated input/output ownership; isolate the conversion cost versus store request grouping.'),
    dict(id='N02_awq_dequantize', case='N02_k3584_p448_g128',
        candidates=['packed_compute_vec1', 'output_vec4'],
        rationale='Contrast packed-input/expanded-compute ownership with output-only vec4 ownership; inspect load, unpacking, layout conversion and store instructions separately.'),
    dict(id='N03_flash_rope', case='N03_FA_D128',
        candidates=['input_vec1', 'paired_vec4'],
        rationale='Compare input-only vec1 ownership with coordinated input/paired-compute vec4; retain the original rotary arithmetic, output layout and launch configuration.'),
    dict(id='N03_mrope', case='N03_MROPE_D128',
        candidates=['Q_vec1', 'QK_vec4'],
        rationale='Compare Q-only vec1 with coordinated Q/K vec4 input ownership; inspect Q/K handling without changing the in-place rotary operator or trig/output layouts.'),
    dict(id='N04_fp8_pack', case='N04_h32_s4096_d128',
        candidates=['v_input_vec2', 'v_compute_input'],
        rationale='Change V input ownership alone versus keeping V computation in input ownership; inspect BF16 conversion, FP8 quantization, transpose and store grouping. Q/K are fixed.'),
    dict(id='N05_kda_gate', case='N05_t2048_h8_d128',
        candidates=['feature_vec2', 'token_vec2'],
        rationale='Hold vec2 fixed and reverse feature-versus-token ownership of input/compute; output/bias ownership, exact elementwise arithmetic and U-only tuned BT/W/stages remain fixed.'),
    dict(id='N06_kv_cache', case='N06_h32_d128',
        candidates=['e2_heads1', 'e2_heads4'],
        rationale='Hold sizePerThread vec2 fixed and redistribute lanes over one versus four heads; inspect token-major loads and HND-strided paged cache stores.'),
]


def declaration(include_cold_pack=False):
    groups = []
    for spec in SELECTIONS:
        group = copy.deepcopy(spec)
        group['variants'] = ['U', 'G0'] + ['GL_'+name for name in spec['candidates']]
        group['modes'] = ['software', 'warm_application']
        if include_cold_pack and spec['id']=='N04_fp8_pack':
            group['modes'].append('cold_isolated')
            group['cold_rationale'] = 'The large strided QKV input plus FP8 outputs exceed H100 L2 capacity; contrast NCU kernel/cache-all isolation with application/cache-none priming. This is not formal timing.'
        groups.append(group)
    return dict(selection_basis='Static case/subtype and declared layout intervention only; no candidate timing files are read.',
        timing_read_policy='This program accepts only a compiler-artifact payload. It never opens timing directories, analysis.json, timing.csv or performance reports.',
        declared_targets=groups,
        optional_cold_pack=bool(include_cold_pack),
        roles='U and G0 are independently profiled controls. Identical Gluon candidates may alias only after exact full-CUBIN, ABI, runtime specialization and launch-resource checks.')


def _member(metadata, filename):
    path = Path(metadata['artifact_directory'])/filename
    assert not path.is_absolute() and '..' not in path.parts
    return path.as_posix()


def _abi(ptx):
    match = re.search(r'\.visible\s+\.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
    assert match, 'Missing frozen PTX entry ABI'
    return re.findall(r'\.param\s+\.(\w+)', match[1])


def _identity(metadata, abi):
    return (metadata['cubin_SHA256'], tuple(abi), tuple(metadata['runtime_argument_names']),
        c.encode(metadata.get('specialized_runtime_values', {})), metadata['num_warps'],
        metadata['dynamic_smem_bytes'], metadata['global_scratch_size'])


def select(payload, include_cold_pack=False):
    """Read only compiler outputs and return exact selected physical records."""
    payload = Path(payload)
    plan = declaration(include_cold_pack)
    with zipfile.ZipFile(payload) as archive:
        compiled = json.loads(archive.read('result.json'))
        assert compiled['status'] == 'OK'
        assert compiled.get('GPU_allocated') is False and compiled.get('GPU_kernel_launches') == 0, (
            'N3 selection must use the CPU-only candidate compiler payload')
        programs = compiled['programs']
        by_case = {}
        by_variant = {}
        for row in programs:
            case = row['case']
            assert row['geometry'] == case['id']
            by_case[case['id']] = case
            key = (case['id'], row['variant'])
            assert key not in by_variant, key
            by_variant[key] = row
        groups, alias_proofs = [], []
        for target in plan['declared_targets']:
            case = by_case[target['case']]
            records, static_evidence, seen = [], [], {}
            for variant in target['variants']:
                metadata = copy.deepcopy(by_variant[(case['id'], variant)])
                if variant.startswith('GL_'):
                    assert metadata['choice']['name'] == variant.removeprefix('GL_')
                cubin = archive.read(_member(metadata, 'kernel.cubin'))
                assert c.sha(cubin) == metadata['cubin_SHA256']
                ptx = archive.read(_member(metadata, 'kernel.ptx'))
                assert c.sha(ptx) == metadata['artifact_SHA256']['kernel.ptx']
                ir = archive.read(_member(metadata, 'kernel.ttgir'))
                assert c.sha(ir) == metadata['artifact_SHA256']['kernel.ttgir']
                abi = _abi(ptx.decode())
                identity = _identity(metadata, abi)
                # Keep U separate from every Gluon program. G0 is the first
                # Gluon representative and can carry an exact candidate alias.
                kind = 'U' if variant=='U' else 'Gluon'
                key = (kind, identity)
                if key in seen:
                    representative = seen[key]
                    representative['aliases'].append(metadata['tag'])
                    representative['logical_variants'].append(variant)
                    alias_proofs.append(dict(group=target['id'], tag=metadata['tag'],
                        representative=representative['metadata']['tag'], kind=kind,
                        full_CUBIN_equal=True, ABI_equal=True,
                        runtime_argument_names_equal=True, scalar_specialization_equal=True,
                        launch_resources_equal=True, cubin_SHA256=metadata['cubin_SHA256'],
                        ABI=abi, logical_metadata=metadata))
                else:
                    representative = dict(metadata=metadata, aliases=[], logical_variants=[variant])
                    records.append(representative)
                    seen[key] = representative
                ir_lines = ir.decode().splitlines()
                static_evidence.append(dict(tag=metadata['tag'], variant=variant,
                    cubin_SHA256=metadata['cubin_SHA256'], PTX_SHA256=c.sha(ptx),
                    ttgir_SHA256=c.sha(ir), ABI=abi,
                    resources=metadata.get('resources', {}), choice=metadata.get('choice'),
                    encoding_definitions=[line.strip() for line in ir_lines
                        if re.match(r'\s*#\w+\s*=\s*#ttg\.', line)],
                    operations=[line.strip().split(' loc(')[0] for line in ir_lines
                        if re.search(r'\b(?:tt\.(?:load|store|reshape|join|split|trans)|ttg\.convert_layout)\b', line)]))
            groups.append(dict(id=target['id'], case=copy.deepcopy(case), records=records,
                modes=target['modes'], logical_variants=target['variants'],
                static_rationale=target['rationale'], static_evidence=static_evidence,
                **({'cold_rationale': target['cold_rationale']} if 'cold_rationale' in target else {})))
    return groups, alias_proofs, plan


def run(payload, stage='stage_n3', include_cold_pack=False, snapshot=None):
    payload = Path(payload).resolve()
    root = c.OUT/stage
    assert not (root/'preregistration').exists(), root
    groups, aliases, plan = select(payload, include_cold_pack)
    metric_source = c.old.OUT/'stage_g8/profile_protocol.json'
    previous = c.read(metric_source)
    metrics = previous['profile_metrics']
    assert len(metrics) == len(set(metrics)) == 26
    names = [str(path.relative_to(c.ROOT)) for path in c.SOURCE.rglob('*')
             if path.is_file() and '__pycache__' not in path.parts]
    names += [str((c.old.SOURCE/name).relative_to(c.ROOT))
              for name in ['common.py', 'driver.py', 'layout_parsing.py']]
    names += ['experiments/tma_reduction_layout/phase8/launch.py',
              'experiments/tma_reduction_layout/phase4/exact_cuda.py']
    snapshot = Path(snapshot) if snapshot else Path('/tmp/nonreduction-n3-worker-snapshot')
    assert not snapshot.exists(), snapshot
    hashes = c.old.snapshot(names, snapshot)
    prior = c.read(c.old.OUT/'stage_g8/preregistration/protocol.json')
    protocol = dict(stage=stage, phase='N3_static_mechanism_profiles',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        profile='chenmiaoming', snapshot=str(snapshot), source_SHA256=hashes,
        compiler=prior['compiler'], native_SHA256=c.NATIVE,
        backend_SHA256=prior['backend_SHA256'], payload_path=str(payload),
        payload_SHA256=c.sha(payload.read_bytes()), profile_groups=groups,
        metrics=metrics, metric_provenance=dict(path=str(metric_source.relative_to(c.ROOT)),
            SHA256=c.sha(metric_source.read_bytes()),
            scope='Exact metric list from completed G8 H100 NCU collection; support on a future worker is still checked by actual collection.'),
        selectors=plan, alias_proofs=aliases, maximum_concurrent_GPU_groups=3,
        formal_timing=False, selection_uses_candidate_timing=False,
        ncu_settings=dict(software=dict(replay='kernel', cache='all', section='InstructionStats', grid='one CTA; all original arguments unchanged'),
            warm_application=dict(replay='application', cache='none', grid='original full grid'),
            cold_isolated=dict(replay='kernel', cache='all', grid='original full grid')),
        correctness=dict(warmups=8, one_binary_per_report=True,
            unchanged_runtime_sizes_and_specialization=True, full_oracle_and_canaries=True,
            software_inactive_storage_unchanged=True,
            inplace_restore='Before every warmup and target, outside profiler range'),
        interpretation='Counter profiles describe fixed standalone kernels. Counts are not weighted latency estimates; cache modes and profiled time remain separate from formal timing. Static contrasts are not timing winners.')
    c.write_once(root/'preregistration/selectors.json', plan)
    c.write_once(root/'preregistration/source_snapshot.zip', c.archive(snapshot))
    c.write_once(root/'preregistration/protocol.json', protocol)
    # Byte-for-byte derived copy supplies profile_run's default payload path.
    c.write_once(root/'compiled_payload.zip', payload.read_bytes())
    logical_reports = sum(len(group['logical_variants'])*len(group['modes']) for group in groups)
    physical_reports = sum(len(group['records'])*len(group['modes']) for group in groups)
    print('N3 frozen', len(groups), 'subtypes;', logical_reports, 'logical /', physical_reports,
          'physical reports; selection used no candidate timing')
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--stage', default='stage_n3')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--include-cold-pack', action='store_true')
    args = parser.parse_args()
    run(args.payload, args.stage, args.include_cold_pack, args.snapshot)


if __name__ == '__main__':
    main()
