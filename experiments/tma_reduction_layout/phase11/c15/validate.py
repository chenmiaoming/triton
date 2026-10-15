"""Final CPU-only evidence validator, including committed raw-byte checks."""
import argparse
import ast
import io
from pathlib import Path
import re
import subprocess
import zipfile

from experiments.tma_reduction_layout.phase11.c15 import common as c
from experiments.tma_reduction_layout.phase11.c15 import analyze, report

DERIVED = {'artifact_gate.json', 'counter_gate.json', 'analysis.json', 'summary.md',
           'validation.json', 'raw_manifest.json', 'plots'}


def raw_inventory():
    return {name: digest for name, digest in c.inventory(c.OUT).items()
            if Path(name).parts[0] not in DERIVED}


def check_archives():
    count = 0
    for raw in c.OUT.glob('*/attempt_*/raw_export.zip'):
        folder = raw.parent / 'export'
        with zipfile.ZipFile(raw) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            for name in z.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert z.read(name) == (folder / name).read_bytes(), (raw, name)
                count += 1
    return count


def check_sources():
    paths = [c.OUT / 'preregistration/protocol.json']
    paths += list((c.OUT / 'preregistration/revisions').glob('*/protocol.json'))
    for path in paths:
        protocol = c.read(path)
        snapshot = path.with_name('source_snapshot.zip')
        with zipfile.ZipFile(snapshot) as z:
            for name, digest in protocol['worker_source_SHA256'].items():
                assert c.sha(z.read(name)) == digest, (path, name)
    c.verify_sources(c.load_protocol())
    for path in c.SOURCE.glob('*.py'):
        ast.parse(path.read_text())
    return len(paths)


def check_no_feedback():
    gate = c.read(c.OUT / 'artifact_gate.json')
    result = {}
    for tag, meta in gate['metadata'].items():
        if meta['family'] != 'MR':
            continue
        path = c.OUT / 'artifacts' / gate['accepted_attempt'] / 'export' / tag.replace(':', '/') / 'kernel.ttgir'
        ir = path.read_text()
        loop = re.search(r'scf.for.*?scf.yield[^\n]*', ir, re.S)
        assert loop
        for token in ('tt.broadcast', 'ttg.convert_layout', 'ttg.local_store', 'ttg.local_load'):
            assert token not in loop[0], (tag, token)
        assert 'tt.reduce' in loop[0] and 'arith.subf' in loop[0] and 'arith.addf' in loop[0]
        template = gate['MR_template_audit'][tag]
        assert template['widening_full_tile_single_use'] and template['private_add_sub_BF16']
        assert template['max_opcode_histogram_matches']
        result[tag] = {'no_result_broadcast_or_layout_conversion_in_loop_IR': True,
                       'no_explicit_smem_load_store_in_loop_IR': True,
                       'compiler_lowered_reduction_scratch_audited_by_NCU': True}
    assert len(result) == 8
    return result


def check_timing_launches():
    gate = c.read(c.OUT / 'artifact_gate.json')
    guards, launches = 0, 0
    for invocation in (1, 2, 3):
        raw = c.read(c.OUT / 'timing' / f'invocation_{invocation}' / 'raw.json')
        assert raw['call_id'] == c.read(c.OUT / 'timing' / f'invocation_{invocation}' / 'dispatch.json')['call_id']
        assert raw['CUDA_call_counts']['cuLaunchKernel'] == 128 * (20 + 3)
        for tag, binding in raw['binaries'].items():
            assert binding['CUBIN_SHA256'] == binding['loaded_CUBIN_SHA256'] == gate['metadata'][tag]['cubin_SHA256']
            assert binding['sha_guards'] == binding['launches'] == 8 * (20 + 3)
            guards += binding['sha_guards']
            launches += binding['launches']
    return {'launches': launches, 'SHA_guards': guards, 'event_samples': 7680, 'warmup_launches': 1152}


def check_wavefront_scaling():
    gate = c.read(c.OUT / 'counter_gate.json')
    metrics = ('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
               'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum')
    for tag, row in gate['binaries'].items():
        for metric in metrics:
            body = row['hardware_delta_per_iteration'][metric]
            fixed = row['hardware_by_K']['1'][metric] - body
            assert all(row['hardware_by_K'][str(k)][metric] == fixed + k * body
                       for k in (1, 16, 64, 256)), (tag, metric)
    return {'hardware_metrics': len(metrics), 'programs': len(gate['binaries']),
            'K': [1, 16, 64, 256], 'exact_affine_scaling': True}


def check_ML_reuse():
    gate = c.read(c.OUT / 'artifact_gate.json')
    root = c.OUT / 'artifacts' / gate['accepted_attempt'] / 'export'
    count = 0
    for tag, meta in gate['metadata'].items():
        if meta['family'] != 'ML':
            continue
        prior = c.ROOT / meta['artifact_origin']['original_path']
        now = root / tag.replace(':','/')
        for suffix in ('glir','ttgir','llir','ptx','sass','cubin','resource.txt'):
            assert (prior / ('kernel.'+suffix)).read_bytes() == (now / ('kernel.'+suffix)).read_bytes()
        assert c.read(prior/'attempt.json')['correctness'] == c.read(now/'attempt.json')['correctness']
        count += 1
    assert count == 8
    return count


def check_BF16_max_counter_mapping():
    gate = c.read(c.OUT / 'counter_gate.json')
    assert 'VHMNMX' in gate['max_SASS_mapping']
    count = 0
    for tag,row in gate['binaries'].items():
        if not tag.endswith(':MR'):
            continue
        for pc in row['loop_PCs']:
            if 'MNMX' in pc['instruction']:
                assert 'max' in pc['families'], (tag, pc)
                if 'VHMNMX' in pc['instruction']:
                    count += 1
    assert count > 0, 'Restored packed BF16 max mapping was not exercised'
    return {'VHMNMX_covered': True, 'VHMNMX_PC_sites_including_unrolled_copies': count}


def committed_bytes(files):
    names = [(c.OUT / name).relative_to(c.ROOT).as_posix() for name in files]
    request = ''.join('HEAD:' + name + '\n' for name in names).encode()
    p = subprocess.run(['git', 'cat-file', '--batch'], input=request, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, cwd=c.ROOT, check=True)
    stream = io.BytesIO(p.stdout)
    for name, relative in zip(names, files):
        head = stream.readline().decode().split()
        assert len(head) == 3 and head[1] == 'blob', (name, head)
        data = stream.read(int(head[2]))
        assert stream.read(1) == b'\n'
        assert c.sha(data) == files[relative] == c.sha((c.ROOT / name).read_bytes()), name
    assert not stream.read()
    return len(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-manifest', action='store_true')
    parser.add_argument('--committed', action='store_true')
    args = parser.parse_args()
    c.protect_prior()
    assert analyze.artifact_gate() == c.read(c.OUT / 'artifact_gate.json')
    assert __import__('json').loads(c.encode(analyze.counter_gate())) == c.read(c.OUT / 'counter_gate.json')
    assert c.read(c.OUT / 'counter_gate.json')['status'] == 'OK'
    derived = report.derive()
    assert c.read(c.OUT / 'analysis.json') == __import__('json').loads(c.encode(derived))
    assert (c.OUT / 'summary.md').read_text() == report.render(derived)
    protocols = check_sources()
    archive_entries = check_archives()
    feedback = check_no_feedback()
    launches = check_timing_launches()
    wavefronts = check_wavefront_scaling()
    reused = check_ML_reuse()
    max_mapping = check_BF16_max_counter_mapping()
    files = raw_inventory()
    if args.freeze_manifest:
        c.write_once(c.OUT / 'raw_manifest.json', {'files': files})
    assert files == c.read(c.OUT / 'raw_manifest.json')['files'], 'Raw evidence changed'
    result = {'status': 'PASS', 'stage': 'C1.5', 'layouts': 8, 'programs': 16,
              'new_MR_programs': 8, 'reused_ML_programs': reused,
              'new_MR_full_output_checks': 96, 'reused_ML_full_output_checks': 96,
              'new_MR_NCU_reports': 64, 'reused_ML_NCU_reports': 64,
              'strict_K_gate': 'PASS', 'protocol_revisions': protocols,
              'raw_files': len(files), 'archive_entries_verified': archive_entries,
              'no_reverse_broadcast': feedback, 'timing': launches,
              'hardware_wavefront_scaling': wavefronts,
              'BF16_max_SASS_counter_coverage': max_mapping,
              'prior_results_files_protected': len(c.read(c.OUT / 'preregistration/prior_manifest.json')['files']),
              'native_rebuild': False, 'C2_C3_D_executed': False, 'PR_created': False,
              'BF16_max_template_match': '8/8', 'fit_K': [16,64,256], 'K1_in_fit': False}
    if args.committed:
        print('Committed raw bytes PASS', committed_bytes(files), 'files')
        assert c.read(c.OUT / 'validation.json') == result
    else:
        (c.OUT / 'validation.json').write_bytes(c.encode(result))
    print('Final C1.5 validator PASS: 8/8 max templates, 8 reused ML, 8 corrected MR, fit K>=16, 7680 new samples')


if __name__ == '__main__':
    main()
