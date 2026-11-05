"""Evidence integrity, source/protocol provenance and exact committed bytes."""
import argparse
import io
import json
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11.c16 import common as c
from experiments.tma_reduction_layout.phase11.c16 import analyze, report

DERIVED = {'artifact_gate.json', 'counter_gate.json', 'analysis.json', 'summary.md',
           'interpretation.md', 'validation.json', 'raw_manifest.json','figure_manifest.json'}


def inventory():
    return {p.relative_to(c.OUT).as_posix(): c.sha(p.read_bytes()) for p in sorted(c.OUT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.relative_to(c.OUT).as_posix() not in DERIVED
            and 'figures' not in p.relative_to(c.OUT).parts}


def sources():
    paths = [c.OUT / 'preregistration/protocol.json'] + sorted((c.OUT / 'preregistration/revisions').glob('*/protocol.json'))
    previous = None
    original = c.read(paths[0])
    for path in paths:
        protocol = c.read(path)
        if previous:
            assert protocol['previous_protocol_SHA256'] == previous
        previous = c.sha(c.encode(protocol))
        for key in ('analysis', 'timing', 'profiles', 'new_cases', 'reused_bindings', 'K', 'baseline'):
            assert protocol[key] == original[key], key
        with zipfile.ZipFile(path.parent / 'source_snapshot.zip') as archive:
            for name, digest in protocol['worker_source_SHA256'].items():
                assert c.sha(archive.read(name.removeprefix('experiments/'))) == digest
    c.verify_sources(c.load_protocol())
    diagnostic=c.read(c.OUT/'preregistration/submission_protocol.json')
    with zipfile.ZipFile(c.OUT/'preregistration/submission_source_snapshot.zip') as archive:
        for name,digest in diagnostic['worker_source_SHA256'].items():
            assert c.sha(archive.read(name.removeprefix('experiments/')))==digest
            assert c.sha((c.ROOT/name).read_bytes())==digest
    return len(paths)


def archives():
    count = 0
    for path in c.OUT.rglob('raw_export.zip'):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert len(names) == len(set(names))
            for name in names:
                assert not name.startswith('/') and '..' not in name.split('/')
                assert archive.read(name) == (path.parent / 'export' / name).read_bytes()
                count += 1
    gate = c.read(c.OUT / 'artifact_gate.json')
    root = c.OUT / 'artifacts' / gate['accepted_attempt'] / 'export'
    with zipfile.ZipFile(c.OUT / 'frozen_payload.zip') as archive:
        assert len(archive.namelist()) == 52
        for name in archive.namelist():
            assert archive.read(name) == (root / name).read_bytes()
    return count


def committed(files):
    names = [(c.OUT / name).relative_to(c.ROOT).as_posix() for name in files]
    request = ''.join('HEAD:' + name + '\n' for name in names).encode()
    p = subprocess.run(['git', 'cat-file', '--batch'], input=request, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, cwd=c.ROOT, check=True)
    stream = io.BytesIO(p.stdout)
    for name, relative in zip(names, files):
        header = stream.readline().decode().split()
        assert len(header) == 3 and header[1] == 'blob', (name, header)
        data = stream.read(int(header[2]))
        assert stream.read(1) == b'\n'
        assert c.sha(data) == files[relative] == c.sha((c.ROOT / name).read_bytes()), name
    assert not stream.read()
    return len(names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze-manifest', action='store_true')
    p.add_argument('--committed', action='store_true')
    args = p.parse_args()
    c.protect_prior()
    assert analyze.artifacts() == c.read(c.OUT / 'artifact_gate.json')
    assert analyze.counters() == c.read(c.OUT / 'counter_gate.json')
    derived = report.derive()
    assert json.loads(c.encode(derived)) == c.read(c.OUT / 'analysis.json')
    assert report.render(derived) == (c.OUT / 'summary.md').read_text()
    figure=c.read(c.OUT/'figure_manifest.json')
    assert figure['analysis_SHA256']==c.sha((c.OUT/'analysis.json').read_bytes())
    for name,digest in figure['files'].items():
        assert c.sha((c.OUT/'figures'/name).read_bytes())==digest
    protocol_revisions, entries = sources(), archives()
    cpu = c.read(c.OUT / 'cpu_preflight/attempt_3/export/result.json')
    assert cpu['status'] == 'OK' and cpu['GPU_allocated'] is False and len(cpu['programs']) == 2
    cpu_protocol = analyze.protocol_for_digest(cpu['protocol_SHA256'])
    name = (c.SOURCE/'kernels.py').relative_to(c.ROOT).as_posix()
    assert cpu_protocol['worker_source_SHA256'][name]==c.load_protocol()['worker_source_SHA256'][name]
    for invocation in (1,2,3):
        raw = c.read(c.OUT / 'timing' / f'invocation_{invocation}' / 'raw.json')
        assert len(raw['binaries']) == 26
        assert raw['CUDA_call_counts']['cuLaunchKernel'] == sum(x['launches'] for x in raw['binaries'].values())
    files = inventory()
    if args.freeze_manifest:
        c.write_once(c.OUT / 'raw_manifest.json', {'files': files})
    assert files == c.read(c.OUT / 'raw_manifest.json')['files']
    result = {'stage': 'C1.6', 'status': 'PASS', 'evidence_validation_not_hypothesis_acceptance': True,
        'new_programs': 2, 'exact_reused_programs': 24, 'new_control_correctness_checks': 4,
        'timing_worker_checks': 234, 'event_samples': 6480, 'NCU_reports': 20,
        'submission_diagnostic_event_samples':800, 'total_event_samples':7280,
        'single_kernel_graph_topologies_verified':10, 'submission_correctness_checks':20,
        'cache_modes': ['warm','cold'], 'application_replay': True, 'Triton_imported_in_timing': False,
        'frozen_model_constants_unchanged': True, 'protocol_revisions': protocol_revisions,
        'raw_files': len(files), 'archive_entries_verified': entries,
        'prior_results_files_protected': len(c.read(c.OUT/'preregistration/prior_manifest.json')['files']),
        'native_rebuild': False, 'C2_started': False, 'PR_created': False}
    if args.committed:
        print('Committed raw bytes PASS', committed(files))
        assert c.read(c.OUT / 'validation.json') == result
    else:
        (c.OUT / 'validation.json').write_bytes(c.encode(result))
    print('C1.6 final evidence validator PASS: 26 binaries, 6480 samples, 20 DRAM reports')


if __name__ == '__main__':
    main()
