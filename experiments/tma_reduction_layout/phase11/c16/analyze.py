"""Offline gates for C1.6; never changes raw evidence."""
import argparse
import csv
import io
import json
import math
import re
from experiments.tma_reduction_layout.phase11.c16 import common as c
from experiments.tma_reduction_layout.phase11.validate_a import csv_metrics


def protocol_for_digest(digest):
    paths = [c.OUT/'preregistration/protocol.json'] + list((c.OUT/'preregistration/revisions').glob('*/protocol.json'))
    return next(value for path in paths if c.sha(c.encode(value:=c.read(path)))==digest)


def artifacts():
    protocol = c.load_protocol()
    accepted, attempts = [], []
    for folder in sorted((c.OUT / 'artifacts').glob('attempt_*/export')):
        result = c.read(folder / 'result.json')
        attempts.append({'attempt': folder.parent.name, 'status': result['status']})
        if result['status'] == 'OK':
            accepted.append((folder, result))
    assert len(accepted) == 1, attempts
    root, result = accepted[0]
    original = protocol_for_digest(result['protocol_SHA256'])
    for key in ('baseline','new_cases','reused_bindings','analysis','timing'):
        assert original[key]==protocol[key], key
    name = (c.SOURCE/'kernels.py').relative_to(c.ROOT).as_posix()
    assert original['worker_source_SHA256'][name]==protocol['worker_source_SHA256'][name]
    assert result['environment']['native_SHA256'] == protocol['baseline']['native_SHA256']
    assert result['environment']['ccache_before'] == result['environment']['ccache_after']
    assert result['GPU_timing'] is False and len(result['new']) == 2
    metadata = result['metadata']
    assert len(metadata) == 26
    evidence = {}
    for tag, meta in metadata.items():
        folder = root / tag.replace(':', '/')
        assert c.sha((folder / 'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
        ptx = (folder / 'kernel.ptx').read_text()
        if meta['family'] != 'TMA':
            assert meta == protocol['reused_metadata'][tag]
            for name, digest in protocol['reused_bindings'][tag]['files'].items():
                assert c.sha((folder / name).read_bytes()) == digest
            continue
        checks = c.read(folder / 'checks.json')
        assert len(checks['checks']) == 2 and all(row['pass'] for row in checks['checks'])
        ir = (folder / 'kernel.ttgir').read_text()
        assert 'async_tma_copy_global_to_local' in ir and 'ttg.local_gather' in ir
        assert '"tt.reduce"' not in ir
        original = root / f'M32_N{meta["N"]}_w{meta["num_warps"]}' / 'vec1/FULL/kernel.ptx'
        pattern = r'\bcp\.async\.bulk\.tensor[^;]*;'
        tma_count, original_count = (len(re.findall(pattern, text)) for text in (ptx, original.read_text()))
        assert tma_count == original_count and tma_count > 0
        evidence[tag] = {'TMA_PTX_sites': tma_count, 'original_TMA_PTX_sites': original_count,
                         'first_row_descriptor_gather': True, 'reduction_absent': True,
                         'correctness': checks['checks']}
    return {'status': 'OK', 'stage': 'C1.6', 'accepted_attempt': root.parent.name,
            'attempts': attempts, 'metadata': metadata, 'controls': evidence,
            'new_programs': 2, 'exact_reused_programs': 24, 'native_rebuild': False}


def counters():
    protocol, gate = c.load_protocol(), c.read(c.OUT / 'artifact_gate.json')
    accepted = [(p,c.read(p/'result.json')) for p in (c.OUT/'profiles').glob('attempt_*/export') if c.read(p/'result.json')['status']=='OK']
    assert len(accepted)==1
    folder,result = accepted[0]
    assert result['status'] == 'OK' and len(result['records']) == 20
    profile_protocol = protocol_for_digest(result['protocol_SHA256'])
    assert profile_protocol['profiles']==protocol['profiles']
    assert result['payload_SHA256'] == c.sha((c.OUT / 'frozen_payload.zip').read_bytes())
    rows = []
    for record in result['records']:
        tag, mode = record['tag'], record['cache']
        meta = gate['metadata'][tag]
        path = folder / tag.replace(':', '/') / mode
        collection, imported = c.read(path / 'collect.json'), c.read(path / 'import.json')
        assert collection['returncode'] == imported['returncode'] == 0
        args = collection['argv']
        assert args[args.index('--replay-mode') + 1] == 'application'
        assert args[args.index('--cache-control') + 1] == 'none'
        bindings = [json.loads(line.split('PROFILE_BINDING ', 1)[1])
                    for line in collection['output'].splitlines() if line.startswith('PROFILE_BINDING ')]
        assert bindings, tag
        for binding in bindings:
            assert binding['tag'] == tag and binding['cache'] == mode and binding['correctness']['pass']
            assert binding['binding']['CUBIN_SHA256'] == binding['binding']['loaded_CUBIN_SHA256'] == meta['cubin_SHA256']
            assert binding['Triton_imported'] is False
            assert binding['protocol_SHA256'] == result['protocol_SHA256']
        raw = csv_metrics(imported['output'])
        csv_rows=list(csv.reader(io.StringIO(imported['output'])))
        header=next(i for i,row in enumerate(csv_rows) if 'Kernel Name' in row)
        columns=csv_rows[header]
        if 'Metric Name' in columns:
            units={dict(zip(columns,row))['Metric Name']:dict(zip(columns,row))['Metric Unit']
                   for row in csv_rows[header+1:] if len(row)==len(columns)}
        else:
            units=dict(zip(columns,csv_rows[header+1]))
        metrics = {name: float(raw[name].replace(',', '')) for name in protocol['profiles']['metrics']}
        assert all(math.isfinite(v) and v >= 0 for v in metrics.values())
        scale={'second':1e9,'msecond':1e6,'usecond':1e3,'nsecond':1,'ns':1}
        ns = metrics['gpu__time_duration.sum'] * scale[units['gpu__time_duration.sum']]
        for name in ('dram__bytes_read.sum','dram__bytes_write.sum'):
            assert units[name]=='byte', (name,units[name])
        assert ns > 0 and metrics['dram__bytes_read.sum'] > 0
        expected_input = 8192 * 8192 * 2
        rows.append({'tag': tag, 'N': meta['N'], 'vec': meta.get('vec'), 'family': meta['family'], 'cache': mode,
            'metrics': metrics, 'metric_units': units, 'duration_ns': ns,
            'report_SHA256': c.sha((path / 'report.ncu-rep').read_bytes()),
            'PROFILE_BINDINGS': bindings, 'expected_logical_input_bytes': expected_input,
            'expected_logical_output_bytes': 256 * 8192 * 4,
            'DRAM_read_fraction_of_logical_input': metrics['dram__bytes_read.sum'] / expected_input,
            'DRAM_read_GBs': metrics['dram__bytes_read.sum'] / ns,
            'DRAM_total_GBs': (metrics['dram__bytes_read.sum'] + metrics['dram__bytes_write.sum']) / ns})
    assert len(rows) == 20
    return {'status': 'OK', 'reports': 20, 'rows': rows,
            'profiler_duration_is_formal_timing': False, 'application_replay_cache_recreated': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['artifacts', 'counters'])
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    c.protect_prior()
    result = artifacts() if args.action == 'artifacts' else counters()
    target = c.OUT / ('artifact_gate.json' if args.action == 'artifacts' else 'counter_gate.json')
    if args.validate:
        assert c.read(target) == json.loads(c.encode(result))
    else:
        c.write_once(target, result)
        if args.action == 'artifacts':
            root = c.OUT / 'artifacts' / result['accepted_attempt'] / 'export'
            import io
            import zipfile
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
                for tag in sorted(result['metadata']):
                    for name in ('kernel.cubin', 'kernel.ptx'):
                        relative = tag.replace(':', '/') + '/' + name
                        z.writestr(relative, (root / relative).read_bytes())
            c.write_once(c.OUT / 'frozen_payload.zip', stream.getvalue())
    print('C1.6', args.action, 'gate PASS')


if __name__ == '__main__':
    main()
