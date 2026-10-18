"""Independently re-import every saved NCU report on the local CPU host."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows, source_blocks


def one(record):
    report = c.ROOT/record['path']
    mode = record['mode']
    key = report.parents[3].name+'_'+mode
    # .../group_0/attempt_1/export/hardware/report.ncu-rep
    target = c.OUT/'offline_imports'/(key+'.json')
    if target.exists():
        result = c.read(target)
    else:
        commands = [['/usr/local/cuda/bin/ncu', '--import', str(report), '--page', 'raw',
                     '--csv', '--print-units', 'base', '--print-fp']]
        if mode == 'software':
            commands.append(['/usr/local/cuda/bin/ncu', '--import', str(report), '--page', 'source',
                             '--print-source', 'sass', '--csv'])
        outputs = []
        for argv in commands:
            process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            outputs.append({'argv': argv, 'returncode': process.returncode, 'output': process.stdout})
            assert process.returncode == 0, process.stdout[-2000:]
        result = {'report': record['path'], 'report_SHA256': c.sha(report.read_bytes()),
                  'CPU_only': True, 'commands': outputs}
        c.write_once(target, result)
    assert result['report_SHA256'] == record['SHA256']
    local = hardware_rows(result['commands'][0]['output'])
    remote = hardware_rows(c.read(report.parent/'import.json')['output'])
    assert local == remote, (record['path'], 'local/remote metric import differs')
    if mode == 'software':
        assert source_blocks(result['commands'][1]['output']) == source_blocks(c.read(report.parent/'source.json')['output'])
    return {'report': record['path'], 'local_import': str(target.relative_to(c.ROOT)),
            'local_import_SHA256': c.sha(target.read_bytes()), 'metric_launches': len(local),
            'mode': mode, 'all_metrics_and_source_match': True}


def main():
    records = c.read(c.OUT/'counter_gate.json')['reports']
    with ThreadPoolExecutor(max_workers=4) as executor:
        imported = list(executor.map(one, records))
    value = {'status': 'OK', 'CPU_only': True, 'GPU_allocations': 0, 'reports': imported,
             'ncu_version': subprocess.check_output(['/usr/local/cuda/bin/ncu', '--version'], text=True)}
    c.derived(c.OUT/'offline_import_validation.json', value)
    print('C2c local NCU re-import OK', len(imported), 'reports; all metrics/source identical')


if __name__ == '__main__':
    main()
