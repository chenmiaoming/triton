"""Re-import the retained NCU reports locally without allocating a GPU."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows, source_blocks
from . import common as c


def one(record):
    report = c.ROOT / record['path']
    geometry = report.parents[3].name
    target = c.OUT / 'stage_g2/offline_imports' / (geometry + '_' + record['mode'] + '.json')
    if target.exists():
        result = c.read(target)
    else:
        commands = [['/usr/local/cuda/bin/ncu', '--import', str(report), '--page', 'raw',
                     '--csv', '--print-units', 'base', '--print-fp']]
        if record['mode'] == 'software':
            commands.append(['/usr/local/cuda/bin/ncu', '--import', str(report), '--page', 'source',
                             '--print-source', 'sass', '--csv'])
        outputs = []
        for argv in commands:
            process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            outputs.append({'argv': argv, 'returncode': process.returncode, 'output': process.stdout})
            assert process.returncode == 0, process.stdout[-2000:]
        result = {'report': record['path'], 'report_SHA256': c.sha(report.read_bytes()),
                  'CPU_only': True, 'commands': outputs}
        c.write_once(target, result)
    assert result['report_SHA256'] == record['SHA256'] == c.sha(report.read_bytes())
    local = hardware_rows(result['commands'][0]['output'])
    remote = hardware_rows(c.read(report.parent / 'import.json')['output'])
    assert local == remote, (record['path'], 'local/remote metric values differ')
    if record['mode'] == 'software':
        assert source_blocks(result['commands'][1]['output']) == source_blocks(c.read(report.parent / 'source.json')['output'])
    return {'report': record['path'], 'local_import': target.relative_to(c.ROOT).as_posix(),
            'local_import_SHA256': c.sha(target.read_bytes()), 'metric_launches': len(local),
            'mode': record['mode'], 'all_metrics_and_source_match': True}


def main():
    records = c.read(c.OUT / 'stage_g2/counter_analysis.json')['reports']
    with ThreadPoolExecutor(max_workers=4) as executor:
        imported = list(executor.map(one, records))
    result = {'status': 'OK', 'CPU_only': True, 'GPU_allocations': 0, 'reports': imported,
              'ncu_version': subprocess.check_output(['/usr/local/cuda/bin/ncu', '--version'], text=True)}
    c.write_once(c.OUT / 'stage_g2/offline_import_validation.json', result)
    print('G2 local NCU re-import OK:', len(imported), 'reports; all metrics/source identical')


if __name__ == '__main__':
    main()
