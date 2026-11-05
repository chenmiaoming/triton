"""Freeze source-only baseline changes and immutable reference bindings."""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import subprocess
from . import common as c

OUT = c.OUT/'stage_g6'
SNAPSHOT = Path('/tmp/operator-gpu-g6-source-snapshot')


def main(attempt, family):
    previous = c.read(c.OUT/'stage_g5/preregistration/protocol_v2.json')
    actual = c.read(c.OUT/'stage_g5/preregistration/actual_protocol.json')
    cases = []
    for item in previous['cases'] + actual['cases']:
        if family != 'all' and item['case']['family'] != family:
            continue
        value = copy.deepcopy(item)
        value['record'] = next(r for r in value.pop('records') if r['tag'].endswith(':gluon_anchor'))
        cases.append(value)
    names = list(previous['worker_source_SHA256'])
    names = [name for name in names if not Path(name).name.startswith('g5_')]
    names += [str(p.relative_to(c.ROOT)) for p in c.SOURCE.glob('g6_*.py')]
    hashes = c.snapshot(names, SNAPSHOT)
    sources = c.OUT/'stage_g5/compiler_source'
    p = dict(stage='G6 source-only baseline', attempt=attempt, family=family,
             frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
             parent_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             compiler=previous['compiler'], native_SHA256=c.NATIVE, GPU_allocated=False,
             compiler_intervention=False, backend_SHA256=c.sha((sources/'third_party/nvidia/backend/compiler.py').read_bytes()),
             profile='chenmiaoming', cases=cases, worker_source_SHA256=hashes,
             acceptance='All ELF allocated section bytes, runtime ABI and resources must match U; retain every failed attempt.',
             cache_policy='Dedicated source-only G6 cache; no Python or native backend modifications.',
             protected_SHA256={str(path.relative_to(c.ROOT)): c.sha(path.read_bytes())
                 for stage in ['stage_g2', 'stage_g3a', 'stage_g3b', 'stage_g4a', 'stage_g4b', 'stage_g5']
                 for path in sorted((c.OUT/stage).rglob('*')) if path.is_file()})
    dest = OUT/f'attempt_{attempt:03}'
    c.write_once(dest/'preregistration/source_snapshot.zip', c.archive(SNAPSHOT))
    c.write_once(dest/'preregistration/protocol.json', p)
    print('G6 frozen:', len(cases), 'programs, attempt', attempt, 'default backend')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--family', choices=['all', 'C01', 'C02', 'C06'], default='all')
    args = parser.parse_args(); main(args.attempt, args.family)
