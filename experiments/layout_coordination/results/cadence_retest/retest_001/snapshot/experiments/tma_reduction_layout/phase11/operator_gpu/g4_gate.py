"""Inspect immutable CPU exports and combine old-B4 artifacts without recompilation."""
from pathlib import Path
import argparse
import shutil
from . import common as c
from .g4_prepare import out, protocol_path, gate_path, payload_path


def main(stage):
    target = out(stage); p = c.read(protocol_path(stage)); c.verify_sources(p)
    candidates = [d for d in sorted((target / 'cpu').glob('attempt_*/export'))
                  if c.read(d / 'result.json')['protocol_SHA256'] == c.sha(c.encode(p))]
    assert len(candidates) == 1, candidates
    root = candidates[0]; result = c.read(root / 'result.json')
    assert result['status'] == 'OK' and not result['GPU_allocated'] and result['native_rebuilds'] == 0
    assert result['protocol_SHA256'] == c.sha(c.encode(p)) and result['ccache_before'] == result['ccache_after']
    package = Path('/tmp') / ('operator-g4-package-' + stage + '-v2'); package.mkdir(exist_ok=False)
    records = result['records'].copy()
    for row in records:
        source = root / row['artifact_directory']; dest = package / row['artifact_directory']
        shutil.copytree(source, dest)
        assert all(c.sha((dest / name).read_bytes()) == digest for name, digest in row['artifact_SHA256'].items())
    if stage == 'g4b':
        oldroot = Path('/tmp/operator-g4-old-g2-payload'); c.extract((c.OUT / 'stage_g2/compiled_payload.zip').read_bytes(), oldroot)
        old = {r['tag']: r for r in c.read(c.OUT / 'stage_g2/compile_gate.json')['records']}
        for group in p['groups']:
            for label, name in group['old_B4_variants'].items():
                row = dict(old[group['case']['id'] + ':' + name]); old_tag = row['tag']
                row.update(tag=group['id'] + ':old_B4:' + label, group=group['id'], provenance='frozen_G2_B4',
                           variant={'name': label}, original_G2_tag=old_tag,
                           stage2_tag=group['id'] + ':actual:stage2')
                row['artifact_directory'] = 'g2/' + row['artifact_directory']
                source = oldroot / old[old_tag]['artifact_directory']; dest = package / row['artifact_directory']
                if not dest.exists(): shutil.copytree(source, dest)
                assert all(c.sha((dest / name).read_bytes()) == digest for name, digest in row['artifact_SHA256'].items())
                records.append(row)
    for name, digest in p['old_evidence_SHA256'].items(): assert c.sha((c.ROOT / name).read_bytes()) == digest, name
    payload = c.archive(package); c.write_once(payload_path(stage), payload)
    gate = {'status': 'OK', 'records': records, 'plans': result['plans'],
            'protocol_SHA256': c.sha(c.encode(p)), 'payload_SHA256': c.sha(payload),
            'CPU_logical_programs': len(result['records']), 'native_rebuilds': 0,
            'unique_CUBINs': len({r['cubin_SHA256'] for r in records}),
            'CPU_export_directory': str(root.relative_to(c.ROOT))}
    c.write_once(gate_path(stage), gate)
    print(stage, 'CPU gate PASS', len(records), 'records;', gate['unique_CUBINs'], 'unique CUBINs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['g4a', 'g4b']); main(parser.parse_args().stage)
