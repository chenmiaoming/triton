"""Independent artifact comparison for each retained source-only attempt."""
from pathlib import Path
import argparse
import re
from . import common as c
from .g5_analyze import elf, facts
from .g6_prepare import OUT


def artifact(root, meta):
    folder = root/meta['artifact_directory']
    for name, digest in meta['artifact_SHA256'].items():
        assert c.sha((folder/name).read_bytes()) == digest
    ptx = (folder/'kernel.ptx').read_text()
    signature = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
    assert signature
    return dict(allocated_sections=elf((folder/'kernel.cubin').read_bytes()),
                ABI=re.findall(r'\.param\s+\.(\w+)', signature[1]),
                resources={k: int(v) for k, v in re.findall(r'\b(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER):(\d+)',
                                                          (folder/'kernel.resource.txt').read_text())},
                facts=facts((folder/'kernel.ttgir').read_text()),
                artifacts=str(folder.relative_to(c.ROOT)))


def references():
    index = {}
    g5 = c.OUT/'stage_g5'
    for namespace in ['pilot/attempt_001', 'matrix/attempt_001', 'actual/attempt_001']:
        root = g5/namespace/'export'
        for raw in c.read(root/'result.json')['programs']:
            if raw['variant'] not in ['U', 'U_actual']:
                continue
            index[raw['case']] = (root/raw['path'], raw['metadata'])
    return index


def analyze(attempt):
    folder = OUT/f'attempt_{attempt:03}'; p = c.read(folder/'preregistration/protocol.json')
    raw = c.read(folder/'export/result.json'); refs = references(); rows = []
    assert raw['protocol_SHA256'] == c.sha(c.encode(p))
    for record in raw['records']:
        case = record['case']; row = dict(case=case, compiled=record['status'] == 'OK')
        if row['compiled']:
            meta = record['metadata']; refroot, refmeta = refs[case]
            value = artifact(folder/'export', meta); reference = artifact(refroot, refmeta)
            row.update(artifact=value, U_reference=reference, metadata=meta,
                       allocated_sections_equal_U=value['allocated_sections'] == reference['allocated_sections'],
                       ABI_equal_U=value['ABI'] == reference['ABI'] and meta['runtime_argument_names'] == refmeta['runtime_argument_names'],
                       resources_equal_U=value['resources'] == reference['resources'] and
                           all(meta[k] == refmeta[k] for k in ['num_warps', 'dynamic_smem_bytes', 'global_scratch_size']))
            row['equivalent'] = all(row[k] for k in ['allocated_sections_equal_U', 'ABI_equal_U', 'resources_equal_U'])
        else:
            row.update(equivalent=False, error=record['error'])
        rows.append(row)
        print(case, 'EQUIVALENT' if row['equivalent'] else 'NOT_EQUIVALENT')
    result = dict(compilation_status=raw['status'], attempt=attempt, rows=rows,
                  equivalent=sum(r['equivalent'] for r in rows), total=len(rows),
                  compiler_intervention=raw['compiler_intervention'], GPU_allocated=raw['GPU_allocated'])
    path = folder/'comparison.json'
    if path.exists(): assert c.read(path) == result
    else: c.write_once(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    analyze(parser.parse_args().attempt)
