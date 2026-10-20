"""CPU preparation integrity and references; does not import Triton or init CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path

from .cases import cases, ownership
from .oracles import validate_cpu_references
from .prepare import SOURCE, semantic_parity, sha, write_once


def validate(attempt, output=None):
    attempt = Path(attempt)
    protocol = json.loads((attempt / 'protocol.json').read_text())
    result = json.loads((attempt / 'result.json').read_text())
    assert result['protocol_SHA256'] == sha((attempt / 'protocol.json').read_bytes())
    assert result['status'] == 'OK' and not result['GPU_allocated']
    assert not result['GPU_correctness_verified'] and not result['timing_collected']
    assert result['native_rebuilds'] == 0
    assert result['native_SHA256'] == result['native_SHA256_after']
    assert result['ccache_before'] == result['ccache_after']
    if result['compiler_policy'] == 'pinned':
        assert result['native_SHA256'] == protocol['formal_compiler']['native_SHA256']
    for name, digest in protocol['source_SHA256'].items():
        assert sha((attempt / 'source' / name).read_bytes()) == digest, name
        # Subsequent source changes must receive another preparation protocol.
        assert sha((SOURCE / name).read_bytes()) == digest, name
    assert semantic_parity() == protocol['semantic_parity']
    assert {r.get('parent', r['id']) for r in protocol['cases']} == {f'C{i:02}' for i in range(1, 9)}
    live = {r['id']: r for r in cases()}
    expected_tags = set()
    for row in protocol['cases']:
        assert live[row['id']]['scope'] == row['scope'] == 'semantic_fixture'
        assert not row['production_adapter_verified']
        expected_tags.add(row['id'] + ':canonical_triton')
        anchors = result['input_layout_anchors'][row['id']]
        assert anchors, row['id']
        expected_tags.update(row['id'] + ':gluon_input_anchor_' + str(i) for i in range(len(anchors)))
        for variant in row['layouts']:
            assert ownership(row['BM'], row['BN'], row['W'], variant['layout']) == variant['ownership']
            expected_tags.add(row['id'] + ':gluon:' + variant['name'])
    assert expected_tags == {r['tag'] for r in result['records']}
    assert len(expected_tags) == len(result['records'])
    aliases = 0
    for row in result['records']:
        assert row['status'] == 'OK' and not row['production_adapter_verified']
        folder = attempt / row['artifact_directory']
        for name, digest in row['artifact_SHA256'].items():
            assert sha((folder / name).read_bytes()) == digest, row['tag']
        cubin = (folder / 'kernel.cubin').read_bytes()
        assert cubin.startswith(b'\x7fELF') and sha(cubin) == row['cubin_SHA256']
        assert not row['dynamic_counts_available']
        if 'identical_compile_alias_of' in row:
            anchor = next(x for x in result['records'] if x['tag'] == row['identical_compile_alias_of'])
            assert anchor['compile_hash'] == row['compile_hash']
            assert anchor['cubin_SHA256'] == row['cubin_SHA256']
            aliases += 1
    references = validate_cpu_references()
    report = {'status': 'PASS', 'compiler_policy': result['compiler_policy'],
              'candidate_families': 8, 'semantic_specializations': len(protocol['cases']),
              'compiled_programs': len(result['records']), 'identical_compile_aliases': aliases,
              'unique_compile_keys': result['new_compile_keys'], 'native_rebuilds': 0,
              'source_parity': protocol['semantic_parity'], 'references': references,
              'GPU_allocated': False, 'GPU_kernel_correctness_verified': False,
              'production_adapters_verified': False, 'whole_graph_bridge_verified': False,
              'timing_collected': False}
    if output is not None:
        write_once(output, report)
    print('PREPARATION VALIDATION PASS:', len(result['records']), 'CPU programs;', len(references['checks']),
          'CPU reference checks;', references['FP8_codebook_checks'], 'FP8 codebook checks; GPU remains untested')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    validate(args.attempt, args.output)


if __name__ == '__main__':
    main()
