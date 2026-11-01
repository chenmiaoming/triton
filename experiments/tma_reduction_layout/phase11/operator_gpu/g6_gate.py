"""Promote a complete source-only baseline matrix and bind actual workloads."""
import argparse
from pathlib import Path
from . import common as c
from .g5_analyze import elf
from .g6_analyze import analyze
from .g6_prepare import OUT


def main(attempt):
    folder = OUT/f'attempt_{attempt:03}'
    p = c.read(folder/'preregistration/protocol.json'); analysis = analyze(attempt)
    assert len(analysis['rows']) == len(p['cases']) == 13
    assert len({r['case'] for r in analysis['rows']}) == 13
    assert {r['case'] for r in analysis['rows']} == {item['case']['id'] for item in p['cases']}
    assert all(r['equivalent'] for r in analysis['rows']), 'Source-only baseline matrix is not yet equivalent'
    c.verify_sources(p)
    bycase = {r['case']: r for r in analysis['rows']}
    original = c.read(c.OUT/'stage_g4b/compile_gate_v2.json')
    groups = c.read(c.OUT/'stage_g4b/preregistration/protocol.json')['groups']
    bindings = []
    for group in groups:
        gid = group['id']; case = group['case']['id']
        if case.startswith('C02'):
            key = case + '_b8_l2048'
        elif '_b1_' in gid:
            key = gid
        else:
            key = case
        repaired = bycase[key]; meta = repaired['metadata']
        u = next(r for r in original['records'] if r['tag'] == gid+':actual:upstream')
        refroot = c.ROOT/original['CPU_export_directory']/u['artifact_directory']
        assert repaired['artifact']['allocated_sections'] == elf((refroot/'kernel.cubin').read_bytes())
        assert all(meta[k] == u[k] for k in ['num_warps', 'runtime_argument_names', 'dynamic_smem_bytes', 'global_scratch_size'])
        bindings.append(dict(group=gid, source_baseline_case=key,
                             original_U_CUBIN_SHA256=u['cubin_SHA256'],
                             source_baseline_CUBIN_SHA256=meta['cubin_SHA256'],
                             baseline_artifacts=repaired['artifact']['artifacts']))
    manifest = dict(status='OK', version='G0_source_aligned_v1', attempt=attempt,
                    compiler=p['compiler'], native_SHA256=p['native_SHA256'], backend_SHA256=p['backend_SHA256'],
                    target=dict(backend='cuda', arch=90, warp_size=32), compiler_intervention=False,
                    default_compiler_pipeline=True, GPU_allocated=False, new_performance_claim=False,
                    protocol_path=str((folder/'preregistration/protocol.json').relative_to(c.ROOT)),
                    protocol_SHA256=c.sha(c.encode(p)),
                    adapter='experiments.tma_reduction_layout.phase11.operator_gpu.g6_compile.arguments',
                    family_functions={'C01': 'g6_layernorm.merge', 'C02': 'g6_sglang._fwd_kernel_stage1', 'C06': 'g6_vllm.reduce_segments'},
                    source_SHA256={name: digest for name, digest in p['worker_source_SHA256'].items()
                                   if Path(name).name.startswith('g6_')},
                    cases=[dict(case=r['case'], metadata=r['metadata'], artifacts=r['artifact']['artifacts'])
                           for r in analysis['rows']], actual_G4b_workloads=bindings,
                    scope='Frozen dtypes/options/ABIs only. New candidates and specializations must pass effective-layout, correctness and ABI gates before timing.')
    c.write_once(OUT/'baseline_manifest.json', manifest)
    print('G6 baseline gate: 13/13 source-only programs, 10/10 actual workload bindings')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    main(parser.parse_args().attempt)
