"""Freeze model constants and controls before C1.6 GPU execution."""
import subprocess
import argparse
from experiments.tma_reduction_layout.phase11.c16 import common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--reason', default='')
    args = parser.parse_args()
    if args.revision > 1:
        old = c.load_protocol()
        revised = {**old, 'revision': args.revision,
                   'previous_protocol_SHA256': c.sha(c.encode(old)),
                   'technical_revision_reason': args.reason,
                   'worker_source_SHA256': c.snapshot()}
        target = c.OUT / 'preregistration/revisions' / str(args.revision)
        c.write_once(target / 'protocol.json', revised)
        c.write_once(target / 'source_snapshot.zip', c.archive(c.SNAPSHOT / 'experiments'))
        print('C1.6 technical revision frozen', args.revision, args.reason)
        return
    previous = c.previous.load_protocol()
    c.previous.protect_prior()
    metadata, origins = {}, {}
    payload_root = c.SNAPSHOT / 'reuse'
    old_b = c.read(c.base.OUT / 'stage_b/artifact_gate.json')
    old_m = c.read(c.base.OUT / 'stage_c15/artifact_gate.json')
    for n, w in ((64, 8), (128, 4)):
        for vec in (1, 2, 4, 8):
            original = f'M32_N{n}_w{w}:bfloat16:max:host:fixed_vec{vec}'
            tag = f'M32_N{n}_w{w}:vec{vec}:FULL'
            folder = c.base.OUT / 'stage_b/artifacts' / old_b['accepted_attempt'] / 'export' / f'M32_N{n}_w{w}' / 'bfloat16/max/host' / f'fixed_vec{vec}'
            metadata[tag] = {**old_b['metadata'][original], 'tag': tag, 'family': 'FULL', 'vec': vec}
            origins[tag] = folder
            for family in ('ML', 'MR'):
                tag = f'M32_N{n}_w{w}:vec{vec}:{family}'
                folder = c.base.OUT / 'stage_c15/artifacts/attempt_1/export' / tag.replace(':', '/')
                metadata[tag] = {**old_m['metadata'][tag], 'full_grid': (8192 // 32) * (8192 // n)}
                origins[tag] = folder
    bindings = {}
    for tag, folder in origins.items():
        dest = payload_root / tag.replace(':', '/')
        dest.mkdir(parents=True, exist_ok=False)
        files = {}
        for name in ('kernel.cubin', 'kernel.ptx', 'kernel.glir', 'kernel.ttgir', 'kernel.llir', 'kernel.sass', 'kernel.resource.txt'):
            value = (folder / name).read_bytes()
            (dest / name).write_bytes(value)
            files[name] = c.sha(value)
        assert files['kernel.cubin'] == metadata[tag]['cubin_SHA256']
        bindings[tag] = {'directory': folder.relative_to(c.ROOT).as_posix(), 'files': files}
    new = [{'tag': f'M32_N{n}_w{w}:control:TMA', 'family': 'TMA', 'M': 32, 'N': n,
            'num_warps': w, 'path': 'host', 'dtype': 'bfloat16', 'full_grid': 256 * (8192 // n),
            'shared_layout': {'swizzle_byte_width': 128, 'element_bitwidth': 16, 'rank': 3, 'transposed': False}}
           for n, w in ((64, 8), (128, 4))]
    old_analysis = c.read(c.base.OUT / 'stage_c15/analysis.json')
    predictions = []
    for t in old_analysis['trends']:
        if t['grid'] != 8448:
            continue
        floor = 61.4 if t['N'] == 64 else 56.1
        for vec in (1, 2, 4, 8):
            cost = t['full_grid_hypothesis']['component_scaled_us'][str(vec)]
            predictions.append({'N': t['N'], 'vec': vec, 'C_onchip_us': cost,
                                'floor_us': floor, 'offset_us': 26.6,
                                'predicted_us': max(floor, 26.6 + cost)})
    protocol = {'stage': 'C1.6', 'date': '2026-10-16', 'scope': 'Bounded overlap/HBM validation only; no C2/model search/PR',
        'authorization': 'User asked to validate the proposed overlap mechanism and its numerical agreement',
        'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
        'baseline': previous['baseline'], 'upstream': previous['upstream'],
        'profile': 'chenmiaoming', 'target': {'gpu': 'H100!:1', 'cc': [9, 0], 'SM_count': 132},
        'reused_metadata': metadata, 'reused_bindings': bindings, 'new_cases': new,
        'new_program_limit': 2, 'reused_programs': 24, 'K': [16, 64, 256], 'micro_grid': 8448,
        'cache_eviction_bytes': 256 * 1024 * 1024,
        'timing': {'workers': 3, 'samples_per_cell': 20, 'visits': 4, 'samples_per_visit': 5,
                   'full_grid_divisors': [4, 2, 1], 'cache_modes': ['warm', 'cold'],
                   'micro_cells': 48, 'full_and_control_cells': 60, 'samples_total': 6480,
                   'warmups': 3, 'one_kernel_per_sample': True, 'graphs': False,
                   'subprocess_JIT': False, 'outcome_based_repeats': False},
        'profiles': {'cases': '8 FULL + 2 TMA, full grid, warm and cold', 'report_limit': 20,
            'replay': 'application', 'cache_control': 'none', 'clock_control': 'none',
            'metrics': ['dram__bytes_read.sum', 'dram__bytes_write.sum',
                        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
                        'lts__t_sector_hit_rate.pct', 'gpu__time_duration.sum',
                        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                        'sm__cycles_elapsed.avg.per_second', 'dram__cycles_elapsed.avg.per_second'],
            'profiler_durations_in_formal_statistics': False},
        'analysis': {'primary': 'Frozen 26.6us offset and 61.4/56.1us platforms; no refit',
            'frozen_predictions': predictions, 'full_grid_error_reference_percent': 3,
            'secondary': 'Same-worker micro slopes plus independently measured TMA control time; offset calibrated on N64 vec4 only; remaining 7 FULL points held out',
            'calibration_point': 'M32_N64_w8:vec4:FULL',
            'secondary_offset_is_empirical_not_pure_hardware_latency': True,
            'HBM_evidence': 'DRAM bytes and throughput, L2 hits, cache perturbation, independent transfer control',
            'counter_interpretation': 'Report achieved bandwidth and all utilization values; do not call flat timing alone proof of HBM saturation',
            'smaller_grids': 'diagnostic only; no arbitrary extension of full-grid offset scaling',
            'three_workers_are_not_three_independent_devices': True},
        'worker_source_SHA256': c.snapshot(), 'native_rebuild': False, 'PR_authorized': False,
        'stop_after': 'C1.6 evidence and report; do not start C2'}
    destination = c.OUT / 'preregistration'
    c.write_once(destination / 'prior_manifest.json', {'files': c.previous.freeze_prior()})
    c.write_once(destination / 'protocol.json', protocol)
    c.write_once(destination / 'source_snapshot.zip', c.archive(c.SNAPSHOT / 'experiments'))
    c.write_once(destination / 'reuse.zip', c.archive(payload_root))
    print('C1.6 frozen before GPU: 2 new transfer controls, 24 exact reused binaries; 6480 event samples; 20 reports')


if __name__ == '__main__':
    main()
