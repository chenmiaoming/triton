"""CPU-only, write-once local A10 cadence remeasurement registration."""
import argparse
import io
import re
from pathlib import Path
import zipfile

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.nonreduction_survey.elf import elf

HERE = Path(__file__).resolve().parent
DEFAULT = c.ROOT / 'experiments/layout_coordination/results/cadence_retest/a10_001'
UUID = 'GPU-69464998-e901-74ba-63ff-ae7022f8aed9'
OLD = {
    'layout': c.ROOT / 'experiments/tma_reduction_layout/results/local_a10/20261015_gpu0_r2',
    'versions': c.ROOT / 'experiments/tma_reduction_layout/results/local_a10_versions/20261015_gpu0_r2',
}


def inventory():
    programs = []
    for panel, root in OLD.items():
        collectors = [root/'collect/1/raw.json'] if panel == 'layout' else [
            root/f'collect/{v}/1/raw.json' for v in ('3.5.1', '3.6.0', '3.7.1')]
        for source in collectors:
            raw = c.read(source)
            assert raw['status'] == 'OK' and not raw['errors']
            for tag, meta in sorted(raw['artifacts'].items()):
                case = meta['case']
                role = ('U' if case.get('variant') == 'canonical' else
                        'G0' if case.get('path') == 'direct' and case.get('variant') == 'vec8' else
                        'candidate') if panel == 'layout' else 'U'
                directory = root/meta['archive_path']
                cubin = (directory/'kernel.cubin').read_bytes()
                assert c.sha(cubin) == meta['CUBIN_SHA256']
                text = (directory/'kernel.ptx').read_text()
                entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', text, re.S)
                assert entry
                abi = re.findall(r'\.param\s+\.(\w+)', entry[1])
                assert len(abi) in (2, 3, 4) and set(abi) == {'u64'}
                programs.append(dict(panel=panel, tag=tag, role=role, metadata=meta,
                    geometry=f"M{case['BM']}_N{case['BN']}_w{case['num_warps']}",
                    original_collector=str(source.relative_to(c.ROOT)),
                    original_collector_SHA256=c.sha(source.read_bytes()), ABI=abi,
                    payload_directory=panel+'/'+meta['archive_path']))
    assert len(programs) == 28
    equivalence = []
    for geom in sorted({x['geometry'] for x in programs}):
        matches = [x for x in programs if x['panel'] == 'layout' and x['geometry'] == geom]
        u = next(x for x in matches if x['role'] == 'U')
        g = next(x for x in matches if x['role'] == 'G0')
        read = lambda x: (OLD['layout']/x['metadata']['archive_path']/'kernel.cubin').read_bytes()
        assert elf(read(u)) == elf(read(g))
        assert u['ABI'] == g['ABI']
        for field in ('num_regs', 'dynamic_smem_bytes', 'num_spills', 'global_scratch_size'):
            assert u['metadata'][field] == g['metadata'][field]
        equivalence.append(dict(geometry=geom, U=u['tag'], G0=g['tag'],
            allocated_ELF_equal=True, ABI_equal=True, resource_metadata_equal=True,
            U_CUBIN_SHA256=u['metadata']['CUBIN_SHA256'], G0_CUBIN_SHA256=g['metadata']['CUBIN_SHA256'],
            scope='Authentic direct-load G0 only. cp_async changes the memory path and is a secondary diagnostic.'))
    return programs, equivalence


def prepare(root=DEFAULT):
    root = Path(root)
    assert not root.exists(), 'Use a new protocol revision; never overwrite a failed attempt.'
    programs, equivalence = inventory()
    protected = {}
    for old in OLD.values():
        for path in sorted(old.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                protected[str(path.relative_to(c.ROOT))] = c.sha(path.read_bytes())
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for program in programs:
            folder = OLD[program['panel']]/program['metadata']['archive_path']
            for name in ('kernel.cubin', 'kernel.ptx'):
                archive.writestr(program['payload_directory']+'/'+name, (folder/name).read_bytes())
    blob = payload.getvalue()
    source_names = [str(p.relative_to(c.ROOT)) for p in sorted(HERE.glob('a10_*.py'))]
    source_names += [str((HERE/'engine.py').relative_to(c.ROOT)),
        'experiments/layout_coordination/clock_control/telemetry.py',
        'experiments/tma_reduction_layout/phase11/operator_gpu/common.py',
        'experiments/tma_reduction_layout/phase11/operator_gpu/driver.py',
        'experiments/tma_reduction_layout/phase11/nonreduction_survey/elf.py',
        'experiments/tma_reduction_layout/phase11/nonreduction_survey/common.py',
        'experiments/layout_coordination/cadence_retest/cpu_verify.py',
        'experiments/tma_reduction_layout/phase8/launch.py',
        'experiments/tma_reduction_layout/phase4/exact_cuda.py',
        'experiments/tma_reduction_layout/local_a10/experiment.py']
    sources = {}
    for name in sorted(set(source_names)):
        data = (c.ROOT/name).read_bytes()
        sources[name] = c.sha(data)
        c.write_once(root/'snapshot'/name, data)
    protocol = dict(status='FROZEN_BEFORE_GPU_CONTEXT_OR_TIMING', stage='A10_cadence_remeasurement',
        target=dict(gpu_uuid=UUID, name='NVIDIA A10', cc=[8,6], physical_index=0,
            hardware_count=1, cohorts='Three sequential fresh processes on ONE physical A10; not three devices.',
            setting_writes=False, require_idle_before_each_process=True,
            other_process_monitoring='Pre-context idle gate and every context boundary; not continuous exclusion.'),
        source_SHA256=sources, protected_original_SHA256=protected,
        timing=dict(seed=2026110501, primes=8, host_gap_seconds=.001,
            telemetry_period_s=.05, warmups=3, visits=10, samples=31, formal_invocations=3,
            fixed_pre_process_settle_seconds=5,
            outcomes_never_trigger_repeats=True, bootstrap_draws=8192, bootstrap_seed=2026110502,
            practical_threshold_percent=3, equivalence_margin_percent=3,
            analysis_unit='Paired per-visit medians, process/visit blocked bootstrap; temporal replicates on one GPU.'),
        correctness=dict(input_shape=[1,8192,8192], output_shape=[256,8192],
            input_dtype='bf16', output_dtype='fp32',
            input_pattern='((linear_index % 97) - 48) / 4, exactly representable BF16',
            oracle='Independent analytic column max over 32 rows; all 2097152 outputs checked.',
            byte_guards=256, readonly_input_digest=True, captured_graph_readback_before_after=True,
            original_two_pattern_correctness='Pinned original collector quarter/random correctness retained; new observations use the frozen quarter pattern.'),
        suites={'a10':dict(programs=programs, G0_equivalence=equivalence,
            payload_SHA256=c.sha(blob), original_programs=28, contexts=4,
            repeat_shams=8, physical_observations=36,
            formal_raw_samples=36*3*10*31)},
        evaluation=dict(primary='Latest direct-load layout comparisons with same-round U and authentic G0.',
            secondary=['cp_async within-path vec comparison; changed memory path is explicit.',
                       'Frozen unmodified release U 3.6.0/3.5.1 and 3.7.1/3.5.1; no G0 exists for older releases.'],
            prior_context='Old/new timing changes also include driver and date differences; do not attribute them uniquely to cadence.',
            no_model_refit=True, no_architecture_transfer_claim=True,
            fixed_rule='Report all point/CI outcomes and sham/G0 equivalence; no success conditioned on a favorable winner.'))
    c.write_once(root/'preregistration/payload.zip', blob)
    c.write_once(root/'preregistration/protocol.json', protocol)
    c.write_once(root/'preregistration/cpu_gate.json', cpu_gate(protocol, blob, root/'snapshot'))
    print(root, 'CPU_FROZEN', len(programs), 'programs', flush=True)
    return protocol


def cpu_gate(p, payload, snapshot=None):
    assert c.sha(payload) == p['suites']['a10']['payload_SHA256']
    for name, digest in p['source_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest, name
        if snapshot is not None:
            assert c.sha((Path(snapshot)/name).read_bytes()) == digest,name
    for name, digest in p['protected_original_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest, name
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len(archive.namelist()) == 56
        for x in p['suites']['a10']['programs']:
            assert c.sha(archive.read(x['payload_directory']+'/kernel.cubin')) == x['metadata']['CUBIN_SHA256']
    assert len(p['suites']['a10']['G0_equivalence']) == 2
    return dict(status='PASS', programs=28, payload_files=56, authentic_G0_pairs=2,
        original_files_protected=len(p['protected_original_SHA256']),
        snapshot_verified=snapshot is not None,
        protocol_SHA256=c.sha(c.encode(p)), GPU_context_created=False, compile_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT)
    prepare(parser.parse_args().root)
