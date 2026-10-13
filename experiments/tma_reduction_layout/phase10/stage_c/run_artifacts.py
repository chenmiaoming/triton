"""Six exact-image workers; compile/correctness/resource evidence, no timing."""
from pathlib import Path
import io
import os
import zipfile
import modal
from experiments.tma_reduction_layout.phase10.stage_c import common as c

app = modal.App('triton-phase10-stage-c-artifacts')
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase10-evidence')
snapshot = Path(os.environ.get('TMA_STAGE_C_SNAPSHOT', '/tmp/tma-phase10-stage-c-snapshot'))


def make_image(base, patch=False):
    image = modal.Image.from_id(base)
    if patch:
        image = image.add_local_file(snapshot / 'test_source.py',
                                     '/opt/triton-src/python/test/unit/cuda/test_tensor_descriptor_cuda.py', copy=True)
    return (image.env({
        'PYTHONPATH': '/opt/triton-src', 'PATH':
        '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
    }).add_local_dir(snapshot / 'experiments', '/opt/triton-src/experiments', copy=True,
                     ignore=['**/__pycache__/**', '**/*.pyc', '**/results/**']))


baseline_image = make_image('im-qzTUZumWu2JXjWic8K82cZ')
patch_image = make_image('im-KaEq6823Uj1qypGzBbwFr9', True)


def collect(target, variant, protocol):
    import re
    import subprocess
    import traceback
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.tools.tensor_descriptor import TensorDescriptor
    from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    from experiments.tma_reduction_layout.phase10.stage_c.kernels import kernel as template
    from experiments.tma_reduction_layout.phase10.stage_c.data import input_tensor, output_reference

    def command(*args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

    spec = protocol['targets'][target]
    torch.cuda.init()
    name, cc = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
    assert spec['name'] in name and cc == spec['cc'], (name, cc)
    native_spec = protocol['compilers'][variant]
    native_sha = c.sha(Path(native.__file__).read_bytes())
    assert native_sha == native_spec['native_SHA256']
    for source, digest in native_spec['source'].items():
        assert c.sha((Path('/opt/triton-src') / source).read_bytes()) == digest, source
    for source, digest in protocol['worker_source_SHA256'].items():
        assert c.sha((c.SOURCE / source).read_bytes()) == digest, source
    cuda = ExactCUDA()
    env = {
        **cuda.environment(), 'target': target, 'variant': variant, 'gpu_name': name, 'compute_capability': cc,
        'call_id': modal.current_function_call_id(), 'native_SHA256': native_sha, 'source': native_spec['source'],
        'image_id': os.environ.get('MODAL_IMAGE_ID'), 'base_image': native_spec['image'], 'torch': torch.__version__,
        'torch_cuda': torch.version.cuda, 'protocol_SHA256': c.sha(c.encode(protocol)), 'patch_SHA256':
        protocol['patch_SHA256'], 'worker_source_SHA256': protocol['worker_source_SHA256'], 'GPU_timing': False,
        'ccache_start': command('ccache', '--print-stats')
    }
    ptxas = Path(get_ptxas(cc[0] * 10 + cc[1]).path)
    env['ptxas'] = {
        'SHA256': c.sha(ptxas.read_bytes()), 'path': str(ptxas), 'version': command(str(ptxas), '--version')
    }
    root = Path('/evidence/stage_c/artifacts') / target / variant / env['call_id']
    root.mkdir(parents=True)
    records = []
    tensors = {}
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda'))
    for case in protocol['cases']:
        tag = c.tag(case, variant)
        directory = root / tag.replace(':', '/')
        directory.mkdir(parents=True)
        record = {**case, 'variant': variant, 'target': target, 'tag': tag, 'status': 'FAILED'}
        compiled = None
        try:
            if case['dtype'] not in tensors:
                tensors[case['dtype']] = input_tensor(case['dtype'])
            x = tensors[case['dtype']]
            out, reference = output_reference(x, case)
            desc = TensorDescriptor.from_tensor(x, [1, case['M'], case['N']])
            arg = desc if case['path'] == 'host' else x
            kernel = triton.jit(template.fn)
            os.environ['TRITON_CACHE_DIR'] = '/tmp/stage_c_fresh/' + tag.replace(':', '/')
            os.environ.pop('TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT', None)
            events = []
            triton.knobs.runtime.jit_post_compile_hook = lambda **event: events.append(event['key'])
            compiled = kernel.warmup(arg, out, BM=case['M'], BN=case['N'], HOST=case['path'] == 'host', OP=case['op'],
                                     MODE=case['mode'], num_warps=case['num_warps'], grid=(1, ))
            assert len(events) == 1, (tag, len(events))
            record['actual_jit_compile_events'] = len(events)
            for ext in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
                value = compiled.asm[ext]
                (directory / ('kernel.' + ext)).write_bytes(value if isinstance(value, bytes) else value.encode())
            info = analyze_ttgir(compiled.asm['ttgir'])
            record['observed_layout'] = info['blocked_encodings'].get(info['local_load_dest_layout'])
            record['observed_axes'] = [
                int(x) for x in re.findall(r'"tt.reduce".*?<\{axis = (\d+)', compiled.asm['ttgir'])
            ]
            expected_axes = [] if case['mode'] == 'copy' else [
                2
            ] if case['mode'] == 'innermost' else [1, 1] if case['mode'] == 'multi_use' else [1]
            assert record['observed_axes'] == expected_axes, (tag, record['observed_axes'])
            for ext, option in [('sass', '-sass'), ('resource.txt', '-res-usage')]:
                (directory / ('kernel.' + ext)).write_text(command('cuobjdump', option,
                                                                   str(directory / 'kernel.cubin')))
            metadata = {
                **case, 'variant': variant, 'target': target, 'tag': tag, 'function_name': compiled.name, 'grid':
                c.grid(case), 'dynamic_smem_bytes': compiled.metadata.shared, 'global_scratch_size':
                compiled.metadata.global_scratch_size, 'global_scratch_align': compiled.metadata.global_scratch_align,
                'tensordesc_meta': compiled.metadata.tensordesc_meta, 'cubin_SHA256': c.sha(compiled.asm['cubin'])
            }
            (directory / 'metadata.json').write_bytes(c.encode(metadata))
            record['cubin_SHA256'] = metadata['cubin_SHA256']
            limit = env['device_limits']['max_shared_memory_per_block_optin']
            if metadata['dynamic_smem_bytes'] > limit:
                record['status'] = 'RESOURCE_UNAVAILABLE'
                raise RuntimeError(f'Dynamic SMEM {metadata["dynamic_smem_bytes"]} > target limit {limit}')
            occupancy = cuda.occupancy(directory / 'kernel.cubin', compiled.name, compiled.metadata.shared,
                                       case['num_warps'], c.sha((directory / 'kernel.resource.txt').read_bytes()))
            (directory / 'occupancy.json').write_bytes(c.encode(occupancy))
            assert occupancy['blocks_per_sm_actual_dynamic_smem'] > 0
            out.fill_(float('nan'))
            compiled[(metadata['grid'], 1, 1)](arg, out, case['M'], case['N'], case['path'] == 'host', case['op'],
                                               case['mode'])
            torch.cuda.synchronize()
            diff = (out.float() - reference.float()).abs().max().item()
            assert diff == 0, (tag, diff)
            record.update(status='OK', correctness_max_abs_diff=diff)
            del out, reference, desc, arg
        except Exception:
            record['error'] = traceback.format_exc()
            (directory / 'error.txt').write_text(record['error'])
            print(record['error'], flush=True)
        record['files'] = {p.name: c.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()}
        (directory / 'attempt.json').write_bytes(c.encode(record))
        records.append(record)
        print(target, variant, tag, record['status'], flush=True)
    env['ccache_end'] = command('ccache', '--print-stats')
    assert env['ccache_start'] == env['ccache_end'], 'Unexpected native rebuild'
    (root / 'environment.json').write_bytes(c.encode(env))
    (root / 'attempts.json').write_bytes(c.encode(records))
    volume.commit()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(root))
    return {'records': records, 'environment': env, 'archive': stream.getvalue()}


@app.function(image=baseline_image, gpu='H100!:1', timeout=3600, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def baseline_sm90(protocol):
    return collect('sm90', 'baseline', protocol)


@app.function(image=patch_image, gpu='H100!:1', timeout=3600, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def patch_sm90(protocol):
    return collect('sm90', 'patch', protocol)


@app.function(image=baseline_image, gpu='B200:1', timeout=3600, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def baseline_sm100(protocol):
    return collect('sm100', 'baseline', protocol)


@app.function(image=patch_image, gpu='B200:1', timeout=3600, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def patch_sm100(protocol):
    return collect('sm100', 'patch', protocol)


@app.function(image=baseline_image, gpu='RTX-PRO-6000:1', timeout=3600, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def baseline_sm120(protocol):
    return collect('sm120', 'baseline', protocol)


@app.function(image=patch_image, gpu='RTX-PRO-6000:1', timeout=3600, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def patch_sm120(protocol):
    return collect('sm120', 'patch', protocol)


@app.local_entrypoint()
def main():
    protocol = c.protocol()
    for target in protocol['targets']:
        for variant in protocol['variants']:
            root = c.OUT / 'artifacts' / target / variant
            if (root / 'raw_export.zip').exists():
                continue
            call = globals()[variant + '_' + target].spawn(protocol)
            c.write_once(root / ('dispatch_' + call.object_id + '.json'),
                         {'call_id': call.object_id, 'profile': os.environ.get('MODAL_PROFILE')})
            result = call.get()
            c.write_once(root / 'raw_export.zip', result['archive'])
            c.write_once(root / 'return.json', {'records': result['records'], 'environment': result['environment']})
            with zipfile.ZipFile(io.BytesIO(result['archive'])) as z:
                z.extractall(root / 'export')
