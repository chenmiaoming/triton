"""CPU-only feasibility check: can a same-semantic scalar BF16 max survive ptxas?"""
from pathlib import Path
import os
import modal
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2c import run as original

app = modal.App('triton-phase11-c2c-packing-feasibility')


@app.function(image=original.image, cpu=4, timeout=1200,
              volumes={'/cache': original.cache, '/evidence': original.evidence}, include_source=False)
def inspect(protocol):
    import subprocess, traceback
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental import gluon as g
    from triton.experimental.gluon import language as gl
    from experiments.tma_reduction_layout.phase11.c2.kernels import advance
    c.verify_sources(protocol)

    from experiments.tma_reduction_layout.phase11.c2c.packing_cpu_kernels import probe

    root = Path('/evidence/stage_c2c/packing_cpu')/modal.current_function_call_id()
    root.mkdir(parents=True)
    result = {'GPU_allocated': False, 'native_SHA256': c.sha(Path(native.__file__).read_bytes()), 'cases': []}
    assert result['native_SHA256'] == protocol['baseline']['native_SHA256']
    result['ccache_before'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    for r in (4, 8, 16, 32):
        for style in (0, 1):
            folder = root/f'R{r}_style{style}'
            folder.mkdir()
            row = {'R': r, 'style': style, 'status': 'FAILED'}
            try:
                constants = {'R': r, 'C': r//2, 'L': gl.BlockedLayout([1, 2, r//2], [32, 1, 1], [4, 1, 1], [2, 1, 0]), 'STYLE': style}
                signature = {'arg': '*bf16', 'out': '*fp32', 'K': 'i32', **{name: 'constexpr' for name in constants}}
                source = gluon.GluonASTSource(probe, signature, constants, {(1,): [['tt.divisibility', 16]]})
                compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32), options={'num_warps': 4})
                for ext in ('glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                    value = compiled.asm[ext]
                    c.write_once(folder/('kernel.'+ext), value if isinstance(value, bytes) else value.encode())
                sass = subprocess.check_output(['cuobjdump', '-sass', str(folder/'kernel.cubin')], text=True)
                c.write_once(folder/'kernel.sass', sass.encode())
                row.update(status='OK', max_SASS_sites=sum('MNMX' in line for line in sass.splitlines()),
                           CUBIN_SHA256=c.sha(compiled.asm['cubin']))
            except Exception:
                row['error'] = traceback.format_exc()
            result['cases'].append(row)
            print('CPU packing feasibility', r, style, row['status'], row.get('max_SASS_sites'), flush=True)
    result['ccache_after'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    assert result['ccache_before'] == result['ccache_after']
    c.write_once(root/'result.json', result)
    original.cache.commit()
    original.evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(attempt: int = 2):
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol = c.read(c.OUT/'packing_supplement/preregistration/protocol.json')
    relative = Path(__file__).resolve().relative_to(c.ROOT)
    snapshot = Path('/tmp/tma-phase11-c2c-snapshot')/relative
    snapshot.write_bytes(Path(__file__).read_bytes())
    kernel = c.SOURCE/'packing_cpu_kernels.py'
    kernel_relative = kernel.relative_to(c.ROOT)
    (Path('/tmp/tma-phase11-c2c-snapshot')/kernel_relative).write_bytes(kernel.read_bytes())
    protocol = {**protocol, 'worker_source_SHA256': {**protocol['worker_source_SHA256'], str(relative): c.sha(snapshot.read_bytes()), str(kernel_relative): c.sha(kernel.read_bytes())}}
    root = c.OUT/'packing_supplement/scalar_feasibility'/f'attempt_{attempt}'
    c.write_once(root/'protocol.json', protocol)
    call = inspect.spawn(protocol)
    c.write_once(root/'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming', 'GPU_allocated': False})
    value = call.get()
    c.write_once(root/'raw_export.zip', value)
    c.extract(value, root/'export')
    print('CPU-only scalar feasibility artifacts retained')
