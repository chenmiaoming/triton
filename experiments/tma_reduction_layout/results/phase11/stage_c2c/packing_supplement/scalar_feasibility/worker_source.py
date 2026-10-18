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

    @g.jit(do_not_specialize=['K'])
    def probe(arg, out, K, R: gl.constexpr, C: gl.constexpr, L: gl.constexpr, STYLE: gl.constexpr):
        pid = gl.program_id(0)
        owner = gl.arange(0, 128, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
        rows = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
        cols = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
        offsets = owner[:, None, None]*R+rows[None, :, None]*C+cols[None, None, :]
        x = gl.load(arg+(pid % 1024)*128*R+offsets)
        step = ((((offsets*13) % 7)+1).to(gl.float32)*.25).to(gl.bfloat16)
        reduced_layout: gl.constexpr = gl.SliceLayout(1, L)
        checksum = gl.full([128], 0, gl.float32, gl.SliceLayout(1, reduced_layout))
        for i in range(K):
            x = advance(x, step, L)
            a, b = gl.split(gl.permute(x, (0, 2, 1)))
            if STYLE == 0:
                partial = gl.inline_asm_elementwise(
                    '{ .reg .b16 t; max.bf16 t, $1, $2; cvt.f32.bf16 $0, t; }',
                    constraints='=f,h,h', args=[a, b], dtype=gl.float32, is_pure=False, pack=1)
            else:
                partial = gl.inline_asm_elementwise(
                    '{ .reg .b32 a,b,v; .reg .b16 l,h; mov.b32 a, {$1,$1}; mov.b32 b, {$2,$2}; max.bf16x2 v,a,b; mov.b32 {l,h},v; cvt.f32.bf16 $0,l; }',
                    constraints='=f,h,h', args=[a, b], dtype=gl.float32, is_pure=False, pack=1)
            partial = gl.convert_layout(partial, reduced_layout)
            checksum += gl.sum(partial, axis=1)
        gl.store(out+pid*128*(R+1)+owner, checksum)
        gl.store(out+pid*128*(R+1)+128+offsets, x.to(gl.float32))

    root = Path('/evidence/stage_c2c/packing_cpu')/modal.current_function_call_id()
    root.mkdir(parents=True)
    result = {'GPU_allocated': False, 'native_SHA256': c.sha(Path(native.__file__).read_bytes()), 'cases': []}
    assert result['native_SHA256'] == protocol['baseline']['native_SHA256']
    result['ccache_before'] = subprocess.check_output(['ccache', '--print-stats'], text=True)
    for r in (4, 32):
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
def main():
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol = c.read(c.OUT/'packing_supplement/preregistration/protocol.json')
    relative = Path(__file__).resolve().relative_to(c.ROOT)
    snapshot = Path('/tmp/tma-phase11-c2c-snapshot')/relative
    snapshot.write_bytes(Path(__file__).read_bytes())
    protocol = {**protocol, 'worker_source_SHA256': {**protocol['worker_source_SHA256'], str(relative): c.sha(snapshot.read_bytes())}}
    root = c.OUT/'packing_supplement/scalar_feasibility'
    c.write_once(root/'protocol.json', protocol)
    call = inspect.spawn(protocol)
    c.write_once(root/'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming', 'GPU_allocated': False})
    value = call.get()
    c.write_once(root/'raw_export.zip', value)
    c.extract(value, root/'export')
    print('CPU-only scalar feasibility artifacts retained')
