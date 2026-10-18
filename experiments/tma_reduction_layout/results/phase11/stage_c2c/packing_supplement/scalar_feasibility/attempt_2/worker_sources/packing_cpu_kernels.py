from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from experiments.tma_reduction_layout.phase11.c2.kernels import advance


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

