"""Pure tensor BF16 scalar-max contrast; no hardware IDs or scalar backdoors."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from experiments.tma_reduction_layout.phase11.c2.kernels import advance


@gluon.jit
def scalar_max(a, b):
    return gl.inline_asm_elementwise('max.bf16 $0, $1, $2;', constraints='=h,h,h',
                                    args=[a, b], dtype=gl.bfloat16, is_pure=True, pack=1)


@gluon.jit(do_not_specialize=['K'])
def fixed_scalar(arg, out, K, R: gl.constexpr, P: gl.constexpr,
                 C: gl.constexpr, L: gl.constexpr):
    pid = gl.program_id(0)
    owner = gl.arange(0, 128, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    rows = gl.arange(0, P, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    offsets = owner[:, None, None]*R+rows[None, :, None]*C+cols[None, None, :]
    x = gl.load(arg+(pid % 1024)*128*R+offsets)
    step = ((((offsets*13) % 7)+1).to(gl.float32)*.25).to(gl.bfloat16)
    reduced_layout: gl.constexpr = gl.SliceLayout(1, L)
    consumer_layout: gl.constexpr = gl.SliceLayout(1, reduced_layout)
    checksum = gl.full([128], 0, gl.float32, consumer_layout)
    for i in range(K):
        x = advance(x, step, L)
        partial = gl.reduce(x, axis=1, combine_fn=scalar_max).to(gl.float32)
        partial = gl.convert_layout(partial, reduced_layout)
        checksum = checksum+gl.sum(partial, axis=1)
    checksum = gl.convert_layout(checksum, gl.SliceLayout(1, gl.SliceLayout(2, L)))
    gl.store(out+pid*128*(R+1)+owner, checksum)
    gl.store(out+pid*128*(R+1)+128+offsets, x.to(gl.float32))
