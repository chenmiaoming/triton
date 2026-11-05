# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# This file is modified and supported by the Moonshot AI Team
"""Exact forward-only U extraction; FLA_USE_FAST_OPS=0, NVIDIA softplus.

The original autotune configs are resolved separately using only U.
"""
import triton
import triton.language as tl

def _generate_softplus(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 20.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p mul.f32            ${out_reg}, ${in_reg}, 1.4426950408889634;
        @!p ex2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p mul.f32            ${out_reg}, ${out_reg}, 0.6931471805599453;
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str

def _generate_constraints(num_pack):
    return ",".join("=r" for i in range(num_pack)) + "," + ",".join("r" for i in range(num_pack))

_NUM_REG = 1

s_softplus: tl.constexpr = tl.constexpr(_generate_softplus(_NUM_REG))

s_constraints: tl.constexpr = tl.constexpr(_generate_constraints(_NUM_REG))

NUM_REG: tl.constexpr = tl.constexpr(_NUM_REG)

@triton.jit
def softplus_nv(x):
    # equivalent to:
    # return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )

@triton.jit
def softplus(x):
    return softplus_nv(x)

@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))

@triton.jit(do_not_specialize=['T'])
def kda_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    yg,
    yb,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)

    b_A = tl.load(A_log + i_h).to(tl.float32) if HAS_A else 1.0

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    p_g = g + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    p_yg = yg + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    # [BT, BD]
    b_g = tl.load(p_g, mask=m_g, other=0.0).to(tl.float32)
    if HAS_BIAS:
        o_b = i_h * D + tl.arange(0, BD)
        b_g = b_g + tl.load(dt_bias + o_b, mask=o_b < H * D, other=0.0).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_yg = -(exp(b_A) if HAS_A else b_A) * softplus(b_g)
    else:
        b_yg = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), mask=m_g)

    if HAS_BETA:
        p_b = beta + i_h + o_t * H
        p_yb = yb + i_h + o_t * H
        b_yb = tl.sigmoid(tl.load(p_b, mask=m_t, other=0.0).to(tl.float32))
        tl.store(p_yb, b_yb.to(p_yb.dtype.element_ty), mask=m_t)
