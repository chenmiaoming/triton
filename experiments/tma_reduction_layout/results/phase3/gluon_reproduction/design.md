# Phase 3 Step C: Gluon Canonical Structural Reproduction Design

## 1. Architectural Concept

Gluon allows explicit specification of shared-memory and distributed-register layouts,
bypassing heuristic compiler inference and opaque compiler barriers.

```python
@gluon.jit
def gluon_canonical_reduction_kernel(
    in_desc, out_ptr,
    register_layout: gl.constexpr,
    shared_layout: gl.constexpr,
    B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr
):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, M, N], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid, 0, 0], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    x = smem.load(register_layout)
    x_f32 = x.to(gl.float32)
    r = gl.max(x_f32, axis=1)
    offs_0 = gl.arange(0, 1, layout=gl.SliceLayout(1, gl.SliceLayout(1, register_layout)))[:, None]
    offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, gl.SliceLayout(1, register_layout)))[None, :]
    gl.store(out_ptr + pid * N + offs_0 * N + offs_n, r)
```

## 2. Explicit Layout Bindings

- `M32_N64_w8 default`: `gl.BlockedLayout([1, 1, 8], [1, 4, 8], [1, 8, 1], [2, 1, 0])`
- `M32_N64_w8 cand4`: `gl.BlockedLayout([1, 1, 4], [1, 2, 16], [1, 8, 1], [2, 1, 0])`
- `M32_N128_w4 default`: `gl.BlockedLayout([1, 1, 8], [1, 2, 16], [1, 4, 1], [2, 1, 0])`
- `M32_N128_w4 cand4`: `gl.BlockedLayout([1, 1, 4], [1, 1, 32], [1, 4, 1], [2, 1, 0])`
- `Shared Layout`: `gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=3, transposed=False)`


Exact matches compare the complete filtered normalized fingerprint: selected opcode families and shuffle/barrier immediates. Most operands/predicates are ignored; matching does not establish dataflow or full PTX/SASS/CUBIN equality. Initial LocalLoad matching checks every opcode/width.
Theoretical residency is not achieved occupancy. Canonical occupancy-query records use a resource-matched recompile, with a different recorded CUBIN SHA from the measured canonical binary. See ../freeze.md for archival limits.