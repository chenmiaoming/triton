# Phase 3 Step D.1: Gluon Repeated-Reduction Isolation Design

## 1. Architectural Concept

The Step D design achieves clean repeated-reduction isolation using compiler barriers:

```python
@gluon.jit(do_not_specialize=["num_reductions"])
def gluon_repeated_reduction_kernel(
    in_desc, out_ptr, num_reductions: gl.int32,
    register_layout: gl.constexpr,
    shared_layout: gl.constexpr,
    B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
    X_CONSTRAINTS: gl.constexpr, R_CONSTRAINTS: gl.constexpr,
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

    for _ in range(0, num_reductions):
        # Prototype X: Input tied barrier (0 explicit PTX insts; induced copies present; no explicit loop SASS MOV observed)
        x_iter = gl.inline_asm("", X_CONSTRAINTS, [x], x.type, is_pure=False)
        # Canonical reduction core
        r = gl.max(x_iter.to(gl.float32), axis=1)
        # Prototype Y: Result sink (0 explicit PTX insts, 0 induced copies)
        gl.inline_asm("", R_CONSTRAINTS, [r], (), is_pure=False)

    gl.store(out_ptr + pid, gl.to_tensor(0.0))
```

## 2. Invariant Properties Established

1. **TMA & Initial LocalLoad**: Issued exactly once outside the loop.
2. **Residency Match**: `blocks_per_sm = 8` for w8, `blocks_per_sm = 16` for w4 across default and cand4.
3. **Zero Register Spills**: `LOCAL=0, STACK=0`.
4. **Single-Binary**: Identical CUBIN SHA256 across all R.
5. **SASS Barrier Overhead**: No explicit MOV/IMAD.MOV observed in the runtime region; no total-cost attribution is made.


Exact sequence compares the complete filtered normalized reduction fingerprint, including selected opcodes and shuffle/barrier immediates. Most operands and predicates are ignored; this is not dataflow/full PTX/SASS/CUBIN equality. Secondary opcode multiset equality does not prove topology.
All structural conditions, complete loop memory signature, resources, and recorded binary bindings are required. The archived gate authorizes no new timing. Indirect live-range, allocation, and scheduler effects remain possible.