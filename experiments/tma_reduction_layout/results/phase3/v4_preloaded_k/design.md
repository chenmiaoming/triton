# Phase 3 Step B v4: Preloaded-Register Runtime-Loop Isolation Design

## 1. Architectural Concept

The v4 design isolates the reduction communication body by preloading the input tile from shared memory
into thread registers before entering the runtime `k_iters` loop.

```python
x = desc.load([pid, 0, 0])
# Materialize shared->register BEFORE loop
x_preloaded = tl.inline_asm_elementwise(
    "mov.b32 $0, $1;",
    "=r,r",
    [x],
    dtype=tl.bfloat16,
    is_pure=False,
    pack=2,
)
acc = tl.zeros([1, N], dtype=tl.float32)
for i in tl.range(0, k_iters, loop_unroll_factor=1, disable_licm=True):
    # In-loop anti-CSE barrier
    x_iter = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=r,r",
        [x_preloaded],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=2,
    )
    r_i = tl.max(x_iter.to(tl.float32), axis=1)
    acc += r_i
offs_n = tl.arange(0, N)
tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))
```

## 2. Invariant Proof Goals

1. **TMA Invariant**: Exactly 1 async TMA copy global-to-local outside `scf.for`.
2. **LocalLoad Invariant**: Exactly 0 tile loads inside `scf.for`. All tile shared loads execute prior to loop entry.
3. **Reduction Body Template Invariant**: The reduction instructions inside the loop match the canonical Step A reduction sequence instruction-for-instruction.
4. **Single-Binary Reusability**: Single CUBIN executes across all K in {1, 2, 4, 8}.
5. **Numerical Correctness**: Numerical results match K * ref_max across all K.
6. **Residency Verification**: cuOccupancyMaxActiveBlocksPerMultiprocessor comparison between candidates.
