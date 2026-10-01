# Phase 3 Step B v5: Preloaded-Register Last-Result Carry Isolation Design

## 1. Architectural Concept

The v5 design eliminates the candidate-dependent accumulator adds by maintaining only the last reduction result across iterations.

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
last = tl.zeros([1, N], dtype=tl.float32)
for i in tl.range(0, k_iters, loop_unroll_factor=1, disable_licm=True):
    x_iter = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=r,r",
        [x_preloaded],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=2,
    )
    last = tl.max(x_iter.to(tl.float32), axis=1)
offs_n = tl.arange(0, N)
tl.store(out_ptr + pid * N + offs_n, tl.reshape(last, [N]))
```
