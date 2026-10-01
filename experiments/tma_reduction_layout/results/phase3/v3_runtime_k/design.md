# Phase 3 Step B v3: Single-Binary Runtime-K Isolation Design

## 1. Core Motivation & Problem Statement
In Phase 3 Step B v1 and v2, static compile-time unrolling (`tl.constexpr` trip counts) introduced critical confounds:
- Varying register pressure across K (e.g. 23 -> 43 regs).
- Divergent PTX / SASS instruction scheduling.
- Initial LocalLoad signature variations across K.
- Potential residency / active warp reduction confounds (Criterion F).

The objective of v3 is **strict mechanism isolation**:
1. Guarantee **exact identical binary execution** across all repeated reduction counts $K \in \{1, 2, 4, 8\}$.
2. Maintain zero register growth, zero spill, identical dynamic shared memory, and identical SM occupancy across K.
3. Keep the TMA descriptor load and the initial LocalLoad strictly invariant outside the loop.
4. Scale only the inner reduction execution via a runtime loop.

## 2. Kernel Design & Non-Specialization Architecture
```python
@triton.jit(do_not_specialize=["k_iters"])
def reduction_kernel_v3(
    a_ptr, out_ptr, stride_b, stride_m,
    k_iters: int, # runtime scalar trip count!
    B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    desc = tl.make_tensor_descriptor(...)
    x = desc.load([pid, 0, 0]) # Invariant LocalLoad outside loop
    acc = tl.zeros([1, N], dtype=tl.float32)
    
    for _ in tl.range(0, k_iters, loop_unroll_factor=1, disable_licm=True):
        x_opaque = tl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=r,r", [x], dtype=tl.bfloat16, is_pure=False, pack=2
        )
        r_i = tl.max(x_opaque.to(tl.float32), axis=1)
        acc += r_i
        
    offs_n = tl.arange(0, N)
    tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))
```

Key Architectural Principles:
1. `do_not_specialize=["k_iters"]`: Prevents Triton JIT from specializing integer values into constants, ensuring runtime scalar passing in IR (`%k_iters: i32`).
2. `loop_unroll_factor=1, disable_licm=True`: Forces MLIR `scf.for` emission with single loop body copy (`loop_body_copy_count = 1`).
3. Opaque Inline ASM Barrier: `mov.b32 $0, $1;` with `is_pure=False` prevents cross-iteration hoisting, value propagation, or reassociation, while maintaining exact candidate symmetry between `default` and `cand4`.
4. Single-Binary Warmup Compilation: Kernel is compiled once at $K=1$. Subsequent invocations at $K \in \{1, 2, 4, 8\}$ reuse the exact same CUBIN binary.

## 3. Loop Body Functional Decomposition
Inside the PTX loop body, instructions are strictly partitioned into 5 functional regions:
1. `sunk_localload_region`: Shared memory load (`ld.shared`) placed inside loop by compiler lowering.
2. `opaque_identity_region`: Opaque elementwise register barriers (`mov.b32`).
3. `canonical_reduction_region`: BF16->FP32 conversion, warp-level shuffles (`shfl.sync.bfly.b32`), comparisons (`max.f32`), and CTA shared memory barriers for `default`.
4. `accumulator_region`: Floating-point accumulation (`add.rn.f32`) into `acc`.
5. `loop_control_region`: Counter increment (`add.s32`), comparison (`setp`), and loop branch (`bra`).

## 4. Criteria Verification Matrix
The isolation is accepted if and only if all Criteria A through J pass.
