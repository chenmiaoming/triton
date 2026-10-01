# Phase 3 Step B: Reduction-Communication Amplification Microbenchmark Design

## 1. Executive Research Question & Objective

The objective of this microbenchmark is to test **Hypothesis 2 (H2)**:
> Does the ~36.8% marginal-slope throughput separation between `default` and `cand4` in `M32_N64_w8` systematically amplify when the reduction body is repeatedly executed $K$ times, while holding TMA transfer count and initial LocalLoad strictly constant?

## 2. Experimental Configurations & Candidates

The benchmark evaluates two strictly matched configurations:
1. **Positive Case (`M32_N64_w8`)**: $M=32, N=64, \text{num\_warps}=8$
   - Strong Phase 2 layout sensitivity: default $\approx 3.882$ ns/CTA vs cand4 $\approx 2.454$ ns/CTA ($-36.78\%$ marginal slope).
   - Warp partitions are identical across default and cand4 (`warpsPerCTA[M]=8`).
   - Investigates whether reducing lane partitions from 4 to 2 (pruning 24 butterfly shuffles and 8 cross-warp shuffles) amplifies proportionally with $K$.
2. **Negative Control (`M32_N128_w4`)**: $M=32, N=128, \text{num\_warps}=4$
   - Zero Phase 2 layout sensitivity: default $\approx 2.923$ ns/CTA vs cand4 $\approx 2.920$ ns/CTA ($-0.11\%$ marginal slope).
   - Also undergoes substantial instruction pruning (removes 16 butterfly shuffles ($24 \times K$ vs $8 \times K$) and 8 cross-warp shuffles), but exhibits negligible throughput response.
   - Validates whether communication pruning is execution-regime dependent rather than universally beneficial.

Candidates tested: strictly `default` and `4` (`cand4`).

## 3. Mathematical Amplification Structure $F(v_i, r_i, m, i)$

To prevent LLVM/Triton from algebraically folding or dead-code eliminating repeated reductions, the kernel uses a strict recurrence structure:
```python
m_pat = tl.reshape(tl.arange(0, M), [1, M, 1]).to(tl.float32) * 0.05 + 1.0
v = x  # loaded tile [1, M, N], bfloat16
acc = tl.zeros([1, N], dtype=tl.float32)

for i in tl.static_range(K_ITERS):
    v_f32 = v.to(tl.float32)
    r_i = tl.max(v_f32, axis=1)  # shape [1, N]
    acc += r_i
    if i < K_ITERS - 1:
        v_next_f32 = v_f32 * m_pat + tl.reshape(r_i, [1, 1, N])
        v = v_next_f32.to(tl.bfloat16)

tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))
```

Key design properties:
- **Non-uniformity across $M$**: `m_pat` multiplies each row $m$ by $(1.0 + 0.05 \times m)$, preventing $\max(x + c) = \max(x) + c$ folding.
- **Data dependency**: Iteration $i+1$ directly consumes the reduction output $r_i$ from iteration $i$.
- **Zero SMEM roundtrip**: The update operates strictly on register tiles.
- **Exact layout preservation**: Casting back to `tl.bfloat16` preserves vector packing and enables `cand4` to execute `max.bf16x2` in all $K$ iterations.
- **Single TMA & LocalLoad**: Exactly one TMA descriptor load and one initial LocalLoad sequence are issued before the loop.

## 4. Verification Protocol & Invariants

Before timing, compiled artifacts must satisfy 5 invariants:
1. **TMA count**: exactly 1 TMA descriptor load / bulk copy across all $K$.
2. **Initial LocalLoad**: instruction family and count constant across all $K$ (`1x ld.shared.v4.b32` for default, `2x ld.shared.v2.b32` for cand4 in `M32_N64_w8`).
3. **Zero local memory spill**: `LOCAL bytes == 0`, `STACK == 0`.
4. **Distributed layout**: `threadsPerWarp`, `warpsPerCTA`, `sizePerThread` invariant across all $K$.
5. **Physical register tracking**: recorded per $K$; transitions across occupancy thresholds flagged.

## 5. Timing & Statistical Protocol

- Fixed binary: compiled once with $B_{\text{desc}}=65536$, executed across $B_{\text{run}} \in \{16384, 32768, 65536\}$.
- 10 rounds, 10 samples per condition per round (100 total samples per condition).
- Rotated grid-size order and candidate/K order across rounds.
- 3 sequential remote benchmark invocations on the same physical H100 (`same-device temporal replication`).
- Primary metric: marginal grid-time slope $b_K = \Delta T / \Delta B$ (ns/additional CTA).
- Primary test metric: $\text{gap}(K) = b_{\text{default}}(K) - b_{\text{cand4}}(K)$.
- Amplification slope: $\Delta b / \Delta K$ and $\Delta \text{gap} / \Delta K$ (ns / additional CTA / additional reduction body).

## 6. Predefined Classification Criteria

- `AMPLIFIES`: $\text{gap}(K)$ increases monotonically or linearly with $K$ across all 3 runs with $\Delta \text{gap} / \Delta K > 0$.
- `NO_AMPLIFICATION`: Opcode counts scale with $K$, but $\text{gap}(K)$ remains flat or near zero.
- `CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE`: Initial LocalLoad signature mismatch, non-identical reduction body template, or material register pressure growth.
- `CONFOUNDED`: Register spill, topology change, or compiler simplification occurs.
- `UNSTABLE`: Inconsistent direction across invocations.
