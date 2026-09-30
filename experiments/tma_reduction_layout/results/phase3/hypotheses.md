# Phase 3 Mechanism Isolation: Formal Hypotheses

> [!IMPORTANT]
> In accordance with Phase 3 Evidence Discipline, this document presents exactly 4 candidate mechanism hypotheses.
> Every statement is strictly partitioned into **OBSERVED** (directly witnessed in committed artifacts),
> **DERIVED** (computed from layout or architecture formulas), **HYPOTHESIS** (proposed causal explanation),
> and **FALSIFICATION TEST** (concrete differential experiment capable of disproving the hypothesis).

---

## Hypothesis 1: Regime Dichotomy (Memory-Bandwidth Saturation vs SM Communication Bottleneck)

- **OBSERVED**:
  - In `M32_N128_w4` (8 KiB tile), all candidates achieve identical marginal slope of `2.92 ns/additional CTA` (within 0.12% delta).
  - At 2.92 ns/CTA, `M32_N128_w4` achieves **2.80 TB/s input rate** (2.98 TB/s total DRAM traffic with output), which is **88.9% of H100 theoretical peak HBM3 bandwidth** (3.35 TB/s).
  - In `M32_N64_w8` (4 KiB tile), default achieves `3.88 ns/CTA` (**1.05 TB/s**, only 31.5% of peak bandwidth), while cand2 achieves `2.25 ns/CTA` (**1.82 TB/s**, 54.3% of peak bandwidth).
  - Pruning 24 shuffles and 10 barriers in `M32_N128_w4` produces zero runtime change, while pruning 24 shuffles and 4 barriers in `M32_N64_w8` produces a 36.8% runtime reduction.

- **DERIVED**:
  - Minimum DRAM transfer time for 8192 bytes input + 512 bytes output at 89% peak bandwidth (2.98 TB/s) is: `8704 bytes / 2.98 TB/s = 2.92 ns`.
  - Minimum DRAM transfer time for 4096 bytes input + 256 bytes output at 2.98 TB/s is: `4352 bytes / 2.98 TB/s = 1.46 ns`.
  - In `M32_N128_w4`, CTA duration equals the DRAM transfer floor. In `M32_N64_w8`, default CTA duration (3.88 ns) exceeds the DRAM transfer floor by 2.6x.

- **HYPOTHESIS**:
  - Layout candidate selection only exhibits material marginal throughput sensitivity (>10%) when the kernel operates in an **SM-bound/communication-bound regime** (far below physical memory bandwidth saturation).
  - When a workload is **memory-bandwidth saturated** (~85%+ peak DRAM bandwidth), all reductions in SM arithmetic, intra-warp shuffle, and cross-warp synchronization are completely hidden behind the physical memory bus latency and transfer limit.

- **FALSIFICATION TEST**:
  - Controlled differential sweep over tile size N with fixed warps (e.g., N=16, 32, 64, 128, 256).
  - *Falsification condition*: If any configuration operating at >80% peak DRAM bandwidth exhibits >15% layout slope separation, H1 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

---

## Hypothesis 2: Lane-Partitioning Pruning Dominates Over Warp-Partitioning in SM-Bound Regimes

- **OBSERVED**:
  - In `M32_N64_w8`, transitioning `default -> cand4` holds `warpPart[M]=8` constant while halving `lanePart[M]` from 4 to 2, achieving a **36.78% slope reduction** (`3.882 -> 2.454 ns`).
  - In contrast, transitioning `cand2 -> cand1` holds `lanePart[M]=1` constant while halving `warpPart[M]` from 8 to 4, achieving only a **1.28% slope reduction** (`2.254 -> 2.225 ns`).
  - In `default`, `lanePart[M]=4` forces `derived_M_elems_per_thread = 1`, which completely prevents thread-local reduction before communication.
  - In `cand4`, `lanePart[M]=2` provides 2 elements on M per thread, enabling **2x `max.bf16x2`** packed local reduction, eliminating 12 intra-warp shuffles and halving cross-warp exchange rounds.

- **DERIVED**:
  - Reducing `lanePart[M]` by 2x allows packed register-level SIMD folding (`max.bf16x2`) before any thread-to-thread communication, cutting total warp communication by 60%.

- **HYPOTHESIS**:
  - The primary driver of the large `default -> cand4` throughput improvement is the enablement of packed thread-local reduction and the elimination of intra-warp shuffle stages, while cross-warp combine topology differences contribute only secondary gains once lane reduction is eliminated.

- **FALSIFICATION TEST**:
  - Microbenchmark B (isolating thread-local reduction vs intra-warp shuffle vs cross-warp combine with fixed LocalLoad).
  - *Falsification condition*: If isolating cross-warp combine while maintaining intra-warp shuffles achieves equal or greater slope reduction than eliminating intra-warp shuffles, H2 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

---

## Hypothesis 3: LocalLoad Vectorization Penalty is Fully Masked by Communication Pruning

- **OBSERVED**:
  - `default` issues 1x `ld.shared.v4.b32` (128-bit vector load), while `cand4` issues 2x `ld.shared.v2.b32` (64-bit vector loads) and `cand1` issues 8x `ld.shared.b16` (scalar loads).
  - Despite issuing 2x or 8x narrower load instructions, `cand4`, `cand2`, and `cand1` all run substantially faster than `default` in `M32_N64_w8`.

- **DERIVED**:
  - 8 scalar loads require 8 separate instruction issues and address generations vs 1 issue for `ld.shared.v4`.
  - However, the 128-bit vector load enforces a distributed layout with `threadsPerWarp[M]=4`, which incurs 40 shuffles, 40 float maxes, and 14 barriers.

- **HYPOTHESIS**:
  - In TMA reduction workloads, the instruction issue penalty of narrower LocalLoad instructions is negligible compared to the latency and synchronization penalty imposed by the wider layout's reduction communication.
  - Narrower layout policies trade a trivial load-issue penalty for an enormous reduction in inter-thread communication.

- **FALSIFICATION TEST**:
  - Microbenchmark A (amplifying LocalLoad K times without reduction communication).
  - *Falsification condition*: If K*LocalLoad slope differences between `ld.shared.v4` and `ldmatrix` / `ld.shared.v2` exceed the shuffle/barrier latency differences observed in reduction, H3 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

---

## Hypothesis 4: Epilogue Register-Shuffle Layout Conversion Eliminates CTA Barrier Overhead

- **OBSERVED**:
  - In `cand4`, post-reduction layout conversion uses shared memory: `1x st.shared.v4.b32`, `2x bar.sync 0`, and `1x ldmatrix.x1`, requiring 10 total barriers.
  - In `cand2`, post-reduction layout conversion is performed entirely in registers via `2x shfl.sync.idx.b32` and `1x selp.b32`, requiring only 8 total barriers.
  - The marginal slope improves from `2.4543 ns` (cand4) to `2.2537 ns` (cand2) — an ~8.2% relative improvement.

- **DERIVED**:
  - `bar.sync 0` is a CTA-wide barrier that synchronizes all 256 threads across 8 warps.
  - `shfl.sync.idx` is intra-warp only, synchronizing only the 32 threads within a single warp without CTA-wide stall.

- **HYPOTHESIS**:
  - The ~0.20 ns/CTA gain from `cand4 -> cand2` is substantially driven by eliminating the 2 epilogue CTA-wide barriers and shared memory roundtrip, rather than being solely an artifact of the `ldmatrix` LocalLoad.

- **FALSIFICATION TEST**:
  - Microbenchmark C (comparing epilogue layout conversion via shared memory vs register shuffle while holding reduction body constant).
  - *Falsification condition*: If removing the epilogue shared-memory roundtrip and 2 barriers accounts for less than 20% of the observed ~0.20 ns delta between cand4 and cand2, H4 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`
