# Phase 3 Mechanism Isolation: Formal Hypotheses

> [!IMPORTANT]
> In accordance with Phase 3 Evidence Discipline, this document presents exactly 4 candidate mechanism hypotheses.
> Every statement is strictly partitioned into **OBSERVED** (directly witnessed in committed artifacts and empirical runs),
> **DERIVED** (computed from layout or architecture formulas), **HYPOTHESIS** (proposed causal explanation),
> **FALSIFICATION TEST** (concrete differential experiment capable of disproving the hypothesis),
> and **STATUS**.

---

## Hypothesis 1: Bandwidth-Roof / Overlap Hypothesis

- **OBSERVED**:
  - In `M32_N128_w4` (8 KiB input tile), current empirical marginal slopes range from `2.9681` to `2.9809` ns/additional CTA (0.431% span relative to default).
  - At the current default slope of 2.9771 ns/CTA, `M32_N128_w4` has a logical input rate of **2751.7 GB/s** and logical I/O rate of 2923.7 GB/s.
  - In `M32_N64_w8` (4 KiB input tile), default has marginal slope `3.9449 ns/CTA` (**1038.3 GB/s** logical input rate), while cand2 has `2.2766 ns/CTA` (**1799.2 GB/s** logical input rate).
  - Pruning shuffles/barriers in `M32_N128_w4` coincides with near-parity grid slopes, while pruning 24 shuffles and 4 barriers in `M32_N64_w8` coincides with a 37.00% marginal-slope reduction.
  - Actual DRAM/HBM traffic, cache hit rates, and hardware memory utilization are **UNKNOWN** (not measured via hardware counters).

- **DERIVED**:
  - Arithmetic logical-byte scaling: 8704 bytes / 2.9771 ns = 2.9237 TB/s logical rate. Half the logical bytes at this assumed rate gives 1.4885 ns; this is not a measured transfer floor.
  - The primary default marginal grid slope (3.9449 ns/CTA) is 2.650 times that arithmetic half-byte scaling. Neither quantity is single-CTA latency or a measured hardware floor.

- **HYPOTHESIS**:
  - The negative case (`M32_N128_w4`) may be limited by a memory-system throughput roof or pipeline overlap that hides reductions in SM-side communication cost.
  - Layout candidate selection only exhibits material marginal throughput sensitivity (>10%) when the workload is not bottlenecked by memory-system transfer limits.

- **FALSIFICATION TEST**:
  - Controlled differential sweep over tile size N with fixed warps (e.g., N=16, 32, 64, 128, 256) paired with a streaming read/copy benchmark on the same device and buffer sizes.
  - *Falsification condition*: If a configuration operating near the empirical logical throughput ceiling exhibits >15% layout slope separation, H1 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

---

## Hypothesis 2: Lane-Partitioning Pruning Dominates Over Warp-Partitioning in SM-Sensitive Regimes

- **OBSERVED**:
  - In the audited `M32_N64_w8` `default -> cand4` artifact, transitioning `lanePart[M]` from 4 to 2 (while holding `warpPart[M]=8` constant) coincides with whole-kernel shuffles dropping from 40 to 16 (-60%), barriers dropping from 14 to 10 (-28.6%), and marginal slope dropping from 3.9449 ns to 2.4854 ns (-37.00%).
  - In contrast, transitioning `cand2 -> cand1` holds `lanePart[M]=1` constant while halving `warpPart[M]` from 8 to 4, coinciding with only a -0.96% slope change (2.2766 -> 2.2548 ns).
  - In `default`, `lanePart[M]=4` forces `derived_M_elems_per_thread = 1`, which prevents thread-local reduction before communication.
  - In `cand4`, `lanePart[M]=2` provides 2 elements on M per thread, enabling **2x `max.bf16x2`** packed local reduction.

- **DERIVED**:
  - In `M32_N64_w8`, halving `lanePart[M]` enables packed SIMD reduction before inter-thread communication, cutting intra-warp reduction stages from 2 to 1 and eliminating an entire cross-warp exchange round.

- **HYPOTHESIS**:
  - The primary driver of the large `default -> cand4` throughput improvement is the enablement of packed thread-local reduction and the elimination of intra-warp shuffle stages, while cross-warp combine topology differences contribute only secondary gains once lane reduction is eliminated.

- **FALSIFICATION TEST**:
  - Microbenchmark B (isolating thread-local reduction vs intra-warp shuffle vs cross-warp combine with fixed LocalLoad).
  - *Falsification condition*: If isolating cross-warp combine while maintaining intra-warp shuffles achieves equal or greater slope reduction than eliminating intra-warp shuffles, H2 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

### Hypothesis Lineage & Decomposition (Phase 3 Step E Update)

- **Hypothesis 2a (H2a)**: Composite reduction-body structure materially contributes to positive default-vs-cand4 throughput separation.
  - **Lineage**: Refinement of H2 to composite reduction body level.
  - **Evidence**: Phase 3 Step E single-binary repeated reduction isolates $\Delta g(1) = 1.0929 \pm 0.0118$ ns/additional CTA on `M32_N64_w8`.
  - **Magnitude Attribution**: The isolated one-reduction differential has a magnitude equal to 74.88% of the canonical default-vs-cand4 marginal-slope gap (cross-harness descriptive magnitude comparison, not an additive causal decomposition).
  - **Sub-Hypothesis Status**: `SUPPORTED_AT_REDUCTION_BODY_LEVEL`

- **Hypothesis 2b (H2b)**: Lane-partition pruning dominates warp-partition pruning.
  - **Lineage**: Specific sub-claim of H2 attributing primary gain to lane partition rather than warp partition.
  - **Evidence**: Unisolated; no matched orthogonal lane-only vs warp-only intervention has been evaluated.
  - **Sub-Hypothesis Status**: `UNVERIFIED`

- **Hypothesis 2c (H2c)**: Intra-warp communication dominates cross-warp communication.
  - **Lineage**: Specific sub-claim of H2 attributing gain to intra-warp communication reduction.
  - **Evidence**: Unisolated; composite repeated reduction does not separate thread-local vs shuffle vs smem vs barrier components.
  - **Sub-Hypothesis Status**: `UNVERIFIED`

---

## Hypothesis 3: LocalLoad-Cost vs Reduction-Communication Trade-Off

- **OBSERVED**:
  - cand1 contains 8 scalar shared-load PTX instructions, whereas default contains 1 vector shared-load PTX instruction.
  - In `M32_N64_w8`, `default` issues 1x `ld.shared.v4.b32` (128-bit vector load), while `cand4` issues 2x `ld.shared.v2.b32` (64-bit vector loads) and `cand1` issues 8x `ld.shared.b16` (scalar loads).
  - Despite issuing narrower load instructions, `cand4`, `cand2`, and `cand1` all run substantially faster than `default` in `M32_N64_w8`.

- **DERIVED**:
  - The 128-bit vector load enforces a distributed layout with `threadsPerWarp[M]=4`, requiring 40 whole-kernel shuffles and 14 barriers.

- **HYPOTHESIS**:
  - The cost added by narrower LocalLoad lowering may be smaller than the communication cost removed by the associated layout change in `M32_N64_w8`.
  - Narrower layout policies trade a load-issue penalty for a substantial reduction in inter-thread communication.

- **FALSIFICATION TEST**:
  - Microbenchmark A (amplifying LocalLoad K times without reduction communication).
  - *Falsification condition*: If K*LocalLoad slope differences between `ld.shared.v4` and narrower lowerings exceed the communication/barrier differences observed in reduction, H3 is falsified.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

---

## Hypothesis 4: Epilogue-Conversion Contribution Hypothesis

- **OBSERVED**:
  - In `cand4`, post-reduction layout conversion uses shared memory: `1x st.shared.v4.b32`, `2x bar.sync 0`, and `1x ldmatrix.x1`, requiring 10 total barriers.
  - In `cand2`, post-reduction layout conversion is performed entirely in registers via `2x shfl.sync.idx.b32` and `1x selp.b32`, requiring only 8 total barriers.
  - The marginal slope improves from `2.4854 ns` (cand4) to `2.2766 ns` (cand2) — a 8.40% relative slope change.

- **DERIVED**:
  - `bar.sync 0` is a CTA-wide barrier synchronizing all 256 threads across 8 warps.
  - `shfl.sync.idx` performs warp-local register data exchange under the instruction's synchronization-mask semantics; unlike bar.sync, it is not a CTA-wide barrier.

- **HYPOTHESIS**:
  - The post-reduction conversion change may contribute materially to the `cand4 -> cand2` throughput improvement.

- **FALSIFICATION TEST**:
  - Microbenchmark C (comparing epilogue layout conversion via shared memory vs register shuffle while holding reduction body constant).
  - Reports the exact ratio of the isolated epilogue conversion delta relative to the original `cand4 -> cand2` delta.

- **STATUS**: `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH`

Current canonical comparison is derived from sample medians and OLS on B={16384,32768,65536}, averaged across three runs. Raw SHA256: `75087a0ea79d359e4b1820be729295db3f0d7cf392cdfde2a888b63dbb453e9b`; last-change commit: `a57bff355124dd3d80e6ff4116e4796d1eeb48de`. H2a uses the declared retrospective Step E decision rule; integrity checks do not require support. See the Phase 3 freeze note for occupancy and archival limits.