# Historical timing impact and remeasurement boundaries

This is a new audit of old evidence, not an edit or replacement of any historical raw sample, compiler artifact, model fit or blind prediction. A host gap before a timed event can alter the GPU state even when the host work itself is outside the event interval. Observed visit-to-visit variation alone does not identify that cause. Conversely, initial warmups, frozen CUBINs or paired order alone do not prove immunity.

The screening statistic in `legacy_timing_screen.json` is `100 × (maximum visit median / minimum visit median − 1)` within one process and one original program/grid. It is descriptive, has no multiplicity correction and is not a hypothesis test or a replacement performance estimate.

| Evidence family | Original timing method and current status | Action in this cadence round |
| --- | --- | --- |
| Early Phase 1–2 exploration | Initial warmups and single launches bounded by CUDA events, with Python launch/wrapper work between launches. No fixed host gap or in-graph priming. The smallest-grid sweep was already exploratory. | Preserve as historical exploration. It does not certify the current source-aligned operator baseline or a general compiler selector. Do not retime every superseded experiment. |
| Phase 3 structural / frozen-binary exploration | Some measured binaries retain hashes/PTX/SASS but not executable CUBIN bytes. Cross-harness timing ratios were already descriptive. | Retain structural evidence and its original limits. An exact-binary replay cannot be claimed when the executable bytes were not archived. |
| Phases 4–8 exact-binary controls | Driver-only direct event timing, archive/in-memory hash guards and initial condition warmups. Phase 8 also captures boundary device telemetry. No common eight-prime graph or fixed per-observation cadence. | Historical descriptor/layout controls remain evidence about those source programs. Current H100 natural FULL conclusions are rechecked by the complete C2d/D1 anchors rather than treating every previous phase as a new experiment. Boundary telemetry is not per-sample frequency control. |
| Phase 9 cross-architecture fixed-vector work | Architecture-specific archived binaries, direct event timing with per-launch guards and initial warmups. This was a fixed-vector intervention, not current source-aligned U/G0 equivalence. | No architecture-independent performance claim is inherited. Any future cross-architecture claim needs fresh same-round controls under the new protocol on that architecture. |
| Phase 10 compiler prototype | Real C++ compiler prototype and assembler-controlled comparisons; not the erroneous production Gluon source. Its instruction-proxy selection rule was subsequently abandoned. Phase 10 Stage A reference comparisons also contain toolchain distinctions that must not be erased. | Retain as superseded prototype evidence. No automatic rerun or reinstatement of the abandoned policy. Its old quantitative cross-architecture win/loss counts must not be presented as current corrected selector validation. |
| Local A10 layout and release panels | Source-valid ordinary `tl.load` and explicit direct/cp_async paths; old single-launch event timing has host hash/dispatch work between observations. The current ordinary-load background claim still relies on these panels. | New standalone `a10_001` binds all 22 layout and six release CUBINs, checks authentic latest U/G0, and adds independent release-U shams. This is one physical GPU with three sequential process cohorts. |
| Phase 11 C1/C2 repeated-loop calibration and C3–C3.4 fits | Microbenchmarks have explicit loop/fixture semantics; a K-loop sample differs from a loop-free FULL sample. Frozen model calibration and held-out predictions are historical fixed objects. | Do not refit them or silently substitute new FULL timings for calibration slopes. New C2d/D1 measurements are diagnostic rechecks after D1 exposure, not a new blind test. Any future model update must use an explicit new stage. |
| C2d/D1 loop-free natural FULL and transfer controls | Frozen source-valid CUBINs, three footprints, single-kernel graphs and legacy correctness/hash guards. Old timing ranges are mostly small; that does not establish cadence immunity. | Complete anchors replay: 139 original program conditions × three footprints = 417 observations across 57 contexts, plus independent default shams. These are explicit-layout references, not invented Triton U/G0 pairs. |
| M3 consumer/output experiments | Source-valid frozen native/fixed/broadcast kernels. Some original q curves exhibit large visit ranges. | Complete 405-condition replay, retaining original q/grid/consumer parameters, plus N-first repeat shams. No authentic ordinary-Triton U/G0 is asserted. |
| M2, accepted M4a revision 3, M4b | Existing natural FULL timing did not show the same broad >3% visit-range signal. They retain their original context and are not numerically corrected merely because related M3 programs were replayed. | Static lowering, counter structure and legality are retained. New timing claims about their exact modified programs require an explicit fresh measurement. Do not call M3 coverage a replay of M2/M4. |
| Production / nonreduction Gluon survey | Early programs with an incorrect Gluon baseline are already superseded. Current source-aligned programs have exact U/G0 gates and archived binaries. | Remeasure the current corrected full matrices with U/G0 in the same observation round; never repair old wrong programs by relabeling their timings. |

## Descriptive legacy screening

| Original panel | Program/process/grid conditions | Visit range >3% | >10% | Full-grid conditions >3% |
| --- | ---: | ---: | ---: | ---: |
| Local A10 layouts | 66 | 22 | 0 | 22/66 |
| Local A10 releases | 42 | 21 | 0 | 21/42 |
| Phase 10 Stage C, SM90 | 1116 | 634 | 103 | 16/372 |
| Phase 10 Stage C, SM100 | 1116 | 1032 | 592 | 290/372 |
| Phase 10 Stage C, SM120 | 936 | 725 | 511 | 101/312 |

The largest local A10 layout range was 7.17% (`32×64/w8 direct vec2`, process 2), and the largest release range was 5.70% (`3.6.0`, `32×64/w8`, process 4). Phase 10's largest variations primarily concern fractional grids. These are reasons to require a new cadence before reusing precise performance rankings, not proof that every old speedup was noise. The explicitly abandoned Phase 10 model is not resurrected or remeasured here.

## Evidence that the cadence finding does not replace

Exact archived CUBIN bytes, allocated ELF equivalence, IR/layout ownership, static instruction structure, queried register/local-memory allocation and occupancy capacity are not timing estimates. Discrete NCU instruction/wavefront/sector totals remain observations of their profiled program. They do not become evidence about a corrected program with different source. Profiled elapsed time, throughput percentages and stall percentages retain their original profiling context; this round does not claim they were collected under the new timing cadence.

The old C2/M3 guard reads use extracted `/tmp` files; they must not be described as Modal Volume I/O. Separate guard diagnostics did not demonstrate a universal `/tmp`-guard effect. This distinction is retained even when a conservative replay is appropriate.

## Source pointers

- `phase2_benchmark.py`, `phase3_structural_analysis.py` and the Phase 3 artifact-equivalence/freeze reports describe the early harness limits.
- `phase8/timing_contract.py` and `phase9/timing_contract.py` pin initial three-launch warmups and per-sample SHA guards.
- `phase10/stage_c/run_timing.py` places guards before each direct-event observation; its original raw files and hashes are enumerated in `legacy_timing_screen.json`.
- `local_a10/experiment.py` and `local_a10_versions` collectors preserve the ordinary-load source/version controls and archived artifacts. The new A10 protocol binds their original collector correctness, payload and all old file hashes.
- `cadence_retest/legacy_audit.py` records the later C2d/D1/M2/M3/M4 timing-method audit. New replay scope is defined by each frozen protocol, not by this appendix.
