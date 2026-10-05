# Phase9 StageA — Cross-architecture preregistration

H100/SM90, B200/SM100 and RTX PRO6000 Blackwell/SM120 are separate strict targets. Twelve retrospective reduction identities and six non-reduction load/store controls are frozen before dispatch. No observed outcome can remove or replace a legal pair. Each target gets its own archived binaries.

Primary outcome: full-kernel time at B_RUN65536; three invocation effects use a fixed3% practical band and df2 descriptive uncertainty. Grid16384/32768 and marginal OLS slopes are secondary. A missing/ambiguous result is not a zero or evidence of safety.

The existing experimental vector4 override tests a candidate, not a production patch. Different targets may change LocalLoad lowering, registers and residency; all valid pairs are retained in explicit resource strata. Historical compiler regression attribution and held-out patch validation remain outstanding.

Reuse the exact Phase6 native core and persistent ccache. Python-only work does not run make. If quota is exhausted stop and ask the user to switch profile. No compiler edit or PR (including draft); user approval is mandatory before PR creation. Stop after the scope report.

| Case | Warps | Harnesses | Reason |
| --- | --- | --- | --- |
| M32_N64_w8 | 8 | reduction, load_copy, store_copy | original canonical positive case |
| M32_N128_w4 | 4 | reduction | initial baseline, narrower lanes |
| M32_N16_w4 | 4 | reduction | same-layout negative control: default already vector4 |
| M32_N256_w4 | 4 | reduction, load_copy, store_copy | wide tile, warp ownership changes |
| M256_N64_w8 | 8 | reduction | Phase5 PRIMARY unresolved-sign counterexample |
| M512_N64_w8 | 8 | reduction | Phase5 PRIMARY unresolved-sign counterexample |
| M128_N32_w16 | 16 | reduction, load_copy, store_copy | Phase8 old positive descriptor-path diagnostic |
| M1024_N32_w16 | 16 | reduction | Phase8 old near-zero/opposed gap diagnostic |
| M2048_N32_w16 | 16 | reduction | Phase7 dominant influence, Phase8 descriptor diagnostic |
| M128_N64_w32 | 32 | reduction | Phase8 new PRIMARY positive case |
| M512_N128_w32 | 32 | reduction | Phase8 PRIMARY residual counterexample, large tile |
| M32_N128_w8 | 8 | reduction | historical warp8 positive case |
