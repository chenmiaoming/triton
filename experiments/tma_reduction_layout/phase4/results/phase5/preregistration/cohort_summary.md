# Phase 5 — Held-Out Validation Preregistration

held-out confirmatory extension for OUTCOME_INFORMED_HYPOTHESIS; Stage C is development/hypothesis-generation data and cannot confirm these hypotheses

24 source transitions; 19 structurally included; 5 excluded. All layout/resource fields are source-rule predictions. No Phase 5 artifact gate or timing has been executed.

Each layout cell is sizePerThread / threadsPerWarp / warpsPerCTA, with order [2,1,0]. Illegal candidates have no predicted layout.

| Case | M | N | Warps | Default layout | Cand4 layout | Include/exclude | Reasons | Resource/artifact state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M16_N128_w4 | 16 | 128 | 4 | [1, 1, 8] / [1, 2, 16] / [1, 4, 1] | [1, 1, 4] / [1, 1, 32] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M16_N128_w8 | 16 | 128 | 8 | [1, 1, 8] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 1, 32] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M16_N16_w4 | 16 | 16 | 4 | [1, 1, 2] / [1, 4, 8] / [1, 4, 1] | ILLEGAL | EXCLUDE | CAND4_ILLEGAL | STRUCTURALLY_EXCLUDED |
| M16_N16_w8 | 16 | 16 | 8 | [1, 1, 1] / [1, 2, 16] / [1, 8, 1] | ILLEGAL | EXCLUDE | CAND4_ILLEGAL | STRUCTURALLY_EXCLUDED |
| M16_N32_w4 | 16 | 32 | 4 | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | EXCLUDE | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | STRUCTURALLY_EXCLUDED |
| M16_N32_w8 | 16 | 32 | 8 | [1, 1, 2] / [1, 2, 16] / [1, 8, 1] | ILLEGAL | EXCLUDE | CAND4_ILLEGAL | STRUCTURALLY_EXCLUDED |
| M16_N64_w4 | 16 | 64 | 4 | [1, 1, 8] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 2, 16] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M16_N64_w8 | 16 | 64 | 8 | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | EXCLUDE | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | STRUCTURALLY_EXCLUDED |
| M256_N128_w4 | 256 | 128 | 4 | [1, 1, 8] / [1, 2, 16] / [1, 4, 1] | [1, 1, 4] / [1, 1, 32] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N128_w8 | 256 | 128 | 8 | [1, 1, 8] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 1, 32] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N16_w4 | 256 | 16 | 4 | [1, 1, 8] / [1, 16, 2] / [1, 4, 1] | [1, 1, 4] / [1, 8, 4] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N16_w8 | 256 | 16 | 8 | [1, 1, 8] / [1, 16, 2] / [1, 8, 1] | [1, 1, 4] / [1, 8, 4] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N32_w4 | 256 | 32 | 4 | [1, 1, 8] / [1, 8, 4] / [1, 4, 1] | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N32_w8 | 256 | 32 | 8 | [1, 1, 8] / [1, 8, 4] / [1, 8, 1] | [1, 1, 4] / [1, 4, 8] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N64_w4 | 256 | 64 | 4 | [1, 1, 8] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 2, 16] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N64_w8 | 256 | 64 | 8 | [1, 1, 8] / [1, 4, 8] / [1, 8, 1] | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N128_w4 | 512 | 128 | 4 | [1, 1, 8] / [1, 2, 16] / [1, 4, 1] | [1, 1, 4] / [1, 1, 32] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N128_w8 | 512 | 128 | 8 | [1, 1, 8] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 1, 32] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N16_w4 | 512 | 16 | 4 | [1, 1, 8] / [1, 16, 2] / [1, 4, 1] | [1, 1, 4] / [1, 8, 4] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N16_w8 | 512 | 16 | 8 | [1, 1, 8] / [1, 16, 2] / [1, 8, 1] | [1, 1, 4] / [1, 8, 4] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N32_w4 | 512 | 32 | 4 | [1, 1, 8] / [1, 8, 4] / [1, 4, 1] | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N32_w8 | 512 | 32 | 8 | [1, 1, 8] / [1, 8, 4] / [1, 8, 1] | [1, 1, 4] / [1, 4, 8] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N64_w4 | 512 | 64 | 4 | [1, 1, 8] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 2, 16] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N64_w8 | 512 | 64 | 8 | [1, 1, 8] / [1, 4, 8] / [1, 8, 1] | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |

Coverage: `{"M": {"16": 3, "256": 8, "512": 8}, "N": {"128": 6, "16": 4, "32": 4, "64": 5}, "lane_transition": {"16->8": 4, "2->1": 6, "4->2": 5, "8->4": 4}, "num_warps": {"4": 10, "8": 9}}`.

M16 is one power-of-two below the development M32/64/128 range; M256/512 are above it. Deliberate extrapolation, never prune for expected PRIMARY membership or performance.

The field compiled_num_warps is required by the reused Stage A selector but is explicitly a predicted source attribute here, not a claim of compilation. Actual Stage B artifacts must confirm it.

M512/N128 retains B_DESC=65536: 8GiB BF16 input per case and 128KiB logical shared tile. Actual compiled shared allocation, register/spill, occupancy and allocation safety remain pending. No descriptor/grid/resource policy is adjusted.

PRIMARY and SECONDARY will be assigned only by the future artifact gate; no predicted PRIMARY count or N64/w8 filter.

Full future sampling, analysis, locked prediction coefficients and hypothesis support/falsification observations are in protocol.json. Stage C cannot confirm Stage D-generated hypotheses.

NO Phase 5 timing observed. NO Phase 5 artifact gate executed. STOP until a separate artifact-gate authorization.
