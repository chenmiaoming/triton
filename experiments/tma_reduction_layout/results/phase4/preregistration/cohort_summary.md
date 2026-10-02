# Phase 4 Stage A — Structural cohort freeze

Source: 30 transitions / 150 historical combinations; structural pool: 20 included, 10 excluded.
Performance-field-independent programmatic cohort selection, frozen before any new Phase 4 mechanism measurements. Membership uses legality and TTGIR layout metadata; resource/reproduction readiness is separate. Historical canonical outcomes existed before this preregistration, so human-level complete outcome blinding cannot be claimed.

Layouts use SPT / TPW / WPC / order, each in [B,M,N] axes. Vec is SPT[N].

| Case | M | N | Warps | Default layout | Cand4 layout | lanePart M | warpPart M | Include | Reasons | Origin |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4 | 128 | 128 | 4 | [1,1,8]/[1,2,16]/[1,4,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,4,1]/[2,1,0] | 2→1 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M128_N128_w8 | 128 | 128 | 8 | [1,1,8]/[1,2,16]/[1,8,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,8,1]/[2,1,0] | 2→1 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M128_N16_w4 | 128 | 16 | 4 | [1,1,8]/[1,16,2]/[1,4,1]/[2,1,0] | [1,1,4]/[1,8,4]/[1,4,1]/[2,1,0] | 16→8 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M128_N16_w8 | 128 | 16 | 8 | [1,1,8]/[1,16,2]/[1,8,1]/[2,1,0] | [1,1,4]/[1,8,4]/[1,8,1]/[2,1,0] | 16→8 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M128_N256_w4 | 128 | 256 | 4 | [1,1,8]/[1,1,32]/[1,4,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,2,2]/[2,1,0] | 1→1 | 4→2 | NO | WARP_PART_CHANGED, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M128_N256_w8 | 128 | 256 | 8 | [1,1,8]/[1,1,32]/[1,8,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,4,2]/[2,1,0] | 1→1 | 8→4 | NO | WARP_PART_CHANGED, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M128_N32_w4 | 128 | 32 | 4 | [1,1,8]/[1,8,4]/[1,4,1]/[2,1,0] | [1,1,4]/[1,4,8]/[1,4,1]/[2,1,0] | 8→4 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M128_N32_w8 | 128 | 32 | 8 | [1,1,8]/[1,8,4]/[1,8,1]/[2,1,0] | [1,1,4]/[1,4,8]/[1,8,1]/[2,1,0] | 8→4 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M128_N64_w4 | 128 | 64 | 4 | [1,1,8]/[1,4,8]/[1,4,1]/[2,1,0] | [1,1,4]/[1,2,16]/[1,4,1]/[2,1,0] | 4→2 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M128_N64_w8 | 128 | 64 | 8 | [1,1,8]/[1,4,8]/[1,8,1]/[2,1,0] | [1,1,4]/[1,2,16]/[1,8,1]/[2,1,0] | 4→2 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M32_N128_w4 | 32 | 128 | 4 | [1,1,8]/[1,2,16]/[1,4,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,4,1]/[2,1,0] | 2→1 | 4→4 | YES | PASS | ANCHOR_EXISTING |
| M32_N128_w8 | 32 | 128 | 8 | [1,1,8]/[1,2,16]/[1,8,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,8,1]/[2,1,0] | 2→1 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M32_N16_w4 | 32 | 16 | 4 | [1,1,4]/[1,8,4]/[1,4,1]/[2,1,0] | [1,1,4]/[1,8,4]/[1,4,1]/[2,1,0] | 8→8 | 4→4 | NO | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M32_N16_w8 | 32 | 16 | 8 | [1,1,2]/[1,4,8]/[1,8,1]/[2,1,0] | illegal / unavailable | 4→None | 8→None | NO | CAND4_ILLEGAL | NEW_GENERALIZATION |
| M32_N256_w4 | 32 | 256 | 4 | [1,1,8]/[1,1,32]/[1,4,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,2,2]/[2,1,0] | 1→1 | 4→2 | NO | WARP_PART_CHANGED, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M32_N256_w8 | 32 | 256 | 8 | [1,1,8]/[1,1,32]/[1,8,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,4,2]/[2,1,0] | 1→1 | 8→4 | NO | WARP_PART_CHANGED, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M32_N32_w4 | 32 | 32 | 4 | [1,1,8]/[1,8,4]/[1,4,1]/[2,1,0] | [1,1,4]/[1,4,8]/[1,4,1]/[2,1,0] | 8→4 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M32_N32_w8 | 32 | 32 | 8 | [1,1,4]/[1,4,8]/[1,8,1]/[2,1,0] | [1,1,4]/[1,4,8]/[1,8,1]/[2,1,0] | 4→4 | 8→8 | NO | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M32_N64_w4 | 32 | 64 | 4 | [1,1,8]/[1,4,8]/[1,4,1]/[2,1,0] | [1,1,4]/[1,2,16]/[1,4,1]/[2,1,0] | 4→2 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M32_N64_w8 | 32 | 64 | 8 | [1,1,8]/[1,4,8]/[1,8,1]/[2,1,0] | [1,1,4]/[1,2,16]/[1,8,1]/[2,1,0] | 4→2 | 8→8 | YES | PASS | ANCHOR_EXISTING |
| M64_N128_w4 | 64 | 128 | 4 | [1,1,8]/[1,2,16]/[1,4,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,4,1]/[2,1,0] | 2→1 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M64_N128_w8 | 64 | 128 | 8 | [1,1,8]/[1,2,16]/[1,8,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,8,1]/[2,1,0] | 2→1 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M64_N16_w4 | 64 | 16 | 4 | [1,1,8]/[1,16,2]/[1,4,1]/[2,1,0] | [1,1,4]/[1,8,4]/[1,4,1]/[2,1,0] | 16→8 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M64_N16_w8 | 64 | 16 | 8 | [1,1,4]/[1,8,4]/[1,8,1]/[2,1,0] | [1,1,4]/[1,8,4]/[1,8,1]/[2,1,0] | 8→8 | 8→8 | NO | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M64_N256_w4 | 64 | 256 | 4 | [1,1,8]/[1,1,32]/[1,4,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,2,2]/[2,1,0] | 1→1 | 4→2 | NO | WARP_PART_CHANGED, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M64_N256_w8 | 64 | 256 | 8 | [1,1,8]/[1,1,32]/[1,8,1]/[2,1,0] | [1,1,4]/[1,1,32]/[1,4,2]/[2,1,0] | 1→1 | 8→4 | NO | WARP_PART_CHANGED, LANE_PART_NOT_REDUCED | NEW_GENERALIZATION |
| M64_N32_w4 | 64 | 32 | 4 | [1,1,8]/[1,8,4]/[1,4,1]/[2,1,0] | [1,1,4]/[1,4,8]/[1,4,1]/[2,1,0] | 8→4 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M64_N32_w8 | 64 | 32 | 8 | [1,1,8]/[1,8,4]/[1,8,1]/[2,1,0] | [1,1,4]/[1,4,8]/[1,8,1]/[2,1,0] | 8→4 | 8→8 | YES | PASS | NEW_GENERALIZATION |
| M64_N64_w4 | 64 | 64 | 4 | [1,1,8]/[1,4,8]/[1,4,1]/[2,1,0] | [1,1,4]/[1,2,16]/[1,4,1]/[2,1,0] | 4→2 | 4→4 | YES | PASS | NEW_GENERALIZATION |
| M64_N64_w8 | 64 | 64 | 8 | [1,1,8]/[1,4,8]/[1,8,1]/[2,1,0] | [1,1,4]/[1,2,16]/[1,8,1]/[2,1,0] | 4→2 | 8→8 | YES | PASS | NEW_GENERALIZATION |

## Coverage (included pool)

```json
{
  "M": {
    "128": 8,
    "32": 5,
    "64": 7
  },
  "N": {
    "128": 6,
    "16": 3,
    "32": 5,
    "64": 6
  },
  "N256_audit": [
    {
      "config_id": "M128_N256_w4",
      "reason_codes": [
        "WARP_PART_CHANGED",
        "LANE_PART_NOT_REDUCED"
      ]
    },
    {
      "config_id": "M128_N256_w8",
      "reason_codes": [
        "WARP_PART_CHANGED",
        "LANE_PART_NOT_REDUCED"
      ]
    },
    {
      "config_id": "M32_N256_w4",
      "reason_codes": [
        "WARP_PART_CHANGED",
        "LANE_PART_NOT_REDUCED"
      ]
    },
    {
      "config_id": "M32_N256_w8",
      "reason_codes": [
        "WARP_PART_CHANGED",
        "LANE_PART_NOT_REDUCED"
      ]
    },
    {
      "config_id": "M64_N256_w4",
      "reason_codes": [
        "WARP_PART_CHANGED",
        "LANE_PART_NOT_REDUCED"
      ]
    },
    {
      "config_id": "M64_N256_w8",
      "reason_codes": [
        "WARP_PART_CHANGED",
        "LANE_PART_NOT_REDUCED"
      ]
    }
  ],
  "lanePart_M_transitions": {
    "16->8": 3,
    "2->1": 6,
    "4->2": 6,
    "8->4": 5
  },
  "num_warps": {
    "4": 11,
    "8": 9
  },
  "origin": {
    "ANCHOR_EXISTING": 2,
    "NEW_GENERALIZATION": 18
  },
  "sizePerThread_transitions": {
    "[1, 1, 8]->[1, 1, 4]": 20
  },
  "warpPart_M_values": {
    "4": 11,
    "8": 9
  }
}
```

Exclusion reason counts are nonexclusive; a case can satisfy more than one exclusion.

```json
{
  "CAND4_ILLEGAL": 1,
  "DEFAULT_ILLEGAL": 0,
  "IDENTICAL_LAYOUT": 3,
  "LANE_PART_NOT_REDUCED": 9,
  "REDUCTION_AXIS_MISMATCH": 0,
  "UNSUPPORTED_LAYOUT": 0,
  "WARP_COUNT_MISMATCH": 0,
  "WARP_PART_CHANGED": 6
}
```

All six N256 cases change warpPart M and do not reduce lanePart M; this is a structural exclusion, not performance filtering.
The two existing anchors are retained. New generalization cases are reported separately in future association sensitivity analyses.
Current pre-timing eligible count: 0. Actual Phase 4 CUBIN/occupancy/provenance closure is absent; pending cases remain in the frozen pool.
Historical sweep reduce_ops metadata is empty. Source AST supplies the initial [1,M,N]/M-axis contract; Stage B must check actual TTGIR per artifact. No missing historical metadata is invented.
Stop after Stage A review. No timing, H100/Modal work, or push.
