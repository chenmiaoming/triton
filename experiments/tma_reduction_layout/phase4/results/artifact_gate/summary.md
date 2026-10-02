# PHASE 4 STAGE B REPORT

Outcome-blind structural eligibility; no Stage C execution or new timing.

GPU: NVIDIA H100 80GB HBM3 / GPU-6a4bf87b-0b64-03f1-fa0f-0c03264fd383; CC [9, 0].
Counts: {'PRIMARY': 3, 'SECONDARY': 15, 'EXCLUDE_FROM_TIMING': 2, 'PENDING': 0}
Generated: {'canonical': 40, 'single': 40, 'repeated': 40}; exported: {'canonical': 40, 'single': 40, 'repeated': 40}

E = EXACT_SEQUENCE_EQUIVALENT; P = PIPELINED_OPCODE_EQUIVALENT; M = fingerprint mismatch; U = unavailable.
Load encodings are default/cand4. Blocks/SM refer to repeated actual dynamic SMEM.

|case|origin|single d/4|single LL|repeated d/4|preload d/4|spill|blocks d/4|closure|class|reasons|
|---|---|---|---|---|---|---|---|---|---|---|---|
|M128_N128_w4|NEW_GENERALIZATION|E/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|6/6|True|SECONDARY|—|
|M128_N128_w8|NEW_GENERALIZATION|E/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|6/6|True|SECONDARY|—|
|M128_N16_w4|NEW_GENERALIZATION|P/P|True|P/P|EXACT/EXACT|0|16/16|True|SECONDARY|—|
|M128_N16_w8|NEW_GENERALIZATION|P/P|True|P/P|EXACT/EXACT|0|8/8|True|SECONDARY|—|
|M128_N32_w4|NEW_GENERALIZATION|P/P|True|P/P|EXACT/EXACT|0|16/16|True|SECONDARY|—|
|M128_N32_w8|NEW_GENERALIZATION|P/E|True|P/E|EXACT/EXACT|0|8/8|True|SECONDARY|—|
|M128_N64_w4|NEW_GENERALIZATION|P/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|12/13|True|EXCLUDE_FROM_TIMING|RESIDENCY_MISMATCH|
|M128_N64_w8|NEW_GENERALIZATION|E/E|True|E/E|EXACT/DIFFERENT_ENCODING|0|8/8|True|PRIMARY|—|
|M32_N128_w4|ANCHOR_EXISTING|E/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|16/16|True|SECONDARY|—|
|M32_N128_w8|NEW_GENERALIZATION|E/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|8/8|True|SECONDARY|—|
|M32_N32_w4|NEW_GENERALIZATION|P/P|True|P/P|EXACT/EXACT|0|16/16|True|SECONDARY|—|
|M32_N64_w4|NEW_GENERALIZATION|P/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|16/16|True|SECONDARY|—|
|M32_N64_w8|ANCHOR_EXISTING|E/E|True|E/E|EXACT/DIFFERENT_ENCODING|0|8/8|True|PRIMARY|—|
|M64_N128_w4|NEW_GENERALIZATION|E/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|13/13|True|EXCLUDE_FROM_TIMING|RESIDENCY_MISMATCH|
|M64_N128_w8|NEW_GENERALIZATION|E/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|8/8|True|SECONDARY|—|
|M64_N16_w4|NEW_GENERALIZATION|P/P|True|P/P|EXACT/EXACT|0|16/16|True|SECONDARY|—|
|M64_N32_w4|NEW_GENERALIZATION|P/P|True|P/P|EXACT/EXACT|0|16/16|True|SECONDARY|—|
|M64_N32_w8|NEW_GENERALIZATION|P/E|True|P/E|EXACT/EXACT|0|8/8|True|SECONDARY|—|
|M64_N64_w4|NEW_GENERALIZATION|P/E|True|P/E|EXACT/DIFFERENT_ENCODING|0|16/16|True|SECONDARY|—|
|M64_N64_w8|NEW_GENERALIZATION|E/E|True|E/E|EXACT/DIFFERENT_ENCODING|0|8/8|True|PRIMARY|—|

## Exact-binary closure

Each exported bundle archives actual ELF CUBIN bytes and its SHA. Occupancy loads a buffer read from that file and queries the bound function using the checked CUDA driver API. R0/R1 smoke calls the same CompiledKernel directly after one warmup; hashes before/after both launches are retained. Export failures are preserved and cannot be eligible.

Canonical/single deterministic four-tile prefix smoke compares float32 max over M. The descriptor retains B_DESC=65536; other tiles are uninitialized and not launched/read. Repeated zero output proves launch safety only.

All backward/self edges, complete loop memory signatures, terminal exchanges, tied copy pairs and explicit SASS MOV/IMAD.MOV observations are retained in cohort_after_gate.json. Filtered opcode equality does not establish topology/dataflow equality. No zero-overhead barrier claim.

R=0 subtraction controls fixed one-time harness differential. It does not prove absence of interactions via register allocation, live ranges, instruction scheduling, or compiler decisions.

For canonical and repeated, runtime CUBIN SHA must equal archived timing-eligible SHA; mismatch ABORT BEFORE TIMING. Stage C not started.

## Resource field interpretation

Root case resources.static_smem_bytes is the driver attribute; cuobjdump_reported_shared_bytes retains raw SHARED. Raw metadata and frozen body-helper resources retain legacy parser names. Here driver static=0 and cuobjdump SHARED=1024; both are archived separately without assuming equality or using a proxy occupancy calculation.

## Structural observations

```json
{
  "single_pipelined": [
    "M128_N16_w4:default",
    "M128_N16_w4:4",
    "M128_N16_w8:default",
    "M128_N16_w8:4",
    "M128_N32_w4:default",
    "M128_N32_w4:4",
    "M128_N32_w8:default",
    "M128_N64_w4:default",
    "M32_N32_w4:default",
    "M32_N32_w4:4",
    "M32_N64_w4:default",
    "M64_N16_w4:default",
    "M64_N16_w4:4",
    "M64_N32_w4:default",
    "M64_N32_w4:4",
    "M64_N32_w8:default",
    "M64_N64_w4:default"
  ],
  "repeated_pipelined": [
    "M128_N128_w4:default",
    "M128_N128_w8:default",
    "M128_N16_w4:default",
    "M128_N16_w4:4",
    "M128_N16_w8:default",
    "M128_N16_w8:4",
    "M128_N32_w4:default",
    "M128_N32_w4:4",
    "M128_N32_w8:default",
    "M128_N64_w4:default",
    "M32_N128_w4:default",
    "M32_N128_w8:default",
    "M32_N32_w4:default",
    "M32_N32_w4:4",
    "M32_N64_w4:default",
    "M64_N128_w4:default",
    "M64_N128_w8:default",
    "M64_N16_w4:default",
    "M64_N16_w4:4",
    "M64_N32_w4:default",
    "M64_N32_w4:4",
    "M64_N32_w8:default",
    "M64_N64_w4:default"
  ],
  "single_localload_failures": [],
  "canonical_layout_drift": [],
  "residency_mismatches": {
    "M128_N64_w4": {
      "canonical": {
        "default": [
          13,
          52
        ],
        "4": [
          12,
          48
        ],
        "matched": false
      },
      "repeated": {
        "default": [
          12,
          48
        ],
        "4": [
          13,
          52
        ],
        "matched": false
      }
    },
    "M64_N128_w4": {
      "canonical": {
        "default": [
          12,
          48
        ],
        "4": [
          13,
          52
        ],
        "matched": false
      }
    }
  }
}
```

## Eligible coverage

```json
{
  "PRIMARY": {
    "M": {
      "128": 1,
      "32": 1,
      "64": 1
    },
    "N": {
      "64": 3
    },
    "num_warps": {
      "8": 3
    },
    "lanePart_transition": {
      "4->2": 3
    }
  },
  "SECONDARY": {
    "M": {
      "128": 6,
      "32": 4,
      "64": 5
    },
    "N": {
      "128": 5,
      "16": 3,
      "32": 5,
      "64": 2
    },
    "num_warps": {
      "4": 9,
      "8": 6
    },
    "lanePart_transition": {
      "16->8": 3,
      "2->1": 5,
      "4->2": 2,
      "8->4": 5
    }
  }
}
```

## Provenance and frozen files

source_bindings.json, local_source_provenance.json, uploaded_source.zip and modal_dispatch.json bind source bytes, local Git/diff, uploaded subset, frozen protocol/pool/schedule and resolved image/function identity. Per-bundle attempt.json binds immutable exports. Stage A master files are byte-identical. The original Stage A validator restricts new paths; Stage B outputs therefore live under phase4/results/artifact_gate.

## No-performance attestation

No new timing / latency / throughput / speedup / slope was collected or used. H2a/H2b/H2c are unchanged. The schedule preview is a stable filter and was not executed. Local commit only; no push; STOP before Stage C.
