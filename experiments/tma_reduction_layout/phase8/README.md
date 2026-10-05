# Phase8 — Same-binary descriptor-path validation

Stages A–D are complete. Read the [final report](../results/phase8/final_validation/summary.md) and [fixed protocol](../results/phase8/stage_a/protocol.json).

The Gluon kernel uses an unspecialized, block-uniform runtime mode to choose a host descriptor or device descriptor construction. Both paths join before one common LocalLoad/reduction/output store. Each layout candidate has one archived CUBIN shared across both modes. No source-level hardware IDs, lane-varying scalars, compiler changes or production heuristic are introduced.

All16 unseen identities remain in the preregistered pool; seven failed the original structural rules before compilation. Allnine admitted unseen cases are PRIMARY. StageC retained36,000 diagnostic samples and eight separate NCU reports; StageD retained81,000 prospective samples. Every scalar event sample binds actual mode, module/function identity and exact launch SHA.

The native Phase6 image and persistent ccache/triton-home are reused. Formal timing loads archived ELF through CUDA driver calls without Triton/JIT/compiler imports. Raw samples, original compiler returns, source snapshots, reports, failures and byte manifests are immutable; analysis and validation outputs are derived.

Final offline validation:

```sh
python3 -m experiments.tma_reduction_layout.phase8.validate_final
```

After committing, verify first-commit bytes without writing any results:

```sh
python3 -m experiments.tma_reduction_layout.phase8.validate_final --committed
```

The final suite includes independent medians/OLS, actual-mode and SHA checks, original-report counter roundtrips, four isolated integrity probes, and prior validators replayed in independent checkouts. NCU worker UUID was absent from its original export and is recorded as unavailable; all formal worker UUIDs are retained. Explicit base-unit offline report views handle NCU2026 auto-scaling without GPU recollection. H2b/H2c remain UNVERIFIED. Work stops after StageD.
