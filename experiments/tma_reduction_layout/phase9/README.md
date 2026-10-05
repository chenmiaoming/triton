# Phase9 — Cross-architecture reproduction before a compiler fix

Stages: A preregistration; B strict target/toolchain/artifact gates; C frozen-binary timing; D architecture scope decision and final closure. Commit each completed stage and push the experiment branch. Do not create a PR, including a draft, without asking the user first. No production compiler change is part of this phase.

Use the exact Phase6 native core image and persistent ccache. All earlier results, raw samples and original compiler artifacts are immutable. Only new Phase9 files and clearly derived Phase9 summaries may be written. Stop on quota exhaustion and ask the user to switch profile; do not silently switch profiles.

```sh
python3 -m experiments.tma_reduction_layout.phase9.preregister --validate
```

The protocol fixes twelve retrospective reduction cases and six non-reduction TMA load/store controls before observations. Each SM90/SM100/SM120 target has separate compilation and three exact-binary timing invocations. This tests cross-architecture transfer of the candidate4 contrast; it is neither a historical compiler bisect nor held-out validation of a production patch.
