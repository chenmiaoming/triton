# Pre-timing parser corrections

No raw compiler/export bytes or preregistered gates were changed. Both repairs occurred before any Phase7 timing.

1. The archived TTGIR is lowered and contains `ttng.tensormap_create`, scratch allocation and fenceproxy acquire, rather than the earlier `tt.make_tensor_descriptor`. Audit the actual lowered operation and publication.
2. For native output, there is no `ttg.convert_layout`. PTX may delay the terminal reduction result `ld.shared` until the store debug location. Use the already-existing Phase6 `body_instructions(..., minimal_output=True)` boundary at the first global store for these native outputs. Canonical outputs retain the source-bound boundary before their explicit output-layout conversion. This includes terminal exchanges instead of selecting any desired subsequence; each bundle retains its original collector annotation, corrected fingerprint, operand inventory hash and boundary policy.

Concrete witness: `host_native/M128_N32_w16/4/kernel.ptx` lines258–274 writes the reduction result to shared memory, then loads that same result at line272 after the `.loc` store marker, before the first global store. The original collector annotation misses the terminal load. Canonical output loads it before the explicit output conversion. Full shared-load/store/barrier/convert/max/shuffle opcode sequences or complete multisets are required for every candidate/cell after correction.
