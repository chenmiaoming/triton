# TMA Reduction Layout: Instruction-Level Evidence & Phase Annotations

This document details the observed instruction ranges and semantic phases for all candidates compiled from the current commit, providing verified evidence for PTX and SASS codegen.

## Candidate `default`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.v4.b32', 'count': 4, 'line_range': [224, 227], 'raw_examples': ['ld.shared.v4.b32 \t{%r39, %r40, %r41, %r42}, [%r38];', 'ld.shared.v4.b32 \t{%r43, %r44, %r45, %r46}, [%r38+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 12, 'cvt.f32.bf16': 8, 'max.f32': 24}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 166, 'max_virtual_b16_regs': 9, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': {'source': 'cuobjdump -res-usage', 'value': 32}, 'cuobjdump_shared_bytes': {'source': 'cuobjdump -res-usage', 'value': 1024}, 'triton_launch_shared_bytes': {'source': 'compiled.metadata.shared', 'value': 8200}, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| tma_setup_and_descriptor | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 36-211 |
| initial_local_load | Initial loading of TMA-loaded shared memory tile into registers (4 x ld.shared.v4.b32) | Lines 224-227 |
| thread_local_reduction_arithmetic | Thread-local reduction arithmetic: 12 x max.bf16x2 and 8 x cvt.f32.bf16 | Lines 232-255 |
| cross_thread_reduction_communication | Cross-lane & cross-warp reduction communication: intra-warp butterfly shuffles (lines 259-280), cross-warp shared exchange (lines 290-301), and cross-warp butterfly shuffles (lines 302-371) | Lines 258-371 |
| post_reduction_convert_layout | ttg.convert_layout shared-memory redistribution converting reduction slice layout to blocked1 layout via st.shared/ld.shared/ldmatrix | Lines 374-405 |
| global_store | st.global.b32 storing final 128 float elements to output buffer | Line 408 |

## Candidate `8`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.v4.b32', 'count': 4, 'line_range': [224, 227], 'raw_examples': ['ld.shared.v4.b32 \t{%r39, %r40, %r41, %r42}, [%r38];', 'ld.shared.v4.b32 \t{%r43, %r44, %r45, %r46}, [%r38+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 12, 'cvt.f32.bf16': 8, 'max.f32': 24}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 166, 'max_virtual_b16_regs': 9, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': {'source': 'cuobjdump -res-usage', 'value': 32}, 'cuobjdump_shared_bytes': {'source': 'cuobjdump -res-usage', 'value': 1024}, 'triton_launch_shared_bytes': {'source': 'compiled.metadata.shared', 'value': 8200}, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| tma_setup_and_descriptor | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 36-211 |
| initial_local_load | Initial loading of TMA-loaded shared memory tile into registers (4 x ld.shared.v4.b32) | Lines 224-227 |
| thread_local_reduction_arithmetic | Thread-local reduction arithmetic: 12 x max.bf16x2 and 8 x cvt.f32.bf16 | Lines 232-255 |
| cross_thread_reduction_communication | Cross-lane & cross-warp reduction communication: intra-warp butterfly shuffles (lines 259-280), cross-warp shared exchange (lines 290-301), and cross-warp butterfly shuffles (lines 302-371) | Lines 258-371 |
| post_reduction_convert_layout | ttg.convert_layout shared-memory redistribution converting reduction slice layout to blocked1 layout via st.shared/ld.shared/ldmatrix | Lines 374-405 |
| global_store | st.global.b32 storing final 128 float elements to output buffer | Line 408 |

## Candidate `4`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.v2.b32', 'count': 8, 'line_range': [227, 234], 'raw_examples': ['ld.shared.v2.b32 \t{%r42, %r43}, [%r39];', 'ld.shared.v2.b32 \t{%r44, %r45}, [%r39+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 14, 'cvt.f32.bf16': 4, 'max.f32': 8}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 118, 'max_virtual_b16_regs': 5, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': {'source': 'cuobjdump -res-usage', 'value': 25}, 'cuobjdump_shared_bytes': {'source': 'cuobjdump -res-usage', 'value': 1024}, 'triton_launch_shared_bytes': {'source': 'compiled.metadata.shared', 'value': 8200}, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:25 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| tma_setup_and_descriptor | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 36-211 |
| initial_local_load | Initial loading of TMA-loaded shared memory tile into registers (8 x ld.shared.v2.b32) | Lines 227-234 |
| thread_local_reduction_arithmetic | Thread-local reduction arithmetic: 14 x max.bf16x2 and 4 x cvt.f32.bf16 | Lines 241-260 |
| cross_thread_reduction_communication | Cross-warp reduction communication: cross-warp shared exchange (lines 266-275), 8 x shfl.sync.bfly.b32 and max.f32 (lines 276-309) | Lines 266-309 |
| post_reduction_convert_layout | ttg.convert_layout shared-memory redistribution converting reduction slice layout to blocked1 layout via st.shared/ld.shared/ldmatrix (lines 312-331) | Lines 312-331 |
| global_store | st.global.b32 storing final 128 float elements to output buffer | Line 334 |

## Candidate `2`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ldmatrix.sync.aligned.m8n8.x4.shared.b16', 'count': 4, 'line_range': [229, 232], 'raw_examples': ['ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r44, %r45, %r46, %r47}, [%r43];', 'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r48, %r49, %r50, %r51}, [%r43+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 15, 'cvt.f32.bf16': 2, 'max.f32': 2}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 111, 'max_virtual_b16_regs': 3, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': {'source': 'cuobjdump -res-usage', 'value': 23}, 'cuobjdump_shared_bytes': {'source': 'cuobjdump -res-usage', 'value': 1024}, 'triton_launch_shared_bytes': {'source': 'compiled.metadata.shared', 'value': 8200}, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:23 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| tma_setup_and_descriptor | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 36-211 |
| initial_local_load | Initial loading of TMA-loaded shared memory tile into registers (4 x ldmatrix.sync.aligned.m8n8.x4.shared.b16) | Lines 229-232 |
| thread_local_reduction_arithmetic | Thread-local reduction arithmetic: 15 x max.bf16x2 and 2 x cvt.f32.bf16 | Lines 236-256 |
| cross_thread_reduction_communication | Cross-warp reduction communication: shared exchange (lines 264-275), 2 x shfl.sync.bfly.b32 and 2 x max.f32 (lines 276-281) | Lines 264-281 |
| post_reduction_convert_layout | ttg.convert_layout shared-memory redistribution converting reduction slice layout to blocked1 layout via st.shared/ld.shared (lines 284-308) | Lines 284-308 |
| global_store | st.global.b32 storing final 128 float elements to output buffer | Line 311 |

## Candidate `1`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.b16', 'count': 32, 'line_range': [220, 265], 'raw_examples': ['ld.shared.b16 \t%rs1, [%r34];', 'ld.shared.b16 \t%rs2, [%r34+1024];']}`
- **Arithmetic Mix**: `{'max.bf16': 31, 'cvt.f32.bf16': 1}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 49, 'max_virtual_b16_regs': 64, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': {'source': 'cuobjdump -res-usage', 'value': 32}, 'cuobjdump_shared_bytes': {'source': 'cuobjdump -res-usage', 'value': 1024}, 'triton_launch_shared_bytes': {'source': 'compiled.metadata.shared', 'value': 8200}, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| tma_setup_and_descriptor | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 36-211 |
| initial_local_load | Initial loading of TMA-loaded shared memory tile into registers (32 x ld.shared.b16) | Lines 220-265 |
| thread_local_reduction_arithmetic | Thread-local reduction tree: 31 x max.bf16 and 1 x cvt.f32.bf16 (reduction completes entirely in thread-local bf16) | Lines 270-301 |
| cross_thread_reduction_communication | Reduction is 100% thread-local; no intra-warp shuffles or cross-warp shared exchange are generated | no standalone PTX sequence identified |
| post_reduction_convert_layout | Thread-local scalar reduction output is directly assigned to the target store register without shared-memory redistribution | no standalone PTX sequence identified |
| global_store | st.global.b32 storing final 128 float elements to output buffer | Line 308 |
