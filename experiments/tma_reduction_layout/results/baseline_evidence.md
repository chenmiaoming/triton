# TMA Reduction Layout: Instruction-Level Evidence & Phase Annotations

This document details the observed instruction ranges and semantic phases for all candidates compiled from the current commit, providing verified evidence for PTX and SASS codegen.

## Candidate `default`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.v4.b32', 'count': 4, 'line_range': [224, 227], 'raw_examples': ['ld.shared.v4.b32 \t{%r39, %r40, %r41, %r42}, [%r38];', 'ld.shared.v4.b32 \t{%r43, %r44, %r45, %r46}, [%r38+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 12, 'cvt.f32.bf16': 8, 'max.f32': 24}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 166, 'max_virtual_b16_regs': 9, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': 32, 'shared_memory_bytes': 1024, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| A. TMA setup & descriptor lifecycle | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 43-211 |
| B. Initial LocalLoad (shared -> registers) | Initial loading of TMA-loaded shared memory tile into registers | Lines 224-227 |
| C. Cross-lane & cross-warp reduction communication | Intra-warp butterfly shuffles and cross-warp shared memory partial exchanges | Lines 259-365 |
| D. Post-reduction layout conversion | ttg.convert_layout converting 1D reduction slice layout to 1D blocked layout | Lines 378-405 |
| E. Global store | st.global.b32 storing final 128 elements to output buffer | Lines 408-408 |

## Candidate `8`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.v4.b32', 'count': 4, 'line_range': [224, 227], 'raw_examples': ['ld.shared.v4.b32 \t{%r39, %r40, %r41, %r42}, [%r38];', 'ld.shared.v4.b32 \t{%r43, %r44, %r45, %r46}, [%r38+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 12, 'cvt.f32.bf16': 8, 'max.f32': 24}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 166, 'max_virtual_b16_regs': 9, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': 32, 'shared_memory_bytes': 1024, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| A. TMA setup & descriptor lifecycle | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 43-211 |
| B. Initial LocalLoad (shared -> registers) | Initial loading of TMA-loaded shared memory tile into registers | Lines 224-227 |
| C. Cross-lane & cross-warp reduction communication | Intra-warp butterfly shuffles and cross-warp shared memory partial exchanges | Lines 259-365 |
| D. Post-reduction layout conversion | ttg.convert_layout converting 1D reduction slice layout to 1D blocked layout | Lines 378-405 |
| E. Global store | st.global.b32 storing final 128 elements to output buffer | Lines 408-408 |

## Candidate `4`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.v2.b32', 'count': 8, 'line_range': [227, 234], 'raw_examples': ['ld.shared.v2.b32 \t{%r42, %r43}, [%r39];', 'ld.shared.v2.b32 \t{%r44, %r45}, [%r39+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 14, 'cvt.f32.bf16': 4, 'max.f32': 8}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 118, 'max_virtual_b16_regs': 5, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': 25, 'shared_memory_bytes': 1024, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:25 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| A. TMA setup & descriptor lifecycle | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 43-211 |
| B. Initial LocalLoad (shared -> registers) | Initial loading of TMA-loaded shared memory tile into registers | Lines 227-234 |
| C. Cross-lane & cross-warp reduction communication | Intra-warp butterfly shuffles and cross-warp shared memory partial exchanges | Lines 266-331 |
| D. Post-reduction layout conversion | ttg.convert_layout converting 1D reduction slice layout to 1D blocked layout | N/A |
| E. Global store | st.global.b32 storing final 128 elements to output buffer | Lines 334-334 |

## Candidate `2`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ldmatrix.sync.aligned.m8n8.x4.shared.b16', 'count': 4, 'line_range': [229, 232], 'raw_examples': ['ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r44, %r45, %r46, %r47}, [%r43];', 'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r48, %r49, %r50, %r51}, [%r43+1024];']}`
- **Arithmetic Mix**: `{'max.bf16x2': 15, 'cvt.f32.bf16': 2, 'max.f32': 2}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 111, 'max_virtual_b16_regs': 3, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': 23, 'shared_memory_bytes': 1024, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:23 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| A. TMA setup & descriptor lifecycle | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 43-211 |
| B. Initial LocalLoad (shared -> registers) | Initial loading of TMA-loaded shared memory tile into registers | Lines 229-232 |
| C. Cross-lane & cross-warp reduction communication | Intra-warp butterfly shuffles and cross-warp shared memory partial exchanges | Lines 264-308 |
| D. Post-reduction layout conversion | ttg.convert_layout converting 1D reduction slice layout to 1D blocked layout | N/A |
| E. Global store | st.global.b32 storing final 128 elements to output buffer | Lines 311-311 |

## Candidate `1`

- **TTGIR LocalLoad Layout**: `blocked`
- **Initial LocalLoad**: `{'instruction': 'ld.shared.b16', 'count': 32, 'line_range': [220, 265], 'raw_examples': ['ld.shared.b16 \t%rs1, [%r34];', 'ld.shared.b16 \t%rs2, [%r34+1024];']}`
- **Arithmetic Mix**: `{'max.bf16': 31, 'cvt.f32.bf16': 1}`
- **Virtual Registers**: `{'max_virtual_b32_regs': 49, 'max_virtual_b16_regs': 64, 'max_virtual_b64_regs': 17, 'max_virtual_pred_regs': 12, 'note': 'Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.'}`
- **Physical Resources**: `{'physical_regs_per_thread': 32, 'shared_memory_bytes': 1024, 'local_memory_bytes': 0, 'stack_bytes': 0, 'raw_text': 'Resource usage:\n Common:\n  GLOBAL:0\n Function kernel:\n  REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568 TEXTURE:0 SURFACE:0 SAMPLER:0'}`

### Semantic Phases (Annotated PTX)

| Phase | Description | Observed PTX Line Range |
| :--- | :--- | :--- |
| A. TMA setup & descriptor lifecycle | TMA descriptor creation, mbarrier setup, proxy fencing, and async bulk copy | Lines 43-211 |
| B. Initial LocalLoad (shared -> registers) | Initial loading of TMA-loaded shared memory tile into registers | Lines 220-265 |
| C. Cross-lane & cross-warp reduction communication | Intra-warp butterfly shuffles and cross-warp shared memory partial exchanges | N/A |
| D. Post-reduction layout conversion | ttg.convert_layout converting 1D reduction slice layout to 1D blocked layout | N/A |
| E. Global store | st.global.b32 storing final 128 elements to output buffer | Lines 308-308 |
