# Separate StageC Nsight diagnostic counters

SINGLE_CASE_SINGLE_PROFILE_PER_BINARY_DESCRIPTIVE_ONLY

| Binary | Registers/thread | Executed instructions | DRAM read bytes | DRAM write bytes | Replay duration ns (diagnostic) | Passes |
| --- | --- | --- | --- | --- | --- | --- |
| M128_N32_w16:canonical:4 | 22 | 44957205.0 | 136324352.0 | 5862656.0 | 96480.0 | 1 |
| M128_N32_w16:canonical:default | 31 | 66563286.0 | 136326144.0 | 5696000.0 | 145632.0 | 1 |
| M128_N32_w16:device_canonical:4 | 22 | 44883678.0 | 136325120.0 | 5689856.0 | 95904.0 | 1 |
| M128_N32_w16:device_canonical:default | 31 | 66539676.0 | 136324864.0 | 5693952.0 | 145664.0 | 1 |
| M128_N32_w16:device_native:4 | 22 | 43382004.0 | 136323840.0 | 5897728.0 | 88352.0 | 1 |
| M128_N32_w16:device_native:default | 31 | 66035813.0 | 136325376.0 | 6030592.0 | 139360.0 | 1 |
| M128_N32_w16:host_canonical:4 | 22 | 36612623.0 | 134225408.0 | 4054272.0 | 66176.0 | 1 |
| M128_N32_w16:host_canonical:default | 31 | 58283302.0 | 134237696.0 | 4052224.0 | 116064.0 | 1 |
| M128_N32_w16:host_native:4 | 22 | 36311164.0 | 134224128.0 | 4097280.0 | 63072.0 | 1 |
| M128_N32_w16:host_native:default | 31 | 56916072.0 | 134239744.0 | 4093952.0 | 104384.0 | 1 |

All commands, settings, original CSV text and binary .ncu-rep reports remain frozen. Source contrasts are retained in counter_results.json.

- Ten independent executable launches in one separate H100 profiler worker; one report per binary, no variance estimate.
- Kernel replay/cache flush changes context; clocks uncontrolled; profiler duration is diagnostic only.
- Instruction/DRAM counters cover the entire kernel and compiler responses, not pure descriptor/store costs.
- The chosen old case is outcome-informed; counters are not a held-out confirmatory test.
