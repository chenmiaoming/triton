# Phase8 StageC — Separate same-CUBIN NCU diagnostics

Single report per mode/binary; aggregate diagnostics, no variance or pure bank-conflict/descriptor latency claim.

| Physical binary/mode | l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum | l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum | dram__bytes_read.sum | dram__bytes_write.sum | smsp__inst_executed.sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M128_N32_w16:switch:4:mode0 | 34854.0 | 516147.0 | 3442726.0 | 3661875.0 | 134227456.0 | 4388608.0 | 38294116.0 |
| M128_N32_w16:switch:4:mode1 | 40438.0 | 280160.0 | 3579382.0 | 3638880.0 | 136325120.0 | 5797888.0 | 44852468.0 |
| M128_N32_w16:switch:default:mode0 | 29552.0 | 371938.0 | 5534576.0 | 6663394.0 | 134228480.0 | 4202752.0 | 59475973.0 |
| M128_N32_w16:switch:default:mode1 | 25819.0 | 212405.0 | 5661915.0 | 6716853.0 | 136326400.0 | 5490176.0 | 66788828.0 |
| M2048_N32_w16:switch:4:mode0 | 0.0 | 0.0 | 19136512.0 | 3145728.0 | 2147545088.0 | 4779520.0 | 64634515.0 |
| M2048_N32_w16:switch:4:mode1 | 0.0 | 0.0 | 19267584.0 | 3358720.0 | 2149607424.0 | 6794752.0 | 68944766.0 |
| M2048_N32_w16:switch:default:mode0 | 0.0 | 0.0 | 21233664.0 | 6291456.0 | 2147517440.0 | 4775680.0 | 81498532.0 |
| M2048_N32_w16:switch:default:mode1 | 0.0 | 0.0 | 21364736.0 | 6504448.0 | 2149610752.0 | 6819328.0 | 85971757.0 |

All original stdout/return codes/commands/NVR reports/source and payload snapshots are retained. A missing field is unavailable evidence, never a zero. Profiler duration is excluded from formal timing.

Metadata deviation: the original profiler export recorded H100 name/CC and dispatch identity but did not capture its worker UUID. UUID is unavailable and is not substituted from another run. All three formal timing worker UUIDs are recorded separately. No GPU recollection was performed to repair this metadata omission.

