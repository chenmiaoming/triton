# Modal H100 锁频权限实测

**结果：3 个不同 UUID 的 H100 worker 均拒绝修改 GPU/SM 频率，未执行锁频计时。**
容器内 euid=0；三台 driver 均为 610.57.04。使用 `gpu="H100!"` 固定型号，账户 profile 为 `chenmiaoming`。

| Worker | GPU UUID | 请求 GPU clock | 返回码 | 结果 |
|---:|---|---:|---:|---|
| 0 | `GPU-2d1503b1-69dd-1e30-ac61-c9a81afa2c9d` | 1500,1500 MHz | 4 | 无修改权限 |
| 1 | `GPU-07b0936a-7bc5-e984-5ef2-9b64d2ce6a55` | 1500,1500 MHz | 4 | 无修改权限 |
| 2 | `GPU-0716ad1d-8289-f194-6182-65de01f6ccbc` | 1500,1500 MHz | 4 | 无修改权限 |

实际命令以每台分配的 UUID 为目标：

```text
nvidia-smi -i GPU-... --lock-gpu-clocks=1500,1500
```

三台均返回：

```text
The current user does not have permission to change clocks for GPU ...
Terminating early due to previous errors.
```

1500 MHz 来自各台实际 supported-clock 表。保存了身份、完整 `nvidia-smi --help`、supported clocks、设置前/尝试后/最终 XML、精确命令、返回码、stdout/stderr、source SHA 与 Modal call id。没有成功写入，所以未执行 reset，也没有修改功率上限或 persistence。
设置前空闲 SM clock 为 345 MHz、memory 为 2619 MHz；这些是空闲查询，不是负载下的频率证明。

## 原问题中的“相同 CUBIN”

U 与 GL 是两份不同程序；U 的完整 CUBIN 在两种 AWQ 尺寸间相同，GL 的完整 CUBIN 也分别相同。它们之间不是同一份 CUBIN。
运行时 `num_cols` 从 448 变为 1024，`num_rows` 从 3584 变为 8192，grid 从 1568 变为 8192 CTA，行地址步幅、指针、数据与总工作集也不同。相同机器代码不意味着相同运行任务。
G0 两份文件仅有非 allocated `.debug_line` 中一个字节不同，allocated code/data、ABI 和资源相同。详细来源仍见此前 `reference_audit.json`，不改旧证据。

## 对旧计时的限制

旧 N4/N5 来自不同 worker 批次，没有记录 clocks/power/temperature，无法回溯证明实际频率相同。本轮权限探针也没有在负载下进行计时，不能据此排除 DVFS，更不能将旧跨批次的排序变化唯一归因于 footprint 或 cache。
原始测量及其区间保持不动；后续采用同卡、同函数、匹配缓存条件的复测再辨识这些因素。具体协议见 [MATCHED_RETEST_PLAN.md](../../../clock_control/MATCHED_RETEST_PLAN.md)。

## 平台和官方文档边界

[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/) 说明 GPU locked-clock 操作要求管理权限；容器 uid=0 不能替代实际成功返回和负载验证。Hopper 的显存锁频走 deferred 路径，需要下一次 GPU 初始化；本轮没有写 deferred 配置或尝试宿主机 reset/重载驱动。
[Modal GPU 文档](https://modal.com/docs/guide/gpu) 中的 `H100!` 用于固定设备类型。该探针只证明本轮三台 H100 的权限结果，不把它扩大为所有 Modal 型号/账户永远不可锁频。

本轮没有运行新 CUDA benchmark、编译 Triton 或修改 C++；所有权限失败结果完整保留。CPU validator 的 OK 表示探针证据完整，不表示锁频成功。
