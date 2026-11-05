# 独立核查补充

三台 worker 的 source、protocol、call id、supported clocks、目标 UUID、返回码与错误文字经独立只读核查一致。1500 MHz 是实际支持的频率，失败属于权限拒绝。

设置前后可见的应用频率字段报告 deprecated，不能把 `-ac` 当成已验证替代方案。未进行其他时钟控制尝试、deferred 设置、reset、功率或 persistence 修改。

host 与 worker 的 UTC 时钟约有 50 秒偏差。协议在源码控制流中先 write、后 spawn；本探针的先后关系依赖这一因果顺序及保存的源文件，而不跨主机按 UTC 排序。后续负载 telemetry 与样本窗口应使用同一 worker 内的 monotonic timestamps。

“同一对 CUBIN”指 U 跨尺寸相同、GL 跨尺寸相同，U 与 GL 自身不同。G0 跨尺寸的一个字节差异位于非 allocated `.debug_line`，不改变运行代码。运行时参数、步幅、grid 和输入仍不同。
