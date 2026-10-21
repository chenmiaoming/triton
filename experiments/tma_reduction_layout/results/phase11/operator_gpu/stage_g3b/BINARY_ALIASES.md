# 报告覆盖与精确二进制别名

每个几何有 11 个逻辑归并标签、10 个唯一 CUBIN/ABI 执行身份，另有原始 dx kernel。
Gluon anchor 与 `vec8:n_first` 是完全相同的 CUBIN、参数名/顺序、专化值、warp 数和 dynamic shared bytes；不重复采集该物理程序。

初始 profile 子集只显式列出 anchor，后续补齐按同一执行身份去重。因此部分原始 job 不包含 `vec8:n_first` 字面别名，但它由同一份 anchor 报告覆盖。最终 validator 从编译记录核对完整执行身份，不用“相同 SASS 大概相同”替代精确 SHA。

总计三个几何 × 10 个身份 × software/warm_application 两种模式 = 60 份报告，覆盖 33 个逻辑归并标签的两种模式。Native-output 等拥有不同 CUBIN 的标签保留其独立报告。
