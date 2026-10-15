## 第 1 页：面向访存与部分归约的 GPU 张量布局选择

本次汇报已完成的问题表征、源码与编译产物审计、Gluon 受控布局对照和本地 A10 版本对照。新微基准 C1、正式硬件校准、静态选择模型和新留出集验证均尚未执行。研究目标是识别值得换布局的部分规约配置，同时保留默认布局已经较好的情况。
源码版本：固定 upstream 20d9d98fead78c2f65e1b3ba5a7881d86a836dad；本地 Triton 3.7.1；本次观察 main 4eda6cc91e86553a2517d1c764ea2967985553ad。

## 第 2 页：切入点：取数布局与部分归约的局部性

不要把“默认布局存在更快候选”与“某版本升级后发生回归”混为一谈。H100 TMA 与 A10 普通 load 的软件版本和测量批次不同，分别报告，不用一个收益区间概括。vec 只是布局的一部分；现阶段尚未分离各项硬件成本，也没有经过校准的选择器。

## 第 3 页：代表性 Triton 程序：TMA 后逐列 max

代码保留三维 tile [1,BM,BN]，所以 axis=1 是 M，而不是二维代码中的 axis=0。host 端构造 TensorDescriptor(x, list(x.shape), list(x.stride()), [1,BM,BN])。原始配置启动 16384 个 CTA，num_warps=4；每个 tile 的部分结果写入 [256,8192] FP32 输出。该页只说明 TMA 程序，不能把它当作 A10 上执行了 TMA 的证据。普通 load 代码见第18页。

## 第 4 页：计算模式：逐列归约，消去 M、保留 N

每列的 BM 个元素合并成一个最大值，消去 M，保留 N。矩阵例子的输出 5、9、8、7 正确。这是分块 partial reduction，不是一次完成整个 M=8192 的最终列最大值。后面改变的是元素的硬件归属，数学语义保持不变。

## 第 5 页：两种取数路径，归约算术都使用寄存器中的值

原普通 load 诊断的 direct vec1 在 32×128/w4 上没有 LDS/STS/SHFL/BAR，max 直接在寄存器上执行；direct vec8 的 LDG 也直接进入寄存器，但后续规约/转换有 shared 操作。cp.async 是另一条显式 shared staging 对照，不能把它叫 TMA。这里的 max 是编译器生成的寄存器算术；没有声称在 global memory 中直接做规约。

## 第 6 页：三种输入布局：参数、重复覆盖与贡献线程

本表所有参数、基本覆盖、重复次数和贡献线程数均经复核。历史两列是原 issue 的 TMA encoding，不是普通 load 在旧版本的 encoding。普通 load 的被测 3.5.1/3.6.0/3.7.1 都已选择 vec8。第三列是 Phase10 的早期 vec2 原型，只用于说明不同 ownership，不代表已验证的现行成本模型。N 优先时 vec2 的列0属于 warp0 和 warp2。

## 第 7 页：第 0 列的 32 个值分别归哪个线程？

保持用户原先要求的完整32行归属图。TMA旧布局每列1个线程；vec8每列8个线程跨4 warp；早期vec2每列2个线程跨2 warp。vec8的同一W0:L0持有M=0/8/16/24，完整tile上各持有N=0..7，共32元素。vec2的第0列偶数行是W0:L0、奇数行W2:L0。此图只是输入元素归属，不表示实际指令顺序或已证明的时间。行标签约9pt，现场可放大讲解；具体数学不变。

## 第 8 页：TMA 与普通 load：两个入口，都是先构造访存布局

E中的除法是源码的整除。128单位是bit，而不是bytes或TMA交易大小。普通load的简式在单一BF16连续指针场景下解释，严格源码还在相关同shape同order内存操作间取最大值，再受E限制。不推广到atomic特殊限制，不把SPT当动态load次数。两条路径的源码入口不同，但最终都构造BlockedEncodingAttr。source_audit中固定保存今日main SHA，关键五个transform文件与实验upstream逐字相同；没有计时新main。

## 第 9 页：同样选 vec8，为什么同列贡献线程会从 8 变成 32？

N128/vec8只需16个线程位置，因此一个warp的剩余2个lane位置用于M；4个warp全部用于M。N64/vec8只需8个位置，剩余4个lane位置用于M；8个warp用于M，因此同一列有32个贡献线程。输入布局与后续lowering成本并非一一对应，不能仅靠贡献线程数预测总时间。audit.py在CPU复现builder，对照六个已存release/shape TTGIR全部一致；不是新增instrumented compiler trace。

## 第 10 页：后续 pass 为什么没有把本例改回线程局部布局？

图为相关pass的简化关系，实际pipeline多次运行RemoveLayoutConversions，且包含其他pass与TMA lowering。OptimizeThreadLocality还处理特定reshape/gather；本例没有归约前允许重排的reshape。所核对版本没有名为shouldOptimizeThreadLocality的函数，具体guards位于runOnOperation。RemoveLayoutConversions也有rematerialization局部代价估计，因此不说整个编译器没有成本机制。缺口是对本例候选输入布局的完整、经过校准的联合时间比较。

## 第 11 页：需要比较完整路径，同时满足资源约束

现阶段已经看到不同lowering、转换数与通信结构，但尚未把总耗时可靠分解为独立分项。资源变化影响吞吐和延迟隐藏，不能把所有阶段时间简单相加，也不能把两个不同kernel的耗时差当纯规约延迟。

## 第 12 页：A10 普通 load：默认布局与固定候选的对照

所有时间为同批三个worker中位数的中位数。32×64/w8原始Triton默认422.400；Gluon vec8 421.888、vec1 279.440。33.8%=1−279.440/421.888，而不是把另一轮版本实验的436µs与这里279µs混算。32×128/w4还测了vec8 within-warp等变体，详见完整证据。接近已测最快候选是描述性观察，不等于全空间最优。计时与NCU分开；既有NCU不能直接把每项总性能差归因于单一机制。

## 第 13 页：H100 TMA：收益同样依赖形状与 warp 数

第一表是worker中位数的中位数；百分比来自七次worker配对log-ratio，使用同条件六个比较的Bonferroni/t区间。因此不能直接用第一表两个跨worker中位数相除复算百分比。3%是预声明实用差异阈值。这里固定host路径；device descriptor的41.297%保存在历史结果，不混入原始路径汇报。已有默认Triton↔matching-layout Gluon bridge证据支持入口对应；这仍不是新compiler patch已实现这些收益的证明。各候选资源与lowering共同变化尚需后续机制校准。

## 第 14 页：早期模型的不足：逻辑分数不能替代硬件成本

默认PTX 120行起先执行max.bf16x2，127行起执行cvt.f32.bf16；候选127行起packed max、143行起转换。因此两者都是先BF16局部合并再转换，而不是只有vec2能延后转换。旧加载分4/16与两种不同指令静态条数4/4没有可直接换算的物理单位，删除了“模型低估/高估4倍”。两份代码最初加载段的4条PTX不能证明硬件成本相等。旧等权规则已弃用，不安排继续证明其正确性或错误性。

## 第 15 页：先验证微基准能测到什么，再校准子图有效成本

ML/MR需要可观察的结果和合法block-uniform runtime循环；不能重复max(x)同一个不变张量就当重复执行。人工反馈成本不直接转嫁到原kernel；不能简单相减得纯max或纯转换延迟。MR若与原kernel的BF16 packed模板不匹配，只能校准匹配部分。MO若转换被折叠，只保留诊断并使用真实相邻子图联合辨识。D=1/2和驻留限制区分依赖延迟与吞吐。纯寄存器子图不需要descriptor。将来扩展普通load时需单独的global/cache取数模板，不能直接套shared-load参数。

## 第 16 页：模型在编译期做选择，完整 kernel 的实测用于评价

这是待验证模型路线，不是已实现选择器。θ应具有延迟或服务速率等可解释单位，不给每个layout分别自由拟合权重。寄存器与驻留使用早期静态估计和不确定区间，不能偷用测试binary的真实资源信息。regret中的实测最小值是有噪声的评价oracle，需报告不确定性。首版优先验证H100未见shape，之后才考虑B200与其他可用架构；旧数据用于调优后不能继续称held-out。

## 第 17 页：已完成问题表征；下一步是校准可行性与选择模型

实际进度为Phase11 A/B、两轮本地A10与本次源码/文档审计已完成；新的C1尚未执行，模型尚未构建。不把早期Phase10的StageC与这次新的校准C1混淆。目前已有显著优化机会和负方向对照，但没有把所有机理项测清，也没有证明跨架构泛化。创建PR仍需用户另行许可；此次任务只更新报告和整理代码机制。

## 第 18 页：A10 上运行的普通 Triton DSL

该函数源码与已完成A10试验中的普通Triton anchor一致；新报告未新编译或执行GPU。BM=32，BN128/w4的grid=16384；BN64/w8的grid=32768。没有descriptor，没有手工shared allocation，也没有Gluon显式布局；shared通信由后端按实际规约/转换需求生成。

## 第 19 页：A10：没有版本回归，不代表默认布局总是合适

原始配置的3.6/3.5比值1.00012，95% t区间[0.99882,1.00141]；补充配置0.99989，[0.99973,1.00006]。3%阈值在测试前冻结，结论仅限一张A10和两个配置。相同shape的三个release最终TTGIR均核对load和reduction输入编码；order在size1 batch维上的变化不改变有效M/N归属。这些时间不是今日main的性能测量，也不替代原GB300的TMA验证。

## 第 20 页：源码版本、原始证据与后续方案索引

所有源码文件已随报告归档并记录SHA256。S1为Coalesce.cpp，S2为CoalesceUtils/Utility，S3为TritonGPUAttrDefs.td builder，S4为RemoveLayoutConversions，S5为OptimizeThreadLocality。详细函数和不可变commit链接见mechanism_audit.md。报告输入PPTX也保存SHA；旧raw samples、compiler artifacts与NCU reports保持不变。
