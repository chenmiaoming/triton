# 固定预热与采样间隙复测：旧证据与新证据汇总

状态：**OK**；552,420 个目标计时样本。此文件只汇总已有冻结分析，不调参、不重新选模型、不改变旧数据。

| Suite | Contexts | 原始样本 | U/G0 实用等价（逻辑比较） | Sham 实用等价（逻辑比较） |
|---|---:|---:|---:|---:|
| legacy | 216 | 552,420 | 0/0 | 189/189 |

## M3 配对机制证据

全部 189 个相同 vec、consumer、q、grid 的 within/N-first 配对保留。没有把这些显式 Gluon 对照改称 Triton U/G0。

summary.json 另给出按 q 分层的三网格服务斜率。旧原始样本按新 visit 估计量重算的点值与旧报告的 flat-sample median 斜率分别命名；没有给旧点值虚构 CI。

## 解释边界

- This is derived postprocessing of frozen measurements, not a newly selected cost model or sampling protocol.
- A support transition means a difference in pointwise CI resolution between two execution contexts. It does not prove the old gain was caused by idle or that a new gain would hold under every real workload.
- Old N4/G8 CI-supported thresholds use their actual stored confidence intervals. Legacy/anchor published point slopes do not acquire invented old confidence intervals.
- All losing, unchanged and control-inconclusive outcomes remain in the source analyses. Aliases do not create additional independent observations.
- Each suite has three independently allocated formal GPU workers. Raw sample counts and logical comparison counts are not independent GPU counts.
