"""Freeze M4 structural hypotheses and bounded experiment design on CPU only."""
import subprocess
from experiments.tma_reduction_layout.phase11.c2d import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, access
from .static import GEOMETRIES, CONSUMERS, load_plan, plan, shared_offset


def main():
    source = c.BASE / 'phase11/m4'
    root = c.BASE / 'results/phase11/stage_m4_preparation'
    assert not root.exists()
    all_plans = []
    inputs_checked = 0
    rows = []
    for m, n, w in GEOMETRIES:
        for swizzle in (0, 32, 64, 128):
            assert {shared_offset(row, col, n, swizzle) for row in range(m) for col in range(n)} == set(range(m * n))
        for case in candidates(m, n, w):
            old = access(m, n, w, case['register_layout'])
            new = load_plan(case, 128)
            for key in ('ideal_wavefronts', 'wavefronts', 'bank_conflicts'):
                assert old[key] == new[key]
            inputs_checked += 1
            for consumer in CONSUMERS:
                output = plan(case, consumer)
                if consumer in ('broadcast', 'centered'):
                    vec = case['register_layout']['size_per_thread'][2]
                    assert output['store']['predicted_sector_requests_per_CTA'] == (m*n//8)*(2 if vec == 8 else 1)
                all_plans.append(output)
            if case['register_layout']['size_per_thread'][2] == 2:
                values = [load_plan(case, swizzle) for swizzle in (128, 64, 32, 0)]
                rows.append('| ' + case['id'] + ' | ' + ' | '.join(
                    f"{item['wavefronts']} / {item['bank_conflicts']}" for item in values) + ' |')
    names = subprocess.check_output(['git', 'ls-files', '-z', 'experiments/tma_reduction_layout'], cwd=c.ROOT).decode().split('\0')
    prior = {name: c.sha((c.ROOT/name).read_bytes()) for name in names if name}
    c.write_once(root/'prior_manifest.json', {'files': prior})
    c.write_once(root/'operation_plans_v2.json', {'stage': 'M4 CPU preparation', 'CPU_only': True,
        'previous_version': 'stage_m3/preregistration/static_predictions.json (unchanged)',
        'plans': all_plans, 'M4a_load_contrasts': [plan(case, 'native', swizzle)
           for m,n,w in GEOMETRIES for case in candidates(m,n,w)
           if case['register_layout']['size_per_thread'][2] == 2 for swizzle in (128,64,32,0)],
        'structural_checks': {'input_layouts_matching_previous_128B_address_model': inputs_checked,
                              'shared_address_bijections': 12,
                              'dense_store_sector_cases': inputs_checked*2},
        'not_a_cost_model': True, 'new_GPU_observations_used': False})
    for name in ('lib/Dialect/TritonGPU/IR/Dialect.cpp', 'lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp'):
        blob = subprocess.check_output(['git', 'show', c.PIN+':'+name], cwd=c.ROOT)
        c.write_once(root/'compiler_reference'/name, blob)
    report = '''# M4 CPU 准备：扩充操作计划与实验预绑定

新增 operation_plans_v2.json；M3 的计划和全部历史证据保持不变。本阶段无 GPU
编译/计时，无选择器拟合。新计划包含转换混合置换门限、warp/指令级 store 分组、
消费者需要的结果副本、单次返回读取的剪枝条件，以及真实二维消费者对宽化输入
活跃值的影响。完整实验边界见 phase11/m4/PLAN.md。

M4a 在寄存器布局和 native 消费者固定时改变 128/64/32/0B shared swizzle；
下面是编译前地址预测（wavefronts / bank conflicts）。预测以逻辑 vec 分组为条件，
必须由 SASS 和 NCU 检查真实指令分组，不能直接用计数除频率当延迟。

| 布局 | 128B | 64B | 32B | 0B |
|---|---|---|---|---|
''' + '\n'.join(rows) + '''

M4b 在两个原开发几何上测全部合法唯一布局，保留 native 1D 与原广播诊断，新增
真实 x.to(float32)-column_max(x.to(float32))。后者宽化输入存在第二个消费者，
BF16 packed-max、寄存器、驻留容量都可能改变，不能提前把变化只归到 MO。
不同消费者有不同输出字节数，分别在同语义内比较候选；不会把二维程序与一维
max 的时间直接作布局加速比。输出缓冲区按 BM*BN stride 分配。

加载侧支持 12 个 shared-address 双射检查，并与旧 128B 地址模型逐布局一致；
dense store 请求预测保留 vec8 的每指令 sector 重复，未将其当作 DRAM 写字节数。
旧 M3 H1/H2/H3 是已有结构证据；新增 shared-layout 和真实消费者的性能仍待测。

H4 拟检验加载吞吐与同步依赖的不同驻留响应，但 stall 百分比是相关诊断，不能
孤立证明某个 barrier 的延迟。根据 NVIDIA 的 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)，
short scoreboard 和 MIO 涉及多种操作；需联合源行、指令和 wavefront 检查。
若合法 swizzle 无法独立改变冲突，或同时改变 MR/寄存器，保留该负结果。

后续停点：完成 M4a、M4b 的报告/validator/分阶段 commit 与 push，等待复核。
'''
    c.write_once(root/'REPORT.md', report.encode())
    files = [source/'static.py', source/'PLAN.md', source/'prepare.py'] + [p for p in root.rglob('*') if p.is_file()]
    c.write_once(root/'freeze.json', {'CPU_only': True, 'parent_commit': subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'files': {p.relative_to(c.ROOT).as_posix(): c.sha(p.read_bytes()) for p in files},
        'native_rebuilds': 0})
    print('M4 CPU preparation OK:', inputs_checked, 'input layouts;', len(all_plans),
          'consumer plans;', len(prior), 'prior files frozen; no GPU compilation/timing')


if __name__ == '__main__':
    main()
