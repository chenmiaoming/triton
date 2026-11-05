"""CPU-only validation and report for the real clock-permission probe."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

SOURCE = Path(__file__).resolve()
REPO = SOURCE.parents[3]
ROOT = REPO/'experiments/layout_coordination/results/clock_control/probe_001'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def inspect(root=ROOT):
    root = Path(root)
    protocol = json.loads((root/'preregistration/protocol.json').read_text())
    source = (root/'preregistration/source.py').read_bytes()
    assert sha(source) == protocol['source_SHA256']
    assert source == (SOURCE.parent/'probe_run.py').read_bytes()
    assert protocol['GPU'] == 'H100!' and protocol['profile'] == 'chenmiaoming'
    assert protocol['workers'] == 3
    rows = []
    hashes = {str(path.relative_to(REPO)): sha(path.read_bytes())
              for path in sorted(root.rglob('*')) if path.is_file()}
    for invocation in range(protocol['workers']):
        folder = root/f'invocation_{invocation:03}'
        raw = json.loads((folder/'raw.json').read_text())
        dispatch = json.loads((folder/'dispatch.json').read_text())
        assert dispatch['protocol_SHA256'] == sha(json.dumps(protocol, sort_keys=True).encode())
        assert dispatch['profile'] == 'chenmiaoming' and dispatch['call_id'].startswith('fc-')
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['source_SHA256'] == protocol['source_SHA256']
        assert not raw['formal_timing'] and raw['kernel_launches'] == 0
        assert raw['memory_clock_lock_attempted'] is False
        assert raw['effective_under_load_verified'] is False
        env = raw['environment'];uuid = env['gpu_uuid']
        assert uuid.startswith('GPU-') and 'H100' in env['name'] and 'H200' not in env['name']
        commands = {row['role']: row for row in raw['commands']}
        assert len(commands) == len(raw['commands'])
        assert set(commands) == {'identity','help','before','supported_clocks',
                                 'lock_gpu_clocks','after_attempt','final_state'}
        clocks = {graphics for _,graphics in raw['supported_clock_pairs']}
        frequency = min(clocks, key=lambda value: (abs(value-protocol['target_MHz']),value))
        assert raw['requested_graphics_MHz'] == frequency == 1500
        lock = commands['lock_gpu_clocks']
        assert lock['argv'] == ['nvidia-smi','-i',uuid,
                               '--lock-gpu-clocks='+str(frequency)+','+str(frequency)]
        # A denied request is a valid observed outcome, not a successful lock.
        denied = lock['returncode'] == 4 and 'does not have permission' in lock['stdout']
        assert denied and raw['clock_request_accepted'] is False
        assert raw['restore_passed'] is None and 'restore_gpu_clocks' not in commands
        for role in ['before','after_attempt','final_state']:
            command = commands[role]
            assert command['returncode'] == 0 and command['argv'] == ['nvidia-smi','-i',uuid,'-q','-x']
            gpu = ET.fromstring(command['stdout']).find('gpu')
            assert gpu.findtext('uuid') == uuid
        initial = ET.fromstring(commands['before']['stdout']).find('gpu')
        rows.append(dict(invocation=invocation,**env,euid=raw['euid'],requested_MHz=frequency,
            returncode=lock['returncode'],message=lock['stdout'].strip(),permission_denied=denied,
            initial_idle_clocks={entry.tag:entry.text for entry in initial.find('clocks')},
            effective_under_load_verified=False,formal_timing=False))
    assert len({row['gpu_uuid'] for row in rows}) == protocol['workers']
    return dict(status='OK',CPU_only=True,GPU_allocations=0,worker_count=len(rows),
        distinct_GPU_count=len(rows),permission_denied_count=sum(row['permission_denied'] for row in rows),
        effective_fixed_clock_benchmark_performed=False,rows=rows,
        input_SHA256=hashes,auditor_source_SHA256=sha(SOURCE.read_bytes()),
        scope='These three Modal H100 workers denied SM-clock writes. Does not establish a policy for every Modal GPU or profile.')


def render(result):
    lines = ['# Modal H100 锁频权限实测','',
        '**结果：3 个不同 UUID 的 H100 worker 均拒绝修改 GPU/SM 频率，未执行锁频计时。**',
        '容器内 euid=0；三台 driver 均为 '+result['rows'][0]['driver_version']+'。使用 `gpu="H100!"` 固定型号，账户 profile 为 `chenmiaoming`。','',
        '| Worker | GPU UUID | 请求 GPU clock | 返回码 | 结果 |','|---:|---|---:|---:|---|']
    for row in result['rows']:
        lines.append(f'| {row["invocation"]} | `{row["gpu_uuid"]}` | {row["requested_MHz"]},{row["requested_MHz"]} MHz | {row["returncode"]} | 无修改权限 |')
    lines += ['', '实际命令以每台分配的 UUID 为目标：','',
        '```text','nvidia-smi -i GPU-... --lock-gpu-clocks=1500,1500','```','',
        '三台均返回：','',
        '```text','The current user does not have permission to change clocks for GPU ...',
        'Terminating early due to previous errors.','```','',
        '1500 MHz 来自各台实际 supported-clock 表。保存了身份、完整 `nvidia-smi --help`、supported clocks、设置前/尝试后/最终 XML、精确命令、返回码、stdout/stderr、source SHA 与 Modal call id。没有成功写入，所以未执行 reset，也没有修改功率上限或 persistence。',
        '设置前空闲 SM clock 为 345 MHz、memory 为 2619 MHz；这些是空闲查询，不是负载下的频率证明。','',
        '## 原问题中的“相同 CUBIN”','',
        'U 与 GL 是两份不同程序；U 的完整 CUBIN 在两种 AWQ 尺寸间相同，GL 的完整 CUBIN 也分别相同。它们之间不是同一份 CUBIN。',
        '运行时 `num_cols` 从 448 变为 1024，`num_rows` 从 3584 变为 8192，grid 从 1568 变为 8192 CTA，行地址步幅、指针、数据与总工作集也不同。相同机器代码不意味着相同运行任务。',
        'G0 两份文件仅有非 allocated `.debug_line` 中一个字节不同，allocated code/data、ABI 和资源相同。详细来源仍见此前 `reference_audit.json`，不改旧证据。','',
        '## 对旧计时的限制','',
        '旧 N4/N5 来自不同 worker 批次，没有记录 clocks/power/temperature，无法回溯证明实际频率相同。本轮权限探针也没有在负载下进行计时，不能据此排除 DVFS，更不能将旧跨批次的排序变化唯一归因于 footprint 或 cache。',
        '原始测量及其区间保持不动；后续采用同卡、同函数、匹配缓存条件的复测再辨识这些因素。具体协议见 [MATCHED_RETEST_PLAN.md](../../../clock_control/MATCHED_RETEST_PLAN.md)。','',
        '## 平台和官方文档边界','',
        '[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/) 说明 GPU locked-clock 操作要求管理权限；容器 uid=0 不能替代实际成功返回和负载验证。Hopper 的显存锁频走 deferred 路径，需要下一次 GPU 初始化；本轮没有写 deferred 配置或尝试宿主机 reset/重载驱动。',
        '[Modal GPU 文档](https://modal.com/docs/guide/gpu) 中的 `H100!` 用于固定设备类型。该探针只证明本轮三台 H100 的权限结果，不把它扩大为所有 Modal 型号/账户永远不可锁频。','',
        '本轮没有运行新 CUDA benchmark、编译 Triton 或修改 C++；所有权限失败结果完整保留。CPU validator 的 OK 表示探针证据完整，不表示锁频成功。','']
    return '\n'.join(lines)


if __name__ == '__main__':
    result = inspect()
    for path,value in [(ROOT/'validation.json',json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n'),
                       (ROOT/'REPORT.md',render(result))]:
        with path.open('x') as handle:handle.write(value)
    print(json.dumps({key:result[key] for key in ['status','worker_count','permission_denied_count','effective_fixed_clock_benchmark_performed']}))
