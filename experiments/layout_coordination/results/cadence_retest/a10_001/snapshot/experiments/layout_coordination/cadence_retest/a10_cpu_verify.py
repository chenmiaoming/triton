"""Explicit GPU-free tests for the separate local A10 adapter and analysis."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from . import a10_runtime as runtime,engine,cpu_verify as fake
from .a10_prepare import UUID,inventory


def a10_session_test(root):
    clock=fake.Clock();device=fake.Device(clock)
    device.environment=dict(gpu_uuid=UUID,cc=[8,6],SM_count=72,name='NVIDIA A10')
    p=fake.protocol();p['suites']={'a10':dict(payload_SHA256='0'*64)}
    p['target']=dict(gpu_uuid=UUID,cc=[8,6],name='NVIDIA A10')
    assert runtime.A10Session.measure is engine.Session.measure
    assert runtime.A10Session.finish is engine.Session.finish
    items=[];states=[]
    for tag in ('3.5.1:U','3.6.0:U','3.7.1:U'):
        item,state=fake.item(device,root/'kernel.cubin',tag)
        items.append(item);states.append(state)
    with fake.patched(clock),patch.object(runtime,'NVML',fake.Reader),patch.object(runtime,'Sampler',fake.Sampler),patch.object(runtime,'collect_env_readonly',lambda reader:{}):
        session=runtime.A10Session(device,p,root/'session',1)
        cells=session.measure(dict(id='CPU_fake_three_release_controls'),items)
        result=session.finish([])
    assert result['status']=='OK' and len(cells)==6
    assert len(result['graph_proofs'])==3 and len(result['checks'])==9
    assert all(state['graph_reads']==2 for state in states)
    assert {x['sham_of'] for x in cells if x['role']=='U_repeat'}=={x['tag'] for x in items}
    # There is no architecture-name spoofing or call to the H100 initializer.
    device.environment['cc']=[9,0]
    try:runtime.A10Session(device,p,root/'rejected',1)
    except AssertionError:pass
    else:raise AssertionError('A10 initializer accepted CC90')
    return dict(name='legitimate_SM86_initializer_and_unchanged_engine',status='PASS',
        original_graphs=3,independent_cells=6,wrong_architecture_rejected=True)


def oracle_test():
    source=np.frombuffer(runtime.quarter_input(),np.uint16)
    expected=runtime.expected_output()
    assert source.size==8192*8192 and expected.shape==(256,8192)
    # Verify every phase, many tiles, boundary addresses and both BN geometries
    # using actual BF16 bits independently of the analytic phase table.
    checks=0
    for pm in (0,1,17,128,255):
        for column in list(range(97))+[127,128,4095,8191]:
            indices=(pm*32+np.arange(32))*8192+column
            values=(source[indices].astype(np.uint32)<<16).view(np.float32)
            assert expected[pm,column]==values.max()
            checks+=1
    return dict(name='full_BF16_input_and_independent_oracle',status='PASS',
        checked_columns=checks,expected_outputs=expected.size,input_bytes=source.nbytes)


def analysis_test(root,programs):
    from .a10_report import analyze
    p=dict(suites={'a10':dict(programs=programs)},
        timing=dict(bootstrap_seed=123,bootstrap_draws=128))
    c.write_once(root/'preregistration/protocol.json',p)
    for invocation in (1,2,3):
        folder=root/f'timing/invocation_{invocation}'
        shards=[]
        for panel in ('layout','versions'):
            for geometry in sorted({x['geometry'] for x in programs}):
                cells=[]
                for program in [x for x in programs if x['panel']==panel and x['geometry']==geometry]:
                    tag=program['tag'];time=100.
                    if panel=='layout' and tag.endswith('direct:vec1'):time=70.
                    if panel=='versions' and tag.startswith('3.6.0:'):time=104.
                    visits=[dict(samples_us=[time]*31) for _ in range(10)]
                    cells.append(dict(tag=tag,visits=visits))
                    if program['role']=='U':cells.append(dict(tag=tag+':repeat',visits=visits))
                name=panel+'_'+geometry+'.json'
                c.write_once(folder/name,dict(context=dict(id=panel+':'+geometry),cells=cells))
                shards.append(dict(path=name))
        c.write_once(folder/'result.json',dict(shards=shards))
    result=analyze(root)
    assert all(x['equivalent_within_3percent'] for x in result['controls'])
    direct=[x for x in result['layout_comparisons'] if x['tag'].endswith('direct:vec1')]
    assert len(direct)==2 and all(x['certified_vs_U_and_G0'] for x in direct)
    assert all(abs(x['vs_U']['gain_percent']-30)<1e-8 for x in direct)
    assert all(x['practical_slower'] for x in result['release_comparisons'] if x['version']=='3.6.0')
    assert all(x['certified_regression'] for x in result['release_comparisons'] if x['version']=='3.6.0')
    assert all(not x['certified_vs_U_and_G0'] for x in result['layout_comparisons'] if x['scope']=='cp_async_changed_path')
    # A bad release sham must preserve the raw CI while removing the claim.
    for path in root.glob('timing/invocation_*/versions_*.json'):
        shard=c.read(path)
        for cell in shard['cells']:
            if cell['tag'].startswith('3.6.0:') and cell['tag'].endswith(':repeat'):
                for visit in cell['visits']:visit['samples_us']=[208.]*31
        path.write_bytes(c.encode(shard))
    unqualified=analyze(root)
    assert all(x['practical_slower'] and not x['certified_regression'] and not x['controls_equivalent']
        for x in unqualified['release_comparisons'] if x['version']=='3.6.0')
    return dict(name='paired_analysis_and_changed_path_boundaries',status='PASS',
        controlled_gain_percent=30,release_regression_detected=True,changed_path_not_certified=True)


def early_failure_test(root):
    p=dict(source_SHA256={},suites={'a10':dict(payload_SHA256='0'*64)})
    device=fake.Device(fake.Clock())
    with patch.object(runtime,'cpu_gate',lambda *args:None),patch.object(runtime.c,'extract',lambda *args:None),\
         patch.object(runtime,'Device',lambda:device),patch.object(runtime,'A10Session',side_effect=RuntimeError('CPU injected initial NVML/session failure')):
        result=runtime.run(p,b'',root,0)
    assert result['status']=='FAILED' and not result['GPU_timing_started']
    assert result['phase']=='device_or_session_initialization' and device.closed
    assert c.read(root/'result.json')==result
    return dict(name='structured_early_failure_and_device_cleanup',status='PASS',GPU_dispatches=0)


def run():
    programs,pairs=inventory()
    assert len(programs)==28 and len(pairs)==2
    counts={panel:sum(x['panel']==panel for x in programs) for panel in ('layout','versions')}
    assert counts==dict(layout=22,versions=6)
    assert sum(x['role']=='U' for x in programs)==8
    assert sum(x['role']=='G0' for x in programs)==2
    with tempfile.TemporaryDirectory(prefix='a10-cpu-') as name:
        root=Path(name);(root/'kernel.cubin').write_bytes(b'CPU fake, not executable')
        tests=[a10_session_test(root),oracle_test(),analysis_test(root/'analysis',programs),early_failure_test(root/'early_failure')]
    return dict(status='PASS',GPU_contexts=0,GPU_dispatches=0,compile_calls=0,
        scope='CPU fake-driver/NVML tests; not GPU eligibility.',original_programs=28,
        authentic_allocated_ELF_pairs=2,repeat_shams=8,tests=tests)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run()
    c.write_once(args.output,result)
    print(json.dumps(dict(status=result['status'],tests=len(result['tests']))))
