"""GPU-free behavioral tests of capture order, shams and failure evidence.

Every driver/NVML/device object below is an explicit CPU fake. This does not
claim GPU correctness or platform validation; real eligibility runs remain
mandatory before timing.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes as C
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from . import engine


def value(x):
    return x.value if hasattr(x,'value') else x


class Clock:
    def __init__(self):
        self.now=1_000_000_000
        self.sleeps=[]
    def monotonic_ns(self):
        self.now+=10_000
        return self.now
    def sleep(self,seconds):
        assert seconds>=0
        self.sleeps.append(seconds)
        self.now+=int(seconds*1e9)


class Driver:
    """Emulate capture bookkeeping and replay CPU actions, never libcuda."""
    def __init__(self,clock):
        self.clock=clock;self.counts=Counter();self.codes=set();self.calls=[]
        self.serial=10;self.capture=None;self.graphs={};self.executables={}
        self.nodes={};self.actions={};self.captures=[];self.replay_log=[]
    def handle(self):
        self.serial+=1
        return self.serial
    def event(self):
        return C.c_void_p(self.handle())
    def operation(self,kind,name,action=lambda:None):
        if self.capture is None:
            action()
        else:
            node=self.handle();self.nodes[node]=(kind,name,action)
            self.capture.append(node)
    def call(self,name,types,*args):
        self.counts[name]+=1;self.codes.add(0);self.calls.append(name)
        if name=='cuStreamBeginCapture':
            assert self.capture is None
            self.capture=[]
        elif name=='cuEventRecordWithFlags':
            self.operation(7,'event:'+str(value(args[0])))
        elif name=='cuStreamEndCapture':
            graph=self.handle();self.graphs[graph]=list(self.capture)
            self.captures.append([self.nodes[n][:2] for n in self.capture])
            self.capture=None
            C.cast(args[1],C.POINTER(C.c_void_p))[0]=graph
        elif name=='cuGraphInstantiateWithFlags':
            handle=self.handle();self.executables[handle]=value(args[1])
            C.cast(args[0],C.POINTER(C.c_void_p))[0]=handle
        elif name=='cuGraphGetNodes':
            nodes=self.graphs[value(args[0])]
            C.cast(args[2],C.POINTER(C.c_size_t))[0]=len(nodes)
            if args[1] is not None:
                for i,node in enumerate(nodes):args[1][i]=node
        elif name=='cuGraphNodeGetType':
            C.cast(args[1],C.POINTER(C.c_int))[0]=self.nodes[value(args[0])][0]
        elif name=='cuGraphLaunch':
            handle=value(args[0]);self.replay_log.append(handle)
            for node in self.graphs[self.executables[handle]]:self.nodes[node][2]()
            self.clock.now+=200_000
        elif name in ('cuGraphExecDestroy','cuGraphDestroy','cuEventDestroy_v2','cuStreamSynchronize'):
            pass
        else:
            raise AssertionError('Unexpected fake-driver API: '+name)
    def elapsed_us(self,first,last):
        return 6.0


class Kernel:
    def __init__(self,path):
        self.path=path;self.blob=path.read_bytes();self.guards=0;self.graph_launches=0
    def guard(self):
        assert self.path.read_bytes()==self.blob
        self.guards+=1


class Device:
    def __init__(self,clock):
        self.driver=Driver(clock);self.context=C.c_void_p(1);self.stream=C.c_void_p(2)
        self.kernels=[];self.closed=False
        self.environment=dict(gpu_uuid='GPU-CPU-FAKE',cc=[9,0],SM_count=132,name='CPU fake H100')
    def close(self):
        self.closed=True


class Reader:
    def __init__(self,uuid):self.uuid=uuid;self.closed=False
    def close(self):self.closed=True


class Sampler:
    def __init__(self,reader,period_s):self.reader=reader;self.contexts=[];self.started=False
    def set_context(self,**labels):self.contexts.append(labels)
    def start(self):self.started=True
    def stop(self):
        return [dict(sequence=0,monotonic_ns=1,mock=True,context=c) for c in self.contexts]


def item(device,path,tag,role='U',copies=1,zeros=3):
    """A three-kernel in-place CPU pipeline requiring restoration each time."""
    state=dict(input=3,output=0,direct_checks=0,graph_reads=0)
    kernel=Kernel(path);device.kernels.append(kernel)
    def prepare():
        for _ in range(copies):
            device.driver.operation(1,'restore',lambda:state.update(input=3))
    def launch():
        for i in range(zeros):
            device.driver.operation(2,'zero'+str(i),lambda:state.update(output=0))
        device.driver.operation(0,'kernel0',lambda:state.update(input=state['input']+1))
        device.driver.operation(0,'kernel1',lambda:state.update(output=2*state['input']))
        device.driver.operation(0,'kernel2',lambda:state.update(input=state['input']+1))
    def check():
        prepare();launch();state['direct_checks']+=1
        return {'pass':state['output']==8,'complete_output':True}
    def check_graph():
        before=len(device.driver.replay_log)
        state['graph_reads']+=1
        checked={'pass':state['output']==8,'complete_output':True}
        assert len(device.driver.replay_log)==before
        return checked
    return dict(tag=tag,aliases=[],role=role,kernels=[kernel],prepare=prepare,launch=launch,
        check=check,check_graph=check_graph,
        expected_nodes=dict(kernels=3,memsets=3,copies=1)),state


@contextmanager
def patched(clock,graph=None):
    replacements=dict(NVML=Reader,Sampler=Sampler,collect_env_readonly=lambda reader:dict(mock=True),time=clock)
    if graph is not None:replacements['Graph']=graph
    originals={k:getattr(engine,k) for k in replacements}
    try:
        for name,replacement in replacements.items():setattr(engine,name,replacement)
        yield
    finally:
        for name,original in originals.items():setattr(engine,name,original)


def protocol():
    source=Path(engine.__file__).resolve()
    return dict(source_SHA256={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
        suites={'cpu':{'payload_SHA256':'0'*64}},
        timing=dict(seed=911,primes=8,visits=2,samples=3,warmups=1,
                    host_gap_seconds=.001,telemetry_period_s=.05))


def capture_tests(root):
    clock=Clock();device=Device(clock)
    good,state=item(device,root/'kernel.cubin','pipeline')
    graph=engine.Graph(device,good,8)
    operations=device.driver.captures[-1]
    events=[i for i,(kind,_) in enumerate(operations) if kind==7]
    assert len(events)==2
    first,last=events
    assert operations[first-1]==(1,'restore')
    assert [name for _,name in operations[first+1:last]]==['zero0','zero1','zero2','kernel0','kernel1','kernel2']
    assert sum(name=='restore' for _,name in operations[:first])==9
    assert sum(kind==0 for kind,_ in operations[:first])==24
    assert sum(kind==1 for kind,_ in operations[first+1:last])==0
    graph.run();assert state['output']==8 and state['input']==5
    graph.close()
    rejected=[]
    for label,actual_copies,actual_zeros in [('missing_restore',0,3),('missing_pipeline_zero',1,2)]:
        malformed,_=item(Device(clock),root/'kernel.cubin',label,copies=actual_copies,zeros=actual_zeros)
        bad_device=Device(clock)
        malformed,_=item(bad_device,root/'kernel.cubin',label,copies=actual_copies,zeros=actual_zeros)
        try:engine.Graph(bad_device,malformed,8)
        except AssertionError:rejected.append(label)
        else:raise AssertionError('Missing node was accepted: '+label)
    assert rejected==['missing_restore','missing_pipeline_zero']
    return dict(name='capture_order',status='OK',captured_nodes=len(operations),
        prime_iterations=8,timed_semantic_iterations=1,
        restores_before_start=True,three_pipeline_zeros_inside_target_events=True,
        missing_nodes_rejected=rejected)


def session_test(root,*,eligibility=False,fail_during_sample=False,fail_cleanup=False):
    clock=Clock();device=Device(clock);created=[]
    real_graph=engine.Graph
    class TracedGraph(real_graph):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);created.append(self)
            self.code={'U1':100,'U2':200,'G0':300}[self.item['tag']]
        def run(self):
            if fail_during_sample and self.item['tag']=='U1' and self.replays+1==11:
                raise RuntimeError('CPU injected mid-visit replay failure')
            super().run()
            return self.code+self.replays/1000
        def close(self):
            super().close()
            if fail_cleanup and self.item['tag']=='U1':
                raise RuntimeError('CPU injected graph cleanup failure')
    items=[];states=[]
    for tag,role in [('U1','U'),('U2','U'),('G0','G0')]:
        record,state=item(device,root/'kernel.cubin',tag,role)
        items.append(record);states.append(state)
    evidence=root/('eligibility' if eligibility else 'partial' if fail_during_sample else 'cleanup' if fail_cleanup else 'formal')
    error=None
    with patched(clock,TracedGraph):
        session=engine.Session(device,protocol(),evidence,0 if eligibility else 1,'cpu')
        try:session.measure(dict(id='mock_pipeline'),items)
        except RuntimeError as exc:error=str(exc)
        result=session.finish({'mock':dict(test_only=True)},error=error)
    assert device.closed and session.reader.closed
    shard=json.loads((evidence/result['shards'][0]['path']).read_text())
    cells={row['tag']:row for row in shard['cells']}
    assert set(cells)=={'U1','U2','G0','U1:repeat','U2:repeat'}
    assert len(created)==3, 'Repeats must reuse exact existing graph instances'
    assert len({id(graph) for graph in created})==3
    assert all(record['direct_checks']==1 for record in states)
    if fail_during_sample:
        assert result['status']=='FAILED' and result['partial_errors']
        assert all(len(row['visits'])==1 for row in cells.values())
        partial=shard['incomplete_visit']
        retained=sum(len(rows) for rows in partial['observations'].values())
        assert partial['visit']==1 and retained>0
        assert 'never a complete formal estimate' in partial['scope']
        return dict(name='partial_failure',status='OK',completed_visit_samples=15,
                    incomplete_observations_preserved=retained,error_recorded=True)
    if fail_cleanup:
        assert result['status']=='FAILED' and result['partial_errors']
        assert all(len(row['visits'])==2 for row in cells.values())
        return dict(name='cleanup_failure',status='OK',all_completed_samples_preserved=30)
    assert result['status']=='OK' and not result['partial_errors']
    assert all(record['graph_reads']==2 for record in states)
    assert len(result['checks'])==9
    assert {record['phase'] for record in result['checks']}=={'before','graph_before','graph_after'}
    assert all(record['pass'] for record in result['checks'])
    if eligibility:
        assert not result['formal_timing'] and all(row['visits']==[] for row in cells.values())
        assert all(graph.replays==2 for graph in created)
    else:
        for tag,code in [('U1',100),('U2',200),('G0',300),('U1:repeat',100),('U2:repeat',200)]:
            assert len(cells[tag]['visits'])==2
            samples=[x for visit in cells[tag]['visits'] for x in visit['samples_us']]
            assert len(samples)==6 and all(int(x)==code for x in samples)
            assert all(gap>=1_000_000 for visit in cells[tag]['visits'] for gap in visit['host_gaps_ns'])
        for tag in ('U1','U2'):
            original=[x for visit in cells[tag]['visits'] for x in visit['samples_us']]
            repeat=[x for visit in cells[tag+':repeat']['visits'] for x in visit['samples_us']]
            assert set(original).isdisjoint(repeat), 'Sham observations must not reuse samples'
        assert [g.replays for g in created]==[16,16,10]
        assert clock.sleeps
    trace=gzip.decompress((evidence/'telemetry.jsonl.gz').read_bytes())
    assert trace and b'"mock":true' in trace
    return dict(name='eligibility_no_formal_samples' if eligibility else 'multiple_U_shams',status='OK',
        original_graph_instances=3,independent_observation_cells=5,
        same_graph_repeat_count=2,captured_output_checks=6,
        formal_samples=0 if eligibility else 30,requested_host_gap_ns=1_000_000)


def run():
    source=Path(engine.__file__).resolve()
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='cadence-cpu-verify-') as name:
        root=Path(name)
        (root/'kernel.cubin').write_bytes(b'CPU fake frozen bytes; not an executable CUBIN')
        results=[capture_tests(root),session_test(root),session_test(root,eligibility=True),
                 session_test(root,fail_during_sample=True),session_test(root,fail_cleanup=True)]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    return dict(status='OK',scope='CPU fake-driver/NVML behavioral verification only; no GPU correctness or platform result.',
        GPU_dispatches=0,CUDA_libraries_loaded=False,tests=results,
        engine_source_SHA256=digest,
        reviewed_sources={str(Path(__file__).with_name(name)):hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('engine.py','run.py','prepare.py','cpu_verify.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=run()
    with args.output.open('xb') as handle:
        handle.write((json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=summary['status'],tests=len(summary['tests']),output=str(args.output)),sort_keys=True))
