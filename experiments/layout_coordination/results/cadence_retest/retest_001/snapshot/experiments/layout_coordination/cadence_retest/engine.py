"""Uniform GPU priming, input restoration and sampling for frozen binaries."""
import ctypes as C
import gzip
import json
from pathlib import Path
import random
import sys
import time
import traceback

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.layout_coordination.clock_control.telemetry import NVML, Sampler, collect_env_readonly


class Graph:
    """Eight unmeasured semantic iterations, then exactly one measured one.

    Input restoration precedes each iteration, outside the target events.
    Internal operator work (e.g. backward zeroing) remains in launch().
    """
    def __init__(self, device, item, primes):
        self.item, self.driver, self.stream = item, device.driver, device.stream
        d = self.driver
        self.first, self.last = d.event(), d.event()
        self.graph, self.executable = C.c_void_p(), C.c_void_p()
        self.primes, self.replays = primes, 0
        d.call('cuStreamBeginCapture', [C.c_void_p, C.c_int], self.stream, 0)
        for iteration in range(primes + 1):
            if item.get('prepare'):
                item['prepare']()
            if iteration == primes:
                d.call('cuEventRecordWithFlags', [C.c_void_p, C.c_void_p, C.c_uint], self.first, self.stream, 1)
            item['launch']()
        d.call('cuEventRecordWithFlags', [C.c_void_p, C.c_void_p, C.c_uint], self.last, self.stream, 1)
        d.call('cuStreamEndCapture', [C.c_void_p, C.POINTER(C.c_void_p)], self.stream, C.byref(self.graph))
        d.call('cuGraphInstantiateWithFlags', [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64],
               C.byref(self.executable), self.graph, 0)
        count = C.c_size_t()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
               self.graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
               self.graph, nodes, C.byref(count))
        self.node_types = []
        for node in nodes:
            kind = C.c_int()
            d.call('cuGraphNodeGetType', [C.c_void_p, C.POINTER(C.c_int)], node, C.byref(kind))
            self.node_types.append(kind.value)
        expected = item['expected_nodes']
        desired = ([0] * ((primes+1)*expected['kernels']) + [7,7]
                   + [2] * ((primes+1)*expected.get('memsets',0))
                   + [1] * ((primes+1)*expected.get('copies',0)))
        assert sorted(self.node_types) == sorted(desired), (item['tag'], self.node_types, desired)
        for kernel in {id(k):k for k in item['kernels']}.values():
            if hasattr(kernel,'captures'):kernel.captures += 1

    def run(self):
        self.driver.call('cuGraphLaunch', [C.c_void_p,C.c_void_p], self.executable,self.stream)
        self.driver.call('cuStreamSynchronize', [C.c_void_p], self.stream)
        value = self.driver.elapsed_us(self.first,self.last)
        self.replays += 1
        for kernel in self.item['kernels']:
            if hasattr(kernel,'graph_launches'):
                kernel.graph_launches += self.primes+1
        return value

    def close(self):
        self.driver.call('cuGraphExecDestroy',[C.c_void_p],self.executable)
        self.driver.call('cuGraphDestroy',[C.c_void_p],self.graph)
        for event in (self.first,self.last):
            self.driver.call('cuEventDestroy_v2',[C.c_void_p],event)


class Session:
    def __init__(self,device,protocol,root,invocation,suite):
        self.device,self.protocol,self.root,self.invocation,self.suite=device,protocol,Path(root),invocation,suite
        for name,digest in protocol['source_SHA256'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest,name
        self.root.mkdir(parents=True,exist_ok=True)
        self.rng=random.Random(protocol['timing']['seed']+invocation)
        self.last_completion=None
        self.reader=NVML(device.environment['gpu_uuid'])
        self.sampler=Sampler(self.reader,period_s=protocol['timing']['telemetry_period_s'])
        self.sampler.set_context(phase='setup',suite=suite)
        self.sampler.start()
        self.result=dict(status='FAILED',suite=suite,invocation=invocation,
            formal_timing=invocation>0,protocol_SHA256=c.sha(c.encode(protocol)),
            payload_SHA256=protocol['suites'][suite]['payload_SHA256'],
            environment={**device.environment,'readonly':collect_env_readonly(self.reader)},
            clock_control='unlocked',compile_calls=0,native_rebuilds=0,
            profiler_attached=False,shards=[],checks=[],graph_proofs=[],
            partial_errors=[],sampling=dict(primes=protocol['timing']['primes'],
                requested_host_gap_seconds=protocol['timing']['host_gap_seconds'],
                target_semantic_iterations=1,inter_prime_target_gap='GPU graph dependency; no host submission'))
        assert device.environment['cc']==[9,0] and device.environment['SM_count']==132
        assert 'H100' in device.environment['name'] and 'H200' not in device.environment['name']

    def _guard(self,items):
        seen=set()
        for item in items:
            for kernel in item['kernels']:
                if id(kernel) not in seen:
                    kernel.guard();seen.add(id(kernel))

    def _observe(self,graph):
        requested=self.protocol['timing']['host_gap_seconds']
        if self.last_completion is not None:
            deadline=self.last_completion+int(requested*1e9)
            while True:
                remaining=deadline-time.monotonic_ns()
                if remaining<=0:break
                time.sleep(remaining/1e9)
        launch=time.monotonic_ns()
        actual_gap=launch-self.last_completion if self.last_completion else 0
        elapsed=graph.run()
        completed=time.monotonic_ns()
        self.last_completion=completed
        assert elapsed>0
        return elapsed,actual_gap,launch,completed

    def measure(self,context,items):
        assert isinstance(context,dict) and context.get('id')
        assert items and len({i['tag'] for i in items})==len(items)
        originals=list(items)
        graphs={};cells={};orders=[]
        active_observations=None;active_visit=None;active_start=None;active_round_orders=[]
        self.sampler.set_context(phase='correctness_before',suite=self.suite,case=context['id'])
        try:
            for item in originals:
                check=item['check']()
                self.result['checks'].append(dict(context=context,tag=item['tag'],
                    role=item['role'],phase='before',**check))
                assert check['pass'],(context,item['tag'],check)
            self._guard(originals)
            for item in originals:
                graphs[item['tag']]=Graph(self.device,item,self.protocol['timing']['primes'])
            self.sampler.set_context(phase='graph_correctness_before',suite=self.suite,case=context['id'])
            for item in originals:
                self._observe(graphs[item['tag']])
                check=item['check_graph']()
                self.result['checks'].append(dict(context=context,tag=item['tag'],
                    role=item['role'],phase='graph_before',**check))
                assert check['pass'],(context,item['tag'],'captured graph',check)
            bases=[i for i in originals if i['role']=='U']
            if not bases:bases=[i for i in originals if i['role']=='default']
            items=list(originals)
            if bases:
                for base in bases:
                    sham={**base,'tag':base['tag']+':repeat','aliases':[],
                        'role':'U_repeat' if base['role']=='U' else 'default_repeat',
                        'sham_of':base['tag']}
                    items.append(sham)
                    graphs[sham['tag']]=graphs[base['tag']]
            for item in items:
                graph=graphs[item['tag']]
                cells[item['tag']]=dict(context=context,tag=item['tag'],aliases=item.get('aliases',[]),
                    role=item['role'],sham_of=item.get('sham_of'),visits=[])
            # Eligibility input 0 exercises each exact graph once, without
            # producing a formal estimate or reusing these timings later.
            if self.invocation==0:
                pass
            else:
                for visit in range(self.protocol['timing']['visits']):
                    self._guard(originals)
                    self.sampler.set_context(phase='warmup',suite=self.suite,case=context['id'],visit=visit)
                    warm_order=list(originals);self.rng.shuffle(warm_order)
                    for item in warm_order:
                        for _ in range(self.protocol['timing']['warmups']):self._observe(graphs[item['tag']])
                    self.sampler.set_context(phase='timing',suite=self.suite,case=context['id'],visit=visit)
                    observations={i['tag']:[] for i in items};round_orders=[]
                    start=time.monotonic_ns()
                    active_observations=observations;active_visit=visit;active_start=start
                    active_round_orders=round_orders
                    for _ in range(self.protocol['timing']['samples']):
                        order=list(items);self.rng.shuffle(order)
                        round_orders.append([i['tag'] for i in order])
                        for item in order:observations[item['tag']].append(self._observe(graphs[item['tag']]))
                    end=time.monotonic_ns()
                    for tag,rows in observations.items():
                        columns=list(zip(*rows))
                        cells[tag]['visits'].append(dict(visit=visit,samples_us=list(columns[0]),
                            host_gaps_ns=list(columns[1]),launch_ns=list(columns[2]),
                            completion_ns=list(columns[3]),start_ns=start,end_ns=end))
                    orders.append(dict(visit=visit,round_orders=round_orders))
                    active_observations=None
                    self.device.driver.calls.clear()
            self.sampler.set_context(phase='correctness_after',suite=self.suite,case=context['id'])
            for item in originals:
                self._observe(graphs[item['tag']])
                check=item['check_graph']()
                self.result['checks'].append(dict(context=context,tag=item['tag'],
                    role=item['role'],phase='graph_after',**check))
                assert check['pass'],(context,item['tag'],check)
            self._guard(originals)
        except Exception:
            self.result['partial_errors'].append(traceback.format_exc())
            raise
        finally:
            cleanup_failed=False
            for item in originals:
                if item['tag'] in graphs:
                    graph=graphs[item['tag']]
                    self.result['graph_proofs'].append(dict(context=context,tag=item['tag'],
                        aliases=item.get('aliases',[]),role=item['role'],node_types=graph.node_types,
                        primes=graph.primes,replays=graph.replays,
                        expected_nodes_per_semantic_iteration=item['expected_nodes']))
                    try:graph.close()
                    except Exception:
                        cleanup_failed=True
                        self.result['partial_errors'].append(traceback.format_exc())
            shard=dict(context=context,cells=list(cells.values()),orders=orders)
            if active_observations is not None:
                shard['incomplete_visit']=dict(visit=active_visit,start_ns=active_start,
                    end_ns=time.monotonic_ns(),round_orders=active_round_orders,
                    observations={tag:[dict(samples_us=row[0],host_gap_ns=row[1],
                        launch_ns=row[2],completion_ns=row[3]) for row in rows]
                        for tag,rows in active_observations.items()},
                    scope='Completed observations from a technically failed visit; never a complete formal estimate.')
            name='cells/'+c.sha(c.encode(context))[:20]+'.json'
            blob=c.encode(shard);c.write_once(self.root/name,blob)
            self.result['shards'].append(dict(path=name,SHA256=c.sha(blob),context=context,
                physical_cells=len(cells),logical_cells=sum(1+len(x.get('aliases',[])) for x in cells.values()),
                raw_samples=sum(len(v['samples_us']) for cell in cells.values() for v in cell['visits'])))
            self.device.driver.calls.clear()
            if cleanup_failed:raise RuntimeError('Graph cleanup failed; partial shard preserved')
        print('Cadence',self.suite,self.invocation,context['id'],'cells',len(cells),'PASS',flush=True)
        return list(cells.values())

    def finish(self,bindings,error=None):
        if error:self.result['error']=error
        self.result['bindings']=bindings
        cleanup=[]
        try:
            rows=self.sampler.stop()
            data=b''.join((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode() for row in rows)
            blob=gzip.compress(data,mtime=0)
            c.write_once(self.root/'telemetry.jsonl.gz',blob)
            self.result['telemetry']=dict(path='telemetry.jsonl.gz',SHA256=c.sha(blob),rows=len(rows),
                scope='Sampled nominal clocks; averaging windows may span several observations, not per-kernel clock proof.')
        except Exception:cleanup.append(traceback.format_exc())
        for resource in (self.reader,self.device):
            try:resource.close()
            except Exception:cleanup.append(traceback.format_exc())
        self.result['cleanup_errors']=cleanup
        self.result['driver_return_codes']=sorted(getattr(self.device.driver,'codes',set()))
        self.result['driver_call_counts']=dict(getattr(self.device.driver,'counts',{}))
        self.result['Triton_imported']=any(n=='triton' or n.startswith('triton.') for n in sys.modules)
        okay=(not error and not cleanup and not self.result['partial_errors']
              and self.result['driver_return_codes']==[0] and not self.result['Triton_imported'])
        self.result['status']='OK' if okay else 'FAILED'
        c.write_once(self.root/'result.json',self.result)
        return self.result
