"""NCU coverage/calibration on three architectures; no performance samples."""
import argparse
import os
import subprocess
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11 import common as c

app = modal.App('triton-phase11-instrument-coverage')
SNAPSHOT = Path(os.environ.get('TMA_PHASE11_SNAPSHOT', '/tmp/tma-phase11-stage-a-snapshot'))
image = (modal.Image.from_id('im-gCqknrtnwsoW0fVOiG5XF9')
         .env({'PYTHONPATH':'/opt/triton-src', 'PATH':'/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
         .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                        ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase11-evidence', create_if_missing=True)
METRICS = [
    'l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'smsp__inst_executed.sum',
]


def probe(target, protocol):
    import sys
    import traceback
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    from experiments.tma_reduction_layout.phase11.calibration import KINDS, ptx
    c.verify_sources(protocol)
    root = Path('/evidence/stage_a') / target / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    result = {'target':target, 'call_id':modal.current_function_call_id(), 'commands':{},
              'protocol_SHA256':c.sha(c.encode(protocol)), 'formal_timing_samples':False}
    def run(label, argv, timeout=120):
        try:
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            record = {'argv':argv, 'returncode':p.returncode, 'output':p.stdout}
        except (subprocess.TimeoutExpired, OSError) as e:
            record = {'argv':argv, 'returncode':'EXECUTION_FAILURE', 'output':str(e)}
        c.write_once(root / (label + '.json'), record)
        result['commands'][label] = record
        return record
    try:
        cuda = ExactCUDA()
        import ctypes as C
        name_buffer = C.create_string_buffer(256)
        cuda.call('cuDeviceGetName',[C.c_void_p,C.c_int,C.c_int],name_buffer,256,cuda.device)
        name, cc = name_buffer.value.decode(), [cuda.attr(75),cuda.attr(76)]
        spec = protocol['targets'][target]
        assert spec['name'] in name and cc == spec['cc'], (name, cc)
        result['environment'] = {**cuda.environment(), 'name':name, 'cc':cc,
            'native_compiler_loaded':False, 'base_image':protocol['tool_image'],
            'image_id':os.environ.get('MODAL_IMAGE_ID')}
        result['ccache_before'] = run('ccache_before', ['ccache','--print-stats'])['output']
        run('nvidia_smi', ['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,driver_version','--format=csv,noheader'])
        run('ncu_version', ['ncu','--version'])
        run('ncu_sections', ['ncu','--list-sections'])
        queried = run('ncu_metrics', ['ncu','--query-metrics','--query-metrics-mode','all','--csv'])
        metrics = [m for m in METRICS if m in queried['output']]
        result['metric_mapping'] = {'available':metrics, 'unavailable':[m for m in METRICS if m not in metrics]}
        assert len(metrics) >= 3, result['metric_mapping']
        ptxas = Path('/usr/local/cuda/bin/ptxas')
        result['ptxas'] = {'path':str(ptxas),'SHA256':c.sha(ptxas.read_bytes()),
                          'version':run('ptxas_version',[str(ptxas),'--version'])['output']}
        for kind in KINDS:
            c.write_once(root / (kind + '.ptx'), ptx(kind, cc[0]*10+cc[1]).encode())
            compiled = run(kind+'_compile', [str(ptxas), '--gpu-name',f'sm_{cc[0]*10+cc[1]}',
                                            str(root/(kind+'.ptx')), '-o', str(root/(kind+'.cubin'))])
            assert compiled['returncode'] == 0, compiled
            run(kind+'_sass', ['cuobjdump','-sass',str(root/(kind+'.cubin'))])
            run(kind+'_resources', ['cuobjdump','-res-usage',str(root/(kind+'.cubin'))])
            exe = [sys.executable,'-m','experiments.tma_reduction_layout.phase11.calibration',
                   '--root',str(root),'--kind',kind]
            run(kind+'_correctness', exe)
            for group, flags in [('hardware',['--metrics',','.join(metrics)]),
                                 ('software',['--section','InstructionStats'])]:
                report = root / (kind+'_'+group)
                argv = ['ncu','--replay-mode','kernel','--cache-control','none','--clock-control','none',
                        '--profile-from-start','off','--launch-count','1',*flags,'--export',str(report),*exe]
                run(kind+'_'+group+'_collect', argv, 180)
                if report.with_suffix('.ncu-rep').exists():
                    run(kind+'_'+group+'_import', ['ncu','--import',str(report.with_suffix('.ncu-rep')),
                        '--page','raw','--csv','--print-units','base','--print-fp'])
                    run(kind+'_'+group+'_source', ['ncu','--import',str(report.with_suffix('.ncu-rep')),
                        '--page','source','--print-source','sass','--csv'])
        result['status'] = 'PROBE_RETURNED'
    except Exception:
        result['status'] = 'FAILED'
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    result['ccache_after'] = run('ccache_after', ['ccache','--print-stats'])['output']
    result['native_rebuild'] = False
    result['ccache_available'] = result['commands']['ccache_before']['returncode'] == 0
    c.write_once(root / 'result.json', result)
    volume.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', timeout=1800, volumes={'/cache':cache,'/evidence':volume},
              single_use_containers=True, include_source=False)
def sm90(protocol):
    return probe('sm90', protocol)


@app.function(image=image, gpu='B200:1', timeout=1800, volumes={'/cache':cache,'/evidence':volume},
              single_use_containers=True, include_source=False)
def sm100(protocol):
    return probe('sm100', protocol)


@app.function(image=image, gpu='RTX-PRO-6000:1', timeout=1800, volumes={'/cache':cache,'/evidence':volume},
              single_use_containers=True, include_source=False)
def sm120(protocol):
    return probe('sm120', protocol)


@app.local_entrypoint()
def main(targets: str = 'sm90,sm100,sm120', attempt: int = 1, revision: int = 1):
    prereg = c.OUT/'stage_a/preregistration'
    protocol = c.read((prereg if revision == 1 else prereg/'revisions'/str(revision))/'protocol.json')
    c.protect_prior()
    for target in targets.split(','):
        root = c.OUT/'stage_a/probes'/target/f'attempt_{attempt}'
        assert not root.exists(), root
        call = globals()[target].spawn(protocol)
        c.write_once(root/'dispatch.json', {'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),
                                          'image_id':image.object_id})
        try:
            data = call.get()
        except Exception as e:
            c.write_once(root/'retrieval_failure.json', {'error':repr(e),'call_id':call.object_id})
            raise
        c.write_once(root/'raw_export.zip', data)
        c.extract(data, root/'export')
        print(target, c.read(root/'export/result.json')['status'], flush=True)


if __name__ == '__main__':
    raise SystemExit('Use modal run -m experiments.tma_reduction_layout.phase11.run_stage_a')
