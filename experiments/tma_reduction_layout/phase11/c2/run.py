"""CPU compile then bounded frozen-binary Modal jobs, no native rebuild."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11.c2 import common as c

BATCH=os.environ.get('TMA_C2_BATCH','a')
SNAPSHOT=Path('/tmp/tma-phase11-c2'+BATCH+'-snapshot')
app=modal.App('triton-phase11-c2'+BATCH+'-calibration')

def image_for(identity):
    env={'PYTHONPATH':'/opt/triton-src','TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache','TMA_C2_BATCH':BATCH}
    if identity==c.BASE_IMAGE:env['PATH']='/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
    return modal.Image.from_id(identity).env(env).add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,ignore=['**/__pycache__/**','**/*.pyc'])

image=image_for(c.BASE_IMAGE)
tools_image=image_for(c.TOOLS_IMAGE).pip_install('numpy==2.2.6')
cache=modal.Volume.from_name('triton-build-cache');volume=modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image,timeout=1800,volumes={'/cache':cache,'/evidence':volume},include_source=False)
def preflight(protocol,reused):
    import subprocess,traceback
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from experiments.tma_reduction_layout.phase11.c2.kernels import micro,output_path,parameter
    c.verify_sources(protocol)
    root=Path('/evidence')/('stage_'+protocol['stage'].lower())/'cpu_preflight'/modal.current_function_call_id()
    root.mkdir(parents=True,exist_ok=False);c.extract(reused,root)
    result={'status':'FAILED','GPU_allocated':False,'call_id':modal.current_function_call_id(),
            'protocol_SHA256':c.sha(c.encode(protocol)),'metadata':{},'new_compiles':0}
    def command(*args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
    try:
        assert c.sha(Path(native.__file__).read_bytes())==protocol['baseline']['native_SHA256']
        result['native_SHA256']=c.sha(Path(native.__file__).read_bytes())
        result['ccache_before']=command('ccache','--print-stats')
        for case in protocol['cases']:
            tag=case['tag'];folder=root/tag.replace(':','/')
            if tag in protocol.get('previous_metadata',{}) or 'reused_tag' in case:
                meta=protocol.get('previous_metadata',{}).get(tag,case)
                assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
            else:
                layout=gl.BlockedLayout(**case['layout'])
                constants={'BM':case['M'],'BN':case['N'],'W':case['num_warps'],'FAMILY':case['family'],'D':case['D'],'L':layout}
                if case['family']=='MO':
                    fn=output_path;signature={'arg':'*fp32','out':'*fp32','BN':'constexpr','W':'constexpr','L':'constexpr'}
                    constants={key:constants[key] for key in ('BN','W','L')}
                elif case['ABI_kind']=='pointer_K':
                    fn=parameter;signature={'arg':'*bf16','out':'*fp32','K':'i32',**{key:'constexpr' for key in constants}}
                else:
                    fn=micro;shared=gl.NVMMASharedLayout(**case['shared_layout']);constants['S']=shared
                    signature={'desc':f'tensordesc<bf16[1,{case["M"]},{case["N"]}],{shared}>','out':'*fp32','K':'i32',**{key:'constexpr' for key in constants}}
                src=gluon.GluonASTSource(fn,signature,constants,{(1,):[['tt.divisibility',16]]})
                compiled=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':case['num_warps']})
                for ext in ('glir','ttgir','llir','ptx','cubin'):
                    value=compiled.asm[ext];c.write_once(folder/('kernel.'+ext),value if isinstance(value,bytes) else value.encode())
                for ext,flag in (('sass','-sass'),('resource.txt','-res-usage')):
                    c.write_once(folder/('kernel.'+ext),command('cuobjdump',flag,str(folder/'kernel.cubin')).encode())
                meta={**case,'cubin_SHA256':c.sha(compiled.asm['cubin']),'function_name':compiled.name,
                      'dynamic_smem_bytes':compiled.metadata.shared,'global_scratch_size':compiled.metadata.global_scratch_size,
                      'tensordesc_meta':compiled.metadata.tensordesc_meta}
                result['new_compiles']+=1
            result['metadata'][tag]=meta;c.write_once(folder/'c2_metadata.json',meta)
            print('CPU-only',tag,'REUSED' if 'reused_tag' in case or tag in protocol.get('previous_metadata',{}) else 'COMPILED',flush=True)
        result['metadata'].update(protocol['anchors'])
        result['ccache_after']=command('ccache','--print-stats')
        assert result['ccache_before']==result['ccache_after']
        assert result['new_compiles']==protocol.get('attempt_compile_count',protocol['new_programs'])
        result['status']='OK'
    except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
    c.write_once(root/'result.json',result);cache.commit();volume.commit()
    return c.archive(root)


@app.function(image=image,gpu='H100!:1',timeout=1800,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def collect(protocol,metadata,payload,previous_checks):
    from experiments.tma_reduction_layout.phase11.c2.timing import eligibility
    c.verify_sources(protocol)
    root=Path('/tmp')/modal.current_function_call_id();c.extract(payload,root)
    result=eligibility(protocol,metadata,root,previous_checks)
    result.update({'call_id':modal.current_function_call_id(),'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload)})
    c.write_once(Path('/evidence')/('stage_'+protocol['stage'].lower())/'eligibility'/modal.current_function_call_id()/'raw.json',result)
    volume.commit();return result


@app.function(image=tools_image,gpu='H100!:1',timeout=3000,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def profiles(protocol,gate,payload,group_index):
    import subprocess,sys,traceback
    c.verify_sources(protocol)
    root=Path('/evidence')/('stage_'+protocol['stage'].lower())/'profiles'/modal.current_function_call_id()
    root.mkdir(parents=True,exist_ok=False);archives=root/'archives';c.extract(payload,archives)
    groups=gate['profile_groups'];group=groups[group_index]
    result={'status':'FAILED','call_id':modal.current_function_call_id(),'group':group_index,'records':[],
            'protocol_SHA256':c.sha(c.encode(protocol)),'payload_SHA256':c.sha(payload),'formal_timing_sample':False}
    def run(path,argv):
        try:
            p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=1500)
            value={'argv':argv,'returncode':p.returncode,'output':p.stdout}
        except Exception as e:value={'argv':argv,'returncode':'EXECUTION_FAILURE','output':repr(e)}
        c.write_once(path,value);return value
    try:
        run(root/'ncu_version.json',['ncu','--version'])
        for mode in ('software','hardware'):
            cells=[]
            for tag in group:
                meta=gate['metadata'][tag]
                ks=[1] if meta['family']=='MO' else protocol['profile_K']
                reps=1 if mode=='software' else protocol['profiles']['hardware_repeats']
                if not meta.get('previous_batch'):
                    for k in ks:
                        for repeat in range(reps):cells.append({'tag':tag,'K':k,'grid':1,'q':'natural','repeat':repeat})
                if protocol['stage']=='C2b':
                    if mode=='hardware':
                        cells.append({'tag':tag,'K':256 if meta['family']!='MO' else 1,
                                      'grid':64*protocol['target']['SM_count'],'q':'natural','repeat':0})
                    for q in ('1','2','4'):
                        if gate['resources'][tag]['reservations'][q].get('feasible'):
                            cells.append({'tag':tag,'K':256 if meta['family']!='MO' else 1,
                                          'grid':64*protocol['target']['SM_count'] if mode=='hardware' else 1,
                                          'q':q,'repeat':0})
            folder=root/mode;folder.mkdir()
            job={'cells':cells,'metadata':{t:gate['metadata'][t] for t in group}}
            c.write_once(folder/'job.json',job)
            flags=['--section','InstructionStats'] if mode=='software' else ['--metrics',','.join(protocol['profiles']['metrics'])]
            value=run(folder/'collect.json',['ncu','--replay-mode','kernel','--cache-control','none','--clock-control','none',
                '--profile-from-start','off',*flags,'--export',str(folder/'report'),sys.executable,'-m',
                'experiments.tma_reduction_layout.phase11.c2.profile_one','--root',str(archives),'--job',str(folder/'job.json')])
            assert value['returncode']==0,value['output'][-6000:]
            assert (folder/'report.ncu-rep').exists()
            imported=run(folder/'import.json',['ncu','--import',str(folder/'report.ncu-rep'),'--page','raw','--csv','--print-units','base','--print-fp'])
            assert imported['returncode']==0,imported['output'][-6000:]
            if mode=='software':
                source=run(folder/'source.json',['ncu','--import',str(folder/'report.ncu-rep'),'--page','source','--print-source','sass','--csv'])
                assert source['returncode']==0,source['output'][-6000:]
            result['records'].append({'mode':mode,'launches':len(cells),'report_SHA256':c.sha((folder/'report.ncu-rep').read_bytes())})
            print('C2'+BATCH,'profile group',group_index,mode,'PASS',len(cells),'launches',flush=True)
        result['status']='OK'
    except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
    c.write_once(root/'result.json',result);volume.commit()
    # Frozen payload already archived once; preserve only new profiling evidence.
    import shutil;shutil.rmtree(archives)
    return c.archive(root)


@app.function(image=image,gpu='H100!:1',timeout=3600,volumes={'/evidence':volume},single_use_containers=True,include_source=False)
def timing(invocation,protocol,gate,payload):
    from experiments.tma_reduction_layout.phase11.c2.timing import execute
    result=execute(invocation,protocol,gate,payload,modal.current_function_call_id())
    c.write_once(Path('/evidence')/('stage_'+protocol['stage'].lower())/'timing'/modal.current_function_call_id()/'raw.json',result)
    volume.commit();return result


@app.local_entrypoint()
def main(action:str='preflight',attempt:int=1,invocation:int=1,group:int=0):
    protocol=c.load_protocol(BATCH);c.protect_prior(BATCH)
    if action=='preflight':
        folder=c.out(BATCH)/'cpu_preflight'/f'attempt_{attempt}';function=preflight
        reuse=c.out(BATCH)/protocol.get('attempt_reuse_path','preregistration/reuse.zip')
        args=(protocol,reuse.read_bytes())
    else:
        compile_gate=c.read(c.out(BATCH)/'compile_gate.json')
        if action=='collect':
            folder=c.out(BATCH)/'eligibility'/f'attempt_{attempt}';function=collect
            previous_checks=[]
            if BATCH=='b':
                prior_gate=c.read(c.out('a')/'artifact_gate.json')
                previous_checks=[r for r in c.read(c.out('a')/'eligibility'/prior_gate['eligibility_attempt']/'raw.json')['checks'] if r['pass']]
            for old in sorted((c.out(BATCH)/'eligibility').glob('attempt_*/raw.json')):
                for check in c.read(old)['checks']:
                    if check['pass'] and check['tag'] not in protocol.get('technical_recompile_tags',[]) and check not in previous_checks:previous_checks.append(check)
            args=(protocol,compile_gate['metadata'],(c.out(BATCH)/'compiled_payload.zip').read_bytes(),previous_checks)
        else:
            gate=c.read(c.out(BATCH)/'artifact_gate.json');assert gate['status']=='OK'
            payload=(c.out(BATCH)/'frozen_payload.zip').read_bytes()
            if action=='profiles':
                folder=c.out(BATCH)/'profiles'/f'group_{group}'/f'attempt_{attempt}';function=profiles;args=(protocol,gate,payload,group)
            else:
                assert action=='timing' and c.read(c.out(BATCH)/'counter_gate.json')['status']=='OK'
                folder=c.out(BATCH)/'timing'/f'invocation_{invocation}';function=timing;args=(invocation,protocol,gate,payload)
    assert not folder.exists(),folder
    call=function.spawn(*args)
    c.write_once(folder/'dispatch.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),
                 'action':action,'protocol_SHA256':c.sha(c.encode(protocol))})
    try:value=call.get()
    except Exception as e:c.write_once(folder/'retrieval_failure.json',{'call_id':call.object_id,'error':repr(e)});raise
    if isinstance(value,bytes):
        c.write_once(folder/'raw_export.zip',value);c.extract(value,folder/'export');status=c.read(folder/'export/result.json')['status']
    else:c.write_once(folder/'raw.json',value);status=value['status']
    print('C2'+BATCH,action,status,flush=True)
