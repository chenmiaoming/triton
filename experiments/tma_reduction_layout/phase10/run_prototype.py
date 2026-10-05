"""CPU-only prototype checks over the exact Stage A current-main image."""
from pathlib import Path
import io
import os
import zipfile
import modal
from experiments.tma_reduction_layout.phase10 import common as c

app=modal.App('triton-phase10-compiler-prototype')
cache=modal.Volume.from_name('triton-build-cache')
volume=modal.Volume.from_name('triton-phase10-evidence')
checkout=Path(os.environ.get('TMA_PHASE10_CHECKOUT','/tmp/triton-tma-current-main'))
native_file=Path('lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp')
test_file=Path('test/TritonGPU/optimize-locality.mlir')
gpu_test_file=Path('python/test/unit/cuda/test_tensor_descriptor_cuda.py')
image=(modal.Image.from_id('im-qzTUZumWu2JXjWic8K82cZ')
    .add_local_file(checkout/native_file,'/opt/triton-src/'+str(native_file),copy=True)
    .add_local_file(checkout/test_file,'/opt/triton-src/'+str(test_file),copy=True)
    .add_local_file(checkout/gpu_test_file,'/opt/triton-src/'+str(gpu_test_file),copy=True)
    .add_local_file(Path(__file__).with_name('build_prototype.py'),'/opt/phase10-prototype-build.py',copy=True)
    .run_commands('python3 /opt/phase10-prototype-build.py',volumes={'/cache':cache})
    .uv_pip_install('pytest','pytest-instafail','pytest-xdist')
    .env({'PYTHONPATH':'/opt/triton-src','PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(Path(os.environ.get('TMA_PHASE10_SCRIPTS','/tmp/tma-phase10-prototype-scripts'))/'experiments',
        '/opt/triton-src/experiments',copy=True,ignore=['**/results/**','**/__pycache__/**','**/*.pyc']))


@app.function(image=image,timeout=1800,volumes={'/evidence':volume,'/cache':cache},single_use_containers=True,include_source=False)
def check(expected):
    import subprocess
    import sys
    import json
    import hashlib
    import triton._C.libtriton as native
    sys.path.insert(0,'/opt/triton-src/python')
    from build_helpers import get_cmake_dir
    root=Path('/opt/triton-src')
    build=Path(get_cmake_dir())
    out=Path('/evidence/stage_b')/modal.current_function_call_id()
    out.mkdir(parents=True)
    result={'call_id':modal.current_function_call_id(),'status':'OK','checks':[],
        'native_SHA256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        'image_id':os.environ.get('MODAL_IMAGE_ID'),'GPU_timing':False}
    for name,digest in expected.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert actual==digest,(name,actual,digest)
        path=out/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/name).read_bytes())
    for p in Path('/opt/phase10-prototype-build').iterdir():
        if p.is_file():(out/p.name).write_bytes(p.read_bytes())
    output=subprocess.run([str(build/'bin/triton-opt'),str(root/'test/TritonGPU/optimize-locality.mlir'),
        '-split-input-file','-tritongpu-optimize-thread-locality','-canonicalize'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/'optimized_locality.mlir').write_text(output.stdout)
    # The generated build-tree tests resolve current source and LLVM lit config.
    for tag,command in [('targeted',['lit','-v','test/TritonGPU/optimize-locality.mlir']),
                        ('coalesce',['lit','-v','test/TritonGPU/coalesce.mlir']),
                        ('full_lit',['ninja','check-triton-lit-tests'])]:
        completed=subprocess.run(command,cwd=build,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/(tag+'.log')).write_text(completed.stdout)
        result['checks'].append({'tag':tag,'command':command,'cwd':str(build),'return_code':completed.returncode})
        print(completed.stdout[-12000:],flush=True)
        if completed.returncode:result['status']='FAILED';break
    (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    volume.commit()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    return {'result':result,'zip':stream.getvalue()}


@app.function(image=image,gpu='H100!:1',timeout=1800,volumes={'/evidence':volume,'/cache':cache},single_use_containers=True,include_source=False)
def check_gpu(expected):
    import subprocess
    import json
    import hashlib
    import torch
    import triton._C.libtriton as native
    root=Path('/opt/triton-src')
    for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    assert 'H100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0)==(9,0)
    command=['python3','-m','pytest','-s','--tb=short','python/test/unit/cuda/test_tensor_descriptor_cuda.py::test_tensor_descriptor_partial_reduction_layout']
    completed=subprocess.run(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    out=Path('/evidence/stage_b')/modal.current_function_call_id();out.mkdir(parents=True)
    (out/'pytest.log').write_text(completed.stdout)
    print(completed.stdout,flush=True)
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    result={'call_id':modal.current_function_call_id(),'status':'OK' if completed.returncode==0 else 'FAILED',
        'command':command,'return_code':completed.returncode,'GPU_timing':False,'gpu_name':torch.cuda.get_device_name(0),
        'gpu_environment':ExactCUDA().environment(),'native_SHA256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');volume.commit()
    return {'result':result,'log':completed.stdout}


@app.local_entrypoint()
def main():
    import uuid
    expected={str(p):c.sha((checkout/p).read_bytes()) for p in [native_file,test_file,gpu_test_file]}
    root=c.OUT/'stage_b/attempts'/uuid.uuid4().hex
    call=check.spawn(expected)
    c.write_once(root/'dispatch.json',{'call_id':call.object_id,'profile':os.environ.get('MODAL_PROFILE'),'expected_source':expected})
    result=call.get()
    c.write_once(root/'raw_export.zip',result['zip']);c.write_once(root/'return.json',result['result'])
    with zipfile.ZipFile(io.BytesIO(result['zip'])) as z:z.extractall(root/'export')
    assert result['result']['status']=='OK','Native/lit failure archived'
    gpu=check_gpu.remote(expected)
    c.write_once(root/'gpu_result.json',gpu['result']);c.write_once(root/'pytest.log',gpu['log'].encode())
    assert gpu['result']['status']=='OK','GPU correctness failure archived'
    c.write_once(c.OUT/'stage_b/accepted.json',{'attempt':str(root.relative_to(c.ROOT)),'source':expected,'result':result['result'],'GPU_correctness':gpu['result']})
