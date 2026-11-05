"""CPU preflight and bounded H100 frozen-binary jobs using existing caches."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11.c2d import common as c

SNAPSHOT = Path('/tmp/tma-phase11-c2d-snapshot')
app = modal.App('triton-phase11-c2d-full-calibration')


def image_for(identity):
    environment = {'PYTHONPATH': '/opt/triton-src',
                   'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache'}
    if identity == c.BASE_IMAGE:
        environment['PATH'] = '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
    return modal.Image.from_id(identity).env(environment).add_local_dir(
        SNAPSHOT/'experiments', '/opt/triton-src/experiments', copy=True,
        ignore=['**/__pycache__/**', '**/*.pyc'])


image = image_for(c.BASE_IMAGE)
tools_image = image_for(c.TOOLS_IMAGE)
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={'/cache': cache, '/evidence': evidence}, include_source=False)
def preflight(protocol, reuse_blob):
    import subprocess
    import traceback
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from experiments.tma_reduction_layout.phase11.c2.kernels import micro
    from experiments.tma_reduction_layout.phase11.c2c.kernels import streaming, resident_row, resident_mc, row_scaffold, fixed_local, one_pass
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c2d/cpu_preflight')/modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    c.extract(reuse_blob, root)
    result = {'status': 'FAILED', 'GPU_allocated': False, 'metadata': {},
              'call_id': modal.current_function_call_id(), 'new_compiles': 0, 'reused': 0,
              'protocol_SHA256': c.sha(c.encode(protocol))}

    def command(*argv):
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)

    try:
        result['native_SHA256'] = c.sha(Path(native.__file__).read_bytes())
        assert result['native_SHA256'] == protocol['baseline']['native_SHA256']
        result['ccache_before'] = command('ccache', '--print-stats')
        for case in protocol['cases']:
            folder = root/case['tag'].replace(':', '/')
            if case.get('reuse_source'):
                assert c.sha((folder/'kernel.cubin').read_bytes()) == case['cubin_SHA256']
                meta = case
                result['reused'] += 1
            else:
                w = case['num_warps']
                if case['kind'] == 'local':
                    fn = fixed_local
                    constants = {'R': case['R'], 'P': case['p'], 'C': case['columns'],
                                 'VARIANT': case['variant'], 'L': gl.BlockedLayout(**case['layout'])}
                    signature = {'arg': '*bf16', 'out': '*fp32', 'K': 'i32'}
                else:
                    constants = {'BM': case['M'], 'BN': case['N'], 'W': w}
                    if case['family'] == 'SCAFFOLD':
                        fn = row_scaffold
                        signature = {'arg': '*bf16', 'out': '*fp32', 'K': 'i32'}
                    else:
                        shared = gl.NVMMASharedLayout(**case['shared_layout'])
                        constants['S'] = shared
                        signature = {'desc': f'tensordesc<bf16[1,{case["M"]},{case["N"]}],{shared}>',
                                     'out': '*fp32', 'K': 'i32'}
                        if case['family'] == 'ROW':
                            fn = resident_row
                        else:
                            constants['L'] = gl.BlockedLayout(**case['layout']) if 'layout' in case else gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, w], [2, 1, 0])
                            if case['family'] == 'MC':
                                fn = resident_mc
                            else:
                                fn = streaming
                                constants['REDUCE'] = case['family'] == 'FULL'
                        if case['family'] == 'TRANSFER':
                            # The transfer control does not use a reduction
                            # layout; retain a valid uniform row layout constant.
                            constants['L'] = gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, w], [2, 1, 0])
                        if case.get('one_pass'):
                            fn = one_pass
                            constants['L'] = gl.BlockedLayout(**case['layout']) if 'layout' in case else gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, w], [2, 1, 0])
                            constants['REDUCE'] = case['family'] in ('FULL', 'ONE_FULL')
                            signature.pop('K')
                signature.update({key: 'constexpr' for key in constants})
                source = gluon.GluonASTSource(fn, signature, constants, {(1,): [['tt.divisibility', 16]]})
                compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32), options={'num_warps': w})
                for ext in ('glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                    value = compiled.asm[ext]
                    c.write_once(folder/('kernel.'+ext), value if isinstance(value, bytes) else value.encode())
                for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
                    c.write_once(folder/('kernel.'+ext), command('cuobjdump', flag, str(folder/'kernel.cubin')).encode())
                meta = {**case, 'cubin_SHA256': c.sha(compiled.asm['cubin']),
                        'function_name': compiled.name, 'dynamic_smem_bytes': compiled.metadata.shared,
                        'global_scratch_size': compiled.metadata.global_scratch_size,
                        'tensordesc_meta': compiled.metadata.tensordesc_meta,
                        'compile_cache_key': compiled.hash}
                result['new_compiles'] += 1
            c.write_once(folder/'c2d_metadata.json', meta)
            result['metadata'][case['tag']] = meta
            print('C2d CPU', case['tag'], 'REUSED' if case.get('reuse_source') else 'COMPILED', flush=True)
        result['ccache_after'] = command('ccache', '--print-stats')
        assert result['ccache_before'] == result['ccache_after']
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root/'result.json', result)
    cache.commit()
    evidence.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', timeout=2400,
              volumes={'/evidence': evidence}, single_use_containers=True, include_source=False)
def collect(protocol, metadata, payload):
    from experiments.tma_reduction_layout.phase11.c2d.runtime import eligibility
    c.verify_sources(protocol)
    root = Path('/tmp')/modal.current_function_call_id()
    c.extract(payload, root)
    result = eligibility(protocol, metadata, root)
    result.update(call_id=modal.current_function_call_id(), protocol_SHA256=c.sha(c.encode(protocol)),
                  payload_SHA256=c.sha(payload))
    c.write_once(Path('/evidence/stage_c2d/eligibility')/modal.current_function_call_id()/'raw.json', result)
    evidence.commit()
    return result


@app.function(image=image, gpu='H100!:1', timeout=5400,
              volumes={'/evidence': evidence}, single_use_containers=True, include_source=False)
def timing(invocation, protocol, gate, payload):
    from experiments.tma_reduction_layout.phase11.c2d.runtime import execute
    c.verify_sources(protocol)
    result = execute(invocation, protocol, gate, payload, modal.current_function_call_id())
    c.write_once(Path('/evidence/stage_c2d/timing')/modal.current_function_call_id()/'raw.json', result)
    evidence.commit()
    return result


@app.function(image=tools_image, gpu='H100!:1', timeout=3000,
              volumes={'/evidence': evidence}, single_use_containers=True, include_source=False)
def profiles(protocol, gate, payload, group_index):
    import subprocess
    import sys
    import traceback
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c2d/profiles')/modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = root/'artifacts'
    c.extract(payload, artifacts)
    group = gate['profile_groups'][group_index]
    result = {'status': 'FAILED', 'group': group_index, 'records': [],
              'call_id': modal.current_function_call_id(), 'protocol_SHA256': c.sha(c.encode(protocol)),
              'formal_timing': False}

    def command(path, argv):
        try:
            process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
            value = {'argv': argv, 'returncode': process.returncode, 'output': process.stdout}
        except Exception as error:
            value = {'argv': argv, 'returncode': 'EXECUTION_FAILURE', 'output': repr(error)}
        c.write_once(path, value)
        return value

    try:
        command(root/'ncu_version.json', ['ncu', '--version'])
        for mode in ('software', 'hardware'):
            cells = []
            for tag in group:
                meta = gate['metadata'][tag]
                if mode == 'software':
                    ks = [1] if meta.get('original_once') else protocol['profiles']['MC_software_K']
                    cells += [{'tag': tag, 'K': k, 'grid': 1, 'q': 'natural'} for k in ks]
                else:
                    qs = protocol['profiles']['q_conditions']
                    for q in qs:
                        if q != 'natural' and not gate['resources'][tag]['reservations'][q].get('feasible'):
                            continue
                        single = meta.get('original_once')
                        grid = (8192//meta['M'])*(8192//meta['N']) if single else 64*132
                        cells.append({'tag': tag, 'K': 1 if single else 64, 'grid': grid, 'q': q})
            folder = root/mode
            folder.mkdir()
            job = {'cells': cells, 'metadata': {tag: gate['metadata'][tag] for tag in group}}
            c.write_once(folder/'job.json', job)
            flags = ['--section', 'InstructionStats'] if mode == 'software' else ['--metrics', ','.join(protocol['profiles']['metrics'])]
            value = command(folder/'collect.json', ['ncu', '--replay-mode', 'kernel', '--cache-control', 'none',
                '--clock-control', 'none', '--profile-from-start', 'off', *flags,
                '--export', str(folder/'report'), sys.executable, '-m',
                'experiments.tma_reduction_layout.phase11.c2d.profile_one', '--root', str(artifacts), '--job', str(folder/'job.json')])
            assert value['returncode'] == 0, value['output'][-6000:]
            assert (folder/'report.ncu-rep').exists()
            value = command(folder/'import.json', ['ncu', '--import', str(folder/'report.ncu-rep'), '--page', 'raw',
                             '--csv', '--print-units', 'base', '--print-fp'])
            assert value['returncode'] == 0, value['output'][-3000:]
            if mode == 'software':
                value = command(folder/'source.json', ['ncu', '--import', str(folder/'report.ncu-rep'),
                                 '--page', 'source', '--print-source', 'sass', '--csv'])
                assert value['returncode'] == 0, value['output'][-3000:]
            result['records'].append({'mode': mode, 'launches': len(cells),
                                      'report_SHA256': c.sha((folder/'report.ncu-rep').read_bytes())})
            print('C2d NCU group', group_index, mode, 'PASS', len(cells), flush=True)
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root/'result.json', result)
    # CUBIN payload is already preserved in CPU preflight and its archive.
    import shutil
    shutil.rmtree(artifacts)
    evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(action: str = 'preflight', attempt: int = 1, invocation: int = 1, group: int = 0):
    protocol = c.protocol()
    c.protect_prior()
    if action == 'preflight':
        folder = c.OUT/'cpu_preflight'/f'attempt_{attempt}'
        function, args = preflight, (protocol, (c.OUT/'preregistration/reuse.zip').read_bytes())
    else:
        compiled = c.read(c.OUT/'compile_gate.json')
        payload = (c.OUT/'compiled_payload.zip').read_bytes()
        if action == 'collect':
            folder = c.OUT/'eligibility'/f'attempt_{attempt}'
            function, args = collect, (protocol, compiled['metadata'], payload)
        else:
            gate = c.read(c.OUT/'artifact_gate.json')
            assert gate['status'] == 'OK'
            if action == 'profiles':
                folder = c.OUT/'profiles'/f'group_{group}'/f'attempt_{attempt}'
                function, args = profiles, (protocol, gate, payload, group)
            else:
                assert action == 'timing'
                folder = c.OUT/'timing'/f'invocation_{invocation}'
                function, args = timing, (invocation, protocol, gate, payload)
    assert not folder.exists(), folder
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    call = function.spawn(*args)
    c.write_once(folder/'dispatch.json', {'call_id': call.object_id, 'profile': os.environ['MODAL_PROFILE'],
                 'action': action, 'protocol_SHA256': c.sha(c.encode(protocol))})
    try:
        result = call.get()
    except Exception as error:
        c.write_once(folder/'retrieval_failure.json', {'call_id': call.object_id, 'error': repr(error)})
        raise
    if isinstance(result, bytes):
        c.write_once(folder/'raw_export.zip', result)
        c.extract(result, folder/'export')
        status = c.read(folder/'export/result.json')['status']
    else:
        c.write_once(folder/'raw.json', result)
        status = result['status']
    print('C2d', action, status, flush=True)
