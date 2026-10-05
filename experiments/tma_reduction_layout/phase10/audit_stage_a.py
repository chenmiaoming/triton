"""Recompute Stage A protocol, archive, ABI, launch, and timing checks."""
import math
from pathlib import Path
import statistics
import zipfile
from experiments.tma_reduction_layout.phase10 import common as c
from experiments.tma_reduction_layout.phase4.artifact_gate import ttgir_contract


def run():
    stage=c.OUT/'stage_a'
    protocol=c.read(stage/'protocol.json')
    assert c.sha((stage/'upstream_source.tar').read_bytes())==protocol['source_tar_SHA256']
    for name,digest in c.read(stage/'prior_manifest.json').items():
        assert c.sha((c.ROOT/name).read_bytes())==digest,f'Prior evidence changed: {name}'
    results=[];sample_total=0;manifest={}; compiler_identities={}
    kernel_sha=c.sha((c.ROOT/"experiments/tma_reduction_layout/phase10/kernels.py").read_bytes())
    for target in protocol['targets']:
        metadata={}
        native={};assemblers={}
        expected_research=c.read(c.ROOT/f"experiments/tma_reduction_layout/results/phase9/stage_b/{target}/export/environment.json")["native_extension"]["SHA256"]
        for compiler in ['main','research']:
            root=stage/'artifacts'/target/compiler
            archive=root/'raw_export.zip'
            manifest[str(archive.relative_to(c.ROOT))]=c.sha(archive.read_bytes())
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    data=z.read(name)
                    assert data==(root/'export'/name).read_bytes(),name
                    manifest[str((root/'export'/name).relative_to(c.ROOT))]=c.sha(data)
            env=c.read(root/'export/environment.json')
            assert env['ccache_start']==env['ccache_end']
            if compiler=='main': assert env['source_archive_SHA256']==protocol['source_tar_SHA256']
            assert env['kernel_source_SHA256']==kernel_sha
            assert env['compute_capability']==protocol['targets'][target]['cc']
            assert protocol['targets'][target]['name'] in env['gpu_name']
            native[compiler]=env['native_extension_SHA256']
            assemblers[compiler]=env['ptxas']
            if compiler=='research':assert native[compiler]==expected_research
            attempts=c.read(root/'export/attempts.json')
            assert len(attempts)==(6 if compiler=='main' else 12)
            for row in attempts:
                assert row['status']=='OK' and row['correctness_max_abs_diff']==0 and row['compile_calls']==1
                assert row['actual_jit_compile_events']==1
                expected_vector=4 if row['variant']=='research_vector4' else 8
                directory=root/'export'/row['tag'].replace(':','/')
                observed=ttgir_contract((directory/'kernel.ttgir').read_text(),[1,row['M'],row['N']],row['num_warps'])
                assert row['observed_layout']==observed
                assert observed['blocked']['sizePerThread']==[1,1,expected_vector]
                for name,digest in row['files'].items():assert c.sha((directory/name).read_bytes())==digest
                meta=c.read(directory/'metadata.json')
                assert meta['cubin_SHA256']==row['cubin_SHA256']==c.sha((directory/'kernel.cubin').read_bytes())
                occupancy=c.read(directory/'occupancy.json')
                assert occupancy['queried_cubin_sha256']==meta['cubin_SHA256']
                assert occupancy['blocks_per_sm_actual_dynamic_smem']>0
                metadata[row['tag']]=meta
        compiler_identities[target]={'native':native,'assemblers':assemblers,'same_assembler':assemblers['main']['sha256']==assemblers['research']['sha256']}
        medians={};uuids=[];calls=[];processes=[]
        payload=stage/'timing'/target/'payload.zip'
        manifest[str(payload.relative_to(c.ROOT))]=c.sha(payload.read_bytes())
        assert c.read(payload.with_name('metadata.json'))==metadata
        with zipfile.ZipFile(payload) as z:
            assert len(z.namelist())==36
            for tag,meta in metadata.items():
                assert c.sha(z.read(tag.replace(':','/')+'/kernel.cubin'))==meta['cubin_SHA256']
        for invocation in range(1,4):
            root=stage/'timing'/target
            path=root/f'raw_invocation_{invocation}.json'
            raw=c.read(path)
            manifest[str(path.relative_to(c.ROOT))]=c.sha(path.read_bytes())
            assert raw['status']=='OK' and raw['target']==target and raw['invocation']==invocation
            assert raw['protocol_SHA256']==c.sha(c.encode(protocol))
            assert raw['payload_SHA256']==c.sha((root/'payload.zip').read_bytes())
            assert raw['cuda_return_codes']==[0] and raw['compilation']=={'triton_imported':False,'subprocess_calls':0,'compiler_calls':0}
            uuid=raw['environment']['gpu_uuid'];uuids.append(uuid)
            calls.append(raw['call_id']);processes.append(raw['process_identity'])
            assert raw['environment']['compute_capability']==protocol['targets'][target]['cc']
            assert protocol['targets'][target]['name'] in raw['environment']['gpu_name']
            tags=sorted(metadata)
            assert len(raw['warmups'])==18 and len(raw['visits'])==180
            for record in raw['warmups']:
                assert record['count']==3 and record['correctness_max_abs_diff']==0 and record['gpu_uuid']==uuid
                assert record['binary']['cubin_SHA256']==metadata[record['tag']]['cubin_SHA256']
            samples={tag:[] for tag in tags}
            for i,visit in enumerate(raw['visits']):
                r=i//18
                offset=((invocation-1)*17+r*13)%18
                order=tags[offset:]+tags[:offset]
                tag=order[i%18]
                assert visit['tag']==tag and visit['round']==r+1 and visit['gpu_uuid']==uuid
                assert visit['binary']['cubin_SHA256']==metadata[tag]['cubin_SHA256']
                assert len(visit['samples_us'])==len(visit['launch_SHA256'])==5
                assert all(x==metadata[tag]['cubin_SHA256'] for x in visit['launch_SHA256'])
                assert all(math.isfinite(x) and x>0 for x in visit['samples_us'])
                vals=visit['actual_parameters'];alloc=raw['allocations']
                if metadata[tag]['path']=='device':expected=[alloc['input_pointer'],alloc['output_pointer'],alloc['scratch_pointer'],0]
                else:expected=[vals[0],1,8192,8192,8192*8192,8192,1,alloc['output_pointer'],alloc['scratch_pointer'],0]
                assert vals==expected
                if metadata[tag]['path']=='host':assert vals[0]>0 and vals[0]%64==0
                samples[tag]+=visit['samples_us']
            for tag,binding in raw['binaries'].items():
                assert binding['guards']==binding['launches']==53 and binding['cubin_SHA256']==metadata[tag]['cubin_SHA256']
            expected_samples=18*50
            assert raw['cuda_call_counts']['cuLaunchKernel']==18*53
            assert raw['cuda_call_counts']['cuEventElapsedTime']==expected_samples
            assert raw['cuda_call_counts']['cuEventRecord']==expected_samples*2
            assert raw['cuda_call_counts']['cuEventSynchronize']==expected_samples
            sample_total+=expected_samples
            # Independent order-statistic median for the fixed 50 samples.
            medians[invocation]={tag:(sorted(x)[24]+sorted(x)[25])/2 for tag,x in samples.items()}
        assert len(set(calls))==3
        # Modal uses the same hostname/PID within fresh container namespaces.
        # Distinct call IDs plus single_use_containers identify dispatches.
        for case in protocol['cases']:
            for path in protocol['paths']:
                prefix=f'M{case["M"]}_N{case["N"]}_w{case["num_warps"]}:{path}:'
                a,b,d=[prefix+x for x in protocol['variants']]
                effects=[100*(medians[i][a]-medians[i][d])/medians[i][a] for i in range(1,4)]
                mean=statistics.mean(effects);sd=statistics.stdev(effects);half=4.302652729911275*sd/math.sqrt(3)
                results.append({'target':target,'case':prefix.rstrip(':'),'current_main_us':[medians[i][a] for i in range(1,4)],
                    'research_default_us':[medians[i][b] for i in range(1,4)],'research_vector4_us':[medians[i][d] for i in range(1,4)],
                    'current_main_vs_research_vector4_percent':effects,'mean_percent':mean,'process_band':[mean-half,mean+half],
                    'current_main_equals_research_default_CUBIN':metadata[a]['cubin_SHA256']==metadata[b]['cubin_SHA256'],
                    'native_SHA256':native,'physical_UUIDs':uuids,
                    'same_assembler':compiler_identities[target]['same_assembler'],
                    'current_main_vs_research_default_percent':[100*(medians[i][a]-medians[i][b])/medians[i][a] for i in range(1,4)],
                    'research_default_vs_vector4_percent':[100*(medians[i][b]-medians[i][d])/medians[i][b] for i in range(1,4)]})
    result={'rows':results,'samples':sample_total,'compiler_identities':compiler_identities,'meaning':'Current-main residual audit; historical vector4 reference is not a patch to current main.'}
    text='# Phase10 Stage A — Current-main residual audit\n\n'
    text+=f'Clean upstream `{protocol["upstream_HEAD"]}` was built with `make`. All 54 compiler attempts and {sample_total} exact-binary event samples are retained. Each target has three independently dispatched worker processes; each worker compares all 18 frozen binaries.\n\n'
    text+='The comparison uses the original `[1,8192,8192]` BF16→FP32 max geometry on host and device descriptor paths. The vector4 reference uses the historical compiler; its contrast with current main is a diagnostic opportunity, not current-main patch validation.\n\n'
    text+='| Target | Case/path | Current main mean median µs | Research vector4 mean median µs | Diagnostic time decrease % | Process band % | Main/default CUBIN equal |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    for row in results:
        text+=f'| {row["target"]} | {row["case"]} | {statistics.mean(row["current_main_us"]):.4f} | {statistics.mean(row["research_vector4_us"]):.4f} | {row["mean_percent"]:.3f} | [{row["process_band"][0]:.3f}, {row["process_band"][1]:.3f}] | {row["current_main_equals_research_default_CUBIN"]} |\n'
    text+='\nNo historical culprit is inferred from this audit. B200/SM100 and RTX PRO6000/SM120 do not replace the original GB300/SM103 system. The next stage freezes an instruction-count compiler prototype and validates structure/correctness; held-out patch timing remains a separate stage. H100 current main uses ptxas 13.4.59 while its historical compiler uses 12.9.86; this cross-compiler contrast therefore also includes assembler changes. B200 and SM120 use the same assembler SHA on both compiler builds. The analysis retains the within-historical-compiler default/vector4 comparison separately. All prior evidence is unchanged.\n'
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"analysis.json","summary.md","raw_manifest.json"}:
            manifest[str(path.relative_to(c.ROOT))]=c.sha(path.read_bytes())
    return result,text,manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    result,text,manifest=run()
    stage=c.OUT/'stage_a'
    if args.validate:
        assert (stage/'analysis.json').read_bytes()==c.encode(result)
        assert (stage/'summary.md').read_text()==text
        assert c.read(stage/'raw_manifest.json')==manifest
    else:
        c.write_once(stage/'analysis.json',result);c.write_once(stage/'summary.md',text.encode());c.write_once(stage/'raw_manifest.json',manifest)
    print(f'Stage A PASS: {result["samples"]} samples, 54 compiler attempts, prior evidence immutable')
