"""Freeze a CPU-only audit of the entire existing U/G0 operator bridge."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import subprocess
from . import common as c

OUT = c.OUT/'stage_g5'
SNAPSHOT = Path('/tmp/operator-gpu-g5-snapshot-v1')
PIN = '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'


def protocol_path():
    path=OUT/'preregistration/protocol_v2.json'
    return path if path.exists() else OUT/'preregistration/protocol.json'


def main(revision=1):
    g2 = c.read(c.OUT/'stage_g2/compile_gate.json')
    p2 = c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    g3 = c.read(c.OUT/'stage_g3b/cpu/attempt_002/export/result.json')
    p3 = c.read(c.OUT/'stage_g3b/preregistration/protocol_v2.json')
    cases = []
    for case in p2['geometries'] + p3['cases']:
        gate = g3 if case['family'] == 'C01' else g2
        rows = [r for r in gate['records'] if r['tag'] in [case['id']+':upstream',case['id']+':gluon_anchor']]
        assert len(rows) == 2
        cases.append({'case':case,'records':rows,'plan':gate['plans'][case['id']]})
    refs = ['third_party/nvidia/backend/compiler.py','lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp',
            'lib/Dialect/TritonGPU/Transforms/Coalesce.cpp','lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp',
            'lib/Dialect/TritonGPU/Transforms/Utility.cpp','lib/Dialect/TritonGPU/Transforms/LoopAwareCSE.cpp',
            'lib/Dialect/Triton/Transforms/Combine.cpp','lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp',
            'lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp','lib/Dialect/Gluon/Transforms/Canonicalize.cpp',
            'lib/Dialect/TritonGPU/IR/Ops.cpp','python/src/ir.cc','python/triton/compiler/compiler.py']
    compiler_sources = {}
    for name in refs:
        run = subprocess.run(['git','show',PIN+':'+name],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if run.returncode: continue
        dest=OUT/'compiler_source'/name
        if dest.exists():assert dest.read_bytes()==run.stdout
        else:c.write_once(dest,run.stdout)
        compiler_sources[name]=c.sha(run.stdout)
    names = list(p2['worker_source_SHA256'])
    names += [str((c.SOURCE/n).relative_to(c.ROOT)) for n in ['g3b_gluon.py','g3b_triton.py','g5_prepare.py','g5_worker.py','g5_run.py']]
    hashes = c.snapshot(names,SNAPSHOT)
    p = {'stage':'G5 CPU bridge audit','frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
         'parent_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'compiler':PIN,'native_SHA256':c.NATIVE,'profile':'chenmiaoming','GPU_allocated':False,
         'cases':cases,'worker_source_SHA256':hashes,'compiler_source_SHA256':compiler_sources,
         'pilot':['C06_s32_d256','C02_n64_d128','C01_p256_n1024'],
         'pilot_variants':['U','G0','G0_cleanup','G0_rmlc','G0_fullsuffix','U_no_rmlc'],
         'matrix_variants':['U','G0','G0_fullsuffix'],
         'invariants':['No C++ edits or native rebuild','No original sample/artifact overwrite',
                       'No changed math, dtype, tile, warp, source scalar participation or runtime ABI',
                       'A repaired anchor is not called equivalent solely because timings are close',
                       'Check every intermediate encoding, conversion width and loop placement; preserve negative outcomes'],
         'old_evidence_SHA256':{str(path.relative_to(c.ROOT)):c.sha(path.read_bytes())
             for stage in ['stage_g2','stage_g3a','stage_g3b','stage_g4a','stage_g4b']
             for path in sorted((c.OUT/stage).rglob('*')) if path.is_file()}}
    suffix='_v2' if revision==2 else ''
    if revision==2:
        p.update(revision=2,
            revision_reason='Isolate all Python pipeline interventions from production kernel-cache keys; CPU-only restore of three pilot U/G0 keys and complete the other six pairs. No GPU was allocated.',
            pilot_variants=['U_restore_cache','G0_restore_cache'],
            cache_policy='Unmodified restore programs use the original cache; every intervention uses a dedicated cache directory.',
            prior_attempt_SHA256={str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for path in sorted((OUT/'pilot').rglob('*')) if path.is_file()})
    c.write_once(OUT/f'preregistration/source_snapshot{suffix}.zip',c.archive(SNAPSHOT))
    c.write_once(OUT/f'preregistration/protocol{suffix}.json',p)
    print('G5 CPU audit frozen:',len(cases),'U/G0 pairs; no GPU experiment')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,choices=[1,2],default=1);main(parser.parse_args().revision)
