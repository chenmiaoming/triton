"""M1 CPU-only audit. Every preceding source and evidence byte is immutable."""
from pathlib import Path
import subprocess
from experiments.tma_reduction_layout.phase11.c2d import common as old

ROOT,BASE=old.ROOT,old.BASE
SOURCE=Path(__file__).resolve().parent
OUT=BASE/'results/phase11/stage_m1'
read,encode,sha,write_once=old.read,old.encode,old.sha,old.write_once


def derived(path,value):
    path=Path(path);assert path.resolve().is_relative_to(OUT.resolve())
    blob=value if isinstance(value,bytes) else encode(value)
    if (OUT/'freeze.json').exists() and path.name!='validation.json':
        assert path.is_file() and path.read_bytes()==blob,path
    else:old.derived(path,blob)


def protocol():return read(OUT/'protocol.json')


def protect_prior():
    for name,digest in read(OUT/'prior_manifest.json')['files'].items():
        assert sha((ROOT/name).read_bytes())==digest,name


def source(stage,name):return BASE/'results/phase11'/stage/name


def geo(g):return f'{g[0]}x{g[1]}w{g[2]}'


def bind():
    from datetime import datetime,timezone
    assert not (OUT/'protocol.json').exists()
    names=subprocess.check_output(['git','ls-files','-z','experiments/tma_reduction_layout'],cwd=ROOT).decode().split('\0')
    files={n:sha((ROOT/n).read_bytes()) for n in names if n and '/phase11/m1/' not in n and '/stage_m1/' not in n}
    write_once(OUT/'prior_manifest.json',{'files':files})
    p={'stage':'M1','frozen_method_UTC':datetime.now(timezone.utc).isoformat(),
       'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'scope':'Post-D1 explanatory CPU audit; no new selector, no GPU, no blind-test claim for reused D1.',
       'prior_manifest_SHA256':sha((OUT/'prior_manifest.json').read_bytes()),
       'plan_SHA256':sha((SOURCE/'PLAN.md').read_bytes()),
       'eta_grid':[0.,.25,.5,.75,1.], 'fixed_ridge':.1,
       'profile_training':'Nine development geometries only, identical C34 train implementation; scalar eta. Fixed ridge profile plus existing training-only CV-selected ridge profile.',
       'profile_CV':'Reuse frozen C34 scalar inner-CV losses, not D1 outcomes. No likelihood confidence interpretation.',
       'substitutions':['static','measured_q','measured_control','measured_q_and_control'],
       'control_matching':'Same cohort, geometry, worker, GPU UUID and exact measured occupancy capacity. No interpolation for missing q.',
       'substitution_scope':'Recompute with frozen coefficients; all four combinations retained, common-support comparisons explicit; not causal attribution.',
       'paired_scope':'All same-geometry same-vec N-first/within pairs, labelled by ML/local/MR/MO differences; no claim that coarse static flags guarantee identical generated code.',
       'q_conditions':['natural','1','2','4','8'],
       'primary_diagnostic_geometries':[[32,64,8],[32,128,4],[16,256,8]],
       'CPU_only':True,'GPU_allocations':0,'native_rebuilds':0,'new_D1_predictions':False,
       'stop':'Finish audit/report/validator, commit and push. Propose any missing GPU intervention without launching it.'}
    write_once(OUT/'protocol.json',p)
    print('M1 CPU audit method bound; protected',len(files),'prior files',flush=True)


if __name__=='__main__':bind()
