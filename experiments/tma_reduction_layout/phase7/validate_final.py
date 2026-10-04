"""Phase7 final closure: first-commit bytes, trusted prior validators, isolated probes."""
import copy
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c, contracts as ct, timing_contract as tc

DEST=c.OUT/'final_validation'
COMMANDS=(('stage_a','audit_existing.py','--validate'),('prereg','contracts.py','--validate'),
          ('artifacts','audit_artifacts.py','--validate'),('stage_c_raw','timing_contract.py','stage_c'),
          ('stage_c_analysis','analyze.py','stage_c','--validate'),('stage_c_profile','validate_profile.py'),('stage_c_counters','analyze_profile.py','--validate'),('stage_d_raw','timing_contract.py','stage_d'),
          ('stage_d_analysis','analyze.py','stage_d','--validate'))


def first_freeze(stage):
    root=c.OUT/stage;rel=root.relative_to(c.ROOT).as_posix()
    commits=subprocess.check_output(['git','log','--diff-filter=A','--format=%H','--',rel+'/raw_manifest.json'],cwd=c.ROOT,text=True).splitlines()
    c.require(len(commits)==1,'Unique first freeze '+stage);commit=commits[0]
    tree=subprocess.check_output(['git','ls-tree','-rz',commit,'--',rel],cwd=c.ROOT);blobs={}
    for entry in tree.split(b'\0'):
        if entry:
            meta,name=entry.split(b'\t',1);blobs[name.decode()]=meta.decode().split()[2]
    files=set(c.read(root/'raw_manifest.json')['files'])|{'raw_manifest.json'};digests={}
    for name in sorted(files):
        data=(root/name).read_bytes();digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        c.require(blobs.get(rel+'/'+name)==digest,'First-commit bytes changed '+stage+'/'+name);digests[name]=c.sha(data)
    return {'first_raw_commit':commit,'protected_files':len(files),'inventory_SHA256':c.sha(c.encode(digests))}


def prior_phase6():
    with tempfile.TemporaryDirectory(prefix='tma-phase7-prior-') as directory:
        tree=Path(directory)/'trusted'
        subprocess.run(['git','worktree','add','--detach',str(tree),c.BASELINE],cwd=c.ROOT,check=True,capture_output=True)
        try:
            command=[sys.executable,'experiments/tma_reduction_layout/phase6/validate_final.py']
            run=subprocess.run(command,cwd=tree,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            log=DEST/'phase6_trusted.log';log.write_text(run.stdout)
            c.require(run.returncode==0,'Trusted Phase6 final validator')
            src=tree/'experiments/tma_reduction_layout/results/phase6/final_validation'
            target=DEST/'phase6_replay'
            if target.exists():shutil.rmtree(target)
            shutil.copytree(src,target)
            return {'status':'PASS','checkout_commit':c.BASELINE,'command':command,'return_code':run.returncode,
                    'log':log.relative_to(c.ROOT).as_posix(),'log_SHA256':c.sha(log.read_bytes()),
                    'suite_SHA256':c.sha((target/'suite.json').read_bytes()),'scope':'UNCHANGED_CHECKER_AT_TRUSTED_PHASE6_BASELINE'}
        finally:
            subprocess.run(['git','worktree','remove','--force',str(tree)],cwd=c.ROOT,check=True,capture_output=True)


def probes():
    cases,binaries,plan=tc.inputs('stage_d');raw=c.read(c.OUT/'stage_d/raw_invocation_1.json');altered=copy.deepcopy(raw)
    x=altered['visits'][0]['samples_us'][0];altered['visits'][0]['samples_us'][0]=math.nextafter(x,math.inf)
    tc.validate_invocation(altered,plan['invocations'][0],cases,binaries)
    with tempfile.TemporaryDirectory(prefix='tma-phase7-probe-') as directory:
        isolated=Path(directory);shutil.copytree(c.OUT/'stage_d',isolated/'stage_d');gate=isolated/'stage_b_artifacts';gate.mkdir()
        shutil.copyfile(ct.GATE/'launch_stage_d.json',gate/'launch_stage_d.json')
        c.write(isolated/'stage_d/raw_invocation_1.json',altered)
        with patch.object(c,'OUT',isolated),patch.object(ct,'GATE',gate):
            try:tc.validate_raw('stage_d')
            except ValueError as exc:c.require(str(exc)=='All raw bytes frozen','Positive tamper rejected for byte identity')
            else:raise RuntimeError('Positive sample tamper accepted')
            contract=c.read(gate/'launch_stage_d.json');order=contract['schedule']['invocations'][0]['rounds'][0]['order'];order[0],order[1]=order[1],order[0]
            c.write(gate/'launch_stage_d.json',contract)
            try:tc.inputs('stage_d')
            except ValueError as exc:c.require(str(exc)=='Frozen deterministic complete schedule','Order tamper reason')
            else:raise RuntimeError('Master order tamper accepted')
    return ['POSITIVE_FINITE_RAW_BYTE_TAMPER','FROZEN_MASTER_ORDER_TAMPER']


def report(result):
    gate=c.read(ct.GATE/'gate_results.json');fresh=c.read(c.OUT/'stage_d/results.json');diag=c.read(c.OUT/'stage_c/results.json')
    names=list(c.read(ct.DEST/'protocol.json')['fresh_hypotheses'])
    hypotheses=[[scope,h['n'],*[h[name] for name in names]] for scope,h in sorted(fresh['hypotheses'].items())]
    metrics=[[scope,v['n'],p,e['MAE'],e['RMSE']] for scope,v in fresh['predictive_comparisons'].items() if scope!='SECONDARY' for p,e in v['errors'].items()]
    lines=['# Phase7 completion and final evidence closure','',
        'Stages A–D completed. All 4888 earlier experiment files remain byte-identical. No compiler or production heuristic changes.','',
        'StageA audited complete kernel contexts for17 Phase6 cases without GPU. StageB preregistered the four Gluon descriptor/store interventions, canonical reference, nine diagnostic identities and20 entirely unseen identities before compilation/timing. All290 actual binary attempts and deterministic resource failures are retained.','',
        c.table(['Stage','Eligible','Samples','Physical GPU UUIDs'],[[s,len(x['cases']),result['timing'][s]['samples'],result['timing'][s]['physical_GPU_UUID_count']] for s,x in [('stage_c',diag),('stage_d',fresh)]]),'',
        'StageC is an outcome-informed diagnostic cohort; StageD uses the fixed independent cohort and all three uncalibrated hypotheses. No refit, case removal, threshold tuning or outcome-based rerun. Three separately dispatched processes per timing stage; physicalUUID counts are measured, not assumed.','',
        c.table(['Scope','n',*names],hypotheses),'',c.table(['Scope','n','Fixed predictor','MAE','RMSE'],metrics),'',
        'The all-eligible scope has six cases (one PRIMARY, five SECONDARY); the PRIMARY scope is coverage-inconclusive. All three comparisons satisfy the frozen strict MAE/RMSE decision, but H7_01 improves MAE by only about 0.00009356 ns/CTA (0.0212%). Its SUPPORTED label does not establish a practically large output-store effect. Full-context MAE is about 0.01423 versus host/native 0.44188 ns/CTA. These case-level aggregate errors are heavily influenced by M2048_N32_w16; all five other cases, including near-zero/negative gaps, are retained.','',
        'Gaps/effects/errors use ns/additional CTA. Store/descriptor interventions include compiler register allocation and scheduling responses. Reduction opcode matching and equal theoretical residency do not establish complete machine dataflow equivalence or pure component cost. Interaction is retained per case and invocation.','',
        'The exact completed Phase6 native image was reused with unchanged native/compiler SHA identities and persistent cache. No native rebuild. Formal timing loads archived ELF CUBINs and guards SHA before every launch, with no Triton import/JIT/compiler.','',
        'Profiler availability: '+result['profiler_status']+'. Profiler replay duration is excluded from formal timing.','',
        f"Finalvalidator {result['status']}: nine Phase7 checks, trusted Phase6 final closure, eleven historical validators replayed, two isolated tamper probes, and first-Git-commit raw/artifact byte checks.",'',
        '[Artifact admission](../stage_b_artifacts/summary.md) · [Diagnostic results](../stage_c/summary.md) · [Counters](../stage_c/profiling/counter_summary.md) · [Held-out results](../stage_d/summary.md) · [Validator evidence](suite.json)','',
        'Work stops after Phase7 StageD. H2b/H2c remain UNVERIFIED. No later phase has been started.','']
    return '\n'.join(lines)


def main():
    DEST.mkdir(parents=True,exist_ok=True);protected=c.inventory()
    freezes={s:first_freeze(s) for s in ('stage_b_artifacts','stage_c','stage_d')}
    legacy=c.read(DEST/'legacy/suite.json')
    c.require(legacy['status']=='PASS' and len(legacy['validators'])==11,'All historical validators replayed')
    for record in legacy['validators']:
        c.require(record['return_code']==0 and c.sha((c.ROOT/record['log']).read_bytes())==record['log_SHA256'],'Historical replay log closure')
    prior=prior_phase6();validators=[]
    for name,script,*args in COMMANDS:
        command=[sys.executable,'experiments/tma_reduction_layout/phase7/'+script,*args]
        run=subprocess.run(command,cwd=c.ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=DEST/(name+'.log');log.write_text(run.stdout)
        validators.append({'command':command,'return_code':run.returncode,'log':log.relative_to(c.ROOT).as_posix(),'log_SHA256':c.sha(log.read_bytes())})
        print(name,run.returncode,flush=True);c.require(run.returncode==0,'Final check '+name)
    isolated=probes();c.require(c.inventory()==protected,'Prior byte identity throughout final checks')
    c.require({s:first_freeze(s) for s in freezes}==freezes,'Raw evidence unchanged by validators/probes')
    timing={}
    for stage in ('stage_c','stage_d'):
        raws=[c.read(c.OUT/stage/f'raw_invocation_{i}.json') for i in (1,2,3)]
        timing[stage]={'samples':sum(sum(len(v['samples_us']) for v in r['visits']) for r in raws),
                      'physical_GPU_UUID_count':len({r['environment']['gpu_uuid'] for r in raws}),
                      'GPU_UUID_by_invocation':[r['environment']['gpu_uuid'] for r in raws],
                      'separate_processes':len({r['process_identity'] for r in raws})}
    tool=c.read(ct.GATE/'environment.json')['toolchain']['ncu']
    profiler=c.read(c.OUT/'stage_c/profiling/status.json') if (c.OUT/'stage_c/profiling/status.json').exists() else {'status':'NO_SEPARATE_COUNTER_RUN','tool_observation':tool}
    result={'status':'PASS','validated_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'protected_prior_files':len(protected),'prior_inventory_SHA256':c.sha(c.encode(protected)), 'first_commit_raw_protection':freezes,
        'validators':validators,'trusted_Phase6':prior,'legacy_suite_SHA256':c.sha((DEST/'legacy/suite.json').read_bytes()),
        'isolated_corruption_probes':isolated,'timing':timing,'profiler_status':profiler['status'],
        'stop_after':'PHASE_7_STAGE_D','no_next_phase':True}
    c.write(DEST/'suite.json',result);(DEST/'summary.md').write_text(report(result));print('Final PASS',flush=True)


if __name__=='__main__':main()
