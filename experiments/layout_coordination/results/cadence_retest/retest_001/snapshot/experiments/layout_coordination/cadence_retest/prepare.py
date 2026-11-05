"""Audit historical cadence and preregister complete frozen-binary replays."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
from pathlib import Path
import subprocess
import zipfile

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from .nonreduction_audit import build_audit as n_audit
from .operator_audit import audit as g_audit
from .legacy_audit import audit as l_audit

SOURCE=Path(__file__).resolve().parent
BASE=c.ROOT/'experiments/tma_reduction_layout/results/phase11'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve().relative_to(c.ROOT)),SHA256=digest(path))


def bundle(archives):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as out:
        for namespace,path in archives.items():
            with zipfile.ZipFile(path) as archive:
                assert len(archive.namelist())==len(set(archive.namelist()))
                for name in sorted(archive.namelist()):
                    assert not Path(name).is_absolute() and '..' not in Path(name).parts
                    if not name.endswith('/'):out.writestr(namespace+'/'+name,archive.read(name))
    return stream.getvalue()


def source_files():
    for name in ('engine.py','prepare.py','run.py','analyze.py','validate.py',
                 'nonreduction.py','operators.py','legacy.py','anchors.py','anchors_audit.py','cpu_verify.py'):
        assert (SOURCE/name).is_file(),name
    names={str(p.relative_to(c.ROOT)) for p in SOURCE.glob('*.py')}
    for folder in ('experiments/tma_reduction_layout/phase11',
                   'experiments/layout_coordination/clock_control'):
        names.update(str(p.relative_to(c.ROOT)) for p in (c.ROOT/folder).rglob('*')
            if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts)
    names.update('experiments/tma_reduction_layout/'+s for s in ('phase8/launch.py','phase4/exact_cuda.py'))
    return sorted(names)


def prepare(output):
    from .anchors_audit import audit as a_audit
    from .cpu_verify import run as cpu_verify
    out=Path(output).resolve()
    assert out.is_relative_to(c.ROOT) and not out.exists(),out
    sources=source_files()
    cpu_validation=cpu_verify()
    assert cpu_validation['status']=='OK' and cpu_validation['GPU_dispatches']==0
    audits=dict(nonreduction=n_audit(),operators=g_audit(),legacy=l_audit(),anchors=a_audit())
    nr=BASE/'nonreduction'
    np={name:c.read(nr/stage/'preregistration/protocol.json') for name,stage in
        [('n2','stage_n2'),('n5','stage_n5_timing')]}
    for parent in np.values():
        assert parent['compiler']=='20d9d98fead78c2f65e1b3ba5a7881d86a836dad'
        assert parent['native_SHA256']==c.NATIVE
    npaths={name:Path(parent['payload_path']) for name,parent in np.items()}
    nblob=bundle(npaths)
    gp=BASE/'operator_gpu/stage_g8'
    gparent=c.read(gp/'preregistration/protocol.json');ggate=c.read(gp/'compile_gate.json')
    gblob=(gp/'compiled_payload.zip').read_bytes()
    assert ggate['status']=='OK' and c.sha(gblob)==ggate['payload_SHA256']
    assert gparent['native_SHA256']==c.NATIVE
    lp=BASE/'stage_m3'
    lparent=c.read(lp/'preregistration/protocol.json');lgate=c.read(lp/'artifact_gate.json')
    lblob=(lp/'compiled_payload.zip').read_bytes()
    assert lgate['status']=='OK' and c.sha(c.encode(lparent))==lgate['protocol_SHA256']
    panels=audits['anchors']['panels']
    ablob=bundle({name:BASE/('stage_c2d' if name=='c2d' else 'stage_d1')/'compiled_payload.zip'
        for name in panels})
    blobs=dict(nonreduction=nblob,operators=gblob,legacy=lblob,anchors=ablob)
    suites=dict(
        nonreduction=dict(parent_protocols=np,programs_by_source={name:p['programs'] for name,p in np.items()},
            artifact_prefixes={name:name for name in np},original_payloads={name:ref(path) for name,path in npaths.items()},
            eligibility_by_source={name:ref(nr/stage/'check/attempt_001/invocation_000/export/result.json')
                for name,stage in [('n2','stage_n2'),('n5','stage_n5_timing')]},
            old_raw_refs=[ref(path) for stage in ['stage_n2','stage_n4','stage_n5_timing']
                for path in sorted((nr/stage/'timing/attempt_001').glob('invocation_*/export/result.json'))],
            old_frozen_selection=ref(nr/'stage_n4/selection/decision.json'),
            expected=dict(contexts=16,original_physical=134,original_logical=144,shams=16,physical=150)),
        operators=dict(parent_protocol=gparent,compiled=ggate,original_payload=ref(gp/'compiled_payload.zip'),
            original_eligibility=ref(gp/'final_validation.json') if (gp/'final_validation.json').exists() else ref(gp/'compile_gate.json'),
            old_raw_refs=[ref(gp/f'timing/invocation_{worker}/raw.json') for worker in (1,2,3)],
            expected=dict(contexts=64,original_physical=439,original_logical=517,shams=66,physical=505)),
        legacy=dict(parent_protocol=lparent,gate=lgate,original_payload=ref(lp/'compiled_payload.zip'),
            old_raw_refs=[ref(path) for path in sorted((lp/'timing').glob('invocation_*/raw.json'))],
            expected=dict(contexts=216,original_physical=405,shams=189,physical=594),
            baseline_scope='Explicit Gluon N-first placement reference; no authentic Triton U/G0 pair is invented.'),
        anchors=dict(panels=panels,
            old_raw_refs=[ref(path) for stage in ('stage_c2d','stage_d1')
                for path in sorted((BASE/stage/'timing').glob('invocation_*/raw.json'))],
            expected=dict(contexts=57,original_physical=417,shams=57,physical=474),
            baseline_scope='Actual upstream-default explicit Gluon layout plus all legal layouts and independent transfer controls; seen-case remeasurement, never a new blind test.'))
    for name,entry in suites.items():
        entry.update(payload_path='payloads/'+name+'.zip',payload_SHA256=c.sha(blobs[name]))
    protected_names=subprocess.check_output(['git','ls-files','-z','--',
        'experiments/tma_reduction_layout/results','experiments/tma_reduction_layout/phase11',
        'experiments/layout_coordination','experiments/tma_reduction_layout/phase8/launch.py',
        'experiments/tma_reduction_layout/phase4/exact_cuda.py'],text=True).split('\0')
    protected={name:digest(c.ROOT/name) for name in protected_names if name}
    source_sha={name:digest(c.ROOT/name) for name in sources}
    audit_blob=c.encode(audits);inventory_blob=c.encode(protected);cpu_blob=c.encode(cpu_validation)
    p=dict(stage='cadence_retest',profile='chenmiaoming',GPU='H100!',base_image=c.BASE_IMAGE,
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        compiler='20d9d98fead78c2f65e1b3ba5a7881d86a836dad',native_SHA256=c.NATIVE,
        kernel_compiles=0,native_rebuilds=0,clock_control='unlocked',GPU_configuration_writes=False,
        snapshot='snapshot',source_SHA256=source_sha,
        audit_path='preregistration/audit.json',audit_SHA256=c.sha(audit_blob),
        cpu_validation_path='preregistration/cpu_validation.json',cpu_validation_SHA256=c.sha(cpu_blob),
        protected_inventory_path='preregistration/protected_SHA256.json',
        protected_inventory_SHA256=c.sha(inventory_blob),protected_file_count=len(protected),suites=suites,
        timing=dict(workers=3,visits=10,samples=31,primes=8,warmups=3,
            host_gap_seconds=.001,telemetry_period_s=.05,bootstrap_replicates=8192,
            seed=20261107,practical_threshold_percent=3,
            intervals='Pointwise percentile 95% intervals; no multiplicity correction or family-wise guarantee.',
            estimator='Mean of worker/visit medians. Shared worker/visit bootstrap indices across every context within each suite.',
            ordering='Seeded per-round random interleaving of all physical programs including fresh U/G0 and exact-program repeat observations.'),
        eligibility=dict(invocation=0,formal_timing=False,
            gates='Fresh direct finite oracle, exact graph nodes, captured graph oracle before/after. Prior byte-identical three-pattern eligibility is pinned; no recompile.',
            required_before_formal=True),
        correctness=dict(phases=['before','graph_before','graph_after'],complete_outputs=True,
            readonly_inputs=True,canaries=True,inplace='Immutable D2D snapshots before EVERY prime and target; restoration excluded from target event window.',
            pipelines='Original stage2 and original per-iteration zeroing retained; training zeros remain INSIDE measured semantic iteration.'),
        scope=dict(primary='Controlled warm steady-state repeatability and layout comparisons on the complete latest valid matrices.',
            original_physical_conditions=1395,physical_observations_with_shams=1723,
            formal_target_samples=1723*3*10*31,
            preserve='All earlier raw samples, compiler artifacts, NCU files and frozen models/predictions are immutable.',
            not_repeated='Obsolete incorrect-Gluon programs, failed preparations, CPU-only fits, and static/discrete counter evidence are preserved rather than rerun.',
            legacy_boundary='M2/M4 and earlier K-loop/cold-cache or cross-architecture timing remain historical-context evidence; no universal immunity is inferred. This retest does not relabel their numerical values as controlled warm measurements.',
            no_refit=True,no_new_blind_claim=True,no_PR=True),
        decision_rules=dict(no_sample_filtering=True,no_favorable_outcome_gate=True,
            control='Report U/G0 and exact-program shams independently. A failed equivalence interval prevents claiming a supported layout gain for that comparison; preserve all results.',
            layout_gain='For genuine U/G0 suites, both same-round paired 95% CI lower bounds must exceed +3%. All candidates, controls, negative and inconclusive outcomes remain visible.',
            old_comparison='Old/new cohorts and execution contexts are separate; no paired causal interval or blanket percentage correction is fabricated.',
            retries='Failed GPU eligibility or source/semantic correction requires a new immutable protocol revision; never copy a later gate into attempt_001. Transport-only redispatches retain failed attempts. No repeated sampling until a desired significance or winner is obtained.'))
    assert sum(s['expected']['original_physical'] for s in suites.values())==1395
    assert sum(s['expected']['physical'] for s in suites.values())==1723
    for name in sources:
        blob=(c.ROOT/name).read_bytes()
        assert c.sha(blob)==source_sha[name],('Source changed during freeze',name)
        c.write_once(out/'snapshot'/name,blob)
    for name,blob in blobs.items():c.write_once(out/suites[name]['payload_path'],blob)
    c.write_once(out/p['audit_path'],audit_blob)
    c.write_once(out/p['cpu_validation_path'],cpu_blob)
    c.write_once(out/p['protected_inventory_path'],inventory_blob)
    c.write_once(out/'preregistration/protocol.json',p)
    c.write_once(out/'preregistration/freeze.json',dict(status='OK',CPU_only=True,GPU_dispatches=0,
        protocol_SHA256=c.sha(c.encode(p)),source_files=len(sources),protected_files=len(protected)))
    print('Frozen cadence audit/retest',out,'suites',list(suites),'target samples',p['scope']['formal_target_samples'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    prepare(parser.parse_args().output)
