"""Prebind GPU verification to G6 source baselines and frozen U artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import subprocess
from . import common as c
from .g6_analyze import references, artifact

OUT = c.OUT/'stage_g7'
SNAPSHOT = Path('/tmp/operator-gpu-g7-worker-snapshot')


def main():
    manifest = c.read(c.OUT/'stage_g6/baseline_manifest.json')
    accepted = c.read(c.ROOT/manifest['protocol_path']); c.verify_sources(accepted)
    refs = references(); byform = {r['case']: r for r in manifest['cases']}
    g4p = c.read(c.OUT/'stage_g4b/preregistration/protocol.json')
    g4 = c.read(c.OUT/'stage_g4b/compile_gate_v2.json')
    g2 = c.read(c.OUT/'stage_g2/compile_gate.json')
    g3p = c.read(c.OUT/'stage_g3b/preregistration/protocol_v2.json')
    g3root = c.OUT/'stage_g3b/cpu/attempt_002/export'; g3 = c.read(g3root/'result.json')
    source_items = {i['case']['id']: i for i in accepted['cases']}
    records = []; groups = []; binding_groups = []
    payload = Path('/tmp/operator-g7-payload'); assert not payload.exists()

    def add(meta, original_folder, tag, **fields):
        value = copy.deepcopy(meta); destination = 'artifacts/'+tag.replace(':', '_')
        value.update(tag=tag, artifact_directory=destination, **fields)
        for name, digest in meta['artifact_SHA256'].items():
            blob = (original_folder/name).read_bytes(); assert c.sha(blob) == digest
            c.write_once(payload/destination/name, blob)
        records.append(value)
        return value

    def pair(group_id, form_id, family):
        uroot, umeta = refs[form_id]; gm = byform[form_id]['metadata']
        gfolder = c.ROOT/byform[form_id]['artifacts']; ufolder = uroot/umeta['artifact_directory']
        assert artifact(uroot, umeta)['allocated_sections'] == artifact(gfolder.parent.parent, gm)['allocated_sections']
        for label, meta, folder in [('U', umeta, ufolder), ('G0', gm, gfolder)]:
            add(meta, folder, group_id+':'+label, group=group_id, form_id=form_id, pair_role=label,
                role='merge' if family == 'C01' else 'stage1', stage2=False,
                stage2_tag=group_id+':stage2')

    for item in accepted['cases']:
        case = item['case']
        if case['family'] == 'C01' or '_b' in case['id']:
            continue
        group_id = case['id']+'_default_b4_l129'
        groups.append(dict(id=group_id, case=case, form_id=case['id'], workloads=[dict(batch=4, length=129)]))
        pair(group_id, case['id'], case['family'])
        if case['family'] == 'C02':
            helper = next(r for r in g2['records'] if r['tag'] == case['id']+':stage2')
            add(helper, c.ROOT/g2['artifact_directory']/helper['artifact_directory'], group_id+':stage2',
                group=group_id, pair_role='helper', role='stage2', stage2=True)
    bindings = {r['group']: r for r in manifest['actual_G4b_workloads']}
    for old in g4p['groups']:
        gid = old['id']; form_id = bindings[gid]['source_baseline_case']
        group = dict(id=gid, case=old['case'], form_id=form_id, workloads=old['workloads'])
        groups.append(group); binding_groups.append(group)
        pair(gid, form_id, old['case']['family'])
        if old['case']['family'] == 'C02':
            helper = next(r for r in g4['records'] if r['tag'] == gid+':actual:stage2')
            add(helper, c.ROOT/g4['CPU_export_directory']/helper['artifact_directory'], gid+':stage2',
                group=gid, pair_role='helper', role='stage2', stage2=True)
    for case in g3p['cases']:
        gid = case['id']; pair(gid, gid, 'C01')
        helper = next(r for r in g3['records'] if r['tag'] == gid+':dx')
        add(helper, g3root/helper['artifact_directory'], gid+':dx',
            group=gid, pair_role='helper', role='dx', stage2=False)
    assert len(groups) == 16 and len(binding_groups) == 10
    files = [str(p.relative_to(c.ROOT)) for p in c.SOURCE.glob('*.py')]
    files += list(accepted['worker_source_SHA256'])
    hashes = c.snapshot(files, SNAPSHOT)
    protocol = dict(stage='g7', frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        compiler=accepted['compiler'], native_SHA256=c.NATIVE, backend_SHA256=accepted['backend_SHA256'],
        profile='chenmiaoming', requested_gpu='H100!:1', compiler_intervention=False,
        inference_groups=groups, binding_groups=binding_groups, binding_items=accepted['cases'],
        training_cases=g3p['cases'], worker_source_SHA256=hashes,
        correctness=g4p['correctness'], training_correctness=g3p['correctness'],
        timing=dict(workers=3, visits=5, samples=31, warmups=8, seed=20261101,
                    bootstrap_replicates=4096, practical_threshold_percent=3,
                    no_result_dependent_repeats=True,
                    estimator='mean of worker x paired-visit medians; paired hierarchical bootstrap CI95',
                    decision='CI95 wholly within [-3%,+3%]: practical timing equivalence; otherwise uncertain, never optimize or rerun to pass'),
        bitwise_scope='All inference outputs and fixed-input LN merge; whole LN backward tolerances/canaries are primary because partial accumulation can be nondeterministic.',
        protected_SHA256={str(path.relative_to(c.ROOT)): c.sha(path.read_bytes())
            for stage in ['stage_g2','stage_g3a','stage_g3b','stage_g4a','stage_g4b','stage_g5','stage_g6']
            for path in sorted((c.OUT/stage).rglob('*')) if path.is_file()})
    c.write_once(OUT/'preregistration/protocol.json', protocol)
    c.write_once(OUT/'preregistration/source_snapshot.zip', c.archive(SNAPSHOT))
    blob = c.archive(payload); c.write_once(OUT/'compiled_payload.zip', blob)
    c.write_once(OUT/'compile_gate.json', dict(status='OK', records=records,
        protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(blob), source_forms=13,
        compiled_new_programs=0, compiler_intervention=False))
    print('G7 frozen: 13 forms, 16 inference groups, 3 training geometries; no new layout candidates')


if __name__ == '__main__':
    main()
