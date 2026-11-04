"""Freeze all-case confirmation choices before obtaining new timing data."""
import zipfile
import json
from pathlib import Path
from datetime import datetime,timezone
from . import common as c
from .prepare_measure import run as prepare


def main():
    root=c.OUT/'stage_n4';assert not root.exists()
    analysis_path=c.OUT/'stage_n2/analysis/analysis.json';analysis=c.read(analysis_path)
    protocol_path=c.OUT/'stage_n2/preregistration/protocol.json';p=c.read(protocol_path)
    assert analysis['status']=='OK' and analysis['protocol_SHA256']==c.sha(c.encode(p))
    assert all(r['classification']=='practically_equivalent' for r in analysis['baseline_equivalence']), 'Resolve N2 U/G0 controls before confirmation'
    by={r['tag']:r for r in p['programs']};selected=[];choices=[]
    for row in analysis['cases']:
        case=row['case'];best=row['best_observed_tag'];assert best
        tags=[r['tag'] for r in p['programs'] if r['geometry']==case and r['variant'] in ['U','G0']]+[best]
        assert len(tags)==3 and len(set(tags))==3
        selected.extend(by[tag] for tag in tags)
        choices.append(dict(case=case,candidate=best,exploratory_gain_vs_U=row['best_observed_gain_vs_U_percent']))
    decision=dict(frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        role='Independent same-case validation after exploratory selection; no held-out geometry/model-generalization claim.',
        policy='Select the minimum observed mean GL of every one of the 14 cases, including no-benefit cases; no winner-only case filtering or tuning. Use fresh U/G0 and new independent H100 timing workers. Keep all N2 observations and negative outcomes.',
        inputs=dict(analysis_path=str(analysis_path.relative_to(c.ROOT)),analysis_SHA256=c.sha(analysis_path.read_bytes()),
            protocol_SHA256=c.sha(protocol_path.read_bytes())),choices=choices)
    payload=root/'selection/selected_payload.zip';source=Path(p['payload_path'])
    with zipfile.ZipFile(source) as z:
        result=json.loads(z.read('result.json'));result['programs']=selected
        result['parent_compile_counts']=result.pop('counts')
        result['parent_logical_candidate_policy']=result.pop('logical_candidate_policy')
        result['counts']=dict(cases=len(p['cases']),logical_programs=len(selected),variants=len(p['cases']),failed_records=0)
        result['logical_candidate_policy']='Frozen all-case N2 minimum-mean GL confirmation; unchanged binaries and independent controls'
        result['records']=[dict(status='OK',metadata=r) for r in selected]
        result['cases']=p['cases'];result['selection']=decision
        folders=[r['artifact_directory'].rstrip('/')+'/' for r in selected]
        import io
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as out:
            for name in z.namelist():
                if any(name.startswith(folder) for folder in folders):out.writestr(name,z.read(name))
            out.writestr('result.json',c.encode(result))
    c.write_once(payload,stream.getvalue());c.write_once(root/'selection/decision.json',decision)
    prepare('stage_n4',payload,'N4_independent_fixed_choice_confirmation',True)


if __name__=='__main__':main()
