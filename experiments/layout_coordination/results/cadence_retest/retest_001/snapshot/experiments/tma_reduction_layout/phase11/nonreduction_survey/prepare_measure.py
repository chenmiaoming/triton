"""Freeze driver-only correctness/timing or CPU candidate compilation protocols."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import zipfile
from . import common as c


def run(stage,payload,phase,interleaved=False):
    root=c.OUT/stage;assert not (root/'preregistration').exists()
    payload=Path(payload).resolve()
    with zipfile.ZipFile(payload) as z:raw=json.loads(z.read('result.json'))
    assert raw['status']=='OK'
    cases=raw.get('cases') or list({r['metadata']['case']['id']:r['metadata']['case']
                                  for r in raw['records'] if r.get('status')=='OK'}.values())
    names=[str(f.relative_to(c.ROOT)) for f in c.SOURCE.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    names += [str((c.old.SOURCE/n).relative_to(c.ROOT)) for n in ['common.py','driver.py','layout_parsing.py']]
    names += ['experiments/tma_reduction_layout/phase8/launch.py','experiments/tma_reduction_layout/phase4/exact_cuda.py']
    snapshot=Path('/tmp')/('nonreduction-'+stage)
    hashes=c.old.snapshot(names,snapshot)
    prior=c.read(c.old.OUT/'stage_g8/preregistration/protocol.json')
    p=dict(phase=phase,stage=stage,profile='chenmiaoming',snapshot=str(snapshot),
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        compiler=prior['compiler'],native_SHA256=c.NATIVE,backend_SHA256=prior['backend_SHA256'],
        cases=cases,source_SHA256=hashes,payload_path=str(payload),payload_SHA256=c.sha(payload.read_bytes()),
        protected_SHA256={str(f.relative_to(c.ROOT)):c.sha(f.read_bytes())
                          for f in (c.old.OUT/'stage_g8').rglob('*') if f.is_file()},
        correctness=dict(patterns=['finite','structured','edge'],full_outputs=True,
            guards=True,alias_and_stride_preserved=True,inplace_restore='Before EVERY warmup and sample, outside timing events'),
        timing=dict(seed=7110501,workers=3,visits=5,samples=31,warmups=8,
            bootstrap_replicates=4096,practical_threshold_percent=3,
            estimator='Arithmetic mean of worker/visit medians; shared paired worker/visit bootstrap',
            graph='one kernel, two external timing events; profiler absent',
            baseline_gate='U and G0 independently measured, allocated ELF/ABI/resources already identical'),
        interpretation='Candidate sweep is exploratory. No whole-model, cross-device or historical regression claim.')
    if 'programs' in raw:p['programs']=raw['programs']
    if interleaved:
        p['timing'].update(sample_order='interleaved_rounds',visits=10,
            rationale='Uniform redesign after baseline block-order timing produced seven inconclusive equivalence intervals; original observations preserved. Each round measures every physical program once in a new randomized order, reducing temporal drift confounding. No kernel, output, binary or estimator change.')
    c.write_once(root/'preregistration/protocol.json',p)
    c.write_once(root/'preregistration/source_snapshot.zip',c.archive(snapshot))
    print(stage,phase,len(cases),'cases',len(p.get('programs',[])),'programs')


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True);a.add_argument('--payload',required=True)
    a.add_argument('--phase',required=True);a.add_argument('--interleaved',action='store_true')
    v=a.parse_args();run(v.stage,v.payload,v.phase,v.interleaved)


if __name__=='__main__':main()
