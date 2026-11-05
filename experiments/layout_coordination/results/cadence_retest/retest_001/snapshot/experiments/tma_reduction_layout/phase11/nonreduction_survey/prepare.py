"""Freeze one evidence-bearing revision before dispatch; never overwrite it."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import zipfile,json
from . import common as c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,required=True)
    parser.add_argument('--families',default='N01,N02,N03,N04,N05,N06')
    parser.add_argument('--reference')
    args=parser.parse_args();root=c.OUT/'stage_n1'/f'revision_{args.revision:03}'
    assert not root.exists(),root
    families=args.families.split(',');cases=c.cases(families)
    reference=None
    if args.reference:
        with zipfile.ZipFile(args.reference) as archive:reference=json.loads(archive.read('result.json'))
        if 'selected_cases' in reference:cases=[x for x in reference['selected_cases'] if x['family'] in families]
        elif reference.get('records'):
            resolved={r['case']:r['metadata']['case'] for r in reference['records']
                      if r.get('status')=='OK' and 'case' in r.get('metadata',{})}
            cases=[resolved.get(x['id'],x) for x in cases]
    assert cases and {x['family'] for x in cases}==set(families)
    names=[str(f.relative_to(c.ROOT)) for f in c.SOURCE.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    prior=c.old.SOURCE
    names += [str((prior/n).relative_to(c.ROOT)) for n in ['common.py','driver.py','production_compile.py',
        'production_cases.py','layout_parsing.py','g5_analyze.py']]
    names += ['experiments/tma_reduction_layout/phase8/launch.py']
    names += ['experiments/tma_reduction_layout/phase4/exact_cuda.py']
    snapshot=Path('/tmp')/f'nonreduction-n1-revision-{args.revision:03}'
    hashes=c.old.snapshot(names,snapshot)
    previous=c.read(c.old.OUT/'stage_g8/preregistration/protocol.json')
    protocol=dict(stage='nonreduction_n1',revision=args.revision,
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        parent_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        profile='chenmiaoming',compiler=previous['compiler'],native_SHA256=c.NATIVE,
        backend_SHA256=previous['backend_SHA256'],cases=cases,source_SHA256=hashes,
        GPU_role='Actual Tensor JIT binding and full correctness; no candidate performance comparisons',
        adaptive_authorization='User approved all six families and evidence-based follow-ups on 2026-11-05',
        snapshot=str(snapshot),baseline_gate='Allocated ELF/ABI/resources, true tensor specialization, correctness then timing equivalence',
        protected_SHA256={str(f.relative_to(c.ROOT)):c.sha(f.read_bytes())
            for f in (c.old.OUT/'stage_g8').rglob('*') if f.is_file()})
    if reference is not None:
        protocol.update(reference_payload_path=str(Path(args.reference).resolve()),
            reference_payload_SHA256=c.sha(Path(args.reference).read_bytes()))
    c.write_once(root/'preregistration/source_snapshot.zip',c.archive(snapshot))
    c.write_once(root/'preregistration/protocol.json',protocol)
    print('N1 frozen revision',args.revision,'cases',len(cases),'families',families)


if __name__=='__main__':main()
