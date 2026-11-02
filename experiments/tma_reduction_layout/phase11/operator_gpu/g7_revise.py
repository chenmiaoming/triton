"""Freeze the binding-harness correction before retrying only missing calls."""
import copy
from datetime import datetime, timezone
from . import common as c
from .g7_prepare import OUT, SNAPSHOT


def main():
    p = copy.deepcopy(c.read(OUT/'preregistration/protocol.json'))
    p.update(revision=2, revised_at_UTC=datetime.now(timezone.utc).isoformat(),
        revision_reason='Supply runtime auxiliary defaults in SGLang kwargs. Distinguish empty attribute-list syntax and validated batch64 additional divisibility from executable/ABI equality. No timing outcomes have been observed.',
        binding_only_groups=[g['id'] for g in p['binding_groups'] if g['case']['family'] == 'C02'],
        binding_reuse_policy='Reuse 18 existing compiled real-call outputs after offline reconciliation; retry only 8 SGLang calls that failed before compilation.',
        prior_attempt_SHA256={str(path.relative_to(c.ROOT)): c.sha(path.read_bytes())
            for path in sorted((OUT/'binding/attempt_001').rglob('*')) if path.is_file()})
    names = list(p['worker_source_SHA256'])
    names += [str(path.relative_to(c.ROOT)) for path in c.SOURCE.glob('g7_*.py')]
    p['worker_source_SHA256'] = c.snapshot(names, SNAPSHOT)
    c.write_once(OUT/'preregistration/source_snapshot_v2.zip', c.archive(SNAPSHOT))
    c.write_once(OUT/'preregistration/protocol_v2.json', p)
    gate = c.read(OUT/'compile_gate.json'); gate['protocol_SHA256'] = c.sha(c.encode(p))
    c.write_once(OUT/'compile_gate_v2.json', gate)
    print('G7 V2 frozen: retry 8 missing SGLang calls; preserve/reconcile 18 existing bindings')


if __name__ == '__main__':
    main()
