"""Execute separate frozen U/G0 kernels and audit full output byte equality."""
from . import common as c


def run(protocol, compiled, payload, call_id, invocation=0):
    from .g4_runtime import run as execute, identity
    p = {**protocol, 'groups': protocol['inference_groups']}
    metas = [r for r in compiled['records'] if r['family'] != 'C01']
    for group in p['groups']:
        pair = [r for r in metas if r['group'] == group['id'] and not r['stage2']]
        assert len(pair) == 2 and {r['pair_role'] for r in pair} == {'U','G0'}
        # Existing runner's CUBIN identity deduplication must not merge U/G0.
        assert len({identity(r) for r in pair}) == 2
        if group['form_id'].startswith('C06') and '_b1_' in group['form_id']:
            assert all(w['batch'] == 1 for w in group['workloads'])
    result = execute(p, {'records': metas}, payload, call_id, invocation)
    result['protocol_SHA256'] = c.sha(c.encode(protocol))
    if result['status'] != 'OK':
        return result
    paired = {}
    for check in result['checks']:
        key = (check['group'], check['batch'], check['length'], check['pattern'])
        outputs = {d['name']: d['actual_SHA256'] for d in check['details'] if 'actual_SHA256' in d}
        paired.setdefault(key, {})[check['tag'].rsplit(':',1)[1]] = outputs
    comparisons = []
    for key, sides in paired.items():
        assert set(sides) == {'U','G0'}
        same = sides['U'] == sides['G0']
        comparisons.append(dict(group=key[0], batch=key[1], length=key[2], pattern=key[3],
                                output_SHA256=sides, bitwise_equal=same))
        assert same, comparisons[-1]
    result['bitwise_comparisons'] = comparisons
    assert all(not r['aliases'] for r in result['cells'])
    return result
