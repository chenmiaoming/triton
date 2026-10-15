"""Archive gate and descriptive code-generation reproduction comparisons."""
import argparse
import io
import re
import zipfile
from collections import Counter
from experiments.tma_reduction_layout.phase11 import common as c

FAMILIES = ('max.', 'cvt.', 'shfl.', 'st.shared', 'ld.shared', 'ldmatrix',
            'bar.sync', 'selp.', 'redux.')


def fingerprint(ptx):
    # Whole-kernel signature, explicitly including initial load and output
    # conversion. This is not a reduction-only count or a performance model.
    result = []
    for line in ptx.splitlines():
        stripped = re.sub(r'^\s*@!?%\w+\s+','',line.strip())
        m = re.match(r'([\w.]+)\s+(.*?);',stripped)
        if m and m[1].startswith(FAMILIES):
            immediates = re.findall(r'(?<![%\w])(?:0x[0-9a-fA-F]+|\d+)(?![\w])',m[2])
            result.append(m[1]+(':'+','.join(immediates) if immediates else ''))
    return result


def initial_load_sequence(ptx):
    # Source lines supply the load boundary. Record an unbounded sequence too
    # rather than treating all shared loads as initial-tile loads.
    lines = ptx.splitlines()
    rows = []
    for i,line in enumerate(lines,1):
        m = re.match(r'\s*(?:@!?%\w+\s+)?(ld\.shared[\w.]*|ldmatrix[\w.]*)\s+',line)
        if m:
            rows.append({'line':i,'opcode':m[1]})
    return rows


def resource_key(meta):
    return {k:meta[k] for k in ('cubin_SHA256','function_name','dtype','M','N','num_warps',
                              'path','full_grid','dynamic_smem_bytes','global_scratch_size',
                              'global_scratch_align','tensordesc_meta')}


def derive():
    c.protect_prior()
    from experiments.tma_reduction_layout.phase11.protocol import load_b
    protocol,_ = load_b()
    roots = sorted((c.OUT/'stage_b/artifacts').glob('attempt_*/export'))
    accepted = []
    attempts = []
    for root in roots:
        records = c.read(root/'attempts.json')
        attempts.append({'attempt':root.parent.name,'records':len(records),
                         'statuses':dict(Counter(r['status'] for r in records))})
        if len(records)>=40 and all(r['status']=='OK' for r in records):
            accepted.append(root)
    assert len(accepted)==1, attempts
    root = accepted[0]
    records = c.read(root/'attempts.json')
    by_tag = {r['tag']:r for r in records}
    metadata = {}
    for tag,r in by_tag.items():
        folder = root/tag.replace(':','/')
        for name,digest in r['files'].items():
            assert c.sha((folder/name).read_bytes())==digest, (tag,name)
        m = c.read(folder/'metadata.json')
        assert c.sha((folder/'kernel.cubin').read_bytes())==m['cubin_SHA256']
        checked = c.read(folder/'correctness.json')
        assert set(checked)=={'quarter','random','boundary'} and all(v['pass'] for v in checked.values())
        assert c.read(folder/'occupancy.json')['blocks_per_sm_actual_dynamic_smem']>0
        metadata[tag] = m
    env = c.read(root/'environment.json')
    assert env['protocol_SHA256']==c.sha(c.encode(protocol))
    assert env['native_SHA256']==protocol['baseline']['native_SHA256']
    assert env['ccache_before']==env['ccache_after'] and env['GPU_timing'] is False
    aliases,gluon_groups,bridge_groups,reproduction = {},{},{},{}
    for case in protocol['cases']:
        labels = [c.metadata_tag(case,row['variant']) for row in case['candidates']]
        unique = []
        for tag in labels:
            matching = next((u for u in unique if resource_key(metadata[u])==resource_key(metadata[tag])),None)
            aliases[tag] = matching or tag
            if matching is None:
                unique.append(tag)
        gluon_groups[case['case_id']] = unique
        anchor = c.metadata_tag(case,'upstream_default')
        matches = [tag for tag in labels if by_tag[tag]['observed_layout']==by_tag[anchor]['observed_layout']]
        match = matches[0] if matches else c.metadata_tag(case,'canonical_layout_reproduction')
        assert match in metadata
        bridge_groups[case['case_id']] = [anchor,match]
        a = (root/anchor.replace(':','/')/'kernel.ptx').read_text()
        b = (root/match.replace(':','/')/'kernel.ptx').read_text()
        af,bf = fingerprint(a),fingerprint(b)
        reproduction[case['case_id']] = {'canonical':anchor,'gluon':match,
            'input_layout_equal':by_tag[anchor]['observed_layout']==by_tag[match]['observed_layout'],
            'shared_layout_equal':{k:v for k,v in by_tag[anchor]['observed_shared_layout'].items() if k!='name'}==
                                 {k:v for k,v in by_tag[match]['observed_shared_layout'].items() if k!='name'},
            'all_shared_load_sequences':{'canonical':initial_load_sequence(a),'gluon':initial_load_sequence(b)},
            'whole_kernel_filtered_fingerprint':{'canonical':af,'gluon':bf},
            'filtered_sequence_equal':af==bf,'filtered_multiset_equal':Counter(af)==Counter(bf),
            'scope':'whole kernel includes initial loads/reduction/output; no pure phase latency claim',
            'resources':{'canonical':c.read(root/anchor.replace(':','/')/'occupancy.json'),
                         'gluon':c.read(root/match.replace(':','/')/'occupancy.json')},
            'performance_equivalence_from_layout_alone':False}
    return {'accepted_attempt':root.parent.name,'metadata':metadata,'aliases':aliases,
            'gluon_groups':gluon_groups,'bridge_groups':bridge_groups,'reproduction':reproduction,
            'attempts':attempts,'environment':env,'score_model_used':False,'native_rebuild':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args = parser.parse_args()
    gate = derive()
    if args.validate:
        assert c.read(c.OUT/'stage_b/artifact_gate.json')==gate
    else:
        c.write_once(c.OUT/'stage_b/artifact_gate.json',gate)
        root = c.OUT/'stage_b/artifacts'/gate['accepted_attempt']/'export'
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for tag in sorted(gate['metadata']):
                for ext in ('ptx','cubin'):
                    path = tag.replace(':','/')+'/kernel.'+ext
                    z.writestr(path,(root/path).read_bytes())
        c.write_once(c.OUT/'stage_b/frozen_payload.zip',stream.getvalue())
    print('Artifact gate PASS',len(gate['metadata']),'artifacts, 3 full-output correctness recipes each')


if __name__=='__main__':
    main()
