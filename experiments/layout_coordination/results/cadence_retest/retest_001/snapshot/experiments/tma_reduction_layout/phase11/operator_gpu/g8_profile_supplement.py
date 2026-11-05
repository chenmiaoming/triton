"""Post-timing G8 diagnostics for two newly observed vLLM vec4 winners.

The original 167-report preregistration is immutable.  This separate six-report
protocol adds mechanism diagnostics only; it cannot change formal timing,
candidate coverage, selectors, or confirmatory status of the main experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import common as c

OUT = c.OUT/'stage_g8'
PROTOCOL = OUT/'profile_protocol_supplement.json'
DIRECTORY = OUT/'profiles_supplement'
TARGETS = ('g2_C06_s16_d128', 'g2_C06_s32_d128')
VARIANT = 'vec4:within'


def save(path, value):
    blob = value if isinstance(value, bytes) else c.encode(value)
    if path.exists():
        assert path.read_bytes() == blob, path
    else:
        c.write_once(path, blob)


def prepare():
    original = c.read(OUT/'profile_protocol.json')
    gate = c.read(OUT/'compile_gate.json')
    main = c.read(OUT/'preregistration/protocol.json')
    c.verify_sources(original)
    assert gate['status'] == 'OK'
    assert original['compile_gate_SHA256'] == c.sha(c.encode(gate))
    assert original['parent_protocol_SHA256'] == c.sha(c.encode(main))
    timing_path = OUT/'timing_analysis.json'; timing = c.read(timing_path)
    assert timing['status'] == 'OK'
    raw_paths = [OUT/f'timing/invocation_{i}/raw.json' for i in range(1,4)]
    assert all(c.read(path)['status'] == 'OK' for path in raw_paths)
    bytag = {record['tag']:record for record in gate['records']}
    groups = []
    for group_id in TARGETS:
        original_group = next(group for group in original['profile_groups'] if group['id'] == group_id)
        meta = bytag[group_id+':'+VARIANT]
        assert meta['group'] == group_id and meta['geometry'] == original_group['case']['id']
        assert not any(item['metadata']['tag'] == meta['tag'] for item in original_group['records'])
        groups.append(dict(id='diagnostic_'+group_id, case=original_group['case'],
            records=[dict(metadata=meta,aliases=[])],
            modes=['software','warm_application','cold_isolated']))
    protocol = {**original, 'profile_groups':groups,
        'profile_selectors':[dict(group=g['id'],names=[VARIANT],modes=g['modes']) for g in groups],
        'evidence_role':'post_timing_exploratory_diagnostic',
        'confirmatory':False,
        'selection_reason':'The completed source-aligned timing matrix identified vec4:within as a vLLM winner in these two geometries; this layout was absent from the original NCU selectors.',
        'formal_timing_or_selector_changes':False,
        'original_profile_protocol_SHA256':c.sha(c.encode(original)),
        'post_timing_source_SHA256':{str(Path(__file__).resolve().relative_to(c.ROOT)):c.sha(Path(__file__).read_bytes())},
        'selection_evidence_SHA256':{str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for path in [timing_path,*raw_paths]},
        'report_count':6,
        'execution_order':'After all original 11 profile groups complete; at most two H100 workers; no new compilation or formal timing.'}
    save(PROTOCOL,protocol)
    print('G8 supplemental diagnostic protocol frozen: 2 layouts, 6 reports; post-timing exploratory only')


def collect(attempt):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol = c.read(PROTOCOL); c.verify_sources(protocol)
    for name,digest in protocol['post_timing_source_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest,name
    for name,digest in protocol['selection_evidence_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest,name
    original = c.read(OUT/'profile_protocol.json')
    assert protocol['original_profile_protocol_SHA256'] == c.sha(c.encode(original))
    # Do not overlap supplemental workers with the main 167-report collection.
    for group in original['profile_groups']:
        raw = c.read(OUT/'profiles'/group['id']/'attempt_001/export/result.json')
        assert raw['status'] == 'OK' and raw['protocol_SHA256'] == c.sha(c.encode(original))
    payload = (OUT/'compiled_payload.zip').read_bytes()
    assert c.sha(payload) == protocol['payload_SHA256']
    from .g8_profile import app, collect as run
    calls = []
    with app.run():
        for index,group in enumerate(protocol['profile_groups']):
            folder = DIRECTORY/group['id']/f'attempt_{attempt:03}'
            assert not folder.exists(),folder
            call = run.spawn(protocol,payload,index)
            c.write_once(folder/'dispatch.json',dict(call_id=call.object_id,
                profile='chenmiaoming',group=group['id'],
                evidence_role=protocol['evidence_role'],confirmatory=False,
                protocol_SHA256=c.sha(c.encode(protocol)),payload_SHA256=c.sha(payload),
                max_concurrent_groups=2))
            calls.append((folder,call))
        def receive(item):
            folder,call = item; blob = call.get()
            c.write_once(folder/'raw_export.zip',blob); c.extract(blob,folder/'export')
            result = c.read(folder/'export/result.json')
            print('G8 supplemental NCU',result['group'],result['status'],flush=True)
            return result['status']
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(receive,calls))
        assert statuses == ['OK','OK'],statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action',choices=['prepare','collect'],default='prepare')
    parser.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args()
    if args.action == 'prepare':prepare()
    else:collect(args.attempt)


if __name__ == '__main__':main()
