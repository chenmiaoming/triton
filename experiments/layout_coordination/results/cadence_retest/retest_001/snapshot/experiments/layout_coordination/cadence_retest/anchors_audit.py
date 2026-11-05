"""Exact natural-FULL/T_control inventory of the seen C2d and D1 matrices."""
from collections import defaultdict
from pathlib import Path

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.d1.runtime import cells_for


BASE = c.ROOT/'experiments/tma_reduction_layout/results/phase11'


def selected_cells(panel):
    gate = panel['gate']
    result = [row for row in cells_for(panel['parent_protocol'], gate)
              if row['q']=='natural' and gate['metadata'][row['tag']].get('original_once')]
    assert all(gate['metadata'][row['tag']]['family'] in ('ONE_FULL','ONE_TRANSFER')
               and row['K']==1 for row in result)
    if 'cells' in panel:
        assert c.encode(result) == c.encode(panel['cells'])
    return result


def default_tag(meta):
    m,n,w = meta['M'],meta['N'],meta['num_warps']
    assert m*n % (32*w) == 0
    vector = min(8, m*n//(32*w))
    return f'M{m}_N{n}_w{w}:vec{vector}:n_first:ONE_FULL'


def groups(panel):
    metadata = panel['gate']['metadata']
    result = defaultdict(list)
    for cell in selected_cells(panel):
        meta = metadata[cell['tag']]
        key = meta['M'],meta['N'],meta['num_warps'],cell['grid']
        result[key].append(cell)
    for key,rows in result.items():
        meta = metadata[rows[0]['tag']]
        assert sum(row['tag']==default_tag(meta) for row in rows)==1
        assert sum(metadata[row['tag']]['family']=='ONE_TRANSFER' for row in rows)==1
        assert all(row['q']=='natural' and row['K']==1 for row in rows)
        assert key[3] % ((8192//key[0])*(8192//key[1])) == 0
    return sorted(result.items())


def panel(name):
    assert name in ('c2d','d1')
    folder = BASE/('stage_'+name)
    gate_path = folder/'artifact_gate.json'
    gate = c.read(gate_path)
    assert gate['status']=='OK'
    candidates = [p for p in (folder/'preregistration').rglob('protocol.json')
                  if c.sha(p.read_bytes())==gate['protocol_SHA256']]
    assert candidates
    protocol_path = min(candidates,key=lambda p:(len(p.parts),str(p)))
    parent = c.read(protocol_path)
    for source,digest in parent['worker_source_SHA256'].items():
        assert c.sha((c.ROOT/source).read_bytes())==digest,source
    payload_path = folder/'compiled_payload.zip'
    data = dict(parent_protocol=parent,gate=gate,namespace=name,
        payload_SHA256=c.sha(payload_path.read_bytes()))
    rows = selected_cells(data)
    program_tags = {row['tag'] for row in rows}
    expected = (210,70,27) if name=='c2d' else (207,69,30)
    assert (len(rows),len(program_tags),len(groups(data)))==expected
    # Independently compare each timing worker's actual accepted conditions.
    key = lambda row:(row['tag'],row['q'],row['grid'],row['K'])
    references = []
    for raw_path in sorted((folder/'timing').glob('invocation_*/raw.json')):
        raw = c.read(raw_path)
        assert raw['status']=='OK' and raw['protocol_SHA256']==gate['protocol_SHA256']
        assert raw['payload_SHA256']==data['payload_SHA256']
        actual = [row for row in raw['cells'] if row['q']=='natural'
                  and gate['metadata'][row['tag']].get('original_once')]
        assert {key(row) for row in actual}=={key(row) for row in rows}
        references.append(dict(path=str(raw_path.relative_to(c.ROOT)),SHA256=c.sha(raw_path.read_bytes()),
                               gpu_uuid=raw['environment']['gpu_uuid']))
    assert len(references)==3
    data.update(cells=rows,
        source=dict(protocol_path=str(protocol_path.relative_to(c.ROOT)),
            protocol_SHA256=gate['protocol_SHA256'],gate_path=str(gate_path.relative_to(c.ROOT)),
            gate_SHA256=c.sha(gate_path.read_bytes()),payload_path=str(payload_path.relative_to(c.ROOT)),
            raw=references),
        selected_programs=len(program_tags),physical_conditions=len(rows),
        contexts=len(groups(data)),default_rule='vec=min(8,BM*BN/(32W)), N-first, ONE_FULL',
        scope='Seen-case natural FULL/T_control remeasurement. No source-aligned U/G0 identity or new blind result is claimed.')
    return data


def audit():
    panels = {name:panel(name) for name in ('c2d','d1')}
    assert sum(p['selected_programs'] for p in panels.values())==139
    assert sum(p['physical_conditions'] for p in panels.values())==417
    assert sum(p['contexts'] for p in panels.values())==57
    return dict(status='OK',panels=panels,original_physical_conditions=417,
        contexts=57,default_repeats=57,physical_observations_per_visit=474,
        original_models_unchanged=True,D1_new_blind_claim=False)
