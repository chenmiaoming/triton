"""Offline gates, paired increments, and composition identification diagnostics."""
import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path
import re
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.m2 import common as c
from experiments.tma_reduction_layout.phase11.c2.analyze import source_blocks, hardware_rows, FAMILIES


def fit(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    a = np.column_stack((np.ones(len(xs)), xs))
    beta = np.linalg.lstsq(a, ys, rcond=None)[0]
    residual = ys-a@beta
    return {'intercept': float(beta[0]), 'slope': float(beta[1]),
            'max_abs_residual': float(np.max(np.abs(residual))),
            'xs': xs.tolist(), 'ys': ys.tolist()}


def compile_gate():
    protocol = c.protocol()
    c.protect_prior()
    roots = [path for path in (c.OUT/'cpu_preflight').glob('attempt_*/export')
             if c.read(path/'result.json')['status'] == 'OK']
    assert roots, 'No successful CPU preflight'
    root = max(roots, key=lambda path: int(path.parent.name.split('_')[-1]))
    raw = c.read(root/'result.json')
    assert raw['native_SHA256'] == protocol['baseline']['native_SHA256']
    assert raw['ccache_before'] == raw['ccache_after']
    assert len(raw['metadata']) == len(protocol['cases'])
    allowed = {tuple(row[:3]) for row in protocol['pair_geometries_vectors']}
    audits = {}
    for tag, meta in raw['metadata'].items():
        if meta['kind'] != 'local':
            assert (meta['M'], meta['N'], meta['num_warps']) in allowed, tag
        folder = root/tag.replace(':', '/')
        assert c.sha((folder/'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
        ptx = (folder/'kernel.ptx').read_text()
        sass = (folder/'kernel.sass').read_text()
        resources = (folder/'kernel.resource.txt').read_text()
        audits[tag] = {'max_PTX_sites': dict(Counter(re.findall(r'\bmax\.(?:bf16x2|bf16|f32)\b', ptx))),
            'physical_registers_from_cuobjdump': [int(v) for v in re.findall(r'REG:(\d+)', resources)],
            'local_storage_bytes_from_cuobjdump': [int(v) for v in re.findall(r'LOCAL:(\d+)', resources)],
            'stack_bytes_from_cuobjdump': [int(v) for v in re.findall(r'STACK:(\d+)', resources)],
            'local_load_store_SASS_sites': len(re.findall(r'\b(?:LDL|STL)(?:\.|\s)', sass)),
            'SASS_opcode_sites': dict(Counter(re.findall(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)', sass))),
            'CUBIN_SHA256': meta['cubin_SHA256'], 'artifact_directory': str(folder.relative_to(c.ROOT))}
    gate = {'status': 'OK', 'metadata': raw['metadata'], 'static_audit': audits,
            'artifact_directory': str(root.relative_to(c.ROOT)),
            'GPU_allocated_for_compile': False, 'native_rebuilds': 0,
            'new_compiles': raw['new_compiles'], 'reused': raw['reused'],
            'protocol_SHA256': c.sha(c.encode(protocol))}
    c.derived(c.OUT/'compile_gate.json', gate)
    # Preserve only the frozen binary payload, not duplicate result JSONs.
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as output:
        for tag in gate['metadata']:
            folder = root/tag.replace(':', '/')
            for path in sorted(folder.glob('kernel.*')):
                output.write(path, path.relative_to(root).as_posix())
    c.derived(c.OUT/'compiled_payload.zip', buffer.getvalue())
    print('M2 CPU gate OK', len(gate['metadata']), 'programs; reused', raw['reused'])


def artifact_gate():
    protocol = c.protocol()
    gate = c.read(c.OUT/'compile_gate.json')
    roots = [path for path in (c.OUT/'eligibility').glob('attempt_*/raw.json') if c.read(path)['status'] == 'OK']
    assert roots, 'No successful correctness/resource census'
    path = max(roots, key=lambda path: int(path.parent.name.split('_')[-1]))
    raw = c.read(path)
    assert all(row['pass'] for row in raw['checks']) and not raw['Triton_imported']
    assert raw['driver_return_codes'] == [0]
    metadata = gate['metadata']
    selected = list(metadata)
    seen = {}
    aliases = {}
    tags = []
    for tag in selected:
        digest = metadata[tag]['cubin_SHA256']
        if digest in seen:
            aliases[tag] = seen[digest]
        else:
            seen[digest] = tag
            tags.append(tag)
    size = protocol['profiles']['group_limit']
    # Separate single-pass and loop programs: their grid/K Cartesian allocation
    # would otherwise reserve an unlaunched full-grid K64 output tensor.
    groups = []
    for single in (True, False):
        subset = [tag for tag in tags if bool(metadata[tag].get('original_once')) == single]
        groups += [subset[index:index+size] for index in range(0, len(subset), size)]
    result = {'status': 'OK', 'metadata': metadata, 'resources': raw['resources'],
        'environment': raw['environment'], 'checks': len(raw['checks']),
        'eligibility_path': str(path.relative_to(c.ROOT)), 'profile_groups': groups,
        'profile_aliases': aliases, 'protocol_SHA256': c.sha(c.encode(protocol)),
        'local_storage_boundaries': [tag for tag, row in raw['resources'].items() if row['attributes']['local_bytes']]}
    c.derived(c.OUT/'artifact_gate.json', result)
    print('M2 artifact gate OK', len(metadata), 'programs;', len(groups), 'NCU groups;',
          len(result['local_storage_boundaries']), 'local-storage boundaries')


def counters():
    gate = c.read(c.OUT/'artifact_gate.json')
    rows = []
    dynamic = []
    reports = []
    for index, group in enumerate(gate['profile_groups']):
        roots = [path for path in (c.OUT/'profiles'/f'group_{index}').glob('attempt_*/export')
                 if c.read(path/'result.json')['status'] == 'OK']
        assert roots, ('Missing successful NCU group', index)
        root = max(roots, key=lambda path: int(path.parent.name.split('_')[-1]))
        for mode in ('software', 'hardware'):
            folder = root/mode
            job = c.read(folder/'job.json')
            collect = c.read(folder/'collect.json')
            markers = re.findall(r'PROFILE_BINDING (\{[^\n]+\})', collect['output'])
            assert markers, folder
            import json
            binding = json.loads(markers[-1])
            assert not binding['Triton_imported'] and binding['compile_calls'] == 0
            assert len(binding['records']) == len(job['cells'])
            assert all(row['correctness']['pass'] for row in binding['records'])
            for cell, record in zip(job['cells'], binding['records']):
                assert all(cell[key] == record[key] for key in ('tag', 'grid', 'q', 'K'))
                assert record['binding']['CUBIN_SHA256'] == gate['metadata'][cell['tag']]['cubin_SHA256']
            reports.append({'path': str((folder/'report.ncu-rep').relative_to(c.ROOT)),
                'SHA256': c.sha((folder/'report.ncu-rep').read_bytes()), 'mode': mode,
                'launches': len(job['cells']), 'environment': binding['environment']})
            imported = list(csv.reader(io.StringIO(c.read(folder/'import.json')['output'])))
            head_index = next(i for i, row in enumerate(imported) if all(
                key in row for key in ('ID', 'Kernel Name', 'Grid Size', 'Block Size')))
            head = imported[head_index]
            launch_rows = {}
            for values in imported[head_index+1:]:
                if len(values) != len(head):
                    continue
                item = dict(zip(head, values))
                if item['ID'].isdigit():
                    launch_rows.setdefault(int(item['ID']), item)
            assert len(launch_rows) == len(job['cells'])
            for launch_id, cell in enumerate(job['cells']):
                item = launch_rows[launch_id]
                meta = gate['metadata'][cell['tag']]
                assert item['Kernel Name'] == meta['function_name']
                assert [int(value) for value in re.findall(r'\d+', item['Grid Size'])] == [cell['grid'], 1, 1]
                assert [int(value) for value in re.findall(r'\d+', item['Block Size'])] == [meta['num_warps']*32, 1, 1]
                assert item['CC'] == '9.0'
            if mode == 'software':
                blocks = source_blocks(c.read(folder/'source.json')['output'])
                assert len(blocks) == len(job['cells']), (folder, len(blocks), len(job['cells']))
                for cell, pcs in zip(job['cells'], blocks):
                    counts = {family: sum(row['count'] for row in pcs.values() if family in row['families']) for family in FAMILIES}
                    opcodes = Counter()
                    for row in pcs.values():
                        tokens = row['instruction'].split()
                        opcode = tokens[1] if tokens[0].startswith('@') else tokens[0]
                        opcodes[opcode] += row['count']
                    dynamic.append({**cell, 'counts': counts, 'opcode_counts': dict(opcodes),
                        'per_PC': pcs, 'group': index, 'GPU_UUID': binding['environment']['gpu_uuid']})
            else:
                measurements = hardware_rows(c.read(folder/'import.json')['output'])
                assert len(measurements) == len(job['cells'])
                for cell, measurement in zip(job['cells'], measurements):
                    rows.append({**cell, 'metrics': measurement, 'group': index,
                                 'GPU_UUID': binding['environment']['gpu_uuid']})
    affine, single_pass = [], []
    by_tag = defaultdict(list)
    for row in dynamic:
        by_tag[row['tag']].append(row)
    for tag, values in by_tag.items():
        values.sort(key=lambda row: row['K'])
        if gate['metadata'][tag].get('original_once'):
            assert len(values) == 1 and values[0]['K'] == 1
            single_pass.append(values[0])
            continue
        assert [row['K'] for row in values] == [16, 64]
        low, high = values
        components = {}
        for family in FAMILIES:
            slope = (high['counts'][family]-low['counts'][family])/48
            components[family] = {'per_iteration_warp_instructions': slope,
                'fixed_warp_instructions': low['counts'][family]-16*slope,
                'K16': low['counts'][family], 'K64': high['counts'][family]}
        affine.append({'tag': tag, 'families': components,
                       'reverse_broadcast_fixture': False})
    result = {'status': 'OK', 'hardware': rows, 'dynamic': dynamic, 'affine': affine,
              'single_pass': single_pass,
              'reports': reports, 'profile_aliases': gate['profile_aliases'],
              'instruction_scope': 'FULL/control: genuine loop-free K1 counts. MC: K16/K64 affine decomposition. Whole-kernel shared traffic includes input, reduction and output conversion; no unsupported attribution to ML alone.',
              'NCU_duration_used_for_timing': False}
    c.derived(c.OUT/'counter_gate.json', result)
    print('M2 counter gate OK', len(reports), 'reports;', len(dynamic), 'dynamic and', len(rows), 'hardware launches')


def services():
    protocol = c.protocol()
    gate = c.read(c.OUT/'artifact_gate.json')
    metadata = gate['metadata']
    records, oneshot, sources = [], [], []
    sample_count = 0
    for invocation in range(1, protocol['timing']['workers']+1):
        path = c.OUT/'timing'/f'invocation_{invocation}'/'raw.json'
        raw = c.read(path)
        assert raw['status']=='OK' and not raw['Triton_imported'] and raw['driver_return_codes']==[0]
        assert raw['protocol_SHA256']==c.sha(c.encode(protocol))
        sources.append({'path':str(path.relative_to(c.ROOT)), 'SHA256':c.sha(path.read_bytes()),
                        'environment':raw['environment']})
        grouped = defaultdict(dict)
        for row in raw['cells']:
            assert len(row['samples_us'])==protocol['timing']['samples_per_cell']
            assert row['correctness']['pass'] and all(value>0 for value in row['samples_us'])
            sample_count += len(row['samples_us'])
            for identity in [row['cell_id']]+row['aliases']:
                tag, qtext, gtext, ktext = identity.split('|')
                grouped[tag,qtext[1:]][int(gtext[1:]),int(ktext[1:])] = st.median(row['samples_us'])
        for (tag,qcond), values in grouped.items():
            meta = metadata[tag]
            capacity = gate['resources'][tag]['reservations'][qcond]['capacity']
            common = {'tag':tag,'q_condition':qcond,'q':capacity,'invocation':invocation,
                'GPU_UUID':raw['environment']['gpu_uuid'],'family':meta['family'],'kind':meta['kind'],
                'geometry':[meta['M'],meta['N'],meta['num_warps']],
                'cohort':'M2','source':str(path.relative_to(c.ROOT))}
            if not meta.get('original_once'):
                grids = [g*132 for g in protocol['timing']['MC_grid_SM_multipliers']]
                kfits = [fit(protocol['fit_K'],[values[g,k] for k in protocol['fit_K']]) for g in grids]
                gridfit = fit([g/132 for g in grids],[item['slope'] for item in kfits])
                setup = fit([g/132 for g in grids],[item['intercept'] for item in kfits])
                records.append({**common,'service_ns':1000*gridfit['slope'],
                    'K_fits':kfits,'grid_fit':gridfit,'setup_grid_fit':setup,
                    'K1_grid_fit':fit([g/132 for g in grids],[values[g,1] for g in grids]),
                    'global_K_slope_intercept_ns':1000*gridfit['intercept']})
            else:
                full = (8192//meta['M'])*(8192//meta['N'])
                grids = [full*factor for factor in protocol['timing']['full_grid_factors']]
                gridfit = fit([g/132 for g in grids],[values[g,1] for g in grids])
                oneshot.append({**common,'full_grid':full,'full_us':values[full,1],
                    'full_normalized_ns':1000*values[full,1]*132/full,
                    'tile_slope_ns':1000*gridfit['slope'],'global_intercept_us':gridfit['intercept'],
                    'grid_fit':gridfit,'grid_time_us':{str(g):values[g,1] for g in grids},
                    'footprint_scope':'128/256/384 MiB contiguous BF16 input; host descriptor extent 24576x8192'})
    result = {'status':'OK','services':records,'oneshot':oneshot,'timing_sources':sources,
        'event_samples':sample_count,'historical_offset_used':False,
        'M2_compiled':True,'M2_timed':True}
    c.derived(c.OUT/'analysis.json',result)
    print('M2 services OK',len(records),'MC slopes;',len(oneshot),'single-pass slopes;',sample_count,'events')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['compile', 'artifacts', 'counters', 'services'])
    args = parser.parse_args()
    {'compile': compile_gate, 'artifacts': artifact_gate, 'counters': counters, 'services': services}[args.action]()


if __name__ == '__main__':
    main()
