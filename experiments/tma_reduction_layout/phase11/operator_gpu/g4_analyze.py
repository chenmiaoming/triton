"""Paired factorial/binding contrasts and independent offline NCU re-import."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import io
import json
import re
import subprocess
import numpy as np
from . import common as c
from .g4_prepare import out, protocol_path, gate_path, payload_path
from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows, source_blocks, FAMILIES


def save(path, value):
    blob = value if isinstance(value, bytes) else c.encode(value)
    if path.exists(): assert path.read_bytes() == blob, path
    else: c.write_once(path, blob)


def timing(stage):
    target = out(stage); p = c.read(protocol_path(stage)); c.verify_sources(p)
    paired = defaultdict(dict); total = 0; raw_paths = []; environments = []
    for invocation in range(1, 4):
        path = target / f'timing/invocation_{invocation}/raw.json'; raw = c.read(path)
        assert raw['status'] == 'OK' and not raw['Triton_imported'] and raw['compile_calls'] == 0
        assert raw['protocol_SHA256'] == c.sha(c.encode(p)) and raw['driver_return_codes'] == [0]
        assert all(x['pass'] for x in raw['checks'])
        raw_paths.append({'path': str(path.relative_to(c.ROOT)), 'SHA256': c.sha(path.read_bytes())})
        environments.append(raw['environment'])
        for cell in raw['cells']:
            assert len(cell['visits']) == 5 and cell['complete_output_check']['pass']
            for v in cell['visits']:
                assert len(v['samples_us']) == 31 and all(np.isfinite(x) and x > 0 for x in v['samples_us'])
                assert sorted(v['node_types']) == [0] * (2 if cell['mode'] == 'decode' else 1) + [7, 7]
                total += len(v['samples_us'])
            values = [float(np.median(v['samples_us'])) for v in cell['visits']]
            for tag in [cell['tag']] + cell['aliases']:
                key = (cell['group'], cell['batch'], cell['length'], cell['mode'], tag)
                paired[key][invocation] = values
    arrays = {key: np.asarray([value[i] for i in range(1, 4)]) for key, value in paired.items()}
    rows, contrasts = [], []
    for work in sorted({k[:4] for k in arrays}):
        prefix = work[0] + (':actual:' if stage == 'g4b' else ':')
        u = arrays[(*work, prefix + 'upstream')]; g0 = arrays[(*work, prefix + 'gluon_anchor')]
        rng = np.random.default_rng(20261024 + int(c.sha(c.encode(work))[:8], 16))
        wi = rng.integers(0, 3, (4096, 3)); vi = rng.integers(0, 5, (4096, 3, 5))
        def bootstrap(x): return x[wi[:, :, None], vi].mean((1, 2))
        ub, gb = bootstrap(u), bootstrap(g0)
        for key in sorted(k for k in arrays if k[:4] == work):
            v = arrays[key]; vb = bootstrap(v)
            rows.append({'group': work[0], 'batch': work[1], 'length': work[2], 'mode': work[3], 'tag': key[-1],
                'time_us': float(v.mean()), 'worker_visit_medians_us': v.tolist(),
                'gain_vs_U_percent': float(100 * (u.mean() - v.mean()) / u.mean()),
                'gain_vs_G0_percent': float(100 * (g0.mean() - v.mean()) / g0.mean()),
                'gain_vs_U_CI95': np.percentile(100 * (ub - vb) / ub, [2.5, 97.5]).tolist(),
                'gain_vs_G0_CI95': np.percentile(100 * (gb - vb) / gb, [2.5, 97.5]).tolist()})
        if stage == 'g4a':
            a, b, ab = [arrays[(*work, prefix + label)] for label in ('pv_only', 'qk_only', 'coordinated')]
            interaction = ab - a - b + g0
            contrasts.append({'group': work[0], 'batch': work[1], 'length': work[2], 'mode': work[3],
                'contrast': 'coordinated - pv_only - qk_only + anchor',
                'interaction_us': float(interaction.mean()),
                'interaction_CI95_us': np.percentile(bootstrap(interaction), [2.5, 97.5]).tolist()})
        else:
            for name in ('upstream', 'gluon_anchor', 'target'):
                actual = arrays[(*work, prefix + name)]; old = arrays[(*work, work[0] + ':old_B4:' + name)]
                contrasts.append({'group': work[0], 'batch': work[1], 'length': work[2], 'mode': work[3],
                    'contrast': name + ' actual-vs-B4', 'gain_percent': float(100 * (old.mean() - actual.mean()) / old.mean()),
                    'gain_CI95': np.percentile(100 * (bootstrap(old) - bootstrap(actual)) / bootstrap(old), [2.5, 97.5]).tolist()})
    result = {'status': 'OK', 'rows': rows, 'contrasts': contrasts, 'raw_samples': total,
        'logical_conditions': len(rows), 'raw_sources': raw_paths, 'environments': environments,
        'estimator': 'mean of 3 worker x 5 visit medians; paired hierarchical bootstrap 4096',
        'CI_scope': 'exploratory pointwise intervals; no multiplicity correction; no new selector fitting'}
    save(target / 'timing_analysis.json', result)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(target / 'all_timings.csv', stream.getvalue().encode())
    print(stage, 'timing PASS', total, 'samples;', len(rows), 'logical conditions')


def counters(stage):
    assert stage == 'g4a'
    target = out(stage); p = c.read(protocol_path(stage)); c.verify_sources(p)
    rows, jobs = [], []
    metadata = {r['tag']: r for r in c.read(gate_path(stage))['records']}
    for group in p['groups']:
        root = target / 'profiles' / group['id'] / 'attempt_001/export'; result = c.read(root / 'result.json')
        assert result['status'] == 'OK' and result['protocol_SHA256'] == c.sha(c.encode(p))
        for report in result['reports']:
            folder = root / Path(report['path']).parent
            assert c.sha((folder / 'report.ncu-rep').read_bytes()) == report['SHA256']
            found = re.findall(r'PROFILE_BINDING (\{[^\n]+\})', c.read(folder / 'collect.json')['output']); assert found
            binding = json.loads(found[-1]); item = binding['records'][0]
            assert len(binding['records']) == 1 and item['correctness']['pass']
            assert not binding['Triton_imported'] and binding['compile_calls'] == 0 and binding['driver_return_codes'] == [0]
            assert item['binding']['cubin_SHA256'] == item['binding']['loaded_SHA256']
            assert item['tag'] == report['tag'] and item['binding']['cubin_SHA256'] == metadata[report['tag']]['cubin_SHA256']
            metrics = hardware_rows(c.read(folder / 'import.json')['output']); assert len(metrics) == 1
            row = {**report, 'group': group['id'], 'metrics': metrics[0],
                   'CTAs': int(np.prod(item['grid'])), 'resources': item['binding']['attributes'],
                   'environment': binding['environment'], 'report_path': str((folder / 'report.ncu-rep').relative_to(c.ROOT))}
            if report['mode'] == 'software':
                pc = source_blocks(c.read(folder / 'source.json')['output']); assert len(pc) == 1
                row['per_PC'] = pc[0]
                row['counts'] = {f: sum(x['count'] for x in pc[0].values() if f in x['families']) for f in FAMILIES}
                row['opcode_counts'] = dict(Counter({}))
                for x in pc[0].values():
                    match = re.search(r'(?:^|\s)([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)(?:\s|;)', x['instruction'])
                    if match: row['opcode_counts'][match[1]] = row['opcode_counts'].get(match[1], 0) + x['count']
            rows.append(row); jobs.append((folder, report['mode']))
    def one(args):
        folder, mode = args
        dest = target / 'offline_imports' / (folder.relative_to(target).as_posix().replace('/', '_') + '.json')
        commands = [['/usr/local/cuda/bin/ncu', '--import', str(folder / 'report.ncu-rep'), '--page', 'raw', '--csv', '--print-units', 'base', '--print-fp']]
        if mode == 'software': commands.append(['/usr/local/cuda/bin/ncu', '--import', str(folder / 'report.ncu-rep'), '--page', 'source', '--print-source', 'sass', '--csv'])
        if dest.exists(): value = c.read(dest)
        else:
            outputs = []
            for argv in commands:
                process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                assert process.returncode == 0, process.stdout[-1000:]
                outputs.append({'argv': argv, 'returncode': process.returncode, 'output': process.stdout})
            value = {'CPU_only': True, 'report_SHA256': c.sha((folder / 'report.ncu-rep').read_bytes()), 'commands': outputs}; save(dest, value)
        assert value['report_SHA256'] == c.sha((folder / 'report.ncu-rep').read_bytes())
        assert hardware_rows(value['commands'][0]['output']) == hardware_rows(c.read(folder / 'import.json')['output'])
        if mode == 'software': assert source_blocks(value['commands'][1]['output']) == source_blocks(c.read(folder / 'source.json')['output'])
        return {'path': str(dest.relative_to(c.ROOT)), 'values_identical': True, 'SHA256': c.sha(dest.read_bytes())}
    with ThreadPoolExecutor(max_workers=4) as pool: imports = list(pool.map(one, jobs))
    save(target / 'counter_analysis.json', {'status': 'OK', 'rows': rows, 'local_reimports': imports,
         'local_ncu_version': subprocess.check_output(['/usr/local/cuda/bin/ncu', '--version'], text=True)})
    print(stage, 'NCU PASS', len(rows), 'reports, all locally re-imported')


def code(stage):
    target = out(stage); gate = c.read(gate_path(stage)); rows = []
    for meta in gate['records']:
        if meta.get('provenance') != 'CPU_actual_ABI' or meta['stage2']: continue
        folder = c.ROOT / gate['CPU_export_directory'] / meta['artifact_directory']
        ir = (folder / 'kernel.ttgir').read_text(); definitions = [x for x in ir.splitlines() if x.startswith('#')]
        operations = [x.strip() for x in ir.splitlines() if any(s in x for s in ('ttg.convert_layout', 'tt.reduce', 'tt.load', 'tt.store'))]
        command = ['/usr/local/cuda/bin/nvdisasm', '-g', '-c', str(folder / 'kernel.cubin')]
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        assert process.returncode == 0, process.stdout[-1000:]
        dest = target / 'source_correlation_v2' / (meta['tag'].replace(':', '_') + '.json')
        save(dest, {'CPU_only': True, 'argv': command, 'output': process.stdout,
                    'CUBIN_SHA256': meta['cubin_SHA256'], 'returncode': process.returncode})
        rows.append({'tag': meta['tag'], 'CUBIN_SHA256': meta['cubin_SHA256'], 'variant': meta['variant'],
            'encoding_definitions': definitions, 'load_reduce_convert_store': operations,
            'has_line_correlation': bool(re.search(r'Line \d+|line \d+', process.stdout)),
            'source_correlation_path': str(dest.relative_to(c.ROOT)),
            'note': 'Static operation plan and source correlation are not a hardware latency or cost sum.'})
    save(target / 'code_audit_v2.json', {'status': 'OK', 'rows': rows})
    print(stage, 'CPU source correlation PASS', len(rows), 'programs')


def attribution(stage):
    """Join executed PC counts to inline-aware source attribution, never to latency."""
    assert stage == 'g4a'
    target = out(stage); gate = c.read(gate_path(stage))
    metas = {r['tag']: r for r in gate['records']}
    rows = []
    for row in c.read(target / 'counter_analysis.json')['rows']:
        if row['mode'] != 'software' or row['tag'].endswith(':upstream'): continue
        meta = metas[row['tag']]
        path = c.ROOT / gate['CPU_export_directory'] / meta['artifact_directory'] / 'kernel.cubin'
        argv = ['/usr/local/cuda/bin/nvdisasm', '-gi', '-c', str(path)]
        process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        assert process.returncode == 0
        source = ('unknown', 0); mapping = {}
        for text in process.stdout.splitlines():
            loc = re.search(r'//## File "([^"]+)", line (\d+)', text)
            if loc: source = (loc[1], int(loc[2])); continue
            pc = re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*)', text)
            if pc: mapping[f'0x{int(pc[1],16):04x}'] = {'file': source[0], 'line': source[1], 'instruction': pc[2]}
        joined, lines = {}, defaultdict(lambda: {'dynamic_instructions': 0, 'families': defaultdict(int), 'opcodes': defaultdict(int)})
        for pc, item in row['per_PC'].items():
            assert pc in mapping, (row['tag'], pc)
            loc = mapping[pc]; joined[pc] = {**item, 'source': loc}
            key = loc['file'] + ':' + str(loc['line']); cell = lines[key]
            cell['dynamic_instructions'] += item['count']
            for family in item['families']: cell['families'][family] += item['count']
            opcode = re.search(r'(?:^|\s)([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)(?:\s|;)', item['instruction'])
            if opcode: cell['opcodes'][opcode[1]] += item['count']
        save(target / 'inline_source_correlation' / (row['tag'].replace(':','_') + '.json'),
             {'CPU_only': True, 'argv': argv, 'output': process.stdout, 'CUBIN_SHA256': meta['cubin_SHA256']})
        rows.append({'tag': row['tag'], 'per_PC': joined, 'per_source_line': dict(lines),
                     'all_executed_PC_sites_mapped': True,
                     'scope': 'Compiler debug attribution, including inline callers; compiler motion/fusion can blur source boundaries. Counts do not assign phase latency or prove a critical path.'})
    save(target / 'source_attribution.json', {'status': 'OK', 'rows': rows})
    print(stage, 'inline-aware source attribution PASS', len(rows), 'controlled programs')


def plans(stage):
    from .layout_parsing import plan
    target = out(stage); gate = c.read(gate_path(stage)); p = c.read(protocol_path(stage))
    groups = {g['id']: g for g in p['groups']}; rows = []
    def fanout(spec, axis):
        fields = spec['fields']
        return {'lanes': fields['threads_per_warp'][axis], 'warps': fields['warps_per_cta'][axis],
                'threads': fields['threads_per_warp'][axis]*fields['warps_per_cta'][axis]}
    for meta in gate['records']:
        if meta['provenance'] != 'CPU_actual_ABI' or meta['stage2']: continue
        group = groups[meta['group']]; case = group['case']; base = gate['plans'][group['id']]
        root = c.ROOT/gate['CPU_export_directory']/meta['artifact_directory']
        actual = plan((root/'kernel.ttgir').read_text(),case)
        expected = dict(base)
        if not meta['variant'].get('upstream'):
            for role, field in [('pv','V'),('qk','K')]:
                if role not in meta['variant'] or field not in base: continue
                name = meta['variant'][role]
                if name != 'default':
                    expected[field] = {'kind':'blocked','fields':next(x['layout'] for x in case['layouts'] if x['name']==name or name in x.get('aliases',[]))}
        assert all(actual[k]==v for k,v in expected.items() if k in ('K','V','Q','output','statistics')),meta['tag']
        item = {'tag':meta['tag'],'effective_plan':actual,'requested_plan_matches':True,
                'PV_reduction_fanout':fanout(actual['V'],0)}
        if 'K' in actual:item['QK_reduction_fanout']=fanout(actual['K'],1)
        rows.append(item)
    save(target/'effective_layout_audit.json',{'status':'OK','rows':rows})
    print(stage,'effective layout plan PASS',len(rows),'programs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['g4a', 'g4b'])
    parser.add_argument('action', choices=['timing', 'counters', 'code', 'attribution', 'plans']); args = parser.parse_args()
    globals()[args.action](args.stage)
