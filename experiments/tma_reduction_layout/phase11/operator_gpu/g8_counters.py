"""Offline G8 NCU re-import, exact-binary aliases and current PC attribution.

All subprocesses here import an existing report or disassemble a frozen CUBIN.
No GPU is allocated and no kernel or compiler is launched.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
from . import common as c
from .g5_analyze import elf
from .offline_import import hardware_rows, source_blocks
from experiments.tma_reduction_layout.phase11.c2.analyze import FAMILIES

OUT = c.OUT/'stage_g8'
NCU = '/usr/local/cuda/bin/ncu'
NVDISASM = '/usr/local/cuda/bin/nvdisasm'
EXTRA_FAMILIES = {
    'global_load': r'\bLDG(?:\.|\s)',
    'local_load': r'\bLDL(?:\.|\s)',
    'local_store': r'\bSTL(?:\.|\s)',
}


def save(path, value):
    blob = value if isinstance(value, bytes) else c.encode(value)
    if path.exists():
        assert path.read_bytes() == blob, path
    else:
        c.write_once(path, blob)


def profile_protocol_path():
    return OUT/'profile_protocol.json'


def collection_specs():
    """Keep preregistered and post-timing diagnostic provenance separate."""
    original = profile_protocol_path()
    supplement = OUT/'profile_protocol_supplement.json'
    assert supplement.exists(), 'Freeze and finish the six supplemental diagnostics before final counter analysis'
    return [dict(path=original,directory=OUT/'profiles',evidence_role='preregistered_remeasurement'),
            dict(path=supplement,directory=OUT/'profiles_supplement',evidence_role='post_timing_exploratory_diagnostic')]


def opcode(instruction):
    tokens = instruction.strip().split()
    if not tokens:
        return ''
    # nvdisasm keeps the terminator on operand-free instructions (NOP;),
    # whereas NCU's source export omits it (NOP).
    return (tokens[1] if tokens[0].startswith('@') else tokens[0]).rstrip(';')


def static_pc_map(text, entry):
    """Read this CUBIN's own debug lines, retaining inline-callsite records."""
    active = False; mapping = {}; current = []; pending = []; in_source_run = False
    for line in text.splitlines():
        section = re.match(r'\s*\.section\s+\.text\.([^,\s]+)', line)
        if section:
            active = section[1] == entry
            current = []; pending = []; in_source_run = False
            continue
        if re.match(r'\s*\.section\s+', line) and not section:
            active = False
        if not active:
            continue
        location = re.search(r'//## File "([^"]+)", line (\d+)(.*)', line)
        if location:
            if not in_source_run:
                pending = []
            record = dict(file=location[1], line=int(location[2]), raw=line.strip())
            inline = re.findall(r'inlined at "([^"]+)", line (\d+)', location[3])
            if inline:
                record['inlined_at'] = [dict(file=name, line=int(number)) for name, number in inline]
            pending.append(record); in_source_run = True
            continue
        instruction = re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*)', line)
        if instruction:
            if in_source_run:
                current = pending; pending = []; in_source_run = False
            pc = f'0x{int(instruction[1],16):04x}'
            assert pc not in mapping, (entry, pc)
            primary = current[-1] if current else dict(file='unknown', line=0)
            mapping[pc] = dict(instruction=instruction[2].strip(), file=primary['file'],
                               line=primary['line'], source_records=list(current))
    assert mapping, ('No static instruction map', entry)
    return mapping


def abi(meta, artifacts):
    ptx = (artifacts/meta['artifact_directory']/'kernel.ptx').read_text()
    match = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
    assert match, meta['tag']
    return re.findall(r'\.param\s+\.(\w+)', match[1])


def resolve_alias(tag, bytag, case):
    if tag in bytag:
        return bytag[tag]
    group, name = tag.split(':', 1)
    canonical = next((v['name'] for v in case['layouts'] if name in v.get('aliases', [])), None)
    assert canonical is not None, ('Unknown logical alias', tag)
    return bytag[group+':'+canonical]


def audit_aliases(selected, group, bytag, artifacts, protocol):
    """An alias may reuse counters only for equal code, ABI, case and modes."""
    meta = selected['metadata']; binary = artifacts/meta['artifact_directory']/'kernel.cubin'
    digest = c.sha(binary.read_bytes()); assert digest == meta['cubin_SHA256']
    original_sections = elf(binary.read_bytes()); original_abi = abi(meta, artifacts)
    rows = []
    for logical in selected.get('aliases', []):
        other = resolve_alias(logical, bytag, group['case'])
        other_binary = artifacts/other['artifact_directory']/'kernel.cubin'
        assert c.sha(other_binary.read_bytes()) == other['cubin_SHA256'] == digest
        assert elf(other_binary.read_bytes()) == original_sections
        assert other['runtime_argument_names'] == meta['runtime_argument_names']
        assert abi(other, artifacts) == original_abi
        assert other['geometry'] == meta['geometry'] == group['case']['id']
        for key in ['num_warps', 'dynamic_smem_bytes', 'global_scratch_size']:
            assert other[key] == meta[key], (logical, key)
        alias_group, alias_name = logical.split(':', 1)
        selector = next(s for s in protocol['profile_selectors'] if s['group'] == alias_group and alias_name in s['names'])
        requested_modes = list(selector['modes'])
        assert set(requested_modes) <= set(group['modes'])
        rows.append(dict(alias=logical, resolved_tag=other['tag'], representative=meta['tag'],
            CUBIN_SHA256=digest, full_CUBIN_equal=True, allocated_sections_equal=True,
            runtime_names_and_parameter_types_equal=True, launch_resources_equal=True,
            geometry=meta['geometry'], requested_modes=requested_modes))
    return rows


def local_import(folder, report):
    path = folder/'report.ncu-rep'; digest = c.sha(path.read_bytes())
    assert digest == report['SHA256']
    destination = OUT/'offline_imports'/(folder.relative_to(OUT).as_posix().replace('/', '_')+'.json')
    commands = [[NCU, '--import', str(path), '--page', 'raw', '--csv', '--print-units', 'base', '--print-fp']]
    if report['mode'] == 'software':
        commands.append([NCU, '--import', str(path), '--page', 'source', '--print-source', 'sass', '--csv'])
    if destination.exists():
        value = c.read(destination)
    else:
        outputs = []
        for argv in commands:
            process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            assert process.returncode == 0, process.stdout[-3000:]
            outputs.append(dict(argv=argv, returncode=process.returncode, output=process.stdout))
        value = dict(CPU_only=True, GPU_allocations=0, report_SHA256=digest,
                     report_path=str(path.relative_to(c.ROOT)), commands=outputs)
        save(destination, value)
    assert value['report_SHA256'] == digest and value['CPU_only']
    assert len(value['commands']) == len(commands)
    assert all(cmd['returncode'] == 0 for cmd in value['commands'])
    remote = hardware_rows(c.read(folder/'import.json')['output'])
    assert hardware_rows(value['commands'][0]['output']) == remote
    assert len(remote) == 1
    if report['mode'] == 'software':
        assert source_blocks(value['commands'][1]['output']) == source_blocks(c.read(folder/'source.json')['output'])
    return dict(path=str(destination.relative_to(c.ROOT)), SHA256=c.sha(destination.read_bytes()),
                report_SHA256=digest, values_identical=True, source_identical=report['mode']=='software')


def disassemble(meta, artifacts):
    path = artifacts/meta['artifact_directory']/'kernel.cubin'
    digest = c.sha(path.read_bytes()); assert digest == meta['cubin_SHA256']
    destination = OUT/'source_correlation'/(digest+'.json')
    if destination.exists():
        result = c.read(destination)
    else:
        argv = [NVDISASM, '-gi', '-c', str(path)]
        process = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        assert process.returncode == 0, process.stdout[-3000:]
        result = dict(CPU_only=True, CUBIN_SHA256=digest, argv=argv,
                      returncode=process.returncode, output=process.stdout)
        save(destination, result)
    assert result['CUBIN_SHA256'] == digest and result['returncode'] == 0
    return static_pc_map(result['output'], meta['function_name']), dict(
        path=str(destination.relative_to(c.ROOT)), SHA256=c.sha(destination.read_bytes()))


def main():
    original_protocol = c.read(profile_protocol_path()); c.verify_sources(original_protocol)
    collections = []
    for spec in collection_specs():
        protocol = c.read(spec['path']); c.verify_sources(protocol)
        if spec['evidence_role'] == 'post_timing_exploratory_diagnostic':
            assert protocol['original_profile_protocol_SHA256'] == c.sha(c.encode(original_protocol))
            assert protocol['evidence_role'] == spec['evidence_role'] and not protocol['confirmatory']
            assert not protocol['formal_timing_or_selector_changes']
            for name,digest in protocol['post_timing_source_SHA256'].items():
                assert c.sha((c.ROOT/name).read_bytes()) == digest,name
            for name,digest in protocol['selection_evidence_SHA256'].items():
                assert c.sha((c.ROOT/name).read_bytes()) == digest,name
        collections.append({**spec,'protocol':protocol,'SHA256':c.sha(c.encode(protocol))})
    protocol = original_protocol
    gate = c.read(OUT/'compile_gate.json'); assert gate['status'] == 'OK'
    payload_path = OUT/'compiled_payload.zip'; payload = payload_path.read_bytes()
    assert c.sha(payload) == gate['payload_SHA256'] == protocol['payload_SHA256']
    assert protocol['compile_gate_SHA256'] == c.sha(c.encode(gate))
    artifacts = Path('/tmp')/('g8-counter-artifacts-'+c.sha(payload)[:16])
    if not artifacts.exists():
        c.extract(payload, artifacts)
    bytag = {meta['tag']:meta for meta in gate['records']}; assert len(bytag) == len(gate['records'])
    rows = []; jobs = []; reports = []; all_aliases = []; expected = set(); source_cache = {}
    for collection,group in [(collection,group) for collection in collections for group in collection['protocol']['profile_groups']]:
        protocol = collection['protocol']
        assert protocol['payload_SHA256'] == c.sha(payload)
        assert protocol['compile_gate_SHA256'] == c.sha(c.encode(gate))
        root = collection['directory']/group['id']/'attempt_001/export'
        result = c.read(root/'result.json')
        assert result['status'] == 'OK' and result['group'] == group['id']
        assert result['protocol_SHA256'] == c.sha(c.encode(protocol)) and result['payload_SHA256'] == c.sha(payload)
        selections = {s['metadata']['tag']:s for s in group['records']}
        targets = {(tag, mode) for tag in selections for mode in group['modes']}
        observed = [(r['tag'],r['mode']) for r in result['reports']]
        assert len(observed) == len(set(observed)) == len(targets) and set(observed) == targets
        expected |= {(group['id'],tag,mode) for tag,mode in targets}
        aliases = {tag:audit_aliases(selected,group,bytag,artifacts,protocol) for tag,selected in selections.items()}
        all_aliases.extend(item for values in aliases.values() for item in values)
        for report in result['reports']:
            folder = root/Path(report['path']).parent; meta = bytag[report['tag']]
            assert meta == selections[report['tag']]['metadata']
            report_path = folder/'report.ncu-rep'; assert c.sha(report_path.read_bytes()) == report['SHA256']
            binding = c.read(folder/'binding.json')
            if 'binding_SHA256' in report:
                assert c.sha((folder/'binding.json').read_bytes()) == report['binding_SHA256']
            assert not binding['Triton_imported'] and binding['compile_calls'] == 0
            assert binding['driver_return_codes'] == [0] and len(binding['records']) == 1
            actual = binding['records'][0]
            assert actual['tag'] == report['tag'] and actual['correctness']['pass']
            assert actual['binding']['loaded_SHA256'] == actual['binding']['cubin_SHA256'] == meta['cubin_SHA256']
            metrics = hardware_rows(c.read(folder/'import.json')['output']); assert len(metrics) == 1
            ctas = 1
            for size in actual['grid']:
                ctas *= int(size)
            assert ctas > 0
            row = dict(**report, group=group['id'], geometry=group['case']['id'],
                       evidence_role=collection['evidence_role'],
                       collection_protocol_path=str(collection['path'].relative_to(c.ROOT)),
                       collection_protocol_SHA256=collection['SHA256'],
                       report_path=str(report_path.relative_to(c.ROOT)), metrics=metrics[0],
                       metrics_per_CTA={name:value/ctas for name,value in metrics[0].items() if name.endswith('.sum')},
                       CTAs=ctas, counts=dict(CTAs=ctas), resources=actual['binding']['attributes'],
                       environment=binding['environment'], binding=binding,
                       alias_audit=aliases[report['tag']],
                       logical_aliases_for_this_mode=[a['alias'] for a in aliases[report['tag']] if report['mode'] in a['requested_modes']])
            if report['mode'] == 'software':
                blocks = source_blocks(c.read(folder/'source.json')['output']); assert len(blocks) == 1
                dynamic = blocks[0]
                digest = meta['cubin_SHA256']
                if digest not in source_cache:
                    source_cache[digest] = disassemble(meta, artifacts)
                static, source_record = source_cache[digest]
                joined = {}; lines = defaultdict(lambda:dict(dynamic_instructions=0,families=Counter(),opcodes=Counter()))
                counts = {name:0 for name in [*FAMILIES,*EXTRA_FAMILIES]}; opcodes = Counter()
                for pc,item in dynamic.items():
                    assert pc in static, (meta['tag'],pc)
                    assert opcode(static[pc]['instruction']) == opcode(item['instruction']), (meta['tag'],pc,item,static[pc])
                    families = list(item['families']) + [name for name,pattern in EXTRA_FAMILIES.items() if re.search(pattern,item['instruction'])]
                    instruction_opcode = opcode(item['instruction']); opcodes[instruction_opcode] += item['count']
                    for family in families: counts[family] += item['count']
                    location = static[pc]
                    joined[pc] = {**item,'families':families,'source':location}
                    key = location['file']+':'+str(location['line']); summary = lines[key]
                    summary['dynamic_instructions'] += item['count']; summary['opcodes'][instruction_opcode] += item['count']
                    for family in families: summary['families'][family] += item['count']
                row.update(per_PC=joined, opcode_counts=dict(opcodes), per_source_line=dict(lines),
                           source_correlation=source_record, all_executed_PC_sites_mapped=True,
                           instruction_count_unit='dynamic warp instruction instances',
                           source_attribution_scope='Actual frozen CUBIN debug lines and inline records; not phase latency or causal decomposition.')
                row['counts'].update(counts)
                row['counts_per_CTA'] = {name:value/ctas for name,value in counts.items()}
            rows.append(row); jobs.append((folder,report))
            reports.append(dict(path=str(report_path.relative_to(c.ROOT)),SHA256=report['SHA256'],
                                group=group['id'],tag=report['tag'],mode=report['mode'],
                                evidence_role=collection['evidence_role'],
                                collection_protocol_SHA256=collection['SHA256']))
    assert len(rows) == len(expected)
    with ThreadPoolExecutor(max_workers=4) as pool:
        imported = list(pool.map(lambda args:local_import(*args), jobs))
    collection_summary = [dict(path=str(item['path'].relative_to(c.ROOT)),SHA256=item['SHA256'],
        directory=str(item['directory'].relative_to(c.ROOT)),evidence_role=item['evidence_role'],
        report_count=sum(len(g['records'])*len(g['modes']) for g in item['protocol']['profile_groups'])) for item in collections]
    assert [item['report_count'] for item in collection_summary] == [167,6]
    result = dict(status='OK', CPU_only=True, GPU_allocations=0, protocol_SHA256=c.sha(c.encode(original_protocol)),
                  collection_protocols=collection_summary,
                  payload_SHA256=c.sha(payload), rows=rows, reports=reports, local_reimports=imported,
                  alias_audit=all_aliases, report_count=len(rows),
                  local_ncu_version=subprocess.check_output([NCU,'--version'],text=True),
                  local_nvdisasm_version=subprocess.check_output([NVDISASM,'--version'],text=True),
                  metric_normalization='Only additive .sum counters are divided by the actual profiled grid CTA count; pct/ratio/timing metrics remain unchanged.',
                  scope='Dynamic counts and debug-line attribution describe the frozen programs; they do not by themselves assign latency or prove a mechanism.')
    save(OUT/'counter_analysis.json', result)
    print('G8 NCU analysis:',len(rows),'reports; all local re-imports and current-CUBIN PC mappings verified')


if __name__ == '__main__':
    main()
