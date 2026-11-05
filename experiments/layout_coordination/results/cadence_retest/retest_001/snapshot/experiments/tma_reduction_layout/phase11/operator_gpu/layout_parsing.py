"""Read actual canonical load/store encodings; never assume a default vector width."""
import re


def encodings(ir):
    result = {}
    for name, body in re.findall(r'(#\w+)\s*=\s*#ttg\.blocked<\{(.*?)\}>', ir, re.S):
        fields = {}
        for key, values in re.findall(r'(sizePerThread|threadsPerWarp|warpsPerCTA|order)\s*=\s*\[([^]]+)\]', body):
            key = {'sizePerThread': 'size_per_thread', 'threadsPerWarp': 'threads_per_warp',
                   'warpsPerCTA': 'warps_per_cta', 'order': 'order'}[key]
            fields[key] = [int(value.strip()) for value in values.split(',')]
        result[name] = {'kind': 'blocked', 'fields': fields}
    return result


def resolve(text, definitions):
    text = text.strip()
    if text in definitions:
        return definitions[text]
    match = re.fullmatch(r'#ttg\.slice<\{dim = (\d+), parent = (#\w+)\}>', text)
    assert match, ('Unsupported encoding', text)
    return {'kind': 'slice', 'dim': int(match[1]), 'parent': definitions[match[2]]}


def tensor_encoding(line, dimensions, definitions):
    match = re.search(r'tensor<' + 'x'.join(map(str, dimensions)) + r'x.*?,\s*(#ttg\.slice<\{[^}]+\}>|#\w+)>', line)
    assert match, line
    return resolve(match[1], definitions)


def plan(ir, case):
    definitions = encodings(ir)
    bm, bn = case['BM'], case['BN']
    lines = ir.splitlines()
    matrices = [line for line in lines if 'tt.load' in line and f'tensor<{bm}x{bn}x' in line]
    output = next(line for line in lines if 'tt.store' in line and f'tensor<{bn}x!tt.ptr<' in line)
    result = {'output': tensor_encoding(output, [bn], definitions)}
    if case['family'] == 'C02':
        k = next(line for line in matrices if re.search(r'%k(?:_\d+)?\s*=', line))
        v = next(line for line in matrices if re.search(r'%v(?:_\d+)?\s*=', line))
        q = next(line for line in lines if 'tt.load' in line and re.search(r'%q(?:_\d+)?\s*=', line))
        result.update(K=tensor_encoding(k, [bm, bn], definitions),
                      V=tensor_encoding(v, [bm, bn], definitions),
                      Q=tensor_encoding(q, [bn], definitions))
    else:
        result['V'] = tensor_encoding(matrices[0], [bm, bn], definitions)
        stats = [line for line in lines if 'tt.load' in line and '%segm_max_' in line]
        # Triton may duplicate a stat load into the matrix row slice and a
        # contiguous 1D encoding. Keep the latter as a fixed stats control.
        selected = next((line for line in stats if '#ttg.slice' not in line), stats[0])
        result['statistics'] = tensor_encoding(selected, [bm], definitions)
        result['canonical_stat_load_encodings'] = [tensor_encoding(line, [bm], definitions) for line in stats]
    return result


def instantiate(spec, gl):
    if spec['kind'] == 'blocked':
        return gl.BlockedLayout(**spec['fields'])
    return gl.SliceLayout(spec['dim'], instantiate(spec['parent'], gl))
