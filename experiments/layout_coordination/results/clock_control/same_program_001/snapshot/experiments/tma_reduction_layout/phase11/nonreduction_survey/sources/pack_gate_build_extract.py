"""Reproduce exact, dependency-minimal U extractions from pinned source files."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(text, node, decorators=True):
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) if decorators else node.lineno
    return textwrap.dedent('\n'.join(text.splitlines()[first-1:node.end_lineno])) + '\n'


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def body_fingerprint(node):
    return digest(ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False).encode())


def run():
    metadata = json.loads((ROOT/'pack_gate_source_metadata.json').read_text())
    for item in metadata:
        blob = (ROOT/item['local']).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest() == item['github_blob_SHA']
        item['SHA256'] = digest(blob)

    pack = (ROOT/'pack_fused_prep_pinned.py').read_text()
    pack_tree = ast.parse(pack)
    pack_names = ['_divide_rn', '_load_tile', '_quantize', '_pack_qkv']
    pack_out = '# SPDX-License-Identifier: Apache-2.0\n'
    pack_out += '"""Exact target/helper bodies extracted from the pinned SGLang source.\n\nOnly the standalone pack pass is imported; no amax/scales pass is included.\n"""\n'
    pack_out += 'import triton\nimport triton.language as tl\n\nBLOCK_TOKENS = 64\n\n'
    pack_out += '\n'.join(source(pack, function(pack_tree, name)) for name in pack_names)
    (ROOT/'pack_original.py').write_text(pack_out)

    gate = (ROOT/'gate_gate_pinned.py').read_text()
    soft = (ROOT/'gate_softplus_pinned.py').read_text()
    op = (ROOT/'gate_op_pinned.py').read_text()
    gate_tree, soft_tree, op_tree = map(ast.parse, [gate, soft, op])
    gate_out = '\n'.join(gate.splitlines()[:8]) + '\n'
    gate_out += '"""Exact forward-only U extraction; FLA_USE_FAST_OPS=0, NVIDIA softplus.\n\nThe original autotune configs are resolved separately using only U.\n"""\n'
    gate_out += 'import triton\nimport triton.language as tl\n\n'
    for name in ['_generate_softplus', '_generate_constraints']:
        gate_out += source(soft, function(soft_tree, name)) + '\n'
    assignment_names = ['_NUM_REG', 's_softplus', 's_constraints', 'NUM_REG']
    for name in assignment_names:
        node = next(n for n in soft_tree.body if (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        ) or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name))
        gate_out += source(soft, node) + '\n'
    gate_out += source(soft, function(soft_tree, 'softplus_nv')) + '\n'
    nvidia_branch = next(n for n in soft_tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'IS_NVIDIA')
    gate_out += source(soft, function(nvidia_branch, 'softplus')) + '\n'
    exp_branch = next(n for n in op_tree.body if isinstance(n, ast.If) and 'FLA_USE_FAST_OPS' in ast.unparse(n.test))
    exp_node = next(n for n in exp_branch.orelse if isinstance(n, ast.FunctionDef) and n.name == 'exp')
    gate_out += source(op, exp_node) + '\n'
    gate_node = function(gate_tree, 'kda_gate_fwd_kernel')
    gate_out += "@triton.jit(do_not_specialize=['T'])\n" + source(gate, gate_node, decorators=False)
    (ROOT/'gate_original.py').write_text(gate_out)

    extracted = []
    for group, original, names in [('pack', pack_tree, pack_names), ('gate', gate_tree, ['kda_gate_fwd_kernel'])]:
        text = (ROOT/(group+'_original.py')).read_text()
        tree = ast.parse(text)
        records = []
        for name in names:
            a, b = function(original, name), function(tree, name)
            assert ast.dump(a.args) == ast.dump(b.args)
            assert body_fingerprint(a) == body_fingerprint(b)
            forbidden = [n.func.attr for n in ast.walk(b) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                         and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tl'
                         and n.func.attr in ['sum', 'max', 'min', 'reduce', 'cumsum', 'cumprod', 'dot', 'atomic_add', 'atomic_max']]
            assert not forbidden, (name, forbidden)
            records.append(dict(function=name, original_line=a.lineno, body_SHA256=body_fingerprint(a),
                                exact_AST_body=True, exact_signature=True))
        extracted.append(dict(group=group, local=group+'_original.py', SHA256=digest(text.encode()),
                              functions=records))
    result = dict(status='OK', CPU_only=True, GPU_allocations=0, sources=metadata, extractions=extracted,
                  gate_helper_policy='Exact NVIDIA softplus/softplus_nv, pack=1; exact FLA default exp with FLA_USE_FAST_OPS=0.',
                  removed_host_decorators=['triton.heuristics', 'fla_cache_autotune'],
                  autotune_policy='All original BT/W/stages configurations are exported by the adapter for U-only resolution before G0.')
    (ROOT/'pack_gate_provenance.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('Exact U extraction: pack4 + gate1 functions; original Git blobs and AST bodies verified')


if __name__ == '__main__':
    run()
