"""Extract unchanged public kernel bodies; add only explicit Gluon layouts."""
import ast
from pathlib import Path
from . import common as c


SOURCES = {
    'vllm': {'commit': '76c9e23ce7dd4646616e3b282d6263cc47ac17a6',
             'url': 'https://github.com/vllm-project/vllm',
             'path': 'vllm/v1/attention/ops/triton_unified_attention.py'},
    'sglang': {'commit': 'c2e574246485a055c0e51d60d1eae027e5d629c0',
              'url': 'https://github.com/sgl-project/sglang',
              'path': 'python/sglang/kernels/ops/attention/decode_attention.py'},
}


def functions(text, names):
    nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    lines = text.splitlines(keepends=True)
    return '\n'.join(''.join(lines[min([n.lineno for n in nodes[name].decorator_list] +
                                     [nodes[name].lineno]) - 1:nodes[name].end_lineno]) for name in names)


class ToGluon(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'tl':
            node.id = 'gl'
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'triton' and node.attr == 'jit':
            node.value.id = 'gluon'
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in ('reduce_segments', '_fwd_kernel_stage1'):
            node.args.args.append(ast.arg(arg='L', annotation=ast.Attribute(value=ast.Name(id='gl', ctx=ast.Load()),
                                                                           attr='constexpr', ctx=ast.Load())))
            node.args.defaults.append(ast.Constant(value=None))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            return node
        if node.func.value.id != 'gl' or node.func.attr not in ('arange', 'zeros', 'full'):
            return node
        expression = ast.unparse(node.args[1] if node.func.attr == 'arange' else node.args[0])
        row = 'NUM_SEGMENTS_PER_SEQ' in expression or 'BLOCK_N' in expression
        col = any(key in expression for key in ('HEAD_SIZE_PADDED', 'BLOCK_DMODEL', 'BLOCK_DV'))
        if row or col:
            layout = ast.parse(f'gl.SliceLayout({1 if row else 0}, L)', mode='eval').body
            node.keywords.append(ast.keyword(arg='layout', value=layout))
        return node


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'gl':
            node.id = 'tl'
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [key for key in node.keywords if key.arg != 'layout']
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = []
        if node.args.args and node.args.args[-1].arg == 'L':
            node.args.args.pop()
            node.args.defaults.pop()
        return node


def parity(original, explicit):
    def normalize(text):
        return {node.name: ast.dump(Normalize().visit(node), include_attributes=False)
                for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    assert normalize(original) == normalize(explicit), 'Original / Gluon body semantics changed'
    return sorted(normalize(original))


def main():
    original = {name: Path('/tmp/operator-gpu-' + name + '.py').read_bytes() for name in SOURCES}
    helper = Path('/tmp/operator-gpu-vllm-helpers.py').read_bytes()
    for name, blob in original.items():
        c.write_once(c.SOURCE / 'upstream' / (name + '.py.txt'), blob)
    c.write_once(c.SOURCE / 'upstream/vllm_helpers.py.txt', helper)
    vllm = functions(helper.decode(), ('cdiv_fn', 'find_seq_idx')) + '\n' + functions(original['vllm'].decode(), ('reduce_segments',))
    sglang = functions(original['sglang'].decode(), ('tanh', '_fwd_kernel_stage1'))
    stage2 = functions(original['sglang'].decode(), ('_fwd_kernel_stage2',))
    pairs = {}
    for name, body in (('vllm', vllm), ('sglang', sglang)):
        header = ('# SPDX-License-Identifier: Apache-2.0\n'
                  '# Extracted from the pinned upstream source; see upstream/bindings.json.\n'
                  'import triton\nimport triton.language as tl\n'
                  'from types import SimpleNamespace\nfloat8_info = SimpleNamespace(min=-448., max=448.)\n\n')
        c.write_once(c.SOURCE / (name + '_triton.py'), (header + body + '\n' + (stage2 if name == 'sglang' else '')).encode())
        tree = ast.fix_missing_locations(ToGluon().visit(ast.parse(body)))
        explicit = ast.unparse(tree) + '\n'
        pairs[name] = parity(body, explicit)
        gl_header = ('# SPDX-License-Identifier: Apache-2.0\n'
                     '# Layout-only derivative of the pinned upstream source.\n'
                     'from triton.experimental import gluon\n'
                     'from triton.experimental.gluon import language as gl\n'
                     'from types import SimpleNamespace\nfloat8_info = SimpleNamespace(min=-448., max=448.)\n\n')
        c.write_once(c.SOURCE / (name + '_gluon.py'), (gl_header + explicit).encode())
    c.write_once(c.SOURCE / 'upstream/bindings.json', {'sources': SOURCES,
                 'source_SHA256': {name: c.sha(blob) for name, blob in original.items()},
                 'helper_SHA256': c.sha(helper), 'AST_expression_and_control_flow_parity': pairs,
                 'preserved': ['original kernel bodies', 'online softmax', 'paged-index loads',
                               'sequence lookup', 'early returns', 'output precision'],
                 'environment_binding': 'dependency-free extraction under frozen Phase 11 Triton; not installation of full serving libraries'})
    print('BOUND public source bodies:', pairs)


if __name__ == '__main__':
    main()
