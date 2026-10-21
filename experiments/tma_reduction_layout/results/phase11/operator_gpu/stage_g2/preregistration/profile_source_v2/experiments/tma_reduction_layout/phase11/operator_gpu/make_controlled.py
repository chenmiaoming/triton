"""Hold QK/statistics/output encodings fixed while varying the column-reduction tile."""
import ast
from . import common as c
from .bind_sources import ToGluon, Normalize, functions


def expression(text):
    return ast.parse(text, mode='eval').body


class Names(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements

    def visit_Name(self, node):
        return expression(self.replacements[node.id]) if node.id in self.replacements else node


class Controlled(ToGluon):
    def __init__(self, family):
        self.family = family

    def visit_FunctionDef(self, node):
        super().visit_FunctionDef(node)
        if node.name not in ('reduce_segments', '_fwd_kernel_stage1'):
            return node
        for name in (('LSTAT', 'LOUT') if self.family == 'C06' else ('LK', 'LQ', 'LOUT')):
            node.args.args.append(ast.arg(arg=name, annotation=expression('gl.constexpr')))
            node.args.defaults.append(ast.Constant(value=None))
        return node

    def visit_Call(self, node):
        super().visit_Call(node)
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'gl':
            return node
        for key in node.keywords:
            if key.arg != 'layout':
                continue
            shape = ast.unparse(node.args[1] if node.func.attr == 'arange' else node.args[0])
            if self.family == 'C06' and 'NUM_SEGMENTS_PER_SEQ' in shape:
                key.value = expression('LSTAT')
            if self.family == 'C02':
                if 'BLOCK_DMODEL' in shape:
                    key.value = expression('LQ')
                elif 'BLOCK_N' in shape:
                    key.value = expression('gl.SliceLayout(1, LK)')
        if node.func.attr == 'load':
            source = ast.unparse(node.args[0])
            for key in node.keywords:
                if key.arg != 'mask':
                    continue
                if self.family == 'C02' and 'K_Buffer' in source:
                    key.value = Names({'mask_d': 'gl.convert_layout(mask_d, gl.SliceLayout(0, LK))'}).visit(key.value)
                if self.family == 'C02' and 'V_Buffer' in source:
                    key.value = Names({'offs_n': 'gl.convert_layout(offs_n, gl.SliceLayout(1, L))'}).visit(key.value)
                if self.family == 'C06' and 'segm_output_ptr' in source:
                    key.value = Names({'segm_mask': 'gl.convert_layout(segm_mask, gl.SliceLayout(1, L))'}).visit(key.value)
        if self.family == 'C02' and node.func.attr == 'sum' and 'v' in {
                n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name)}:
            node.args[0] = Names({'p': 'gl.convert_layout(p, gl.SliceLayout(1, L))'}).visit(node.args[0])
        if node.func.attr == 'store' and any(name in ast.unparse(node.args[0]) for name in ('Att_Out', 'output_ptr')):
            for i in (0, 1):
                node.args[i] = ast.Call(func=expression('gl.convert_layout'), args=[node.args[i], expression('LOUT')], keywords=[])
            for key in node.keywords:
                if key.arg == 'mask':
                    key.value = ast.Call(func=expression('gl.convert_layout'), args=[key.value, expression('LOUT')], keywords=[])
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if self.family == 'C02':
            if 'offs_buf_k' in names:
                node.value = Names({'offs_d': 'gl.convert_layout(offs_d, gl.SliceLayout(0, LK))'}).visit(node.value)
            if 'offs_buf_v' in names:
                node.value = Names({key: f'gl.convert_layout({key}, gl.SliceLayout(1, L))'
                                    for key in ('kv_loc', 'page_id', 'tok_in_p')}).visit(node.value)
            if 'qk' in names:
                node.value = Names({'q': 'gl.convert_layout(q, gl.SliceLayout(0, LK))'}).visit(node.value)
        elif 'segm_output_offset' in names:
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'arange' and 'NUM_SEGMENTS_PER_SEQ' in ast.unparse(call):
                    for key in call.keywords:
                        if key.arg == 'layout':
                            key.value = expression('gl.SliceLayout(1, L)')
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if self.family == 'C06' and isinstance(node.target, ast.Name) and node.target.id == 'segm_output':
            node.value = Names({'segment_scale': 'gl.convert_layout(segment_scale, gl.SliceLayout(1, L))'}).visit(node.value)
        return node


class RemoveInterventions(Normalize):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'convert_layout':
            return node.args[0]
        node.keywords = [key for key in node.keywords if key.arg != 'layout']
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = []
        while node.args.args[-1].arg in ('L', 'LK', 'LQ', 'LSTAT', 'LOUT'):
            node.args.args.pop()
            node.args.defaults.pop()
        return node


def main():
    parity = {}
    for family, module, names in (('C06', 'vllm', ('cdiv_fn', 'find_seq_idx', 'reduce_segments')),
                                  ('C02', 'sglang', ('tanh', '_fwd_kernel_stage1'))):
        original = functions((c.SOURCE / (module + '_triton.py')).read_text(), names)
        transformed = ast.unparse(ast.fix_missing_locations(Controlled(family).visit(ast.parse(original))))
        def normalized(text):
            return {node.name: ast.dump(RemoveInterventions().visit(node), include_attributes=False)
                    for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
        assert normalized(original) == normalized(transformed)
        header = ('# SPDX-License-Identifier: Apache-2.0\n'
                  '# Controlled layout derivative of the frozen upstream body.\n'
                  'from triton.experimental import gluon\nfrom triton.experimental.gluon import language as gl\n'
                  'from types import SimpleNamespace\nfloat8_info = SimpleNamespace(min=-448., max=448.)\n\n')
        c.write_once(c.SOURCE / (module + '_controlled.py'), (header + transformed + '\n').encode())
        parity[family] = sorted(normalized(original))
    c.write_once(c.SOURCE / 'upstream/controlled_parity.json', {'matched_functions': parity,
                 'scope': 'same expressions/control flow after removing explicit layout operations; not lowering equivalence'})
    print('PASS controlled expression parity', parity)


if __name__ == '__main__':
    main()
