"""Freeze a CPU-only preparation manifest; importing this module allocates no GPU."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from .cases import cases, layouts, ownership

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[3]
PIN = '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'
PIN_NATIVE = 'fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(data)


class StripLayouts(ast.NodeTransformer):
    """Check equal expressions/control flow after removing explicit layout syntax."""
    def visit_Name(self, node):
        if node.id == 'gl':
            node.id = 'tl'
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg != 'layout']
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = []
        node.args.args = [arg for arg in node.args.args if arg.arg != 'L']
        return node


def semantic_parity():
    bodies = []
    for name in ('kernels_triton.py', 'kernels_gluon.py'):
        tree = ast.parse((SOURCE / name).read_text())
        functions = {n.name: ast.dump(StripLayouts().visit(n), include_attributes=False)
                     for n in tree.body if isinstance(n, ast.FunctionDef)}
        bodies.append(functions)
    if bodies[0] != bodies[1]:
        differences = [name for name in set(bodies[0]) | set(bodies[1])
                       if bodies[0].get(name) != bodies[1].get(name)]
        raise ValueError(f'Triton/Gluon expression parity failed: {differences}')
    return {'matched_function_pairs': sorted(bodies[0]), 'pass': True,
            'scope': 'AST expressions and control flow only; not numerical/lowering equivalence'}


def manifest():
    catalog = json.loads((SOURCE / 'candidates.json').read_text())
    originals = [r['id'] for r in catalog['candidates'] if r['origin'] == 'original_five']
    if originals != ['C01', 'C02', 'C03', 'C04', 'C05']:
        raise ValueError('Original five candidates were dropped or reordered')
    rows = cases()
    counts = {}
    for row in rows:
        row['layouts'] = layouts(row['BM'], row['BN'], row['W'])
        for variant in row['layouts']:
            variant['ownership'] = ownership(row['BM'], row['BN'], row['W'], variant['layout'])
        if row['kernel'] in ('rms_backward', 'recurrent_gdn') and row['BN'] != row['N']:
            raise ValueError('Incomplete hidden/K reduction')
        if row['kernel'] in ('segment_merge', 'column_quant') and row['BM'] < row['M']:
            raise ValueError('Incomplete segment/quantization reduction')
        counts[row['id']] = 1 + len(row['layouts'])
    return {
        'stage': 'operator_survey_preparation', 'schema_version': 1,
        'frozen_at_UTC': datetime.now(timezone.utc).isoformat(),
        'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'canonical semantic fixtures plus CPU compile feasibility; not an upstream benchmark',
        'cases': rows, 'base_compile_programs': sum(counts.values()), 'program_counts': counts,
        'automatic_input_layout_anchors': 'add after parsing canonical Triton TTGIR; full bridge remains a GPU gate',
        'semantic_parity': semantic_parity(),
        'primary_future_comparison': ['upstream_original', 'same_layout_gluon_bridge', 'gluon_layout_variants'],
        'production_adapters_ready': False, 'GPU_execution_allowed_by_this_entrypoint': False,
        'GPU_correctness_verified': False, 'timing_collected': False,
        'formal_compiler': {'upstream_commit': PIN, 'native_SHA256': PIN_NATIVE},
        'local_fallback': 'installed 3.7.1 is development-only; never merge it into pinned evidence',
        'C03_fixture_recipe': 'dynamic per-column E4M3FN with FP32 scales and column-major output; not E8M0 MXFP8',
        'source_SHA256': {str(p.relative_to(SOURCE)): sha(p.read_bytes()) for p in sorted(SOURCE.iterdir())
                          if p.is_file() and p.suffix in ('.py', '.md', '.json')},
        'future_gpu_gate': ['pinned toolchain', 'production or subgraph scope labeled',
                            'original library source/dispatch captured', 'complete-output correctness and canaries',
                            'same-layout bridge inspected', 'immutable cubin bindings', 'separate GPU authorization'],
        'stop': 'CPU preparation only; do not start GPU correctness, timing, profiling, model fitting or PR',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preparation outputs are write-once; choose a new directory')
    value = manifest()
    for name in value['source_SHA256']:
        write_once(args.output / 'source' / name, (SOURCE / name).read_bytes())
    write_once(args.output / 'protocol.json', value)
    print(f'PREPARED {len(value["cases"])} specializations / {value["base_compile_programs"]} base programs; GPU work disabled')


if __name__ == '__main__':
    main()
