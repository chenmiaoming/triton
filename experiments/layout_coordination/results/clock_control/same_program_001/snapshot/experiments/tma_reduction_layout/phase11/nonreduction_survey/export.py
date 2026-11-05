"""Standalone immutable artifact exporter; same export body as G8. No kernel imports."""
from pathlib import Path
import subprocess
from . import common as c


def export(compiled, fn, case, tag, root, constants, is_stage2=False):
    folder = Path(root) / 'artifacts' / tag.replace(':', '_')
    hashes = {}
    for ext in ('ttir', 'glir', 'ttgir', 'llir', 'ptx', 'cubin'):
        if ext not in compiled.asm:
            continue
        blob = compiled.asm[ext]
        blob = blob if isinstance(blob, bytes) else blob.encode()
        c.write_once(folder / ('kernel.' + ext), blob)
        hashes['kernel.' + ext] = c.sha(blob)
    for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
        blob = subprocess.check_output(['cuobjdump', flag, str(folder / 'kernel.cubin')])
        c.write_once(folder / ('kernel.' + ext), blob)
        hashes['kernel.' + ext] = c.sha(blob)
    src = compiled.src
    specialized = {path[0]: value for path, value in src.constants.items() if len(path) == 1}
    runtime_names = [name for i, name in enumerate(fn.arg_names) if i not in specialized]
    runtime_constants = {fn.arg_names[i]: value for i, value in specialized.items()
                         if fn.arg_names[i] not in constants and isinstance(value, (int, float, bool, type(None)))}
    meta = {'tag': tag, 'geometry': case['id'], 'family': case['family'], 'num_warps': case['W'],
            'stage2': is_stage2, 'artifact_directory': str(folder.relative_to(root)),
            'artifact_SHA256': hashes, 'cubin_SHA256': c.sha(compiled.asm['cubin']),
            'function_name': compiled.name, 'compile_key': compiled.hash,
            'dynamic_smem_bytes': compiled.metadata.shared,
            'global_scratch_size': compiled.metadata.global_scratch_size,
            'runtime_argument_names': runtime_names, 'specialized_runtime_values': runtime_constants,
            'JIT_signature': src.signature, 'JIT_constants': {str(key): str(value) for key, value in src.constants.items()},
            'JIT_attributes': {str(key): value for key, value in src.attrs.items()}}
    assert meta['global_scratch_size'] == 0
    c.write_once(folder / 'metadata.json', meta)
    return meta
