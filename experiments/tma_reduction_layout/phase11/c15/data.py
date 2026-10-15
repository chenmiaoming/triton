"""C1.5 oracle: BF16 private update first, single-use widening then max."""
import numpy as np
from experiments.tma_reduction_layout.phase11.c1.data import bf16_round, tiles, input_tensor, compare
from experiments.tma_reduction_layout.phase11.c1.data import reference as prior_reference


def reference(case, k, pids, pattern='quarter'):
    if case['family'] == 'ML':
        return prior_reference(case, k, pids, pattern)
    m, n = case['M'], case['N']
    state = tiles(m, n, pids, pattern=pattern)
    step = ((np.arange(m)[:, None] * 13 + np.arange(n)[None, :] * 7) % 7 + 1).astype(np.float32) * .25
    step = bf16_round(step)
    result = np.zeros((len(state), n), dtype=np.float32)
    for _ in range(k):
        # Both arithmetic arms return BF16 before select. Their exact FP32
        # promotion is only for the CPU oracle, not the Gluon source program.
        down, up = bf16_round(state - np.float32(16)), bf16_round(state + step)
        state = np.where(state >= np.float32(8), down, up)
        result = result + state.max(axis=1)
    return result
