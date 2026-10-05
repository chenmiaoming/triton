# Phase9 StageB — Three architecture artifact closure

All108 original compiler attempts are retained. H100 and B200 admit18/18 pairs; RTX PRO6000 admits16/18. Four SM120 binaries export successfully but exceed the actual shared-memory opt-in limit; both large-tile pairs remain explicitly unavailable. No timing or outcome-based selection.

| Target | Actual GPU | Eligible pairs | Attempts | Artifact worker UUID |
| --- | --- | --- | --- | --- |
| sm90 | NVIDIA H100 80GB HBM3 | 18 | 36 | GPU-0fa257da-c7d7-2264-a9eb-3c794d64f2b1 |
| sm100 | NVIDIA B200 | 18 | 36 | GPU-05d546d7-e362-a9e7-dd31-cc8919ae676e |
| sm120 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 16 | 36 | GPU-311dde06-ef22-d676-6d31-818f6caec291 |

Exact Phase6 native core im-joNh6Ry3lq2oFqOues9VFh and native extension SHA reused for all targets. Persistent ccache counters remain unchanged during artifact collection. CUDA13.0.85 cuobjdump/nvdisasm overlay verifies official redistributable archive hashes; actual compiler ptxas is recorded separately per architecture. No native rebuild, compiler edit, or PR. Every original ZIP, uploaded source archive, source manifest, PTX/CUBIN/SASS/resource/occupancy query, untimed correctness smoke and failure is retained.
