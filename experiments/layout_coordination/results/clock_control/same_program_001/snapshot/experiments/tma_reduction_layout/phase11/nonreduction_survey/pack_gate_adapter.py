"""Pinned standalone FP8 packing and forward gate fixtures; CPU only on import."""
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def cases():
    result = []
    for heads, sequence in [(8, 2048), (32, 4096)]:
        result.append(dict(id=f'N04_h{heads}_s{sequence}_d128', family='N04',
            W=8, stages=3, H=heads, S=sequence, D=128, BM=64, BN=128,
            padded_queries=(sequence+127)//128*128, padded_keys=(sequence+31)//32*32,
            source_function='_pack_qkv', input_format='fused_token_major_qkv_views',
            caller_defaults='Original pack caller sets num_warps=8 and leaves num_stages at pinned CUDA default 3.',
            evidence_scope='Standalone fixed-external-scale SM120-origin FP8 pack. H100 is a portability experiment, not its original attention backend.'))
    configs = [dict(BT=bt, W=w, stages=s) for bt in [32, 64, 128]
               for w in [4, 8, 16, 32] for s in [2, 3]]
    for dim in [64, 128]:
        result.append(dict(id=f'N05_t2048_h8_d{dim}', family='N05',
            W=None, stages=None, BT=None, T=2048, H=8, D=dim, BN=dim,
            source_function='kda_gate_fwd_kernel', autotune_configs=copy.deepcopy(configs),
            autotune_resolved=False, FLA_USE_FAST_OPS='0',
            caller_defaults='Resolve the original full BT/W/stages set using U alone before G0 or GL; no layout timings select this configuration.',
            evidence_scope='Forward standalone KDA gate with A/bias, no beta, no chunk-cumsum or backward. FP32 output.'))
    return result


def upstream(case):
    if case['family'] == 'N04':
        from .sources.pack_original import _pack_qkv
        return _pack_qkv
    assert case['family'] == 'N05'
    from .sources.gate_original import kda_gate_fwd_kernel
    return kda_gate_fwd_kernel


def provenance():
    return json.loads((ROOT/'sources/pack_gate_provenance.json').read_text())


def _fixture_pack(case, pattern):
    import torch
    h, s, d = case['H'], case['S'], case['D']
    pq, pk = case['padded_queries'], case['padded_keys']
    generator = torch.Generator(device='cpu').manual_seed(20261105+h+s)
    projection = (torch.randn((s, 3*h*d), generator=generator)*0.5).to(torch.bfloat16)
    if pattern != 'finite':
        index = torch.arange(projection.numel(), dtype=torch.int64).reshape(projection.shape)
        values = ((index % 257)-128).float()/32
        if pattern == 'edge':
            edges = torch.tensor([-1000., -448., -32., -1.0625, -0.015625, -0., 0.,
                                  0.015625, 1.0625, 32., 448., 1000.], dtype=torch.float32)
            values = edges[index % edges.numel()]
        projection.copy_(values.to(torch.bfloat16))
    # Match the pinned DiT preparation test: slices of [S,3*H*D], then
    # [S,H,D] -> [H,S,D].  These read-only views share projection storage.
    q, k, v = [projection[:, i*h*d:(i+1)*h*d].view(s,h,d).permute(1,0,2) for i in range(3)]
    assert q.stride() == k.stride() == v.stride() == (d, 3*h*d, 1)
    scales = (0.031 + torch.arange(3)[:, None].float()*0.007
              + (torch.arange(h)[None, :] % 7).float()*0.002).to(torch.float32)

    def quantize(values, scale):
        return torch.clamp(torch.div(values.float(), scale[:, None, None]), -448., 448.).to(torch.float8_e4m3fn)

    expected_q = torch.zeros((h,pq,d), dtype=torch.float8_e4m3fn)
    expected_k = torch.zeros((h,pk,d), dtype=torch.float8_e4m3fn)
    expected_q[:, :s] = quantize(q, scales[0])
    expected_k[:, :s] = quantize(k, scales[1])
    # Independent explicit 16-token key-order table from the format contract;
    # no reuse of the kernel's arithmetic permutation expression.
    permutation = torch.tensor([0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15])
    source = (torch.arange(pk//16)[:, None]*16+permutation[None, :]).reshape(-1)
    gathered = torch.zeros((h,pk,d), dtype=torch.float32)
    valid = source < s
    gathered[:,valid] = v[:,source[valid]].float()
    expected_v = quantize(gathered, scales[2]).transpose(1,2).contiguous()

    def poison(shape):
        value = torch.empty(shape, dtype=torch.float8_e4m3fn)
        value.view(torch.uint8).fill_(0x7F)
        return value

    return dict(tensors=dict(q=q,k=k,v=v,scales=scales,
                q_fp8=poison((h,pq,d)),k_fp8=poison((h,pk,d)),v_fp8=poison((h,d,pk))),
        scalars=dict(sequence=s), constants=dict(head_stride=d,token_stride=3*h*d,
            padded_queries=pq,padded_keys=pk,heads=h,block_tokens=64),
        grid=(h,(pq+63)//64,1),outputs=['q_fp8','k_fp8','v_fp8'],inplace=[],
        expected=dict(q_fp8=expected_q,k_fp8=expected_k,v_fp8=expected_v),
        tolerances={name:dict(rtol=0.,atol=0.,bitwise=True) for name in ['q_fp8','k_fp8','v_fp8']},
        input_storage_aliases=[['q','k','v']],
        external_scale_policy='Predetermined finite positive FP32 scales, shared unchanged by U/G0/GL. No GPU scale computation.')


def _fixture_gate(case, pattern):
    import torch
    import torch.nn.functional as F
    t, h, d = case['T'], case['H'], case['D']
    generator = torch.Generator(device='cpu').manual_seed(20261105+t+h+d)
    g = (torch.randn((t,h,d), generator=generator)*1.25).to(torch.bfloat16)
    if pattern != 'finite':
        index = torch.arange(g.numel()).reshape(g.shape)
        g = (((index % 97)-48).float()/8).to(torch.bfloat16)
        if pattern == 'edge':
            values = torch.tensor([-80.,-20.,-4.,-0.125,-0.,0.,0.125,4.,19.875,20.,20.125,80.])
            g = values[index % values.numel()].to(torch.bfloat16)
    a_log = (torch.arange(h).float()-h//2)/16
    bias = ((torch.arange(h*d).reshape(h,d) % 31)-15).float()/32
    expected = (-torch.exp(a_log)[None,:,None]*F.softplus(g.float()+bias[None,:,:])).float()
    constants = dict(H=h,D=d,BD=1<<(d-1).bit_length(),HAS_A=True,HAS_BIAS=True,
                     HAS_BETA=False,USE_LOWER_BOUND=False)
    if case.get('BT') is not None:
        constants['BT'] = case['BT']
        grid = ((t+case['BT']-1)//case['BT'],h,1)
    else:
        grid = None  # Original U-only autotune must resolve BT/W/stages first.
    return dict(tensors=dict(g=g,A_log=a_log,dt_bias=bias.reshape(-1),
                yg=torch.full(g.shape,float('nan'),dtype=torch.float32)),
        scalars=dict(T=t,beta=None,yb=None,lower_bound=None),constants=constants,
        grid=grid,outputs=['yg'],inplace=[],expected=dict(yg=expected),
        tolerances=dict(yg=dict(rtol=1.e-5,atol=2.e-6,bitwise=False)),
        baseline_pair_bitwise_required=True,
        oracle_scope='Independent CPU FP32 exp/softplus reference; approximate NVIDIA elementwise helper tolerance is fixed before timing.')


def fixture(case, pattern='finite'):
    assert pattern in ['finite','structured','edge'], pattern
    return _fixture_pack(case,pattern) if case['family'] == 'N04' else _fixture_gate(case,pattern)


def derive_plan(ttgir, case):
    from ..operator_gpu.layout_parsing import encodings, tensor_encoding
    definitions = encodings(ttgir)
    lines = ttgir.splitlines()
    result = dict(family=case['family'],layouts=copy.deepcopy(definitions),
                  loads=[line for line in lines if 'tt.load ' in line],
                  stores=[line for line in lines if 'tt.store ' in line],
                  conversions=[line for line in lines if 'convert_layout' in line],
                  inline_asm=[line for line in lines if 'inline_asm' in line])
    if case['family'] == 'N05':
        bt,bd = case['BT'],1<<(case['D']-1).bit_length()
        load = next(line for line in result['loads'] if f'tensor<{bt}x{bd}x' in line)
        store = next(line for line in result['stores'] if f'tensor<{bt}x{bd}x' in line)
        bias = next(line for line in result['loads'] if f'tensor<{bd}x' in line)
        result.update(input=tensor_encoding(load,[bt,bd],definitions),
                      bias=tensor_encoding(bias,[bd],definitions),
                      output=tensor_encoding(store,[bt,bd],definitions))
    return result


def gluon(case):
    from . import pack_gate_gluon
    return getattr(pack_gate_gluon,case['source_function'])


def layout_constants(case,plan,variant='G0'):
    from .pack_gate_gluon import layout_constants as implementation
    return implementation(case,plan,variant)


def variants(case,plan):
    from .pack_gate_gluon import variants as implementation
    return implementation(case,plan)
