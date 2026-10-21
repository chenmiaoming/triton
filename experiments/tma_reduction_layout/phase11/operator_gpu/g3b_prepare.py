"""Pre-register real tutorial dispatch sizes, all legal layouts and pipeline timing."""
from datetime import datetime,timezone
from pathlib import Path
import ast,subprocess
from . import common as c
from experiments.tma_reduction_layout.phase11.operator_survey.cases import layouts

OUT=c.OUT/'stage_g3b'
SNAPSHOT=Path('/tmp/operator-gpu-g3b-snapshot-v1')
PIN='20d9d98fead78c2f65e1b3ba5a7881d86a836dad'
PROTOCOL_FILE='protocol_v2.json'


def main():
    source=subprocess.check_output(['git','show',PIN+':python/tutorials/05-layer-norm.py'])
    parsed=ast.parse(source)
    extracted=ast.parse((c.SOURCE/'g3b_triton.py').read_text())
    for name in ['_layer_norm_bwd_dx_fused','_layer_norm_bwd_dwdb']:
        a=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in extracted.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
    upstream=OUT/'preregistration/upstream_05-layer-norm.py'
    if upstream.exists():assert upstream.read_bytes()==source
    else:c.write_once(upstream,source)
    cases=[]
    for n,groups in [(1024,256),(4096,128),(8192,96)]:
        cases.append({'id':f'C01_p{groups}_n{n}','family':'C01','P':groups,'N':n,'BM':32,'BN':128,'W':4,
                      'num_stages':3,'layouts':layouts(32,128,4),'pipeline_M':[128,2048]})
    old=c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    names=list(old['worker_source_SHA256'])
    names+=[p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.glob('g3b_*.py')]
    hashes=c.snapshot(names,SNAPSHOT)
    c.write_once(OUT/'preregistration/source_snapshot_v2.zip',c.archive(SNAPSHOT))
    protocol={'stage':'G3b','frozen_at_UTC':datetime.now(timezone.utc).isoformat(),
      'parent_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'target':'H100!','profile':'chenmiaoming','compiler':PIN,'native_SHA256':c.NATIVE,
      'source_url':f'https://github.com/triton-lang/triton/blob/{PIN}/python/tutorials/05-layer-norm.py',
      'upstream_source_SHA256':c.sha(source),'original_functions_AST_equal':True,'cases':cases,
      'dtype':'BF16 inputs, partial buffers and final gradients; FP32 accumulators, as tutorial wrapper',
      'variants':'upstream, fixed-output Gluon anchor, 7 unique vec/placement layouts, native-output vec1 and vec8 diagnostics',
      'timing':{'workers':3,'visits':5,'warmups':8,'samples':31,'seed':20261022,'practical_threshold_percent':3},
      'correctness':{'patterns':['finite','quarter','cancellation'],'BF16_rtol':.008,'BF16_atol':.001,
         'merge_reference':'FP64 CPU sum of the exact BF16 partial inputs, then round BF16',
         'pipeline_reference':'independent dX formula; final gradients checked against FP64 sum of the exact partial buffers produced by original stage1',
         'canaries':256,'all_inputs_unchanged':True},
      'pipeline':'zero locks, DW and DB; original dx_fused then candidate merge; mean/rstd are available backward inputs; device-only, not a training step',
      'profile_variants':['upstream','gluon_anchor','vec1:n_first','vec2:within','vec8:within','vec8:n_first:native'],
      'profile_geometries':['C01_p256_n1024','C01_p96_n8192'],'profile_modes':['software','warm_application'],
      'worker_source_SHA256':hashes,'native_rebuilds':0,'no_version_regression_claim':True,
      'old_G2_manifest_SHA256':c.sha((c.OUT/'stage_g3a/preregistration/protocol.json').read_bytes())}
    protocol.update(revision=2,prior_protocol_SHA256=c.sha((OUT/'preregistration/protocol.json').read_bytes()),
        revision_reason='CPU-only tools image lacks Triton; use the existing compiler image. No training GPU experiment has started.')
    c.write_once(OUT/'preregistration'/PROTOCOL_FILE,protocol)
    print('G3b frozen:',len(cases),'dispatches; real tutorial body and bounded original scratch groups')


if __name__=='__main__':main()
