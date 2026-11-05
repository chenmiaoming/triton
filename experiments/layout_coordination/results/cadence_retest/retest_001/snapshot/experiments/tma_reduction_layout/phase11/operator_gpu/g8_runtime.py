"""Reuse proven driver harnesses on newly compiled source-aligned candidates."""
from pathlib import Path
from . import common as c


def run(p,gate,payload,root,invocation=0):
    from .g4_runtime import run as inference
    from .g3b_runtime import run as training
    root=Path(root)
    a=inference({**p,'groups':p['inference_groups']},
        {'records':[r for r in gate['records'] if r['family']!='C01']},payload,root.name,invocation)
    a['protocol_SHA256']=c.sha(c.encode(p));c.write_once(root/'inference.json',a)
    b=training({**p,'cases':p['training_cases'],'correctness':p['training_correctness']},
        {'records':[r for r in gate['records'] if r['family']=='C01']},payload,root/'training',invocation)
    b['protocol_SHA256']=c.sha(c.encode(p))
    result=dict(status='OK' if a['status']==b['status']=='OK' else 'FAILED',invocation=invocation,
        inference=a,training=b,protocol_SHA256=c.sha(c.encode(p)),payload_SHA256=c.sha(payload))
    c.write_once(root/'result.json',result);return result
