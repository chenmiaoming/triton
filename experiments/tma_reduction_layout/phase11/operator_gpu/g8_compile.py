"""CPU-only source-aligned candidate compilation, preserving exact frozen ABIs."""
import copy
import inspect
from pathlib import Path
import subprocess
import traceback
from . import common as c
from .g6_compile import arguments
from .layout_parsing import instantiate
from .production_compile import export


def layout(case,name):
    return {'kind':'blocked','fields':next(x['layout'] for x in case['layouts']
        if x['name']==name or name in x.get('aliases',[]))}


def run(p,reference_payload,root):
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    c.verify_sources(p);root=Path(root);refs=root/'reference_files';c.extract(reference_payload,refs)
    assert c.sha(reference_payload)==p['reference_payload_SHA256']
    assert c.sha(Path(native.__file__).read_bytes())==c.NATIVE
    assert c.sha(Path(inspect.getfile(CUDABackend)).read_bytes())==p['backend_SHA256']
    result=dict(status='FAILED',records=[],GPU_allocated=False,native_rebuilds=0,
        native_SHA256=c.NATIVE,backend_SHA256=p['backend_SHA256'],compiler_intervention=False,
        protocol_SHA256=c.sha(c.encode(p)),ccache_before=subprocess.check_output(['ccache','--print-stats'],text=True))
    reused={};cache={};jobs=[]
    for g in p['inference_groups']:
        for v in g['variants']:jobs.append((g['id'],g['case'],v.get('form_id',g['form_id']),v,g['panel']))
        if g['case']['family']=='C02':jobs.append((g['id'],g['case'],g['case']['id'],dict(name='stage2',helper=True),g['panel']))
    for case in p['training_cases']:
        for v in case['variants']:jobs.append((case['id'],case,case['id'],v,'g3b'))
        jobs.append((case['id'],case,case['id'],dict(name='dx',helper=True),'g3b'))
    def reuse(key):
        if key not in reused:
            meta=copy.deepcopy(p['reference_records'][key])
            for name,digest in meta['artifact_SHA256'].items():
                data=(refs/meta['artifact_directory']/name).read_bytes();assert c.sha(data)==digest
                c.write_once(root/meta['artifact_directory']/name,data)
            reused[key]=meta
        return copy.deepcopy(reused[key])
    for gid,case,form,v,panel in jobs:
        tag=gid+':'+v['name'];row=dict(tag=tag,status='FAILED')
        try:
            item=p['items'][form];native_out=v.get('native',False)
            pv=None if v.get('pv','default')=='default' else layout(case,v['pv'])
            qk=None if v.get('qk','default')=='default' else layout(case,v['qk'])
            if v.get('helper'):
                meta=reuse(case['id']+':'+v['name'])
            elif v.get('upstream'):
                meta=reuse(form+':U')
            else:
                fn,const,attrs=arguments(item,gl,pv)
                if qk is not None:const['LK']=instantiate(qk,gl)
                if case['family']=='C01':const['NATIVE_OUT']=native_out
                key=c.sha(c.encode(dict(family=case['family'],sig=item['record']['JIT_signature'],
                    constants={n:str(value) for n,value in const.items()},attrs=item['record']['JIT_attributes'],
                    W=case['W'],stages=case['num_stages'])))
                if key not in cache:
                    code=triton.compile(gluon.GluonASTSource(fn,dict(item['record']['JIT_signature']),const,attrs),
                        target=GPUTarget('cuda',90,32),options={'num_warps':case['W'],'num_stages':case['num_stages']})
                    cache[key]=export(code,fn,case,'program_'+str(len(cache)),root,const)
                meta=copy.deepcopy(cache[key])
            meta.update(tag=tag,group=gid,geometry=case['id'],family=case['family'],form_id=form,panel=panel,
                variant=v,role='dx' if v['name']=='dx' else 'merge' if case['family']=='C01' else 'stage2' if v['name']=='stage2' else 'stage1',
                stage2=v['name']=='stage2',stage2_tag=gid+':stage2',native_output=native_out,
                requested_input=pv or item['plan'].get('input',item['plan'].get('V')),
                requested_K=qk or item['plan'].get('K'),requested_output=item['plan']['output'])
            row.update(status='OK',metadata=meta);result['records'].append(meta)
        except Exception:
            row['error']=traceback.format_exc();c.write_once(root/'errors'/(tag.replace(':','_')+'.txt'),row['error'].encode())
            result.setdefault('failures',[]).append(row);print(row['error'],flush=True)
        print('G8 CPU',tag,row['status'],flush=True)
    result['ccache_after']=subprocess.check_output(['ccache','--print-stats'],text=True)
    assert result['ccache_before']==result['ccache_after']
    result.update(status='OK' if not result.get('failures') else 'FAILED',new_unique_Gluon_programs=len(cache),reused_reference_programs=len(reused))
    c.write_once(root/'result.json',result);return result
