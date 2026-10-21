"""Read-only baseline audit: pass transitions, complete layouts, ABI and executable bytes."""
from pathlib import Path
import argparse,difflib,gzip,re,struct
from . import common as c
from .g5_prepare import OUT


def save(path,value):
    blob=value if isinstance(value,bytes) else c.encode(value)
    if path.exists():assert path.read_bytes()==blob,path
    else:c.write_once(path,blob)


def elf(blob):
    assert blob[:5]==b'\x7fELF\x02' and blob[5]==1
    start=struct.unpack_from('<Q',blob,40)[0];size,count,index=struct.unpack_from('<HHH',blob,58)
    h=struct.unpack_from('<IIQQQQIIQQ',blob,start+size*index);names=blob[h[4]:h[4]+h[5]]
    rows=[]
    for i in range(count):
        h=struct.unpack_from('<IIQQQQIIQQ',blob,start+size*i)
        name=names[h[0]:].split(b'\0',1)[0].decode()
        if h[2]&2:
            rows.append({'name':re.sub(r'(?<=\.)[^.]+$', 'ENTRY',name) if name.startswith(('.text.','.nv.constant0.','.nv.shared.')) else name,
                         'type':h[1],'flags':h[2],'size':h[5],
                         'SHA256':c.sha(blob[h[4]:h[4]+h[5]]) if h[1]!=8 else None})
    return sorted(rows,key=lambda x:x['name'])


def strip_locations(text):
    # Debug locations contain nested callsite/argument-name parentheses.
    pieces=[];cursor=0
    for match in re.finditer(r'\s+loc\(',text):
        if match.start()<cursor:continue
        pieces.append(text[cursor:match.start()]);end=match.end();depth=1;quoted=False;escape=False
        while depth:
            char=text[end];end+=1
            if escape:escape=False;continue
            if char=='\\' and quoted:escape=True;continue
            if char=='"':quoted=not quoted
            elif not quoted and char=='(':depth+=1
            elif not quoted and char==')':depth-=1
        cursor=end
    pieces.append(text[cursor:]);return ''.join(pieces)


def normalize(text):
    lines=[line for line in text.splitlines() if not line.startswith('#loc')]
    text=strip_locations('\n'.join(lines))
    defs={m[1]:m[2] for m in re.finditer(r'(?m)^(#[\w]+)\s*=\s*(#ttg\.[^\n]+)',text)}
    text='\n'.join(line for line in text.splitlines() if not re.match(r'^#[\w]+\s*=',line))
    for _ in range(len(defs)+1):
        expanded=re.sub(r'#[a-zA-Z_]\w*',lambda m:defs.get(m[0],m[0]),text)
        if expanded==text:break
        text=expanded
    text=re.sub(r'@(?:reduce_segments|_fwd_kernel_stage1|_layer_norm_bwd_dwdb|merge)\b','@ENTRY',text)
    names={}
    def ssa(match):
        value=match[0]
        if value not in names:names[value]='%v'+str(len(names))
        return names[value]
    text=re.sub(r'%[a-zA-Z0-9_.$]+',ssa,text)
    text=re.sub(r'//[^\n]*','',text)
    return '\n'.join(' '.join(line.split()) for line in text.splitlines() if line.strip())+'\n'


def facts(text):
    definitions={m[1]:m[2] for m in re.finditer(r'(?m)^(#[\w]+)\s*=\s*(#ttg\.[^\n]+)',text)}
    def expanded(line):
        line=line.split(' loc(')[0].strip()
        for key in sorted(definitions,key=len,reverse=True):line=re.sub(re.escape(key)+r'(?!\w)',lambda _:definitions[key],line)
        return line
    result={'loads':[],'conversions':[],'truncations':[],'extensions':[],'reductions':[],
            'make_ranges':[],'for_loops':0,'while_loops':0}
    stack=[]
    for line in text.splitlines():
        if line.startswith('#'):continue
        indentation=len(line)-len(line.lstrip())
        while stack and indentation<=stack[-1]:stack.pop()
        if 'scf.for ' in line:result['for_loops']+=1;stack.append(indentation)
        if 'scf.while ' in line:result['while_loops']+=1
        for key,token in [('loads','tt.load '),('conversions','ttg.convert_layout '),('truncations','arith.truncf '),('extensions','arith.extf '),('reductions','"tt.reduce"'),('make_ranges','tt.make_range ')]:
            if token in line:result[key].append({'text':expanded(line),'inside_for_loop':bool(stack)})
    result['counts']={k:len(v) for k,v in result.items() if isinstance(v,list)}
    return result


def program(folder,raw):
    root=folder/raw['path']/raw['metadata']['artifact_directory'];meta=raw['metadata']
    ir=(root/'kernel.ttgir').read_text();ptx=(root/'kernel.ptx').read_text()
    resources={k:int(v) for k,v in re.findall(r'\b(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER):(\d+)',(root/'kernel.resource.txt').read_text())}
    ptx_sig=re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid',ptx,re.S)
    assert ptx_sig
    abi=re.findall(r'\.param\s+\.(\w+)',ptx_sig[1])
    value={'case':raw['case'],'variant':raw['variant'],'metadata':meta,'facts':facts(ir),
           'resources':resources,'allocated_sections':elf((root/'kernel.cubin').read_bytes()),'ABI':abi,
           'normalized_TTGIR_SHA256':c.sha(normalize(ir).encode()),
           'source_path':str(root.relative_to(c.ROOT)),
           'original_CUBIN_SHA256':raw['original_CUBIN_SHA256'],'matches_original_CUBIN':raw['matches_original_CUBIN'],
           'intervention_steps':[]}
    for before in sorted((folder/raw['path']/'intervention_steps').glob('*.before.mlir')):
        after=before.with_name(before.name.replace('.before.','.after.'))
        a=before.read_text();b=after.read_text()
        value['intervention_steps'].append({'pass':before.name.split('.before.')[0],
            'before':facts(a),'after':facts(b),'changed':normalize(a)!=normalize(b)})
    save(OUT/'normalized_ir'/(raw['case']+'_'+raw['variant']+'.mlir'),normalize(ir).encode())
    return value


def traces(folder,raw):
    path=folder/raw['path']/'passes.mlir.gz'
    if not path.exists():return None
    text=gzip.decompress(path.read_bytes()).decode()
    integrity=c.read(path.parent/'trace_integrity.json');assert c.sha(text.encode())==integrity['uncompressed_SHA256']
    matches=list(re.finditer(r'^// .*?IR Dump (Before|After) ([^\n]+)\n',text,re.M));assert matches
    rows=[]
    for i,m in enumerate(matches):
        ir=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        row={'index':i,'direction':m[1],'pass':m[2],'facts':facts(ir),'normalized_SHA256':c.sha(normalize(ir).encode())}
        if i+1<len(matches):
            n=matches[i+1];after=text[n.end():matches[i+2].start() if i+2<len(matches) else len(text)]
            row['next_before_pass']=n[2];row['after_counts']=facts(after)['counts']
            row['changed']=normalize(ir)!=normalize(after)
            if any(name in m[2] for name in ['TritonGPUCoalesce:','RemoveLayoutConversions','LoopInvariantCodeMotion','LoopAwareCSE']):
                stem=raw['case']+'/'+raw['variant']+f'/{i:03}'
                save(OUT/'pass_checkpoints'/(stem+'.before.mlir'),ir.encode())
                save(OUT/'pass_checkpoints'/(stem+'.after.mlir'),after.encode())
                diff=''.join(difflib.unified_diff(normalize(ir).splitlines(True),normalize(after).splitlines(True),fromfile='before',tofile='after'))
                save(OUT/'pass_checkpoints'/(stem+'.diff'),diff.encode())
        rows.append(row)
    return {'case':raw['case'],'variant':raw['variant'],'steps':rows,'trace_SHA256':integrity['uncompressed_SHA256']}


def main():
    rows=[];trace_rows=[];runs=[]
    for namespace in ['pilot/attempt_001','matrix/attempt_001','literal/attempt_001','literal2/attempt_001','literal3/attempt_001']:
        folder=OUT/namespace/'export'
        if not (folder/'result.json').exists():continue
        result=c.read(folder/'result.json');assert result['status']=='OK' and not result['GPU_allocated'] and result['native_rebuilds']==0
        assert result['ccache_before']==result['ccache_after']
        for raw in result['programs']:
            assert raw['status']=='OK';rows.append(program(folder,raw))
            trace=traces(folder,raw)
            if trace:trace_rows.append(trace)
        runs.append({'path':str((folder/'result.json').relative_to(c.ROOT)),'SHA256':c.sha((folder/'result.json').read_bytes())})
    by={(r['case'],r['variant']):r for r in rows};comparisons=[]
    for case in sorted({r['case'] for r in rows}):
        u=by[(case,'U')]
        for variant in [r['variant'] for r in rows if r['case']==case and r['variant']!='U']:
            r=by[(case,variant)]
            comparisons.append({'case':case,'variant':variant,
                'all_allocated_section_bytes_equal_U':r['allocated_sections']==u['allocated_sections'],
                'instruction_bytes_equal_U':[x for x in r['allocated_sections'] if x['name'].startswith('.text.')]==[x for x in u['allocated_sections'] if x['name'].startswith('.text.')],
                'ABI_equal_U':r['ABI']==u['ABI'] and r['metadata']['runtime_argument_names']==u['metadata']['runtime_argument_names'],
                'resources_equal_U':r['resources']==u['resources'] and r['metadata']['dynamic_smem_bytes']==u['metadata']['dynamic_smem_bytes'],
                'normalized_TTGIR_equal_U':r['normalized_TTGIR_SHA256']==u['normalized_TTGIR_SHA256']})
    save(OUT/'compiler_analysis.json',{'status':'OK','programs':rows,'comparisons':comparisons,'traces':trace_rows,'runs':runs,
         'comparison_boundary':'ELF debug, source filenames and public entry symbol spelling are excluded; every SHF_ALLOC section, full runtime ABI and launch resource allocation are compared.'})
    print('G5 compiler analysis:',len(rows),'programs;',len(trace_rows),'complete pass traces')


if __name__=='__main__':main()
