"""Conservative geometry exposure audit; never reads new timing."""
import itertools
import json
import re
import zipfile
import ast
from collections import defaultdict
from experiments.tma_reduction_layout.phase11.c3 import common as c

POOL = list(itertools.product((8,16,32,64,128), (32,64,128,256,512), (2,4,8)))
TOKEN = re.compile(r'M(\d+)_N(\d+)_w(\d+)')


def triples(value):
    if isinstance(value, dict):
        m = value.get('BM', value.get('BLOCK_M', value.get('M')))
        n = value.get('BN', value.get('BLOCK_N', value.get('N')))
        w = value.get('num_warps', value.get('warps', value.get('W')))
        if all(type(x) is int for x in (m,n,w)):
            yield (m,n,w)
        for k,v in value.items():
            for match in TOKEN.finditer(k):
                yield tuple(map(int, match.groups()))
            yield from triples(v)
    elif isinstance(value, list):
        for v in value:
            yield from triples(v)


def audit():
    import subprocess
    paths = subprocess.check_output(['git','ls-files','-z',
        'experiments/tma_reduction_layout'], cwd=c.ROOT).decode().split('\0')
    evidence = defaultdict(set)
    scanned = []
    def consume(name, data):
        hits = set(tuple(map(int,m.groups())) for m in TOKEN.finditer(name))
        if name.endswith(('.json','.jsonl','.md','.csv','.py','.txt')):
            text = data.decode(errors='replace')
            hits.update(tuple(map(int,m.groups())) for m in TOKEN.finditer(text))
            if name.endswith('.json'):
                try:
                    hits.update(triples(json.loads(text)))
                except (ValueError, TypeError):
                    pass
        for key in hits & set(POOL):
            evidence[key].add(name)
    for name in sorted(n for n in paths if n and '/stage_c3/' not in n and
                       '/phase11/c3/' not in n):
        p = c.ROOT/name
        data = p.read_bytes()
        scanned.append({'path':name,'SHA256':c.digest(data),'bytes':len(data)})
        consume(name, data)
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as z:
                for entry in z.infolist():
                    # Text metadata and all artifact names; no opaque binary search.
                    payload = z.read(entry) if entry.filename.endswith(
                        ('.json','.md','.csv','.py','.txt','.jsonl')) else b''
                    consume(name+'!'+entry.filename, payload)
    rows = []
    for m,n,w in POOL:
        hits=sorted(evidence[(m,n,w)])
        rows.append({'BM':m,'BN':n,'W':w,'case':f'M{m}_N{n}_w{w}',
                     'exposed':bool(hits),'exposure_count':len(hits),
                     'evidence':hits,'static_128B_shape_legal':m>=8 and n>=64})
    return {'scope':'All tracked experiment files, text metadata, source pools and zip members. Any geometry exposure excludes, even unexecuted pool entries. Parameter probes count conservatively.',
            'pool':rows,'scanned':scanned,
            'not_claimed':'External/unrecorded experiments cannot be audited.'}


def supplemental():
    import tarfile
    import subprocess
    from pathlib import Path
    names=subprocess.check_output(['git','ls-files','experiments/tma_reduction_layout'],
                                  cwd=c.ROOT,text=True).splitlines()
    hits=defaultdict(list);archives=[]
    for name in names:
        if '/phase11/c3/' in name or '/stage_c3/' in name:continue
        path=c.ROOT/name
        if name.endswith('.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.Tuple,ast.List)) and len(node.elts)==3 and all(
                    isinstance(x,ast.Constant) and type(x.value) is int for x in node.elts):
                    value=tuple(x.value for x in node.elts)
                    if value in POOL:hits[value].append(name+':'+str(node.lineno))
        if name.endswith(('.tar','.tar.gz','.tgz')):
            with tarfile.open(path) as archive:
                members=archive.getnames()
            project=[n for n in members if 'experiments/tma_reduction_layout/' in n]
            archives.append({'path':name,'members':len(members),'project_experiment_members':project,
                             'scope':'Upstream compiler-source provenance archive, not project timing history.'})
            assert not project,(name,'additional experimental archive requires explicit audit')
    blind=c.read(c.OUT/'preregistration/protocol.json')['blind_cases']
    return {'method':'AST literal three-integer tuples/lists in tracked project Python, excluding C3 itself. Tar provenance archive inventory.',
            'hits':{'_'.join(map(str,k)):v for k,v in sorted(hits.items())},
            'archives':archives,
            'blind_all_zero':all((r['BM'],r['BN'],r['W']) not in hits for r in blind)}


if __name__ == '__main__':
    result=audit()
    c.write(c.OUT/'history_audit.json',result)
    print('audited',len(result['scanned']),'files; unseen legal:',
          [r['case'] for r in result['pool'] if not r['exposed'] and r['static_128B_shape_legal']])
