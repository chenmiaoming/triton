"""Allocated-section fingerprint, identical to the established G8 audit."""
import re,struct
from . import common as c


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
