"""Preserve CPU storage aliases/strides, with guards, for real JIT and driver runs."""
import numpy as np
from . import common as c


def dtype_name(dtype):
    return str(dtype).removeprefix('torch.')


class HostFixture:
    def __init__(self,spec):
        import torch
        self.spec=spec;self.groups=[];self.views={};owners={}
        for name,tensor in spec['tensors'].items():
            assert tensor.device.type=='cpu'
            storage=tensor.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
            if key not in owners:
                raw=torch.empty(0,dtype=torch.uint8).set_(storage,0,(storage.nbytes(),),(1,)).numpy().copy()
                owners[key]=len(self.groups)
                self.groups.append(dict(raw=raw,storage=storage,names=[]))
            group=owners[key];self.groups[group]['names'].append(name)
            self.views[name]=dict(group=group,shape=list(tensor.shape),stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),element_size=tensor.element_size(),dtype=dtype_name(tensor.dtype))
        self.outputs=list(spec['outputs']);self.inplace=list(spec.get('inplace',[]))
        assert set(self.inplace)<=set(self.outputs)
        assert set(self.outputs)==set(spec['expected'])
        self.grid=tuple(spec['grid'])+(1,)*(3-len(spec['grid']))
        assert len(self.grid)==3
        self.scalars=spec.get('scalars',{});self.constants=spec.get('constants',{})

    def metadata(self):
        return dict(views=self.views,outputs=self.outputs,inplace=self.inplace,grid=self.grid,
            initial_storage_SHA256=[c.sha(g['raw'].tobytes()) for g in self.groups],
            expected_SHA256={n:c.sha(t.contiguous().view(__import__('torch').uint8).numpy().tobytes())
                             for n,t in self.spec['expected'].items()})

    def check(self,raw_groups):
        import torch
        checks=[];writable=[np.zeros(len(g['raw']),dtype=bool) for g in self.groups]
        for name in self.outputs:
            v=self.views[name];g=v['group'];dtype=getattr(torch,v['dtype'])
            actual=torch.from_numpy(np.array(raw_groups[g],copy=True)).view(dtype).as_strided(
                v['shape'],v['stride'],v['storage_offset'])
            expected=self.spec['expected'][name]
            mask=self.spec.get('masks',{}).get(name)
            if mask is None:mask=torch.ones(expected.shape,dtype=torch.bool)
            byte_mask=mask.numpy().reshape(-1).repeat(v['element_size'])
            aa=actual.contiguous().view(torch.uint8).numpy().reshape(-1)[byte_mask]
            ee=expected.contiguous().view(torch.uint8).numpy().reshape(-1)[byte_mask]
            bitwise=np.array_equal(aa,ee)
            tol=self.spec.get('tolerances',{}).get(name,{'bitwise':True})
            if tol.get('bitwise',True):okay=bitwise
            else:okay=bool(torch.allclose(actual.float()[mask],expected.float()[mask],rtol=tol['rtol'],atol=tol['atol'],equal_nan=True))
            # Preserve all physical bytes outside selected logical outputs.
            dense=True;extent=1
            for size,stride in sorted(zip(v['shape'],v['stride']),key=lambda x:x[1]):
                if size==1:continue
                dense &= stride==extent;extent*=size
            if dense and v['storage_offset']==0 and extent*v['element_size']==len(writable[g]) and bool(mask.all()):
                writable[g][:]=True
            else:
                positions=np.zeros(tuple(v['shape']),dtype=np.int64)+v['storage_offset']
                for dim,(size,stride) in enumerate(zip(v['shape'],v['stride'])):
                    index=[1]*len(v['shape']);index[dim]=size
                    positions+=np.arange(size,dtype=np.int64).reshape(index)*stride
                positions=positions[mask.numpy()].reshape(-1)*v['element_size']
                for byte in range(v['element_size']):writable[g][positions+byte]=True
            checks.append(dict(name=name,pass_=okay,bitwise_equal=bitwise,elements=int(mask.sum()),
                actual_SHA256=c.sha(aa.tobytes()),expected_SHA256=c.sha(ee.tobytes())))
        for i,g in enumerate(self.groups):
            untouched=np.array_equal(raw_groups[i][~writable[i]],g['raw'][~writable[i]])
            checks.append(dict(storage=i,pass_=bool(untouched),readonly_and_inactive_unchanged=bool(untouched)))
        return {'pass':all(x['pass_'] for x in checks),'details':checks}


class TorchFixture(HostFixture):
    def __init__(self,spec,device='cuda'):
        import torch
        super().__init__(spec);self.owners=[];self.tensors={}
        for g in self.groups:
            guarded=np.concatenate([np.full(256,0xa5,np.uint8),g['raw'],np.full(256,0x5a,np.uint8)])
            self.owners.append(torch.from_numpy(guarded).to(device))
        for name,v in self.views.items():
            self.tensors[name]=self.owners[v['group']].view(getattr(torch,v['dtype'])).as_strided(
                v['shape'],v['stride'],256//v['element_size']+v['storage_offset'])

    def arguments(self):
        return {**self.tensors,**self.scalars,**self.constants}

    def restore(self):
        import torch
        for i,g in enumerate(self.groups):
            self.owners[i][256:-256].copy_(torch.from_numpy(g['raw']))

    def check(self):
        import torch
        torch.cuda.synchronize();raw=[];guards=[]
        for owner in self.owners:
            blob=owner.cpu().numpy();raw.append(blob[256:-256])
            guards.append(bool(np.all(blob[:256]==0xa5) and np.all(blob[-256:]==0x5a)))
        result=super().check(raw);result['guards']=guards;result['pass'] &= all(guards)
        return result


class DriverFixture(HostFixture):
    def __init__(self,spec,device):
        super().__init__(spec);self.device=device;self.bases=[]
        for group in self.groups:
            guarded=bytes([0xa5])*256+group['raw'].tobytes()+bytes([0x5a])*256
            base=device.allocate(len(guarded));device.upload(base,guarded);self.bases.append(base)
        self.scratch=device.allocate(256)
        self.mutable={self.views[n]['group'] for n in self.outputs}
        self.mutated={self.views[n]['group'] for n in self.inplace}

    def arguments(self,meta):
        scalars={**self.scalars,**self.constants};values=[]
        for name in meta['runtime_argument_names']:
            if name in self.views:
                v=self.views[name];values.append(self.bases[v['group']]+256+v['storage_offset']*v['element_size'])
            else:values.append(scalars[name])
        for name,value in meta.get('specialized_runtime_values',{}).items():
            if name in scalars:assert scalars[name]==value,(name,scalars[name],value)
        return values+[self.scratch,0]

    def reset(self,only_inplace=False):
        for i in (self.mutated if only_inplace else self.mutable):
            self.device.upload(self.bases[i]+256,self.groups[i]['raw'].tobytes())

    def check(self):
        raw=[];guards=[]
        for base,group in zip(self.bases,self.groups):
            blob=np.frombuffer(self.device.download(base,len(group['raw'])+512),np.uint8)
            raw.append(blob[256:-256]);guards.append(bool(np.all(blob[:256]==0xa5) and np.all(blob[-256:]==0x5a)))
        result=super().check(raw);result['guards']=guards;result['pass'] &= all(guards)
        return result
