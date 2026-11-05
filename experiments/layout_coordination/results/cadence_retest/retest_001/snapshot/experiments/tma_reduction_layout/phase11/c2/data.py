"""Deterministic BF16 inputs and complete CPU output oracles."""
import hashlib
import numpy as np


def bf16(value):
    value=np.asarray(value,dtype=np.float32)
    bits=value.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def values(linear,pattern):
    linear=np.asarray(linear,dtype=np.int64)
    if pattern=='quarter':
        return (((linear%97)-48)/4).astype(np.float32)
    if pattern=='boundary':
        return np.array([-16,-8,-0.,0.,8,15.75,.25,-.25],dtype=np.float32)[linear%8]
    assert pattern=='random'
    return bf16((((linear*1664525+1013904223)&65535).astype(np.float32)-32768)/8192)


def input_bytes(pattern):
    period={'quarter':97,'random':65536,'boundary':8}[pattern]
    raw=(values(np.arange(period),pattern).view(np.uint32)>>16).astype('<u2').tobytes()
    size=8192*8192*2
    return (raw*(size//len(raw)+1))[:size]


def result_bytes(pattern):
    pm=np.arange(256,dtype=np.int64)[:,None]
    cols=np.arange(8192,dtype=np.int64)[None,:]
    result=np.full((256,8192),-np.inf,dtype=np.float32)
    for row in range(32):
        np.maximum(result,values(pm*32*8192+row*8192+cols,pattern),out=result)
    return result.tobytes()


def tiles(meta,pids,slot=0,pattern='quarter'):
    m,n=meta['M'],meta['N']
    pids=np.asarray(pids,dtype=np.int64)%((8192//m)*(8192//n))
    pm,pn=pids//(8192//n),pids%(8192//n)
    pm=(pm+slot*((8192//m)//2))%(8192//m)
    rows=pm[:,None,None]*m+np.arange(m)[None,:,None]
    cols=pn[:,None,None]*n+np.arange(n)[None,None,:]
    return values(rows*8192+cols,pattern)


def reference(meta,k,pids,pattern='quarter'):
    m,n,d=meta['M'],meta['N'],meta['D']
    family=meta['family']
    tile0=tiles(meta,pids,0,pattern)
    if family=='MO':
        return tile0.max(axis=1).reshape(len(pids),-1)
    tile1=tiles(meta,pids,1,pattern)
    if family=='MC':
        pair=[tile0.max(axis=1),tile1.max(axis=1)]
        return np.stack([pair[i&1] for i in range(k)],axis=1).reshape(len(pids),-1)
    if family=='ML':
        result=np.zeros((len(pids),d,m,n),dtype=np.float32)
        pair=[tile0,tile1]
        for i in range(k):
            for stream in range(d):
                result[:,stream]+=pair[(i+stream)&1]
        return result.reshape(len(pids),-1)
    states=[tile0.copy(),tile1.copy()][:d]
    step=bf16((((np.arange(m)[:,None]*13+np.arange(n)[None,:]*7)%7)+1).astype(np.float32)*.25)
    shape=(len(pids),d,m,n) if family=='E' else (len(pids),d,n)
    result=np.zeros(shape,dtype=np.float32)
    for _ in range(k):
        for stream in range(d):
            state=states[stream]
            states[stream]=np.where(state>=8,bf16(state-16),bf16(state+step))
            result[:,stream]+=states[stream] if family=='E' else states[stream].max(axis=1)
    return result.reshape(len(pids),-1)


def compare(actual,expected):
    actual=np.asarray(actual,dtype=np.float32)
    expected=np.asarray(expected,dtype=np.float32)
    assert actual.shape==expected.shape,(actual.shape,expected.shape)
    return {'pass':bool(np.array_equal(actual,expected)),'elements':int(actual.size),
            'max_abs_error':float(np.max(np.abs(actual-expected))),
            'actual_SHA256':hashlib.sha256(actual.tobytes()).hexdigest(),
            'reference_SHA256':hashlib.sha256(expected.tobytes()).hexdigest()}
