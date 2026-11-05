"""Nonnegative ridge regression; coefficients are fitted to MC, not fixtures."""
import numpy as np


def nnls(a,y,ridge):
    a=np.asarray(a,float);y=np.asarray(y,float)
    scale=np.linalg.norm(a,axis=0);scale[scale==0]=1.
    x=a/scale
    gram=x.T@x+float(ridge)*np.eye(x.shape[1]);rhs=x.T@y
    coef=np.zeros(x.shape[1]);passive=np.zeros(x.shape[1],bool)
    tolerance=1e-9*max(1.,float(np.max(np.abs(rhs))))
    iterations=0
    while iterations<20*x.shape[1]:
        gradient=rhs-gram@coef
        active=np.where(~passive,gradient,-np.inf)
        if np.max(active)<=tolerance:break
        passive[int(np.argmax(active))]=True
        while True:
            indices=np.flatnonzero(passive);trial=np.zeros_like(coef)
            trial[indices]=np.linalg.lstsq(gram[np.ix_(indices,indices)],rhs[indices],rcond=1e-12)[0]
            if np.all(trial[passive]>0):
                coef=trial;break
            negative=passive & (trial<=0)
            denominator=coef[negative]-trial[negative]
            valid=denominator>0
            alpha=float(np.min(coef[negative][valid]/denominator[valid])) if np.any(valid) else 0.
            coef+=alpha*(trial-coef)
            drop=passive & (coef<=tolerance)
            coef[drop]=0.;passive[drop]=False
            iterations+=1
            if iterations>=20*x.shape[1]:break
        iterations+=1
    assert np.min(coef)>=-1e-8
    beta=np.maximum(coef,0.)/scale
    return beta,{'iterations':iterations,'columns':len(beta),
                 'rank':int(np.linalg.matrix_rank(x)),
                 'nonnegative':bool(np.all(beta>=0)),
                 'training_rmse_ns':float(np.sqrt(np.mean((a@beta-y)**2)))}
