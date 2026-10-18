"""Small nonnegative regressions with visible design and extrapolation leverage."""
import numpy as np
from experiments.tma_reduction_layout.phase11.c31.fit import nnls


def fit(fs, ys, ridge=.001):
    names = sorted(fs[0])
    a = np.array([[f[name] for name in names] for f in fs])
    y = np.asarray(ys, float)
    beta, diagnostic = nnls(a, y, ridge)
    scale = np.maximum(np.linalg.norm(a, axis=0), 1.)
    z = a/scale
    inverse = np.linalg.pinv(z.T@z+max(ridge, .001)*np.eye(len(names)))
    lev = np.einsum('ij,jk,ik->i', z, inverse, z)
    return {'feature_names': names, 'coefficients': beta.tolist(), 'diagnostic': diagnostic,
        'scale': scale.tolist(), 'inverse_gram': inverse.tolist(),
        'median_training_leverage': float(np.median(lev)),
        'feature_support': {name: [float(a[:, i].min()), float(a[:, i].max())] for i, name in enumerate(names)}}


def predict(model, f):
    x = np.array([f[name] for name in model['feature_names']])
    return float(np.dot(model['coefficients'], x))


def leverage(model, f):
    x = np.array([f[name] for name in model['feature_names']])/np.array(model['scale'])
    h = float(x@np.array(model['inverse_gram'])@x)
    return {'value': h, 'inflation': float(np.sqrt(max(1., h/max(model['median_training_leverage'], 1e-12)))),
            'outside_features': [name for name in model['feature_names']
                if not model['feature_support'][name][0]-1e-8 <= f[name] <= model['feature_support'][name][1]+1e-8]}
