"""Standalone scientific figures from all predeclared comparisons, no timing work."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as c
from .g4_prepare import out


def main(stage):
    target = out(stage); rows = c.read(target / 'timing_analysis.json')['rows']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    if stage == 'g4a':
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout='constrained')
        for i, geom in enumerate(['C02_n64_d128', 'C02_n64_d64']):
            for j, length in enumerate([256, 2048]):
                ax = axes[i, j]; labels = ['gluon_anchor', 'pv_only', 'qk_only', 'coordinated']
                selected = [next(r for r in rows if r['group'] == geom and r['length'] == length and r['mode'] == 'decode' and r['tag'].endswith(':'+label)) for label in labels]
                means = np.asarray([r['gain_vs_U_percent'] for r in selected]); ci = np.asarray([r['gain_vs_U_CI95'] for r in selected]).T
                ax.bar(np.arange(4), means, color=['#888888', '#35689A', '#B5814D', '#6B9369'], width=.65)
                ax.errorbar(np.arange(4), means, yerr=np.maximum(0, np.vstack((means-ci[0], ci[1]-means))), fmt='none', color='black', capsize=4, lw=1)
                ax.axhline(0, color='black', lw=.8); ax.axhline(3, color='gray', ls=':', lw=.8); ax.axhline(-3, color='gray', ls=':', lw=.8)
                ax.set_xticks(np.arange(4), ['G0', 'PV only', 'QK only', 'Coordinated'])
                ax.set_title(f'd={128 if i==0 else 64}, batch=8, context={length}')
                ax.set_ylabel('Elapsed-time reduction vs original Triton (%)')
                ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        fig.suptitle('SGLang: neighboring layouts do not compose into a universal improvement\nFull two-stage decode; 3 workers × 5 visits; pointwise paired 95% bootstrap intervals', fontsize=13)
        filename = 'factorial_decode'
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout='constrained')
        for ax, geometry in zip(axes, ['C06_s16_d128', 'C06_s32_d256']):
            labels, values, intervals = [], [], []
            for batch in [1, 8, 64]:
                row = next(r for r in rows if r['group'].startswith(geometry+'_b'+str(batch)+'_') and r['tag'].endswith(':actual:target'))
                labels.append(str(batch)); values.append(row['gain_vs_U_percent']); intervals.append(row['gain_vs_U_CI95'])
            means = np.asarray(values); ci = np.asarray(intervals).T
            ax.bar(np.arange(3), means, color='#35689A', width=.6)
            ax.errorbar(np.arange(3), means, yerr=np.maximum(0,np.vstack((means-ci[0],ci[1]-means))), fmt='none', color='black', capsize=4)
            ax.axhline(0,color='black',lw=.8); ax.axhline(3,color='gray',ls=':',lw=.8)
            ax.set_xticks(np.arange(3),labels);ax.set_xlabel('Actual JIT batch');ax.set_ylabel('Elapsed-time reduction vs original Triton (%)')
            ax.set_title(geometry + (' / target vec1:N-first' if 's16' in geometry else ' / target vec4:within'))
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.suptitle('vLLM merge: preselected layouts under actual per-batch JIT\nMerge kernel only; pointwise paired 95% bootstrap intervals',fontsize=13)
        filename = 'actual_jit_merge'
    folder = target/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        path = folder/(filename+'.'+ext);assert not path.exists();fig.savefig(path,dpi=180,metadata={'Creator':'g4_plot.py'})
    plt.close(fig);print(stage,'figure exported')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['g4a','g4b']);main(parser.parse_args().stage)
