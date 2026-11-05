"""Standalone scientific figures from held-geometry predictions only."""
import statistics as st
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tma-c33-mpl-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.tma_reduction_layout.phase11.c33 import common as c, data, fit, resources


def run():
    theta=c.read(c.OUT/'theta_H100.json');dev=c.read(c.OUT/'development.json')
    b=data.load();fit.BUNDLE={'local_tables':theta['fixed_R_tables']}
    directory=c.OUT/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'figure.dpi':160})
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    pts=[]
    for fold in theta['outer_folds']:
        rows=data.medians([r for r in b['full'] if r['geometry']==fold['held'] and r['q_condition']=='natural'])
        for r in rows:
            q=resources.q_for(r['case'],resources.point(fold['resource_point_model'],r['case']))
            p=fit.predict(fold['model'],r['case'],q)
            pts.append((r,p))
    lim=max(max(r['target_ns'],p['tile_ns']) for r,p in pts)*1.06
    xx=np.linspace(0,lim,200);ax.fill_between(xx,.97*xx,1.03*xx,color='.9',label='3% reference band')
    ax.plot(xx,xx,color='.4',lw=.8)
    for w,color in ((4,'#285f8f'),(8,'#b5682d')):
        for family,marker in (('n_first','o'),('within','^')):
            rows=[(r,p) for r,p in pts if r['case']['W']==w and r['case']['placement']==family]
            ax.scatter([r['target_ns'] for r,p in rows],[p['tile_ns'] for r,p in rows],
                s=26,c=color,marker=marker,label=f'{w} warps, {family}',alpha=.8)
    ax.set(xlabel='Measured FULL service (ns/tile/SM)',ylabel='Held-geometry prediction (ns/tile/SM)',
        xlim=(0,lim),ylim=(0,lim),title='Nested leave-one-geometry-out: 61 FULL layouts')
    ax.legend(frameon=False,fontsize=8);ax.set_aspect('equal')
    for ext in ('png','pdf'):fig.savefig(directory/('FULL_LOGO_predictions.'+ext),dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    rows=dev['outer_evaluation'];x=np.arange(len(rows));width=.25
    ax.bar(x-width,[r['nominal_predicted_gain_percent'] for r in rows],width,label='Nominal predicted gain',color='#285f8f')
    ax.bar(x,[r['nominal_observed_gain_percent'] for r in rows],width,label='Same choice, measured gain',color='#8ea7bb')
    ax.bar(x+width,[r['selected_observed_gain_percent'] for r in rows],width,label='Conservative choice, measured gain',color='#515151')
    ax.axhline(3,color='.5',ls='--',lw=.8,label='3% practical threshold');ax.axhline(0,color='.3',lw=.6)
    ax.set_xticks(x,[f'{r["geometry"][0]}x{r["geometry"][1]}\nw{r["geometry"][2]}' for r in rows])
    ax.set(ylabel='Gain relative to upstream default (%)',title='Held-geometry choices; all selected layouts measured')
    ax.legend(frameon=False,fontsize=8,ncol=2)
    for ext in ('png','pdf'):fig.savefig(directory/('FULL_LOGO_gains.'+ext),dpi=200)
    plt.close(fig)
    print('C3.3 figures exported (PNG/PDF)')


if __name__=='__main__': run()
