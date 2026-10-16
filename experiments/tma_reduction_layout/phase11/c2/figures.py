"""Standalone scientific figures and tidy derived service tables."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.tma_reduction_layout.phase11.c2 import common as c


def render(batch):
    a=c.read(c.out(batch)/'analysis.json');gate=c.read(c.out(batch)/'artifact_gate.json')
    root=c.out(batch)/'figures';root.mkdir(exist_ok=True)
    with (root/'services.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(a['aggregate'][0]),lineterminator='\n');writer.writeheader();writer.writerows(a['aggregate'])
    natural={r['tag']:r for r in a['aggregate'] if r['q']=='natural'}
    fig,axes=plt.subplots(2,4,figsize=(15,7.5),constrained_layout=True)
    for row,(n,w) in enumerate(((64,8),(128,4))):
        for col,family in enumerate(('ML','MR','MO','MC')):
            ax=axes[row,col]
            for placement,color,marker in [('n_first','#255f8b','o'),('within','#ad5c24','s')]:
                x,y,lo,hi=[],[],[],[]
                for vec in (1,2,4,8):
                    t=f'M32_N{n}_w{w}:vec{vec}:{placement}:{family}:D1'
                    if t not in natural:continue
                    r=natural[t];x.append(vec);y.append(r['median_ns_per_tile_SM'])
                    lo.append(r['median_ns_per_tile_SM']-r['worker_min_ns']);hi.append(r['worker_max_ns']-r['median_ns_per_tile_SM'])
                ax.errorbar(x,y,yerr=[lo,hi],marker=marker,color=color,label=placement,linewidth=1.2,capsize=3)
            ax.set_title(f'32×{n}, {w} warps — {family}')
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],labels=['1','2','4','8'])
            ax.set_xlabel('vec');ax.set_ylabel('ns / tile on one SM' if batch=='b' else 'ns / tile / SM');ax.grid(alpha=.25)
            if row==0 and col==0:ax.legend(frameon=False)
    fig.suptitle('C2a steady service: K≥16, three-grid normalization, D=1\nError bars: range of three worker estimates; MO includes its input fixture',fontsize=12)
    fig.savefig(root/'core_services.png',dpi=180);plt.close(fig)
    if batch=='b':
        fig,axes=plt.subplots(2,4,figsize=(14,7.2),constrained_layout=True)
        for i,q in enumerate(('natural','1')):
            selected={r['tag']:r for r in a['aggregate'] if r['q']==q}
            for ax,family,label in zip(axes[i],('P','U','W','E'),('thread-local p','within-warp u','cross-warp w','widening/consumer e')):
                for d in (1,2):
                    rows=sorted((gate['metadata'][t]['parameter'],r) for t,r in selected.items() if t.startswith('probe_'+family) and gate['metadata'][t]['D']==d)
                    ax.plot([v for v,r in rows],[r['median_ns_per_tile_SM'] for v,r in rows],'-o',label=f'D={d}')
                ax.set_title('natural' if q=='natural' else 'q=1 CTA/SM capacity')
                ax.set_xlabel(label);ax.set_ylabel('ns / tile on one SM');ax.grid(alpha=.25);ax.legend(frameon=False)
        fig.suptitle('C2b effective parameter subgraphs; private update and consumer costs retained\nTop: natural residency; bottom: q=1. D2 normalized per independent stream.')
        fig.savefig(root/'parameters.png',dpi=180);plt.close(fig)
        counter=c.read(c.out(batch)/'counter_gate.json')
        with (root/'q_occupancy.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(counter['q_achieved_occupancy_checks'][0]),lineterminator='\n')
            writer.writeheader();writer.writerows(counter['q_achieved_occupancy_checks'])
    (c.out(batch)/'figure_manifest.json').write_bytes(c.encode({'analysis_SHA256':c.sha((c.out(batch)/'analysis.json').read_bytes()),
        'files':{p.name:c.sha(p.read_bytes()) for p in sorted(root.iterdir()) if p.is_file()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['a','b'],required=True);render(p.parse_args().batch)
