"""Standalone scientific figure from completed same-program diagnostics."""
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root
    specs=[('same_program_001',['fast_gap','legacy_gap','idle_20ms','gpu_preload'],
        ['Fast','File guard','Idle 20 ms','128 preloads'],'Initial diagnostic'),
        ('wake_001',['fast_gap','idle_20ms','idle_then_same_prime','idle_then_disjoint_prime'],
        ['Fast','Idle 20 ms','Idle + same\nprimer','Idle + disjoint\nprimer'],'Adaptive follow-up')]
    colors=['#2166ac','#b35806','#4d9221']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,(folder,conditions,labels,title) in zip(axes,specs):
        data=json.loads((root/folder/'analysis/analysis.json').read_text())
        index={(r['condition'],r['variant']):np.asarray(r['worker_visit_medians_us']) for r in data['rows']}
        for ci,condition in enumerate(conditions):
            pooled=(index[condition,'A']+index[condition,'B'])/2
            for worker in range(3):
                x=ci+(worker-1)*.14
                ax.scatter(x+np.linspace(-.025,.025,pooled.shape[1]),pooled[worker],
                    s=16,alpha=.5,color=colors[worker],edgecolors='none',
                    label=f'Worker {worker}' if ci==0 else None)
                ax.plot([x-.045,x+.045],[pooled[worker].mean()]*2,color=colors[worker],lw=2)
        ax.set_xticks(range(4),labels)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Target CUDA-event time (µs)')
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Identical AWQ target: same CUBIN, 1024 × 1024 shape and launch',fontsize=12)
    fig.text(.5,.015,'Each dot: mean of A/B visit medians (31 samples per label). Horizontal marks: worker means.\nSeparate GPU cohorts; the two experiments are not pooled. All target graphs contain one kernel and two events.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94))
    for suffix in ('png','pdf'):
        stream=io.BytesIO()
        fig.savefig(stream,format=suffix,dpi=180,bbox_inches='tight')
        path=root/f'same_program_repeatability.{suffix}'
        with path.open('xb') as handle:handle.write(stream.getvalue())
    plt.close(fig)


if __name__=='__main__':main()
