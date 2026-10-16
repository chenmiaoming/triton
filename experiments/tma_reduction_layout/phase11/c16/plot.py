"""Static scientific figure; all values come from the saved derived analysis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tma-phase11-c16-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.tma_reduction_layout.phase11.c16 import common as c


def main():
    data=c.read(c.OUT/'analysis.json')
    colors={64:'#28628A',128:'#23816F'}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    for n,w in ((64,8),(128,4)):
        rows=sorted((r for r in data['primary_frozen_predictions'] if r['N']==n and r['cache']=='warm'),key=lambda r:r['vec'])
        x=np.array([r['C_onchip_us'] for r in rows]); y=np.array([r['actual_us'] for r in rows])
        lo=[r['actual_us']-min(r['worker_medians_us']) for r in rows]
        hi=[max(r['worker_medians_us'])-r['actual_us'] for r in rows]
        axes[0].errorbar(x,y,yerr=[lo,hi],fmt='o',capsize=3,color=colors[n],label=f'32 x {n}, {w} warps: measured')
        xx=np.linspace(0,92,250)
        axes[0].plot(xx,np.maximum(rows[0]['floor_us'],26.6+xx),'--',lw=1.3,color=colors[n])
        positions=({1:(18,65.5),2:(30,63.2),4:(44,71.5),8:(87,114)} if n==64
                   else {1:(20.5,51.5),2:(13.5,51.5),4:(14.5,58.5),8:(30,53.5)})
        for r in rows:
            axes[0].annotate(f"v{r['vec']}",(r['C_onchip_us'],r['actual_us']),xytext=positions[r['vec']],
                             fontsize=8,color=colors[n],arrowprops={'arrowstyle':'-','color':colors[n],'lw':.55})
    axes[0].set(xlabel='Scaled ML + MR proxy (us)',ylabel='Full-kernel event time (us)',
                title='Frozen overlap model vs. new warm cohort',xlim=(10,94),ylim=(42,124))
    axes[0].legend(fontsize=8,loc='upper left',frameon=False)
    rows=sorted((r for r in data['counter_rows'] if r['family']=='FULL' and r['cache']=='warm'),key=lambda r:(r['N'],r['vec']))
    values=[r['metrics']['dram__throughput.avg.pct_of_peak_sustained_elapsed'] for r in rows]
    axes[1].bar(np.arange(8),values,color=[colors[r['N']] for r in rows],width=.65)
    for i,value in enumerate(values):
        axes[1].text(i,value+1,f'{value:.1f}',ha='center',fontsize=8)
    axes[1].set(xticks=np.arange(8),xticklabels=[f"N{r['N']}\nv{r['vec']}" for r in rows],
                ylabel='DRAM sustained-peak utilization (%)',title='Independent NCU measurements',ylim=(0,100))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    folder=c.OUT/'figures'
    folder.mkdir(exist_ok=True)
    for extension in ('png','pdf'):
        fig.savefig(folder/f'overlap_validation.{extension}',dpi=180)
    plt.close(fig)
    manifest={'analysis_SHA256':c.sha((c.OUT/'analysis.json').read_bytes()),
              'files':{p.name:c.sha(p.read_bytes()) for p in sorted(folder.glob('overlap_validation.*'))},
              'caption':'Dashed lines use frozen parameters. Error bars span the three worker medians, not confidence intervals. NCU counters use a separate profiling worker.'}
    (c.OUT/'figure_manifest.json').write_bytes(c.encode(manifest))
    print('C1.6 scientific figure saved: PNG/PDF')


if __name__=='__main__':
    main()
