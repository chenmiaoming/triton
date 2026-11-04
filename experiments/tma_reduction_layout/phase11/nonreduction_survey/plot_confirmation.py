"""Standalone academic figure, derived only from frozen N4 measurements."""
from . import common as c


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    r=c.read(c.OUT/'stage_n4/analysis/analysis.json')
    rows=[x for x in r['rows'] if x['variant'] not in ['U','G0']]
    labels=['Scale 4096 x 128','Scale 11008 x 128','AWQ 3584 x 3584','AWQ 1024 x 1024',
        'RoPE D64','RoPE D128','MRoPE D64','MRoPE D128','FP8 pack H8/S2048','FP8 pack H32/S4096',
        'KDA gate D64','KDA gate D128','KV scatter H8/D64','KV scatter H32/D128']
    assert len(rows)==len(labels)==14
    colors=['#256080' if x['gain_vs_U_CI95'][0]>3 and x['gain_vs_G0_CI95'][0]>3
        else '#b07824' if x['case']=='N05_t2048_h8_d128' else '#777777' for x in rows]
    fig,axes=plt.subplots(1,2,figsize=(10.4,6.6),sharey=True,layout='constrained')
    y=np.arange(len(rows))
    for ax,reference in zip(axes,['U','G0']):
        for i,row in enumerate(rows):
            value=row['gain_vs_'+reference+'_percent'];low,high=row['gain_vs_'+reference+'_CI95']
            ax.errorbar(value,i,xerr=[[max(0,value-low)],[max(0,high-value)]],fmt='o',
                color=colors[i],markersize=4,capsize=2,elinewidth=1)
        ax.axvline(0,color='#aaaaaa',linewidth=.8)
        ax.axvline(3,color='#444444',linewidth=.8,linestyle='--')
        ax.set(xlabel='Kernel time reduction (%)',title='Compared with '+reference,xlim=(-3,8))
        ax.set_yticks(y,labels,fontsize=9);ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Independent fixed-choice confirmation on H100',fontsize=12)
    root=c.OUT/'figures';root.mkdir(parents=True,exist_ok=True)
    for extension in ['svg','png']:
        path=root/('N4_confirmation.'+extension);assert not path.exists()
        fig.savefig(path,dpi=180,metadata={'Creator':'Frozen N4 paired measurement analysis'})
    plt.close(fig)
    c.write_once(root/'N4_confirmation_provenance.json',dict(
        analysis_SHA256=c.sha((c.OUT/'stage_n4/analysis/analysis.json').read_bytes()),
        plotting_source_SHA256=c.sha(__import__('pathlib').Path(__file__).read_bytes()),
        pointwise_CI=True,multiplicity_corrected=False,
        colors='Blue: both lower bounds >3%; orange: D128 gate borderline; gray: other outcomes.'))


if __name__=='__main__':main()
