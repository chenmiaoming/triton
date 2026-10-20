"""Standalone scientific plots and flat tables from the retained CPU audit."""
import csv
import io
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tma-m1-matplotlib')
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.tma_reduction_layout.phase11.m1 import common as c


def table(name,rows):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);c.derived(c.OUT/'tables'/name,out.getvalue().encode())


def save(fig,name):
    target=c.OUT/'figures';target.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None
        fig.savefig(target/(name+'.'+ext),dpi=190,metadata=metadata)
    plt.close(fig)


def run():
    prof=c.read(c.OUT/'identifiability.json');diag=c.read(c.OUT/'pipeline.json');pairs=c.read(c.OUT/'mechanisms.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    rows=prof['CV_curve'];x=[r['eta'] for r in rows]
    fig,ax=plt.subplots(figsize=(6.8,4.2),layout='constrained')
    ax.plot(x,[r['fixed_ridge_inner_CV_loss'] for r in rows],'o-',label='Fixed ridge=0.1')
    ax.plot(x,[r['best_existing_inner_CV_loss'] for r in rows],'s--',label='Ridge chosen by existing inner CV')
    ax.set(xlabel='Scalar eta',ylabel='Development inner-CV joint relative loss',title='Eta profile: no D1 fitting',xticks=x)
    ax.legend(frameon=False);save(fig,'eta_CV_profile')
    table('eta_CV_profile.csv',[{k:r[k] for k in ('eta','best_existing_inner_CV_ridge','best_existing_inner_CV_loss','fixed_ridge_inner_CV_loss')} for r in rows])
    coefficients=[{'profile':r['profile'],'eta':r['eta'],'ridge':r['ridge'],'feature':k,'coefficient':v,
        'RMS_response_term_ns':r['RMS_model_contribution_ns'][k]} for r in prof['fits'] for k,v in r['coefficients'].items()]
    table('coefficients.csv',coefficients)
    fixed=[r for r in prof['fits'] if r['profile']=='fixed_ridge'];names=sorted(fixed[0]['coefficients'])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,8),layout='constrained',gridspec_kw={'width_ratios':[1,1.8]})
    s=fixed[0]['FULL_feature_design']['singular_values'];ax.semilogy(np.arange(1,len(s)+1),s,'o-')
    ax.axhline(fixed[0]['FULL_feature_design']['numerical_tolerance'],ls='--',color='.5',label='Numerical tolerance')
    ax.set(xlabel='Singular-value index',ylabel='Normalized feature-design singular value',title='27 columns; rank 26');ax.legend(frameon=False)
    matrix=np.array([[r['RMS_model_contribution_ns'][n] for r in fixed] for n in names])
    im=bx.imshow(np.log1p(matrix),aspect='auto',cmap='cividis')
    bx.set(xticks=range(len(x)),xticklabels=x,yticks=range(len(names)),yticklabels=names,
        xlabel='Eta (fixed ridge=0.1)',title='Model response terms; not measured stage latencies')
    bx.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=bx,label='log(1 + RMS response term / ns)')
    save(fig,'design_and_response_terms')
    flat=[]
    for r in diag['summary']:
        for support in ('all_available','four_way_common_support'):
            for variant,m in r[support].items():flat.append({'model':r['model'],'cohort':r['cohort'],
                'endpoint':r['endpoint'],'support':support,'variant':variant,**m})
    table('substitution_metrics.csv',flat)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for model,color in (('C34_resource_only','#315879'),('C34_full_auto','#9a6038')):
        row=next(r for r in diag['summary'] if r['model']==model and r['cohort']=='D1' and r['endpoint']=='natural')
        keys=('static','measured_q','measured_q_and_control')
        for ax,field in zip(axes,('FULL_relative_RMSE_percent','paired_ratio_RMSE_pp')):
            ax.plot(range(3),[row['all_available'][k][field] for k in keys],'o-',color=color,label=model)
            ax.set_xticks(range(3),['Frozen','Observed q','Observed q + control'])
        axes[0].set(ylabel='FULL relative RMSE (%)',title='D1: 59 layouts, same denominator')
        axes[1].set(ylabel='Paired-ratio RMSE (percentage points)',title='D1: error in candidate/default ratios')
    axes[0].legend(frameon=False,fontsize=8);save(fig,'D1_substitution_errors')
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    curve_rows=[]
    for i,g in enumerate(c.protocol()['primary_diagnostic_geometries']):
        pair=next(r for r in pairs['pairs'] if r['geometry']==g and r['vec']==2)
        for j,model in enumerate(('C34_resource_only','C34_full_auto')):
            ax=axes[i,j];curves=pair['same_worker_same_q_contrasts'];qs=sorted({r['q'] for r in curves})
            for field,color,lab in (('N_first_FULL_ns','#315879','N-first measured'),('within_FULL_ns','#9a6038','Within measured')):
                med=[st.median(r[field] for r in curves if r['q']==q) for q in qs]
                lo=[min(r[field] for r in curves if r['q']==q) for q in qs];hi=[max(r[field] for r in curves if r['q']==q) for q in qs]
                ax.plot(qs,med,'o-',color=color,label=lab);ax.fill_between(qs,lo,hi,color=color,alpha=.12)
            ax.plot(qs,[st.median(r['control_ns'] for r in curves if r['q']==q) for q in qs],'k:',label='Operational control')
            for tag,color,lab in ((pair['N_first'],'#315879','N-first model at measured q'),(pair['within'],'#9a6038','Within model at measured q')):
                rows=[r for r in diag['records'] if r['model']==model and r['cohort']==pair['cohort'] and r['tag']==tag]
                ax.plot(qs,[st.median(r['predictions']['measured_q']['tile_ns'] for r in rows if r['actual_q']==q) for q in qs],
                        's--',color=color,label=lab)
            ax.set_xscale('log',base=2);ax.set_xticks(qs,[str(q) for q in qs]);ax.set_yscale('log')
            ax.set(xlabel='Measured occupancy capacity q',ylabel='FULL service (ns/tile/SM)',
                title=f'{c.geo(g)} vec2: {model}')
            if i==0 and j==0:ax.legend(frameon=False,fontsize=7)
        for r in pair['same_worker_same_q_contrasts']:
            curve_rows.append({'geometry':c.geo(g),'q':r['q'],'worker':r['worker'],
                'N_first_FULL_ns':r['N_first_FULL_ns'],'within_FULL_ns':r['within_FULL_ns'],
                'control_ns':r['control_ns'],'within_minus_N_first_ns':r['within_minus_N_first_ns'],
                'within_gain_percent':r['within_gain_percent']})
    save(fig,'FULL_q_response');table('matched_q_curves.csv',curve_rows)
    print('M1 scientific plots exported: four PNG/PDF pairs and four CSV tables',flush=True)


if __name__=='__main__':run()
