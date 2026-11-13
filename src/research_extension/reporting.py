"""Publication-style figures generated only from saved extension outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import require,read_json,sha,write_json,LABEL6


def _save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    png=directory/f'{stem}.png'; pdf=directory/f'{stem}.pdf'
    fig.savefig(png,dpi=300,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight')
    plt.close(fig)
    return {png.name:sha(png),pdf.name:sha(pdf)}


def plots(ctx):
    """Create figures and a parameter table from completed, hashed experiments."""
    result_dir=ctx.results; out=result_dir/'plots'; out.mkdir(parents=True,exist_ok=True)
    evaluation=read_json(result_dir/'evaluation.json')
    comparison=pd.read_csv(result_dir/'model_comparison.csv')
    require(set(comparison.variant)==set(['A','B','C','D','E','F','G','I']),
            'Expected all prespecified model variants before plotting')
    metrics_path=result_dir/'seed_metrics.csv'; distributions=result_dir/'similarity_distributions.csv'
    rows=[]
    for r in comparison.to_dict('records'):
        rows.append({k:r[k] for k in ['variant','n_seeds','parameters','accuracy','accuracy_sd','macro_f1','macro_f1_sd',
            'weighted_f1','weighted_f1_sd','fake_precision','fake_recall','fake_f1','roc_auc','pr_auc','average_precision']})
    table=pd.DataFrame(rows); table.to_csv(out/'model_table.csv',index=False)
    labels={'A':'A: multimodal','B':'B: raw cosine','C':'C: margins','D':'D: BRAS','E':'E: random control',
            'F':'F: cosine-only','G':'G: BRAS-only','I':'I: CLIP projection control'}

    fig,ax=plt.subplots(figsize=(8.6,4.8))
    order=list('ABCDEFGI'); table=table.set_index('variant').loc[order].reset_index()
    x=table.macro_f1.to_numpy()*100; err=table.macro_f1_sd.fillna(0).to_numpy()*100
    ax.bar(np.arange(len(table)),x,yerr=err,capsize=3,color=['#718096','#4C78A8','#72B7B2','#E45756','#B279A2','#9D755D','#F2CF5B','#54A24B'])
    ax.set_xticks(np.arange(len(table)),[labels[k] for k in table.variant],rotation=24,ha='right')
    ax.set_ylabel('Macro-F1 (%)'); ax.set_ylim(0,100); ax.grid(axis='y',alpha=.25)
    ax.set_title('Exploratory Fakeddit test performance (mean ± seed SD)')
    files=_save(fig,out,'macro_f1_comparison')

    pred=pd.read_csv(result_dir/'predictions'/'D_42.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,column,title in zip(axes,['cosine','m_text','m_image'],['Paired CLIP cosine','Text-conditioned margin','Image-conditioned margin']):
        for label,name,color in [(0,'Fake (0)','#E45756'),(1,'Real (1)','#4C78A8')]:
            vals=pred.loc[pred.label.eq(label),column].to_numpy()
            ax.hist(vals,bins=35,density=True,histtype='step',linewidth=1.8,label=name,color=color)
        ax.set_title(title); ax.set_xlabel('Feature value'); ax.set_ylabel('Density'); ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Test feature distributions; descriptive, previously inspected test subset')
    files.update(_save(fig,out,'test_similarity_distributions'))

    confusion=read_json(result_dir/'confusion_matrices.json')['matrices']
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.4))
    for ax,key,title in zip(axes,['B_42','D_42'],['B: raw cosine','D: BRAS']):
        matrix=np.asarray(confusion[key],dtype=int)
        ax.imshow(matrix,cmap='Blues')
        for (i,j),value in np.ndenumerate(matrix): ax.text(j,i,str(value),ha='center',va='center',color='black')
        ax.set_xticks([0,1],['Fake','Real']); ax.set_yticks([0,1],['Fake','Real'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    fig.suptitle('Seed-42 exploratory test confusion matrices')
    files.update(_save(fig,out,'paired_confusion_matrices'))

    subgroup=pd.read_csv(result_dir/'subgroup_metrics.csv')
    names=[f'label6_{label}_{name}' for label,name in LABEL6.items()]
    rows=subgroup[(subgroup.seed==42)&subgroup.group.isin(names)&subgroup.variant.isin(['B','D'])]
    if not rows.empty:
        piv=rows.pivot(index='group',columns='variant',values='accuracy').reindex(names)
        counts=rows[rows.variant.eq('B')].set_index('group').reindex(names)['n'].to_numpy()
        fig,ax=plt.subplots(figsize=(10.2,4.8)); x=np.arange(len(names)); width=.36
        ax.bar(x-width/2,piv['B'].to_numpy()*100,width,label='B: raw cosine',color='#4C78A8')
        ax.bar(x+width/2,piv['D'].to_numpy()*100,width,label='D: BRAS',color='#E45756')
        display={0:'True',1:'Satire /\nParody',2:'False\nConnection',3:'Imposter\nContent',
                 4:'Manipulated\nContent',5:'Misleading\nContent'}
        ax.set_xticks(x,[f'{display[label]}\n(n={int(n)})' for label,n in zip(LABEL6,counts)],rotation=0,fontsize=9)
        ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,100); ax.set_title('Seed-42 accuracy by Fakeddit six-way label')
        ax.legend(frameon=False); ax.grid(axis='y',alpha=.25)
        files.update(_save(fig,out,'six_way_accuracy'))

    stress=result_dir/'stress'/'correspondence_rankings.csv'
    if stress.exists():
        df=pd.read_csv(stress); df=df[df.feature.isin(['cosine','m_text','m_image'])]
        pivot=df.pivot(index='feature',columns='condition',values='original_higher_fraction')
        fig,ax=plt.subplots(figsize=(8.2,4.6))
        pivot.reindex(['cosine','m_text','m_image']).plot(kind='bar',ax=ax,color=['#4C78A8','#F58518'])
        ax.set_xticklabels(['Cosine','Text margin','Image margin'],rotation=0)
        ax.set_ylabel('Fraction original score > swapped score'); ax.set_xlabel('Feature')
        ax.set_title('Synthetic correspondence ranking (not factual classification)')
        ax.set_ylim(0,1); ax.grid(axis='y',alpha=.25)
        ax.legend(title='Image replacement',frameon=False,loc='upper left',bbox_to_anchor=(1.02,1.0))
        fig.subplots_adjust(left=.12,right=.73,bottom=.16,top=.86)
        files.update(_save(fig,out,'synthetic_pair_ranking'))

    sensitivity=result_dir/'stress'/'bank_and_direction_sensitivity.csv'
    if sensitivity.exists():
        df=pd.read_csv(sensitivity); df=df[df.condition.str.startswith('bank_size_')]
        if not df.empty:
            fig,ax=plt.subplots(figsize=(6.4,4.0)); ax.plot(df.bank_size,df.macro_f1*100,marker='o')
            ax.set_xlabel('Reference bank size'); ax.set_ylabel('Macro-F1 (%)')
            ax.set_title('Fixed-head reference-bank-size sensitivity')
            ax.grid(alpha=.25); files.update(_save(fig,out,'bank_size_sensitivity'))

    manifest={'status':'completed','source_metrics_sha256':sha(metrics_path),
        'source_evaluation_sha256':sha(result_dir/'evaluation.json'),
        'source_distribution_sha256':sha(distributions),'source_predictions_sha256':sha(result_dir/'predictions'/'D_42.csv'),
        'figures':files,'parameters_and_metrics':'model_table.csv',
        'interpretation':'Figures are generated from saved per-seed experiment outputs; no values were hand-entered.'}
    if stress.exists(): manifest['source_stress_sha256']=sha(stress)
    write_json(out/'figures.json',manifest)
    return {'status':'completed','directory':str(out),'figures':list(files)}
