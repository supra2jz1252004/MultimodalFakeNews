"""Report figures from saved measurements and predictions."""
from pathlib import Path
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from .models import NAMES
from .utils import log_stage

COLORS=['#406c9e','#c48934','#467b65','#9c557b']


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def plots(root: Path):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110,'pdf.fonttype':42})
    directory=root/'results/plots'
    directory.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(root/'data/processed/manifest.csv')
    counts=pd.crosstab(manifest.split,manifest.label).reindex(['train','validation','test'])
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    counts.rename(columns={0:'Fake',1:'Real'}).plot.bar(ax=axes[0],color=[COLORS[3],COLORS[0]],rot=0)
    axes[0].set(title='Retained pairs by official split',xlabel='',ylabel='Samples')
    lengths=pd.read_csv(root/'data/processed/token_lengths.csv')
    axes[1].hist(lengths.bert,bins=35,color=COLORS[0])
    axes[1].set(title='Training BERT token lengths',xlabel='Tokens including special tokens',ylabel='Samples')
    axes[2].hist(np.log2(manifest.width),bins=35,alpha=.6,label='Width',color=COLORS[0])
    axes[2].hist(np.log2(manifest.height),bins=35,alpha=.6,label='Height',color=COLORS[1])
    axes[2].set(title='Decoded image dimensions',xlabel='log2(pixels)',ylabel='Samples')
    axes[2].legend(frameon=False)
    save(fig,directory/'dataset_audit')
    composition=pd.crosstab(manifest.split,manifest.label6,normalize='index').reindex(['train','validation','test'])*100
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    composition.rename(columns={0:'0: real',1:'1: satire/parody',2:'2: false connection',3:'3: imposter',4:'4: manipulated',5:'5: misleading'}).plot.bar(stacked=True,ax=ax,rot=0,colormap='tab20')
    ax.set(title='Retained six-way source categories (classifier target is binary)',xlabel='',ylabel='Percent of split',ylim=(0,100))
    ax.legend(title='Official six-way label',bbox_to_anchor=(1.02,1),loc='upper left',frameon=False,fontsize=8)
    save(fig,directory/'source_composition')
    lock=json.loads((root/'models/locked_config.json').read_text())
    seed=lock['primary_paired_seed']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for color,(kind,name) in zip(COLORS,NAMES.items()):
        history=pd.read_csv(root/f'models/final/{kind}_{seed}.csv')
        axes[0].plot(history.epoch,history.train_loss,color=color,linestyle='--',alpha=.65)
        axes[0].plot(history.epoch,history.validation_loss,color=color,label=name)
        axes[1].plot(history.epoch,history.validation_macro_f1,color=color,label=name)
    axes[0].set(title=f'Loss, seed {seed} (dashed: train; solid: validation)',xlabel='Epoch',ylabel='Cross-entropy')
    axes[1].set(title=f'Validation Macro-F1, seed {seed}',xlabel='Epoch',ylabel='Macro-F1')
    axes[1].legend(frameon=False,fontsize=8)
    save(fig,directory/'training_curves')
    comparison=pd.read_csv(root/'results/metrics/model_comparison.csv').set_index('kind').loc[list(NAMES)]
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    x=np.arange(len(NAMES))
    for i,(metric,title) in enumerate([('accuracy','Accuracy'),('macro_f1','Macro-F1'),('roc_auc','ROC-AUC')]):
        ax.bar(x+(i-1)*.24,comparison[metric],width=.24,yerr=comparison[metric+'_std'],capsize=3,label=title,color=COLORS[i])
    ax.set_xticks(x,list(NAMES.values()))
    ax.set(ylim=(0,1.04),ylabel='Test score',title=f'Mean test metrics across {len(lock["seeds"])} seeds (error bars: sample SD)')
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12))
    save(fig,directory/'model_comparison')
    fig,axes=plt.subplots(2,2,figsize=(8,7),layout='constrained')
    curve_fig,curve_axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,color,(kind,name) in zip(axes.ravel(),COLORS,NAMES.items()):
        pred=pd.read_csv(root/f'results/predictions/{kind}_{seed}.csv')
        ConfusionMatrixDisplay.from_predictions(pred.true_label,pred.predicted_label,labels=[0,1],display_labels=['Fake','Real'],
                                                colorbar=False,cmap='Blues',ax=ax)
        ax.set_title(name)
        fp,tp,_=roc_curve(1-pred.true_label,pred.probability_fake)
        precision,recall,_=precision_recall_curve(1-pred.true_label,pred.probability_fake)
        curve_axes[0].plot(fp,tp,label=name,color=color)
        curve_axes[1].plot(recall,precision,label=name,color=color)
    fig.suptitle(f'Test confusion matrices, prespecified seed {seed}')
    save(fig,directory/'confusion_matrices')
    curve_axes[0].plot([0,1],[0,1],color='gray',linestyle='--',linewidth=1)
    curve_axes[0].set(title=f'Test ROC curves, seed {seed}',xlabel='False-positive rate',ylabel='True-positive rate')
    curve_axes[1].axhline(pred.true_label.eq(0).mean(),color='gray',linestyle='--',linewidth=1,label='Fake prevalence')
    curve_axes[1].set(title=f'Test precision–recall curves, seed {seed}',xlabel='Recall (fake)',ylabel='Precision (fake)')
    curve_axes[1].legend(frameon=False,fontsize=8)
    save(curve_fig,directory/'roc_pr_curves')
    examples=pd.read_csv(root/'results/predictions/error_examples.csv')
    for number,(category,group) in enumerate(examples.groupby('category'),1):
        fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
        for ax in axes.ravel():
            ax.axis('off')
        for ax,row in zip(axes.ravel(),group.itertuples()):
            with Image.open(root/row.image_path) as im:
                rgb=ImageOps.exif_transpose(im).convert('RGB')
                rgb.thumbnail((700,700))
                ax.imshow(rgb)
            caption=f'{row.sample_id}; true={row.true_label}; C={row.baseline_prediction}; D={row.proposed_prediction}\nCLIP={row.similarity:.3f}\n'+textwrap.fill(row.text,38)[:170]
            ax.set_title(caption,fontsize=8)
        fig.suptitle(category+' (0=fake, 1=real; examples span similarity range)')
        save(fig,directory/f'error_examples_{number}')
    log_stage(root,'Report plots',
              'Command: `python run_experiments.py plots`. Saved PNG (200 dpi) and vector PDF: dataset classes/token lengths/image dimensions, '
              'training/validation loss, validation Macro-F1, mean test metrics with seed SD, confusion matrices, ROC/PR curves, and four error-example panels. '
              'All figures read actual manifests, histories, predictions and metric tables; no synthetic curves.')
