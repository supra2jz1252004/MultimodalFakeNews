"""Training-only semantic-similarity exploration before final classifier fitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .feature_extractors import load_features
from .utils import atomic_json, log_stage


def analyze(root: Path, smoke: bool = False):
    frame,features=load_features(root,smoke)
    mask=frame.split.eq('train').to_numpy()
    data=pd.DataFrame({'sample_id':frame.loc[mask,'sample_id'].to_numpy(),
                       'label':frame.loc[mask,'label'].to_numpy(),
                       'similarity':features['similarity'][mask,0]})
    suffix='_smoke' if smoke else ''
    metrics=root/'results/metrics'
    plots=root/'results/plots'
    metrics.mkdir(parents=True,exist_ok=True)
    plots.mkdir(parents=True,exist_ok=True)
    data.groupby('label').similarity.describe().to_csv(metrics/f'similarity_summary{suffix}.csv')
    fake=data.loc[data.label.eq(0),'similarity'].to_numpy()
    real=data.loc[data.label.eq(1),'similarity'].to_numpy()
    result=mannwhitneyu(real,fake,alternative='two-sided',method='asymptotic')
    effect=2*float(result.statistic)/(len(real)*len(fake))-1
    report={'scope':'training only','fake_count':len(fake),'real_count':len(real),
            'mann_whitney_u_real_vs_fake':float(result.statistic),'p_value':float(result.pvalue),
            'p_value_display': 'p < 1e-300 (floating-point underflow)' if result.pvalue==0 else f'p = {result.pvalue:.6g}',
            'rank_biserial_real_minus_fake':effect,
            'effect_definition':'P(real similarity > fake similarity) minus P(real similarity < fake similarity), ties counted equally',
            'interpretation':'Distribution association is not evidence of incremental classifier benefit; distant-supervision labels and source styles can confound it.'}
    atomic_json(metrics/f'similarity_statistics{suffix}.json',report)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    bins=np.linspace(min(fake.min(),real.min()),max(fake.max(),real.max()),36)
    axes[0].hist(fake,bins=bins,density=True,alpha=.6,label=f'Fake (n={len(fake):,})',color='#bb4b4b')
    axes[0].hist(real,bins=bins,density=True,alpha=.6,label=f'Real (n={len(real):,})',color='#337b9e')
    axes[0].set(xlabel='Paired CLIP cosine similarity',ylabel='Density',title='Training similarity distribution')
    axes[0].legend(frameon=False)
    axes[1].boxplot([fake,real],tick_labels=['Fake','Real'],showfliers=False)
    axes[1].set(ylabel='Paired CLIP cosine similarity',title='Median and interquartile range')
    fig.savefig(plots/f'similarity_distribution{suffix}.png',dpi=200)
    fig.savefig(plots/f'similarity_distribution{suffix}.pdf')
    plt.close(fig)
    print(json.dumps(report),flush=True)
    log_stage(root,'Training-only CLIP similarity analysis'+(' smoke' if smoke else ''),
              f'Command: `python run_experiments.py analyze-similarity {"--smoke" if smoke else ""}`. '
              f'{len(fake)} fake and {len(real)} real training pairs. Two-sided Mann–Whitney U={result.statistic:.1f}, '
              f'{report["p_value_display"]}, rank-biserial effect (real minus fake)={effect:.6f}. '
              'Means, medians, SD and quartiles saved in similarity_summary CSV; histogram and box plot saved. '
              'This checks association, not added classification value. Validation and test scores were not used.')
