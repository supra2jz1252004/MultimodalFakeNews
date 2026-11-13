"""Fake-positive metrics, exact paired tests, and bounded common-index bootstrap."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
    precision_recall_curve,auc,average_precision_score,confusion_matrix)
from .common import require

def fast_macro_f1(y,pred):
    tp=((y==0)&(pred==0)).sum(axis=-1);tn=((y==1)&(pred==1)).sum(axis=-1)
    fp=((y==1)&(pred==0)).sum(axis=-1);fn=((y==0)&(pred==1)).sum(axis=-1)
    return (2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))/2

def metrics(y,prob):
    y=np.asarray(y);prob=np.asarray(prob)
    require(len(y)>0 and prob.shape==(len(y),2),'Metric shapes/empty group')
    require(np.isin(y,[0,1]).all() and np.isfinite(prob).all(),'Bad labels/probabilities')
    require(np.allclose(prob.sum(1),1.,atol=1e-5) and ((prob>=0)&(prob<=1)).all(),'Invalid probabilities')
    pred=prob.argmax(1)
    result={'n':len(y),'n_fake':int((y==0).sum()),'n_real':int((y==1).sum()),
        'accuracy':float(accuracy_score(y,pred)),
        'fake_precision':float(precision_score(y,pred,pos_label=0,zero_division=0)),
        'fake_recall':float(recall_score(y,pred,pos_label=0,zero_division=0)),
        'fake_f1':float(f1_score(y,pred,pos_label=0,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
        'weighted_f1':float(f1_score(y,pred,labels=[0,1],average='weighted',zero_division=0)),
        'roc_auc':None,'pr_auc':None,'average_precision':None,
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'small_group':len(y)<100,'single_class':len(np.unique(y))<2}
    if len(np.unique(y))==2:
        pr,re,_=precision_recall_curve(1-y,prob[:,0])
        result.update(roc_auc=float(roc_auc_score(1-y,prob[:,0])),pr_auc=float(auc(re,pr)),
                      average_precision=float(average_precision_score(1-y,prob[:,0])))
    return result

def paired(y,base,proposed,repetitions=5000,seed=20260916):
    y=np.asarray(y);base=np.asarray(base);proposed=np.asarray(proposed)
    require(y.shape==base.shape==proposed.shape and y.ndim==1 and len(y)>0,'Unpaired/empty predictions')
    require(repetitions>0,'Bootstrap repetitions must be positive')
    b=base==y;p=proposed==y
    gained=int((~b&p).sum());lost=int((b&~p).sum())
    result={'n':len(y),'corrected':gained,'new_errors':lost,'both_correct':int((b&p).sum()),'both_wrong':int((~b&~p).sum()),
        'mcnemar_exact_two_sided_p':float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.,
        'delta_macro_f1':float(fast_macro_f1(y,proposed)-fast_macro_f1(y,base)),
        'delta_accuracy':float(p.mean()-b.mean()),'repetitions':repetitions,'seed':seed}
    rng=np.random.default_rng(seed);mf=[];ac=[]
    for start in range(0,repetitions,100):
        idx=rng.integers(0,len(y),size=(min(100,repetitions-start),len(y)))
        yy,bb,pp=y[idx],base[idx],proposed[idx]
        mf.extend((fast_macro_f1(yy,pp)-fast_macro_f1(yy,bb)).tolist())
        ac.extend(((yy==pp).mean(1)-(yy==bb).mean(1)).tolist())
    result['macro_f1_ci95']=np.quantile(mf,[.025,.975]).tolist()
    result['accuracy_ci95']=np.quantile(ac,[.025,.975]).tolist()
    result['scope']='conditional row bootstrap; correlated source events may require cluster bootstrap; seeds not pooled'
    return result

def holm(pvalues):
    order=np.argsort(pvalues);out=np.empty(len(order));highest=0.
    for rank,i in enumerate(order):
        highest=max(highest,min(1.,(len(order)-rank)*pvalues[i]));out[i]=highest
    return out.tolist()
