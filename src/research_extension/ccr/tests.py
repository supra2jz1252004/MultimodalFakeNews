import tempfile,unittest,copy,inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from .core import *
from .fit import choose_threshold,predict_policy,fit_head

def temporary_directory():
    root=Path(__file__).resolve().parents[3]/'results/research_extension/ccr/test_scratch'
    root.mkdir(parents=True,exist_ok=True)
    temp=tempfile.TemporaryDirectory(dir=root)
    require(Path(temp.name).resolve().is_relative_to(root.resolve()),'Temporary test path escaped CCR scratch')
    return temp

class CCRTests(unittest.TestCase):
    def test_utility_sign_and_clipping(self):
        y=np.array([0,1]);a=np.array([.4,.6]);b=np.array([.1,.9])
        self.assertTrue(((ce(y,a)-ce(y,b))>0).all());self.assertTrue(np.isfinite(ce(y,np.array([1.,0.]))).all())
        np.testing.assert_allclose(ce(y,b),-np.log(.9))
    def test_gate_input_allowlist(self):
        self.assertEqual(list(inspect.signature(gate_features).parameters),['p_a','p_b','cosine'])
        z=gate_features(np.array([0.,.3]),np.array([1.,.7]),np.array([.1,.2]))
        self.assertEqual(z.shape,(2,10));self.assertTrue(np.isfinite(z).all())
        self.assertFalse(set(FEATURE_NAMES)&{'label','label6','subreddit','sample_id','group_id'})
    def test_repeated_group_fold_assignment(self):
        groups=['a','b','a','c','b'];folds=fold_assign(groups,5,20261106)
        self.assertEqual(folds[0],folds[2]);self.assertEqual(folds[1],folds[4])
        np.testing.assert_array_equal(folds,fold_assign(groups,5,20261106))
    def test_train_only_partition_and_disjointness(self):
        f=pd.DataFrame({'group_id':[str(i//2) for i in range(500)],'split':'train','label':0})
        folds=fold_assign(f.group_id,5,20261106)
        for k in range(5):
            a,b,c=partition(f,folds,k,20261106);self.assertTrue(np.all(a.astype(int)+b+c==1))
            self.assertFalse(set(f.loc[a,'group_id'])&set(f.loc[c,'group_id']))
        f.loc[0,'split']='test'
        with self.assertRaises(ValueError):partition(f,folds,0,20261106)
    def test_scaler_excludes_internal_validation_and_held(self):
        x=np.arange(30,dtype=np.float32).reshape(10,3);mask=np.arange(10)<5
        a=fit_scaler(x,mask);x[~mask]=1e10;b=fit_scaler(x,mask)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    def test_threshold_ties_prefer_B(self):
        best,_=choose_threshold(np.zeros(20),np.zeros(4),np.array([0,1,0,1]),np.ones(4)*.5,np.ones(4)*.5)
        self.assertEqual(best['b_fraction'],1.)
    def test_paired_identity_and_seed_semantics(self):
        y=np.array([0,1,0,1]);p=np.array([.2,.8,.4,.6]);r=paired_stats(y,p,p,100,123)
        self.assertNotIn('seed',r);self.assertEqual(r['bootstrap_seed'],123)
        self.assertEqual(r['ci95']['log_loss'],[0.,0.]);self.assertEqual(r['mcnemar_exact_two_sided_p'],1.)
    def test_paired_corrected_and_loss_direction(self):
        y=np.array([0,1,0,1]);a=np.array([.8,.8,.4,.2]);b=np.array([.2,.8,.4,.7])
        r=paired_stats(y,a,b,100,42);self.assertEqual(r['corrected'],2);self.assertEqual(r['new_errors'],0)
        self.assertLess(r['delta_candidate_minus_base']['log_loss'],0)
    def test_cluster_bootstrap_preserves_cluster_units(self):
        y=np.array([0,0,1,1]);a=np.full(4,.5);b=np.array([.1,.1,.9,.9])
        r=paired_stats(y,a,b,200,42,np.array(['a','a','b','b']))
        self.assertEqual(r['clusters'],2);self.assertEqual(r['bootstrap_unit'],'known group/component')
        np.testing.assert_allclose(r['ci95']['log_loss'],[np.log(.5/.9)]*2)
    def test_metric_conventions(self):
        y=np.array([0,1]);p=np.array([.1,.9]);r=scores(y,p)
        self.assertAlmostEqual(r['brier'],.01);self.assertAlmostEqual(r['accuracy'],1);self.assertAlmostEqual(r['ece_10'],.1)
        self.assertEqual(scores(y,np.array([.5,.5]))['accuracy'],.5)
    def test_reliability_edges_cover_all_rows(self):
        r=reliability(np.array([0,1,0,1]),np.array([0.,1.,.5,.5]),10)
        self.assertEqual(sum(x['n'] for x in r),4);self.assertEqual(r[9]['n'],2)
    def test_prediction_shape_rejects_unpaired(self):
        with self.assertRaises(ValueError):paired_stats(np.array([0,1]),np.array([.2]),np.array([.2,.8]),100)
    def test_receipt_detects_tampering(self):
        with temporary_directory() as d:
            p=Path(d)/'x.csv';p.write_text('x\n1\n');seal(p);verify(p);p.write_text('x\n2\n')
            with self.assertRaises(ValueError):verify(p)
    def test_fallback_is_exactly_B(self):
        class Est:
            def predict(self,z):return z[:,0]-.5
        bundle={'intercept':0.,'ridge':Est(),'tree':Est()}
        thresholds={k:{'threshold':0.} for k in ['intercept','ridge_candidate','tree','cosine_gate','confidence_gate','entropy_gate']}
        policy={'temperatures':{'A':1.,'B':1.},'thresholds':thresholds,'classifier_seed':42,'random_b_fraction':.5,'guard_passed':False}
        f=pd.DataFrame({'sample_id':['a','b'],'label':[0,1],'subreddit':['x','y']});a=np.array([.2,.8]);b=np.array([.7,.3])
        out,_,_=predict_policy(f,a,b,np.array([.1,.2]),bundle,policy)
        np.testing.assert_array_equal(out['CCR'],b)
        changed=f.copy();changed['label']=1-changed.label;changed['subreddit']='hidden'
        out2,_,_=predict_policy(changed,a,b,np.array([.1,.2]),bundle,policy)
        for k in out:np.testing.assert_array_equal(out[k],out2[k])
    def test_held_labels_do_not_affect_training_or_early_stopping(self):
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,2)
            def forward(self,t,i,sim):return self.fc(torch.cat([t,i,sim],1))
        with temporary_directory() as td:
            root=Path(td);inp=root/'train.npz';inp.write_bytes(b'test input identity');norm=root/'normal.npz';norm.write_bytes(b'fixed');roles=root/'roles.csv';roles.write_bytes(b'fixed roles')
            proto={'head_config':{'lr':.001,'weight_decay':.0001},'head_budget':{'epochs':2,'patience':2,'batch_size':8}}
            x={k:torch.arange(20,dtype=torch.float32).reshape(-1,1)/20 for k in ['text','image','similarity']}
            fit=np.arange(20)<12;val=(np.arange(20)>=12)&(np.arange(20)<16);held=np.arange(20)>=16
            y=torch.tensor([i%2 for i in range(20)]);changed=y.clone();changed[held]=1-changed[held]
            results=[]
            for j,labels in enumerate([y,changed]):
                s=SimpleNamespace(models=root/f'm{j}',protocol=proto,smoke=True,project=root,path=lambda name:inp)
                with patch('src.research_extension.ccr.fit.make_head',side_effect=lambda *args:Tiny()),patch('src.research_extension.ccr.fit.source_hashes',return_value={}):
                    results.append(fit_head(s,None,x,labels,fit,val,held,'A',42,0,norm,roles))
            for key in results[0]['state_dict']:self.assertTrue(torch.equal(results[0]['state_dict'][key],results[1]['state_dict'][key]))
            self.assertEqual(results[0]['history'],results[1]['history']);self.assertEqual(results[0]['best_epoch'],results[1]['best_epoch'])
    def test_csls_sum_equivalence(self):
        s=np.array([.1,.2,.3]);rt=np.array([.05,.12,.1]);ri=np.array([.08,.15,.2])
        np.testing.assert_allclose((s-rt)+(s-ri),2*s-rt-ri)
    def test_repeated_swap_donors_preserve_order(self):
        from .evaluate import repeated_historical
        rows=pd.DataFrame({'sample_id':['r','q','r','r','q']})
        def load(ctx,unique,kinds):
            self.assertEqual(unique.sample_id.tolist(),['r','q'])
            return {'image':np.array([[10.,11.],[20.,21.]])}
        with patch('src.research_extension.ccr.evaluate.load_historical',side_effect=load):
            values=repeated_historical(None,rows,('image',))
        np.testing.assert_array_equal(values['image'],[[10,11],[20,21],[10,11],[10,11],[20,21]])

if __name__=='__main__':unittest.main()
