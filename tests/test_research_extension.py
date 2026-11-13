import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.research_extension.common import fit_scaler,standardize,LABEL6,Context,write_json,require
from src.research_extension.reference import select_bank,margins,valid_candidates
from src.research_extension.features import random_control,unimodal_control
from src.research_extension.statistics import metrics,paired,fast_macro_f1,holm
from src.research_extension.training import make_head,seed_all,extras,VARIANTS,fit_one
from src.research_extension.fresh_holdout import _candidate_rows,_clean_text,_image_key

PROJECT = Path(__file__).resolve().parents[1]
TEMP_ROOT=Path(os.environ.get('BRAS_TEST_TMP',str(Path.cwd()/'work'))).resolve()
TEMP_ROOT.mkdir(parents=True,exist_ok=True)

def frame(n=6):
    return pd.DataFrame({'sample_id':[f'id{i}' for i in range(n)],'split':['train']*n,
        'group_id':[f'g{i}' for i in range(n)],'input_hash':[f'p{i}' for i in range(n)],
        'text_key':[f't{i}' for i in range(n)],'sha256':[f'h{i}' for i in range(n)],
        'pixel_sha256':[f'x{i}' for i in range(n)],'label':[i%2 for i in range(n)]})

def normalized(x):
    x=np.array(x,dtype=np.float32);return x/np.linalg.norm(x,axis=1,keepdims=True)

class ReferenceTests(unittest.TestCase):
    def test_cosine_and_directional_margins(self):
        q=frame(1);q['sample_id']='q';q['group_id']='q';q['input_hash']='q';q['text_key']='q';q['sha256']='q';q['pixel_sha256']='q'
        b=frame(2)
        t=normalized([[1,0]]);v=normalized([[.6,.8]])
        bt=normalized([[0,1],[1,0]]);bv=normalized([[1,0],[-1,0]])
        x,n,k=margins(t,v,q,bt,bv,b,1,1)
        np.testing.assert_allclose(x,[[.6,-.4,-.2]],atol=1e-6)
        self.assertEqual(n.tolist(),[2]);self.assertEqual(k.tolist(),[1])

    def test_bank_deterministic_and_order_invariant(self):
        f=frame(20); a=select_bank(f,7,123)
        shuffled=f.sample(frac=1,random_state=3).reset_index(drop=True)
        b=select_bank(shuffled,7,123)
        self.assertEqual(f.loc[a].sample_id.tolist(),shuffled.loc[b].sample_id.tolist())
        self.assertFalse(np.array_equal(a,select_bank(f,7,124)))

    def test_train_only_and_family_aware(self):
        f=frame(10);f.loc[0:2,'split']='test';f.loc[3:4,'split']='validation';f.loc[5:6,'group_id']='same'
        bank=f.loc[select_bank(f,4)]
        self.assertTrue(bank.split.eq('train').all());self.assertTrue(bank.group_id.is_unique)
        with self.assertRaises(ValueError): select_bank(f,5)

    def test_self_family_pair_image_and_text_exclusion(self):
        q=frame(1);b=frame(8)
        for j,key in enumerate(['group_id','input_hash','text_key','sha256','pixel_sha256'],1): b.loc[j,key]=q.loc[0,key]
        self.assertEqual(valid_candidates(q,b).tolist(),[[False]*6+[True,True]])

    def test_fewer_than_k_and_zero_candidates(self):
        f=frame(2);x=normalized([[1,0],[0,1]])
        out,n,k=margins(x,x,f,x,x,f,16)
        np.testing.assert_allclose(out,[[1,1,1],[1,1,1]])
        self.assertEqual(k.tolist(),[1,1])
        out,n,k=margins(x[:1],x[:1],f.iloc[:1],x[:1],x[:1],f.iloc[:1],16)
        self.assertTrue(np.isnan(out[0,1:]).all());self.assertEqual(k.tolist(),[0])

    def test_empty_bank_missing_data(self):
        f=frame(3);x=normalized([[1,0],[0,1],[1,1]])
        out,n,k=margins(x,x,f,np.empty((0,2),np.float32),np.empty((0,2),np.float32),f.iloc[:0],16)
        self.assertTrue(np.isnan(out[:,1:]).all());self.assertEqual(n.tolist(),[0,0,0])

    def test_reject_test_bank_and_nonfinite(self):
        f=frame(2);x=normalized([[1,0],[0,1]])
        b=f.copy();b.loc[0,'split']='test'
        with self.assertRaises(ValueError): margins(x,x,f,x,x,b)
        x[0,0]=np.nan
        with self.assertRaises(ValueError): margins(x,x,f,x,x,f)

    def test_batch_size_invariance(self):
        f=frame(17);rng=np.random.default_rng(15)
        t=normalized(rng.normal(size=(17,7)));v=normalized(rng.normal(size=(17,7)))
        a=margins(t,v,f,t[:9],v[:9],f.iloc[:9],3,1)
        b=margins(t,v,f,t[:9],v[:9],f.iloc[:9],3,11)
        np.testing.assert_allclose(a[0],b[0],atol=2e-7)
        np.testing.assert_array_equal(a[1],b[1])

class ContractTests(unittest.TestCase):
    def test_normalizer_training_only(self):
        x=np.array([[1,2],[3,2],[1e9,1e9]],np.float32)
        a=fit_scaler(x,[True,True,False]);np.testing.assert_array_equal(a['mean'],[2,2])
        x[-1]=0;b=fit_scaler(x,[True,True,False])
        np.testing.assert_array_equal(a['std'],b['std'])
        self.assertTrue(np.isfinite(standardize(x,a)).all())

    def test_random_controls_reorder_and_labels_independent(self):
        a=random_control(['a','b','c'],8);b=random_control(['c','a','b'],8)
        np.testing.assert_array_equal(a,b[[1,2,0]])
        self.assertFalse(np.array_equal(a,random_control(['a','b','c'],9)))

    def test_dimensions_all_variants(self):
        raw=np.ones((5,1),np.float32);g={'bras':np.ones((5,3),np.float32),'random':np.zeros((5,2),np.float32),'unimodal':np.zeros((5,2),np.float32)}
        for kind,width in VARIANTS.items():self.assertEqual(extras(kind,g,raw).shape,(5,width))

    def test_immutable_contracts(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            p=Path(tmp)/'lock.json';write_json(p,{'a':1},immutable=True);write_json(p,{'a':1},immutable=True)
            with self.assertRaises(ValueError):write_json(p,{'a':2},immutable=True)

    def test_binary_semantics_and_category_mapping(self):
        y=np.array([0,0,1,1]);prob=np.array([[.9,.1],[.8,.2],[.3,.7],[.1,.9]])
        m=metrics(y,prob);self.assertEqual(m['fake_recall'],1);self.assertEqual(m['roc_auc'],1)
        self.assertEqual(LABEL6[2],'False Connection');self.assertEqual(LABEL6[4],'Manipulated Content');self.assertEqual(LABEL6[5],'Misleading Content')

    def test_fresh_holdout_pool_excludes_retained_ids_groups_text_and_urls(self):
        def row(sid,text,url,label,group):
            return {'id':sid,'clean_title':text,'image_url':url,'linked_submission_id':'',
                'hasImage':'true','2_way_label':str(label),'6_way_label':'0' if label==1 else '1',
                'subreddit':'test','created_utc':'0'}
        rows=[row('same_id','same id','https://x.test/id.png',0,'g0'),
            row('same_group','different','https://x.test/group.png',0,'g0'),
            row('same_text','Same title','https://x.test/text.png',0,'g1'),
            row('same_url','other title','https://x.test/id.png?size=small',1,'g2'),
            row('fresh_fake','new fake','https://x.test/fake.png',0,'g3'),
            row('fresh_real','new real','https://x.test/real.png',1,'g4')]
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            root=Path(tmp);raw=root/'data/raw/fakeddit/multimodal_test_public.tsv'
            raw.parent.mkdir(parents=True);pd.DataFrame(rows).to_csv(raw,sep='\t',index=False)
            processed=root/'data/processed';processed.mkdir(parents=True)
            pd.DataFrame({'sample_id':[r['id'] for r in rows],'split':['test']*len(rows),
                'group_id':[r['linked_submission_id'] or {'same_id':'g0','same_group':'g0','same_text':'g1',
                    'same_url':'g2','fresh_fake':'g3','fresh_real':'g4'}[r['id']] for r in rows],
                'eligible':[True]*len(rows)}).to_csv(processed/'metadata_audit.csv.gz',index=False,compression='gzip')
            pd.DataFrame([{'sample_id':'same_id','group_id':'g0','text':'Same title',
                'image_url':'https://x.test/id.png','sha256':'old','pixel_sha256':'old-pixel'}]).to_csv(
                    processed/'manifest.csv',index=False)
            pool,_=_candidate_rows(Context(root,root))
            self.assertEqual(set(pool.sample_id),{'fresh_fake','fresh_real'})
            self.assertEqual(pool.label.value_counts().to_dict(),{0:1,1:1})

    def test_fresh_holdout_normalization(self):
        self.assertEqual(_clean_text('  A\u00a0 title  '),'A title')
        self.assertEqual(_image_key('https://Example.test/path.png?width=120'),'example.test/path.png')

    def test_paired_null_and_single_class_groups(self):
        y=np.array([0,1,1,0]);p=np.array([0,0,1,0]);r=paired(y,p,p,100,4)
        self.assertEqual(r['mcnemar_exact_two_sided_p'],1);self.assertEqual(r['macro_f1_ci95'],[0,0])
        m=metrics(np.zeros(3,int),np.array([[.9,.1]]*3));self.assertIsNone(m['roc_auc']);self.assertTrue(m['single_class'])

    def test_bootstrap_fixed_seed_and_holm(self):
        y=np.array([0,1,1,0]);a=np.array([0,0,1,1]);b=y.copy()
        self.assertEqual(paired(y,a,b,100,4),paired(y,a,b,100,4))
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

@unittest.skipUnless((PROJECT/'src/models.py').exists(),'Historical project required for integration test')
class TrainingTests(unittest.TestCase):
    def test_shared_initialization_and_parameter_counts(self):
        cfg={'dimension':128,'dropout':.3}
        seed_all(42);a=make_head('A',cfg,PROJECT).eval()
        xx=torch.randn(4,768);ii=torch.randn(4,2048);aout=a(xx,ii,torch.empty(4,0))
        counts={}
        for kind in ['A','B','C','D','E','I']:
            seed_all(42);m=make_head(kind,cfg,PROJECT).eval()
            for key,value in a.state_dict().items():
                other=m.state_dict()[key]
                torch.testing.assert_close(value,other[:,:256] if key=='fusion.weight' else other,rtol=0,atol=0)
            torch.testing.assert_close(aout,m(xx,ii,torch.randn(4,VARIANTS[kind])),rtol=0,atol=1e-7)
            counts[kind]=sum(p.numel() for p in m.parameters())
        self.assertEqual(counts['A'],393858);self.assertEqual(counts['D']-counts['A'],384)
        self.assertEqual(counts['D'],counts['E']);self.assertEqual(counts['D'],counts['I'])

    def test_training_resume_matches_uninterrupted(self):
        f=frame(12);f.loc[8:,'split']='validation'
        rng=np.random.default_rng(1)
        features={'text':torch.from_numpy(rng.normal(size=(12,768)).astype(np.float32)),
                  'image':torch.from_numpy(rng.normal(size=(12,2048)).astype(np.float32)),
                  'variants':{'D':torch.from_numpy(rng.normal(size=(12,3)).astype(np.float32))}}
        labels=torch.tensor(f.label.to_numpy())
        protocol={'config':{'dimension':8,'dropout':.3,'lr':.001,'weight_decay':.0001},
                  'budget':{'threads':1,'batch_size':4,'epochs':3,'patience':3}}
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp,patch('src.research_extension.training.lock_protocol',return_value=protocol),patch('src.research_extension.training.binding',return_value={'test':1}):
            a=Context(PROJECT,Path(tmp)/'a',smoke=True);b=Context(PROJECT,Path(tmp)/'b',smoke=True)
            full=fit_one(a,f,features,labels,'D',42,{})
            fit_one(b,f,features,labels,'D',42,{},interrupt_after=1)
            resumed=fit_one(b,f,features,labels,'D',42,{})
            self.assertEqual(full['history'],resumed['history'])
            for key,value in full['state_dict'].items():torch.testing.assert_close(value,resumed['state_dict'][key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
