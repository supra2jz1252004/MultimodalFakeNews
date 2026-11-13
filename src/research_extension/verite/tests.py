"""Synthetic software tests and read-only frozen checkpoint replay; not benchmark evidence."""
import unittest,tempfile,io,zipfile,stat
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
from .core import *
from .audit import zip_members,annotation_frame,pairing_checks,decode_hash,extract_archive
from .features import validate_inputs
from .models import infer_heads
from .statistics import binary_metrics,cluster_swap_test,resampling,boot_pair,paired_table,holm

def temp():
    root=Path(__file__).resolve().parents[3]/'results/research_extension/verite/test_tmp';root.mkdir(parents=True,exist_ok=True)
    folder=tempfile.TemporaryDirectory(dir=root);require(Path(folder.name).resolve().is_relative_to(root.resolve()),'Unsafe cleanup path');return folder
def fixture():
    return pd.DataFrame({'caption':['true1','false1','true1','true2','false2','true2'],
      'image_path':['images/true_1.jpg','images/true_1.jpg','images/false_1.jpg','images/true_2.jpg','images/true_2.jpg','images/false_2.jpg'],
      'label':['true','miscaptioned','out-of-context']*2})
class Tests(unittest.TestCase):
    def test_annotation_mapping(self):
        f=annotation_frame(fixture());self.assertEqual(f.binary_label.tolist(),[1,0,0,1,0,0])
    def test_unknown_label_refused(self):
        f=fixture();f.loc[0,'label']='fake'
        with self.assertRaises(ValueError):annotation_frame(f)
    def test_true_misc_shared_image(self):
        f=annotation_frame(fixture());result,links=pairing_checks(f);self.assertEqual(result['paired_counts'],{'miscaptioned':2,'out-of-context':2})
        f.loc[1,'image_path']='images/true_2.jpg'
        with self.assertRaises(ValueError):pairing_checks(f)
    def test_true_ooc_shared_caption(self):
        f=annotation_frame(fixture());f.loc[2,'caption']='different'
        with self.assertRaises(ValueError):pairing_checks(f)
    def test_missing_ooc_preserves_existing_pair(self):
        f=annotation_frame(fixture().iloc[:-1]);result,links=pairing_checks(f);self.assertEqual(result['paired_counts']['out-of-context'],1)
    def test_graph_links_families_by_caption(self):
        f=annotation_frame(fixture());self.assertEqual(clusters(f).nunique(),2);f.loc[3,'text_key']=f.loc[0,'text_key'];self.assertEqual(clusters(f).nunique(),1)
    def test_graph_links_pixels(self):
        f=annotation_frame(fixture());f['pixel_sha256']='';f.loc[[0,3],'pixel_sha256']='same';self.assertEqual(clusters(f).nunique(),1)
    def test_zip_traversal_refused(self):
        for name in ['../bad.jpg','C:/bad.jpg','/bad.jpg','images/CON.jpg']:
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w') as z:z.writestr(name,b'x')
            with zipfile.ZipFile(data) as z:
                with self.assertRaises(ValueError):zip_members(z)
    def test_raw_zip_backslash_refused(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:z.writestr('images/bad.jpg',b'x')
        raw=data.getvalue().replace(b'images/bad.jpg',b'images'+bytes([92])+b'bad.jpg')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with self.assertRaises(ValueError):zip_members(z)
    def test_zip_symlink_refused(self):
        data=io.BytesIO();info=zipfile.ZipInfo('images/x.jpg');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
        with zipfile.ZipFile(data,'w') as z:z.writestr(info,b'other')
        with zipfile.ZipFile(data) as z:
            with self.assertRaises(ValueError):zip_members(z)
    def test_zip_case_collision_refused(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:z.writestr('images/x.jpg',b'a');z.writestr('images/X.jpg',b'b')
        with zipfile.ZipFile(data) as z:
            with self.assertRaises(ValueError):zip_members(z)
    def test_image_integrity_and_resume(self):
        from PIL import Image
        with temp() as directory:
            root=Path(directory);buf=io.BytesIO();Image.new('RGB',(3,2),(1,20,30)).save(buf,format='PNG');archive=root/'x.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('images/x.png',buf.getvalue())
            first=extract_archive(archive,root/'out');second=extract_archive(archive,root/'out');self.assertTrue(first.equals(second))
            class Mock:images=root/'out'
            f=pd.DataFrame({'image_path':['images/x.png'],'available':[True],**{k:[first.iloc[0][k]] for k in ['sha256','pixel_sha256']}})
            validate_inputs(Mock(),f);(root/'out/images/x.png').write_bytes(b'broken')
            with self.assertRaises(Exception):validate_inputs(Mock(),f)
    def test_image_absence_refused(self):
        class Mock:images=Path('.')
        with self.assertRaises(ValueError):validate_inputs(Mock(),pd.DataFrame({'available':[False]}))
    def test_metrics_sklearn(self):
        rng=np.random.default_rng(44);y=rng.integers(0,2,50);p=np.round(rng.uniform(size=50),2);d=(p>.5).astype(int);w=rng.integers(0,4,50)
        v=binary_metrics(y,p,d,w);self.assertAlmostEqual(v[0],accuracy_score(y,d,sample_weight=w),places=12);self.assertAlmostEqual(v[1],f1_score(y,d,average='macro',sample_weight=w),places=12)
        self.assertAlmostEqual(v[2],roc_auc_score(y,p,sample_weight=w),places=12);self.assertAlmostEqual(v[3],average_precision_score(y,p,sample_weight=w),places=12)
    def test_threshold_tie_is_misleading(self):
        v=binary_metrics(np.array([0,1]),np.array([.5,.5]),np.array([0,0]));self.assertEqual(v[0],.5);self.assertAlmostEqual(v[1],1/3)
    def test_cluster_randomization_one_cluster(self):
        y=np.array([0,1,0,1]);result=cluster_swap_test(y,1-y,y,np.array(['one']*4),reps=500)
        self.assertEqual(result['macro_f1_p'],1.);self.assertEqual(result['accuracy_p'],1.)
    def test_randomization_identity(self):
        y=np.array([0,1]);r=cluster_swap_test(y,y,y,['a','b'],reps=20);self.assertEqual(r['macro_f1_p'],1.)
    def test_bootstrap_reproducibility(self):
        a=resampling(['a','a','b'],30);b=resampling(['a','a','b'],30)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(a[0][:,a[1]][:,0],a[0][:,a[1]][:,1])
    def test_paired_tie_half_credit(self):
        f=annotation_frame(fixture());f['cluster_id']=clusters(f);f['source']='synthetic';f['cosine']=[.5,.5,.1,.4,.8,.4]
        _,links=pairing_checks(f);links['complete']=True;table=paired_table(f,links,'cosine')
        self.assertEqual(table.ranking_half.tolist(),[.5,1.,0.,.5])
        table['cluster_id']='all';w,c,k=resampling(table.cluster_id,20);b=boot_pair(table,w,c);self.assertTrue(np.allclose(b,b[0]))
    def test_holm_correction(self):self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
    def test_bras_reference_train_purity(self):
        from ..reference import margins
        t=np.array([[1,0]],np.float32);b=np.array([[1,0],[0,1]],np.float32);query=pd.DataFrame({'text_key':['same']});bank=pd.DataFrame({'split':['train','train'],'text_key':['same','other']})
        m,n,e=margins(t,t,query,b,b,bank,k=16);np.testing.assert_array_equal(m,[[1,1,1]]);self.assertEqual(n[0],1)
        bank['split']='test'
        with self.assertRaises((ValueError,AssertionError)):margins(t,t,query,b,b,bank)
    def test_cache_binding_and_immutability(self):
        with temp() as directory:
            path=Path(directory)/'x.npz';npz(path,{'contract':1},x=np.array([1.]));h=sha(path);npz(path,{'contract':1},x=np.array([1.]));self.assertEqual(h,sha(path))
            with self.assertRaises(ValueError):npz(path,{'contract':2},x=np.array([1.]))
    def test_preservation_detects_mutation(self):
        with temp() as directory:
            s=Study(directory);path=s.root/'original.pt';path.write_bytes(b'original');write(s.results/'historical_inventory.json',{'files':{'original.pt':{'sha256':sha(path),'bytes':8}}})
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):preservation(s)
    def test_frozen_checkpoint_replay(self):
        import torch
        torch.set_num_threads(6)
        class Mock:root=Path(__file__).resolve().parents[3]
        s=Mock();rng=np.random.default_rng(9);z={'text':rng.normal(size=(4,768)).astype(np.float32),'image':rng.normal(size=(4,2048)).astype(np.float32),'similarity':np.array([[-.1],[.1],[.2],[.3]],np.float32)}
        bras=pd.DataFrame({'cosine':z['similarity'][:,0],'m_text':[-.3,-.1,.0,.1],'m_image':[-.4,-.2,-.1,.0]})
        normal=np.load(s.root/'models/research_extension/bras_b4096_k16_s20260916/normalization.npz')
        for seed in SEEDS:
            actual,_=infer_heads(s,z,bras,seed)
            for kind in ['A','B','C','D']:
                path=s.root/(f"models/final/{'baseline' if kind=='A' else 'proposed'}_{seed}.pt" if kind in ['A','B'] else f'models/research_extension/bras_b4096_k16_s20260916/{kind}_{seed}.pt')
                ck=torch.load(path,weights_only=True,map_location='cpu');w={k:v.numpy() for k,v in ck['state_dict'].items()}
                tx=(z['text']-normal['text_mean'])/normal['text_std'];im=(z['image']-normal['image_mean'])/normal['image_std']
                parts=[np.maximum(tx@w['text.weight'].T+w['text.bias'],0),np.maximum(im@w['image.weight'].T+w['image.bias'],0)]
                if kind!='A':
                    raw=z['similarity'] if kind=='B' else bras[['m_text','m_image'] if kind=='C' else ['cosine','m_text','m_image']].to_numpy(np.float32)
                    parts.append((raw-normal[kind+'_mean'])/normal[kind+'_std'])
                hidden=np.maximum(np.concatenate(parts,1)@w['fusion.weight'].T+w['fusion.bias'],0);logits=hidden@w['output.weight'].T+w['output.bias']
                exp=np.exp(logits-logits.max(1,keepdims=True));p=exp[:,1]/exp.sum(1)
                np.testing.assert_allclose(actual[kind]['probability'],p,atol=2e-6,rtol=2e-6);np.testing.assert_array_equal(actual[kind]['decision'],logits.argmax(1))

if __name__=='__main__':unittest.main()
