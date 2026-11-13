"""Synthetic invariants only; no invented scientific benchmark outcomes."""
import io,pickle,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .data import *
from .statistics import *
from .acquire import NumpyOnly
from .evaluation import transfer,load_features
TEMP_ROOT=Path(__file__).resolve().parents[3]/'results/external_eval/newsclippings/test_tmp'
def temporary():
    TEMP_ROOT.mkdir(parents=True,exist_ok=True)
    temp=tempfile.TemporaryDirectory(dir=TEMP_ROOT)
    require(Path(temp.name).resolve().is_relative_to(TEMP_ROOT.resolve()),'Unsafe test cleanup path')
    return temp

def fixture():
    obj={'source_datasets':{'0':'semantics_clip_text_image'},'annotations':[]}
    for i in range(8):
        obj['annotations'] += [{'id':i,'image_id':i,'falsified':False,'source_dataset':0},
            {'id':i,'image_id':100+i,'falsified':True,'source_dataset':0}]
    f=parse_pairs(obj);f['source']='synthetic';f['cluster_id']=linked_clusters(f)
    f['cosine']=[.8,.2,.5,.5,.1,.7,.9,.4,.2,.1,.3,.5,.8,.6,.4,.2]
    return obj,f
class Tests(unittest.TestCase):
    def test_semantics(self):
        obj,f=fixture();self.assertTrue((f[f.falsified.eq(0)].caption_id==f[f.falsified.eq(0)].image_id).all())
        self.assertEqual(len(f),16);self.assertTrue(f.split.eq('test').all())
    def test_bad_pair_rejected(self):
        obj,_=fixture();obj['annotations'][1]['id']=99
        with self.assertRaises(ValueError):parse_pairs(obj)
    def test_bad_flag_rejected(self):
        obj,_=fixture();obj['annotations'][0]['falsified']=True
        with self.assertRaises(ValueError):parse_pairs(obj)
    def test_duplicate_group_rejected(self):
        obj,_=fixture();obj['annotations']+=obj['annotations'][:2]
        with self.assertRaises(ValueError):parse_pairs(obj)
    def test_split_validation(self):
        obj,_=fixture()
        with self.assertRaises(ValueError):parse_pairs(obj,'invented')
    def test_caption_image_join(self):
        _,f=fixture();lookup={str(i):{'caption':f'text{i}','image_path':f'{i}.jpg','source':'x'} for i in list(range(8))+list(range(100,108))}
        out=join_metadata(f,lookup);self.assertEqual(out.iloc[1].caption,'text0');self.assertEqual(out.iloc[1].filename,'100.jpg')
    def test_selection_order_invariance(self):
        _,f=fixture();f=pd.concat([f.assign(strategy=st,pair_id=f.pair_id+st) for st in STRATEGIES])
        self.assertEqual(select(f,8).pair_id.tolist(),select(f.sample(frac=1,random_state=2),8).pair_id.tolist())
    def test_linked_cross_role_items(self):
        _,f=fixture();f.loc[3,'image_id']='0';clusters=linked_clusters(f)
        self.assertEqual(clusters.iloc[0],clusters.iloc[2])
    def test_article_links_across_roles(self):
        _,f=fixture();f['article_key']='';f['image_article_key']='';f.loc[0,'article_key']='story';f.loc[2,'image_article_key']='story'
        clusters=linked_clusters(f);self.assertEqual(clusters.iloc[0],clusters.iloc[2])
    def test_image_guard(self):
        _,f=fixture()
        with self.assertRaises(ValueError):require_images(f,Path('.'))
    def test_full_transfer_guard(self):
        class Mock:
            def frozen(self):return {}
        with temporary() as d:
            mock=Mock();mock.results=Path(d)
            with self.assertRaisesRegex(ValueError,'no verified'):transfer(mock)
    def test_cosine_sign_and_normalization(self):
        t=np.eye(3,dtype=np.float32);v=t.copy();self.assertTrue(np.allclose(np.einsum('ij,ij->i',t,v),1))
        self.assertEqual(float(t[0]@t[1]),0)
    def test_bras_exclusion(self):
        from ...research_extension.reference import margins
        bank=pd.DataFrame({'split':['train','train'],'text_key':['same','different']})
        query=pd.DataFrame({'text_key':['same']});t=np.array([[1,0]],np.float32);bt=np.array([[1,0],[0,1]],np.float32)
        scores,n,k=margins(t,t,query,bt,bt,bank,k=16)
        self.assertEqual(n.tolist(),[1]);np.testing.assert_array_equal(scores,[[1,1,1]])
        bank['split']='test'
        with self.assertRaises((ValueError,AssertionError)):margins(t,t,query,bt,bt,bank)
    def test_tie_half_credit(self):
        _,f=fixture();p=paired(f,'cosine');self.assertEqual(p.loc[p.pair_id.eq(f.iloc[2].pair_id),'ranking_half'].item(),.5)
        self.assertEqual(p.loc[p.pair_id.eq(f.iloc[2].pair_id),'ranking_strict'].item(),0)
    def test_incomplete_group_rejected(self):
        _,f=fixture()
        with self.assertRaises(ValueError):paired(f.iloc[1:],'cosine')
    def test_auc_ap_independent(self):
        rng=np.random.default_rng(11);y=rng.integers(0,2,100);s=np.round(rng.normal(size=100),1);w=rng.integers(0,5,100)
        self.assertAlmostEqual(auc_formula(y,s,w),roc_auc_score(y,s,sample_weight=w),places=12)
        self.assertAlmostEqual(ap_formula(y,s,w),average_precision_score(y,s,sample_weight=w),places=12)
    def test_cluster_resampling_keeps_group(self):
        _,f=fixture();p=paired(f,'cosine');p['cluster_id']='one_component'
        values=bootstrap(p,25);self.assertTrue(np.allclose(values,point(p)[None,:]))
    def test_bootstrap_reproducibility(self):
        _,f=fixture();p=paired(f,'cosine');np.testing.assert_array_equal(bootstrap(p,40,42),bootstrap(p,40,42))
    def test_identity_comparison(self):
        _,f=fixture();p=paired(f,'cosine');r=comparisons({'cosine':p,'bras_mean':p.copy()},30)[0]
        self.assertTrue(all(x==0 for x in r['effect'].values()));self.assertEqual(r['ranking_cluster_signflip_p'],1)
    def test_holm(self):self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
    def test_immutable_cache(self):
        with temporary() as d:
            p=Path(d)/'x.npz';npz(p,{'source':'x'},a=np.array([1.]));h=sha(p)
            npz(p,{'source':'x'},a=np.array([1.]));self.assertEqual(h,sha(p))
            with self.assertRaises(ValueError):npz(p,{'source':'different'},a=np.array([1.]))
            with self.assertRaises(ValueError):npz(p,{'source':'x'},a=np.array([2.]))
    def test_pickle_guard(self):
        data=pickle.dumps({'x':np.array([1.],dtype=np.float32)},protocol=4);obj=NumpyOnly(io.BytesIO(data)).load();self.assertEqual(obj['x'][0],1.)
        data=pickle.dumps(Path('sensitive'),protocol=4)
        with self.assertRaises(ValueError):NumpyOnly(io.BytesIO(data)).load()
    def test_preservation_detection(self):
        with temporary() as d:
            r=Run(d);p=Path(d)/'frozen.pt';p.write_bytes(b'original');write(r.results/'historical_inventory.json',{'files':{'frozen.pt':{'bytes':8,'sha256':sha(p)}}})
            p.write_bytes(b'modified')
            with self.assertRaises(ValueError):preservation(r)
    def test_no_veracity_accuracy_output(self):
        _,f=fixture();r=summary(paired(f,'cosine'),20)
        self.assertEqual(set(r['metrics']),set(METRICS));self.assertNotIn('fake_news_accuracy',r)
    def test_download_resume(self):
        class Response:
            status_code=206;headers={'Content-Length':'3','Content-Range':'bytes 3-5/6'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
            def iter_content(self,*args):yield b'def'
        with temporary() as d:
            p=Path(d)/'data.bin';p.with_suffix('.bin.part').write_bytes(b'abc')
            with patch('requests.get',return_value=Response()) as get:
                download('https://official.example/file',p,max_bytes=6)
                self.assertEqual(get.call_args.kwargs['headers'],{'Range':'bytes=3-'})
            self.assertEqual(p.read_bytes(),b'abcdef');verify(p,{'url':'https://official.example/file'})
            with patch('requests.get',side_effect=RuntimeError('network must not be called')):download('https://official.example/file',p,max_bytes=6)
    def test_download_wrong_range_refused(self):
        class Response:
            status_code=206;headers={'Content-Range':'bytes 0-2/6'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
        with temporary() as d:
            p=Path(d)/'data.bin';p.with_suffix('.bin.part').write_bytes(b'abc')
            with patch('requests.get',return_value=Response()):
                with self.assertRaisesRegex(ValueError,'resume range'):download('https://official.example/file',p,max_bytes=6)
            self.assertFalse(p.exists());self.assertEqual(p.with_suffix('.bin.part').read_bytes(),b'abc')
    def test_real_image_hash_integrity(self):
        from PIL import Image
        with temporary() as d:
            p=Path(d)/'image.png';Image.new('RGB',(2,2),(15,20,25)).save(p)
            _,f=fixture();f=f.iloc[:2].copy();f['filename']='image.png';f['caption']='synthetic test pixels'
            f=validate_images(f,d);require_images(f,d);self.assertTrue(f.sha256.str.len().eq(64).all())
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):require_images(f,d)
    def test_official_cannot_masquerade_as_pinned(self):
        class Mock:
            def binding(self):return {'protocol_sha256':'fixed'}
        with temporary() as d:
            r=Mock();r.data=Path(d);r.results=Path(d);r.root=Path(d)
            npz(Path(d)/'pinned_features.npz',{'protocol_sha256':'fixed','provenance':'official_precomputed'},ids=np.array(['x']))
            with self.assertRaisesRegex(ValueError,'Pinned features require'):load_features(r,'pinned')
    def test_transfer_metric_names_synthetic_fixture(self):
        """Synthetic software fixture with real frozen heads; never research evidence."""
        from PIL import Image
        from .evaluation import source_contract
        project=Path(__file__).resolve().parents[3]
        class Mock:
            root=project
            def frozen(self):return {}
            def binding(self):return {'protocol_sha256':sha(self.results/'protocol.json')}
        with temporary() as d:
            r=Mock();r.results=Path(d);r.data=Path(d);write(r.results/'protocol.json',{'synthetic_test_only':True})
            _,f=fixture();f['caption']='synthetic software fixture';f['filename']='pixels.png'
            Image.new('RGB',(2,2),(15,20,25)).save(Path(d)/'pixels.png')
            f=validate_images(f,d);csv(r.results/'local_manifest.csv',f)
            write(r.results/'local_input_receipt.json',{'authorized_root':d})
            rng=np.random.default_rng(101)
            arrays={'text':rng.normal(size=(len(f),768)).astype(np.float32),'image':rng.normal(size=(len(f),2048)).astype(np.float32),'similarity':rng.uniform(.1,.4,size=(len(f),1)).astype(np.float32)}
            npz(Path(d)/'pinned_features.npz',{**r.binding(),'feature_provenance':'pinned_local_real_images','manifest_sha256':sha(r.results/'local_manifest.csv'),
                'encoder_contract_sha256':sha(project/'data/features/extraction.json')},ids=f.row_id.to_numpy(dtype=str),**arrays)
            result=transfer(r,d)
            for row in result['seeds']:
                effects=pd.read_csv(r.results/f"transfer/paired_{row['classifier_seed']}.csv")
                for name,value in row['means'].items():self.assertAlmostEqual(value,effects[name].mean(),places=10)
                predictions=pd.read_csv(r.results/f"transfer/predictions_{row['classifier_seed']}.csv")
                np.testing.assert_array_equal(predictions.p_B,predictions.p_CCR_identity)
            self.assertGreater(result['head_loading_and_inference_seconds'],0)

if __name__=='__main__':unittest.main()
