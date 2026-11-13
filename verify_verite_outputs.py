"""Verify saved real outputs independently; replay has no training or encoder fitting."""
from pathlib import Path
import sys,io,unittest,hashlib,subprocess,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from src.research_extension.verite.core import *
from src.research_extension.verite.features import load,validate_inputs
from src.research_extension.verite.statistics import resampling
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,log_loss
s=Study(P);s.frozen();verify_sources(s);f,z=load(s);validate_inputs(s,f);torch.set_num_threads(6)
unit=unittest.defaultTestLoader.loadTestsFromName('src.research_extension.verite.tests');stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unit)
require(result.wasSuccessful(),'Unit regression failure')
metrics=pd.read_csv(s.results/'model_metrics.csv');counts=0;replays=[];all_predictions={}
idx=np.array(sorted(range(len(f)),key=lambda i:hashlib.sha256(f'20261109\0{f.iloc[i].row_id}'.encode()).hexdigest())[:64])
normal=np.load(P/'models/research_extension/bras_b4096_k16_s20260916/normalization.npz');original=np.load(P/'models/normalization.npz')
for seed in SEEDS:
 pred=pd.read_csv(s.results/f'predictions/seed_{seed}.csv',keep_default_na=False);all_predictions[seed]=pred;y=pred.binary_label.to_numpy()
 require(pred.row_id.tolist()==f.row_id.tolist(),'Prediction alignment changed')
 for kind in KINDS:
  prob=pred['p_'+kind].to_numpy();decision=pred['decision_'+kind].to_numpy();row=metrics[(metrics.seed==seed)&(metrics.model==kind)].iloc[0];pp=np.clip(prob,1e-6,1-1e-6)
  independent={'accuracy':accuracy_score(y,decision),'macro_f1':f1_score(y,decision,labels=[0,1],average='macro'),
   'roc_auc_truthful':roc_auc_score(y,prob),'average_precision_truthful':average_precision_score(y,prob),
   'log_loss':log_loss(y,np.c_[1-pp,pp],labels=[0,1]),'brier':float(np.mean((prob-y)**2))}
  for name,value in independent.items():require(abs(row[name]-value)<1e-10,'Independent metric mismatch');counts+=1
  if kind=='CCR':require(np.array_equal(prob,pred.p_B) and np.array_equal(decision,pred.decision_B),'CCR identity control failed');continue
  path=P/(f"models/final/{'baseline' if kind=='A' else 'proposed'}_{seed}.pt" if kind in ['A','B'] else f'models/research_extension/bras_b4096_k16_s20260916/{kind}_{seed}.pt')
  ck=torch.load(path,weights_only=True,map_location='cpu');w={k:v.numpy() for k,v in ck['state_dict'].items()}
  tx=(z['text'][idx]-original['text_mean'])/original['text_std'];im=(z['image'][idx]-original['image_mean'])/original['image_std']
  parts=[np.maximum(tx@w['text.weight'].T+w['text.bias'],0),np.maximum(im@w['image.weight'].T+w['image.bias'],0)]
  if kind!='A':
   raw=z['similarity'][idx] if kind=='B' else pred.iloc[idx][['m_text','m_image'] if kind=='C' else ['cosine','m_text','m_image']].to_numpy(np.float32)
   parts.append((raw-normal[kind+'_mean'])/normal[kind+'_std'])
  logits=np.maximum(np.concatenate(parts,1)@w['fusion.weight'].T+w['fusion.bias'],0)@w['output.weight'].T+w['output.bias']
  e=np.exp(logits-logits.max(1,keepdims=True));p=e[:,1]/e.sum(1);error=float(np.max(np.abs(p-prob[idx])))
  require(error<2e-6 and np.array_equal(logits.argmax(1),decision[idx]),'Independent NumPy checkpoint replay failed')
  replays.append({'seed':seed,'model':kind,'records':len(idx),'max_probability_error':error})
 require(np.array_equal(pred.decision_A,(pred[['logit0_A','logit1_A']].to_numpy().argmax(1))),'Argmax mapping differs')
align=pd.read_csv(s.results/'alignment_predictions.csv');stats=read(s.results/'alignment_statistics.json')
score_error=float(np.max(np.abs(np.einsum('ij,ij->i',z['clip_text'].astype(float),z['clip_image'].astype(float))-align.cosine)))
require(score_error<1e-6,'CLIP vector/scalar mismatch')
for score in SCORES:
 r=stats['discrimination'][score]
 require(abs(roc_auc_score(align.binary_label,align[score])-r['roc_auc_truthful'])<1e-12,'Alignment AUC mismatch')
 require(abs(average_precision_score(align.binary_label,align[score])-r['average_precision_truthful'])<1e-12,'Alignment AP mismatch')
 pairs=pd.read_csv(s.results/f'alignment/paired_{score}.csv');lookup=align.set_index('row_id')[score]
 require(np.allclose(pairs.margin,lookup.loc[pairs.true_row_id].to_numpy()-lookup.loc[pairs.misleading_row_id].to_numpy(),atol=1e-11),'Paired score join mismatch')
 for cat,g in pairs.groupby('category'):
  r=stats['paired'][score][cat];require(abs(((g.margin>0)+.5*(g.margin==0)).mean()-r['ranking_half'])<1e-12,'Ranking recomputation mismatch')
# Independently expand 30 cluster bootstrap samples and use sklearn, not weighted custom formulas.
weights,codes,keys=resampling(f.cluster_id);pred=all_predictions[42];boot=np.load(s.results/'bootstrap/classification_42.npz')
for i in range(30):
 expanded=np.repeat(np.arange(len(f)),weights[i,codes]);y=pred.binary_label.to_numpy()[expanded]
 for model in ['A','B','C','D']:
  decision=pred['decision_'+model].to_numpy()[expanded];value=f1_score(y,decision,labels=[0,1],average='macro')
  require(abs(value-boot[model][i,1])<1e-12,'Cluster-bootstrap expansion mismatch')
hashes={p.relative_to(s.results).as_posix():sha(p) for p in list((s.results/'predictions').glob('*.csv'))+[s.results/'model_metrics.csv',s.results/'paired_classification.json',s.results/'alignment_statistics.json',s.results/'protocol.json']}
commands=[]
for stage in ['audit','lock','extract','evaluate','statistics']:
 begin=time.perf_counter();run=subprocess.run([sys.executable,'-B','run_verite_eval.py',stage],cwd=P,capture_output=True,text=True)
 require(run.returncode==0,'Resume failed: '+stage+'\n'+run.stderr)
 commands.append({'stage':stage,'exit_code':run.returncode,'seconds':time.perf_counter()-begin});print('Verified resume:',stage,flush=True)
for rel,h in hashes.items():require(sha(s.results/rel)==h,'Resume changed measured output: '+rel)
record={'passed':True,'unit_tests':result.testsRun,'independent_classifier_metric_estimates':counts,'alignment_auc_ap_checks':8,
 'paired_ranking_checks':8,'cluster_bootstrap_expansion_checks':120,'checkpoint_replays':replays,
 'actual_VERITE_probabilities_replayed':sum(x['records'] for x in replays),'max_CLIP_vector_scalar_difference':score_error,
 'resumed_stages':commands,'measured_hashes_unchanged':hashes,'images_zip_sha256':sha(s.data/'images.zip'),'annotations_sha256':sha(s.data/'VERITE.csv')}
print('Verified',counts,'classifier metrics;',sum(x['records'] for x in replays),'real frozen-head predictions;',result.testsRun,'tests',flush=True)

print('Existing verification receipt retained; replay verified without overwriting it.', flush=True)
