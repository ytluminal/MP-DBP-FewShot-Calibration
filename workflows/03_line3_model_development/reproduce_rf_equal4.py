"""Reproduce frozen Line 3 RF & Equal4, all S1-S4, both schemes, CV and Test."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,csv,json,hashlib,importlib.util,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
ROOT=Path(__file__).resolve().parent; source=ROOT/'input';out=ROOT/'outputs';out.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--stages',nargs='+',type=int,choices=[1,2,3,4],default=[1,2,3,4]);args=parser.parse_args()
spec=importlib.util.spec_from_file_location('frozen_line3',source/'run.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
ntrain=len(m.Y);ntest=len(m.T55['rows']);m.Xall=np.concatenate([m.Xall,np.asarray(m.T55['rows'],dtype=object)],axis=0)
TIDX=np.arange(ntrain,ntrain+ntest);ytest=np.log1p([float(r[22]) for r in m.T55['rows']])
references={}
for kind in ['CV','Test']:
 references[kind]={}
 with open(ROOT/'reference'/f'Line3_{kind}_Row_Predictions.csv') as f:
  for row in csv.DictReader(f):references[kind][(int(row['Stage']),row['Validation_Scheme'],int(row['Row_1based']))]=row
results=[]
def compare(stage,scheme,typ,model,p,y):
 ref=references[typ]; gold=np.array([float(ref[stage,scheme,i+1][f'{model}_pred_ln1p']) for i in range(len(y))])
 delta=float(np.max(np.abs(p-gold)));scores=m.metrics(y,p);goldscores=m.metrics(y,gold)
 results.append(dict(Stage=stage,Scheme=scheme,Dataset=typ,Model=model,R2=scores[0],RMSE=scores[1],MAE=scores[2],Reference_R2=goldscores[0],Reference_RMSE=goldscores[1],Reference_MAE=goldscores[2],Max_Row_Diff=delta,Within_1e_7=bool(delta<1e-7)))
 print('RESULT',stage,scheme,typ,model,'R2',round(scores[0],9),'ref',round(goldscores[0],9),'maxd',round(delta,9),flush=True)
for stage in args.stages:
 for scheme in ['Publication-grouped','Conventional random']:
  feats=m.ALL_FEAT[:22] if stage==1 else (m.subsets['RF'] if stage==4 else m.ALL_FEAT)
  oof=np.full(ntrain,np.nan);testfold=[]
  for fi,(tr,va) in enumerate(m.FOLDS[scheme]):
   oof[va]=m.model_predict(tr,va,feats,'RF',stage,fi)
   testfold.append(m.model_predict(tr,TIDX,feats,'RF',stage,fi))
   gold=np.array([float(references['CV'][stage,scheme,i+1]['RF_pred_ln1p']) for i in va]);print('FOLD',stage,scheme,fi+1,'maxerr',float(np.max(abs(oof[va]-gold))),flush=True)
  prtest=np.mean(testfold,axis=0);compare(stage,scheme,'CV','RF',oof,m.Y);compare(stage,scheme,'Test','RF',prtest,ytest)
  key=f'{stage}_{scheme.replace(" ","_").replace("-","_")}'
  np.savez_compressed(out/f'RF_Stage{key}.npz',CV=oof,Test=prtest,Test_fold=np.array(testfold),Y_train=m.Y,Y_test=ytest)
  with np.load(ROOT/'verified_base_predictions'/f'CV_Stage{key}.npz') as cv,np.load(ROOT/'verified_base_predictions'/f'Test_Stage{key}.npz') as test:
   for typ,y,parts in [('CV',m.Y,[oof,cv['XGBoost'],cv['LightGBM'],cv['CatBoost']]),('Test',ytest,[prtest,test['XGBoost'],test['LightGBM'],test['CatBoost']])]:
    p=np.mean(np.column_stack(parts),axis=1);compare(stage,scheme,typ,'Equal4',p,y);np.save(out/f'Equal4_{typ}_Stage{key}.npy',p)
with open(out/('RF_Equal4_Audit_stage'+''.join(map(str,args.stages))+'.csv'),'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
meta={'total_results':len(results),'passed':sum(x['Within_1e_7'] for x in results),'max_row_diff':max(x['Max_Row_Diff'] for x in results),'environment':threadpool_info(),'source_code_sha256':hashlib.sha256((source/'run.py').read_bytes()).hexdigest()}
(out/('Run_Summary_stage'+''.join(map(str,args.stages))+'.json')).write_text(json.dumps(meta,indent=2));print('SUMMARY',meta['passed'],'/',meta['total_results'],'MAX',meta['max_row_diff'],flush=True)

if meta['passed'] != meta['total_results']: raise SystemExit('FAIL: Predicted results do not match frozen Figure 2c reference at row-level tolerance 1e-7')
