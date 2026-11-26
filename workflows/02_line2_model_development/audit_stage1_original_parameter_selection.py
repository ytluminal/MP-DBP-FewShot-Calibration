import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
P='/mnt/data/Line2_CV_Comparison/'
ns={}
src=open(P+'run.py').read().split('STAGES=[1,2,3,4]')[0]
exec(compile(src,P+'run.py','exec'),ns)
Y=ns['Y'];refs=ns['refs'];feat=ns['ALL_FEAT'][:22]
models=['LightGBM','CatBoost']
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def candidates(m):
 if m=='LightGBM':
  return [(n,l,lr) for n in [200,500] for l in [7,15,31] for lr in [.03,.1]]
 return [(600,2,.05,7,1.),(600,3,.05,10,1.),(600,4,.05,15,1.),(600,6,.03,15,2.)]

def model_for(m,params):
 if m=='LightGBM':
  n,l,lr=params
  return LGBMRegressor(n_estimators=n,num_leaves=l,learning_rate=lr,subsample=.8,colsample_bytree=.8,reg_lambda=1.,objective='regression',random_state=42,n_jobs=4,verbosity=-1)
 it,dep,lr,l2,rs=params
 return CatBoostRegressor(iterations=it,depth=dep,learning_rate=lr,l2_leaf_reg=l2,random_strength=rs,loss_function='RMSE',random_seed=42,verbose=False,allow_writing_files=False,nan_mode='Min',thread_count=3)

out={'source':'SI_Updating(1).docx Stage1 Text S5-S9, inner GroupKFold, 5 outer folds','models':{}}
for m in models:
 start=time.time(); pred=np.full(len(Y),np.nan)
 outcomes=[]
 for ifold,(outer_train,outer_val) in enumerate(ns['folds']['Publication-grouped']):
  splitter=GroupKFold(n_splits=3)
  combinations=candidates(m)
  scores=[]
  for j,setting in enumerate(combinations):
   f_scores=[]
   for it,iv in splitter.split(outer_train,groups=refs[outer_train]):
    tr=outer_train[it];val=outer_train[iv]
    Xtr,Xval,ci=ns['prepare'](tr,val,feat,m,1)
    md=model_for(m,setting)
    with warnings.catch_warnings():
     warnings.simplefilter('ignore')
     if m=='CatBoost':md.fit(Xtr,Y[tr],cat_features=ci)
     else:md.fit(Xtr,Y[tr])
    f_scores.append(float(r2_score(Y[val],md.predict(Xval))))
   scores.append(float(np.mean(f_scores)))
  bestid=int(np.argmax(scores));best=combinations[bestid]
  tr=outer_train;val=outer_val
  Xtr,Xval,ci=ns['prepare'](tr,val,feat,m,1)
  md=model_for(m,best)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   if m=='CatBoost':md.fit(Xtr,Y[tr],cat_features=ci)
   else:md.fit(Xtr,Y[tr])
  pred[val]=md.predict(Xval)
  foldval=float(r2_score(Y[val],pred[val]))
  record={'fold':ifold+1,'selected_params':best,'selected_inner_mean_r2':scores[bestid],'outer_r2':foldval,'inner_scores':scores}
  outcomes.append(record)
  print(m,ifold+1,'settings',best,'foldR2',round(foldval,7),'elapsed',round(time.time()-start,1),flush=True)
 final_r2=float(r2_score(Y,pred))
 out['models'][m]={'pooled_OOF_r2':final_r2,'folds':outcomes,'expected_SI_r2':0.1836723847 if m=='LightGBM' else 0.3258024638}
 np.savez_compressed(P+f'Audit_Stage1_{m}_GroupedOOF_reselected.npz',y=Y,ypred=pred,ref=refs)
 with open(P+'audit_stage1_results.json','w') as f:json.dump(out,f,indent=2)
 print('FINAL',m,final_r2,'expected',out['models'][m]['expected_SI_r2'],'time',round(time.time()-start,1),flush=True)
