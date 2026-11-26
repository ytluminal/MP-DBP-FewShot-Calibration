import os,json,time,sys, warnings
os.environ['OPENBLAS_NUM_THREADS']='5';os.environ['OMP_NUM_THREADS']='3';os.environ['MKL_NUM_THREADS']='5'
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from scipy.sparse import csr_matrix,hstack
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from docx import Document
from pathlib import Path
warnings.filterwarnings('ignore',category=FutureWarning)
BASE=str(Path(__file__).resolve().parent)+'/'
D55=json.load(open(BASE+'train55.json'))
h=D55['headers'];rows=D55['rows'];n=len(rows)
Xall=np.asarray(rows,dtype=object)
Y=np.log1p(np.asarray([r[22] for r in rows],float))
refs=np.array([int(r[23]) for r in rows]); ref_count=Counter(refs)
assert n==995 and np.isfinite(Y).all()
CATCOLS=['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']
ALL_FEAT=h[:22]+h[24:]
SI=Document(Path(BASE)/'SI_Verified.docx')
SUBSETS={}
for j,model in enumerate(['RF','XGBoost','LightGBM','CatBoost'],1):
 SUBSETS[model]=[s.strip().rstrip('.') for s in SI.tables[21].rows[j].cells[1].text.split(';')]
 assert all(c in ALL_FEAT for c in SUBSETS[model]), (model,[c for c in SUBSETS[model] if c not in ALL_FEAT])
assert [len(SUBSETS[a]) for a in SUBSETS]==[37,15,18,40]
GROUP=[{4,8,9,13,18},{5,10,15,23},{2,12,17,19,25,27},{1,11,24},{3,6,20,26}]
folds={ 'Publication-grouped': [(np.flatnonzero(~np.isin(refs,list(g))),np.flatnonzero(np.isin(refs,list(g)))) for g in GROUP], 'Conventional random':list(KFold(n_splits=5,shuffle=True,random_state=42).split(np.arange(n)))}
for typ, fs in folds.items():
 a=np.concatenate([z[1] for z in fs]); assert len(a)==995 and sorted(a)==list(range(995))
 print(typ,'foldsizes',[len(t) for tr,t in fs],flush=True)

# Frozen 5-fold core configs per SI: Stage 1 configs for RF and XGB (native CatBoost/LGB preprocessed baseline configs not recorded, Stage2 frozen core used as proxy)
C1={
 'RF':[(16,0.5,1),(8,'sqrt',3),(16,0.5,1),(16,0.5,1),(None,0.5,3)],
 'XGBoost':[(500,4,.03),(500,4,.1),(200,6,.1),(200,2,.1),(500,4,.1)]
}
C2={
 'RF':[(None,'sqrt',1),(8,'sqrt',3),(8,'sqrt',3),(None,'sqrt',1),(16,0.5,1)],
 'XGBoost':[(500,6,.1),(200,4,.03),(200,4,.1),(500,4,.1),(200,4,.1)],
 'LightGBM':[(500,31,.1),(200,31,.03),(200,15,.1),(200,7,.1),(200,15,.1)],
 'CatBoost':[(600,3,.05,10,1.),(600,6,.03,15,2.),(600,6,.03,15,2.),(600,6,.03,15,2.),(600,2,.05,7,1.)]
}
for m in ['LightGBM','CatBoost']:C1[m]=C2[m]
# pre-set robustness parameters per frozen SI Table S18
A={'RF':[0,.5,0,.5,0], 'XGBoost':[1,.75,.25,.5,0], 'LightGBM':[.25,.5,0,1,.5], 'CatBoost':[0,.75,.25,0,.75]}
XGB_LOSS=[('reg:pseudohubererror',.25),('reg:pseudohubererror',.25),('reg:pseudohubererror',.25),('reg:pseudohubererror',.5),('reg:squarederror',None)]
CAT_LOSS=[('Huber',.5),('Huber',.3),('Huber',2.),('Huber',.5),('Huber',2.)]
# source input column indexes

def prepare(trainidx, validx, feats, model,stage):
 ind=[h.index(c) for c in feats]
 cat_local=[i for i,c in enumerate(feats) if c in CATCOLS]
 num_local=[i for i in range(len(feats)) if i not in cat_local]
 A0=Xall[np.ix_(trainidx,ind)]
 B0=Xall[np.ix_(validx,ind)]
 if model=='CatBoost':
  def to_data(z):
   out=np.empty(z.shape,dtype=object)
   for i in range(z.shape[1]):
    if i in cat_local: out[:,i]=['__MISSING__' if v is None or v=='' else str(v) for v in z[:,i]]
    else:out[:,i]=[float(v) if v is not None and v!='' else np.nan for v in z[:,i]]
   return out
  return to_data(A0),to_data(B0),cat_local
 # numeric
 def num(z):return np.array([[np.nan if z[i,j] is None or z[i,j]=='' else float(z[i,j]) for j in num_local] for i in range(len(z))],float)
 a=num(A0);b=num(B0)
 # Stage 1 LGB KNN, Stage 2+ LGB KNN, RF and XGB median
 imputer=KNNImputer(n_neighbors=5,weights='distance',metric='nan_euclidean') if model=='LightGBM' else SimpleImputer(strategy='median')
 an=imputer.fit_transform(a);bn=imputer.transform(b)
 scaler=StandardScaler()
 an=scaler.fit_transform(an);bn=scaler.transform(bn)
 # categorical mode; use exact categorical order of selected feature subgroups
 def cats(z):return np.asarray([['__NAN__' if z[i,j] is None or z[i,j]=='' else str(z[i,j]) for j in cat_local] for i in range(len(z))], dtype=object).reshape(len(z),len(cat_local))
 if cat_local:
  ac=cats(A0);bc=cats(B0)
  ac[ac=='__NAN__']=np.nan;bc[bc=='__NAN__']=np.nan
  mode=SimpleImputer(strategy='most_frequent')
  ac=mode.fit_transform(ac);bc=mode.transform(bc)
  enc=OneHotEncoder(handle_unknown='ignore',dtype=np.float64,sparse_output=(model!='RF'))
  ac=enc.fit_transform(ac);bc=enc.transform(bc)
  if model=='RF': return np.hstack([an,ac]), np.hstack([bn,bc]),None
  return hstack([csr_matrix(an),ac],format='csr'),hstack([csr_matrix(bn),bc],format='csr'),None
 if model=='RF':return an,bn,None
 return csr_matrix(an),csr_matrix(bn),None

def build(model,stage,fold):
 pars=(C1 if stage==1 else C2)[model][fold]
 if model=='RF':
  depth,mx,leaf=pars
  return RandomForestRegressor(n_estimators=200,max_depth=depth,max_features=mx,min_samples_leaf=leaf,random_state=42,n_jobs=3)
 if model=='XGBoost':
  ne,md,lr=pars
  cfg={'n_estimators':ne,'max_depth':md,'learning_rate':lr,'subsample':.8,'colsample_bytree':.8,'reg_lambda':1.,'objective':'reg:squarederror','random_state':42,'tree_method':'hist','n_jobs':3,'verbosity':0}
  if stage>=3:
   loss,sl=XGB_LOSS[fold];cfg['objective']=loss
   if sl is not None:cfg['huber_slope']=sl
  return XGBRegressor(**cfg)
 if model=='LightGBM':
  ne,nl,lr=pars
  return LGBMRegressor(n_estimators=ne,num_leaves=nl,learning_rate=lr,subsample=.8,colsample_bytree=.8,reg_lambda=1.,objective='regression',random_state=42,n_jobs=3,verbosity=-1)
 if model=='CatBoost':
  ni,depth,lr,l2,rnd=pars
  args={'iterations':ni,'depth':depth,'learning_rate':lr,'l2_leaf_reg':l2,'random_strength':rnd,'loss_function':'RMSE','random_seed':42,'verbose':False,'allow_writing_files':False,'nan_mode':'Min','thread_count':3}
  if stage>=3:
   loss,delta=CAT_LOSS[fold];args['loss_function']=f'Huber:delta={delta}' if loss=='Huber' else 'RMSE'
  return CatBoostRegressor(**args)

def fit(m,Xtrain,ytr,catidx,stage,fold,tr):
 kwargs={}
 if stage>=3:
  alpha=A[m][fold]
  if alpha>0:
   nref=np.array([ref_count[refs[j]] if False else np.sum(refs[tr]==refs[j]) for j in tr])
   # For grouped or random, count only rows present in active training fold
   w=(nref.astype(float)**(-alpha));w/=w.mean()
   kwargs['sample_weight']=w
 if m=='CatBoost':kwargs['cat_features']=catidx
 mobj=build(m,stage,fold)
 mobj.fit(Xtrain,ytr,**kwargs)
 return mobj

STAGES=[1,2,3,4]
models=['RF','XGBoost','LightGBM','CatBoost']
results=[]
for stage in STAGES:
 for typ,fs in folds.items():
  predictions={m:np.full(n,np.nan) for m in models}
  for i,(tr,va) in enumerate(fs):
   t=time.perf_counter()
   for m in models:
    feats=ALL_FEAT[:22] if stage==1 else (SUBSETS[m] if stage==4 else ALL_FEAT)
    Xtrain,Xvalid,catidx=prepare(tr,va,feats,m,stage)
    md=fit(m,Xtrain,Y[tr],catidx,stage,i,tr)
    pred=md.predict(Xvalid)
    predictions[m][va]=pred
   print(f'Stage {stage} | {typ} | fold{i+1} done {time.perf_counter()-t:.1f}s',flush=True)
  predictions['Equal4']=np.mean(np.column_stack([predictions[m] for m in models]),axis=1)
  for m,pred in predictions.items():
   assert np.isfinite(pred).all()
   rr={'Stage':stage,'CV type':typ,'Model':m,'CV_R2':float(r2_score(Y,pred)),'CV_RMSE':float(np.sqrt(mean_squared_error(Y,pred))),'CV_MAE':float(mean_absolute_error(Y,pred))}
   results.append(rr)
   print('RESULT',rr,flush=True)
  np.savez_compressed(BASE+f'OOF_Stage{stage}_{typ.replace(" ","_").replace("-","_")}.npz',y=Y,**predictions)
 with open(BASE+'partial_results.json','w') as f:json.dump(results,f,indent=2)
print('ALL FINISHED',flush=True)
