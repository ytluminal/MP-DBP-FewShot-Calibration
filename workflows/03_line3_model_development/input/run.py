import os,sys,time,json,re,warnings,argparse,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from docx import Document
warnings.filterwarnings('ignore',category=UserWarning)
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=os.path.dirname(os.path.abspath(__file__))+'/'
D22=json.load(open(ROOT+'train22.json'));D55=json.load(open(ROOT+'train55.json'))
T22=json.load(open(ROOT+'test22.json'));T55=json.load(open(ROOT+'test55.json'))
headers=D55['headers'];r22=D22['rows'];r55=D55['rows'];n=len(r55)
assert n==1026 and len(T22['rows'])==218 and len(T55['rows'])==218
assert D22['headers']==headers[:24] and T22['headers']==T55['headers'][:24]
assert all(a==b[:24] for a,b in zip(r22,r55))
assert all(a==b[:24] for a,b in zip(T22['rows'],T55['rows']))
assert len(set(tuple(str(x) for x in r) for r in r55))>100
Y=np.log1p(np.array([float(r[22]) for r in r55]))
refs=np.array([int(r[23]) for r in r55]);testrefs=np.array([int(r[23]) for r in T55['rows']])
assert set(refs).isdisjoint(set(testrefs)) and len(set(refs))==20 and len(set(testrefs))==7
Xall=np.asarray(r55,dtype=object)
CATCOLS=['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']
ALL_FEAT=headers[:22]+headers[24:]
assert len(ALL_FEAT)==55 and len(set(ALL_FEAT))==55
SI=Document(os.path.join(ROOT,'SI_Conventional_RandomSplit_Line3_Stage1-4.docx'))
subsets={}
for i,model in enumerate(['RF','XGBoost','LightGBM','CatBoost'],1):
 subset=[a.strip().rstrip('.') for a in SI.tables[52].rows[i].cells[6].text.split(';')]
 assert all(x in ALL_FEAT for x in subset), (model,subset)
 subsets[model]=subset
assert [len(subsets[m]) for m in ['RF','XGBoost','LightGBM','CatBoost']]==[14,14,14,28]
GROUPS=[{1,3,8,10},{2,9,18,19},{5,7,17,25},{6,15,27},{16,21,22,23,24}]
FOLDS={'Publication-grouped':[(np.flatnonzero(~np.isin(refs,list(g))),np.flatnonzero(np.isin(refs,list(g)))) for g in GROUPS],
       'Conventional random':list(KFold(n_splits=5,shuffle=True,random_state=42).split(np.arange(n)))}
for typ,fs in FOLDS.items():
 a=np.concatenate([va for tr,va in fs]); assert sorted(a)==list(range(n))
 print(typ,'fold sizes',[len(va) for tr,va in fs],flush=True)
C1={
 'RF':[(16,0.5,1),(None,0.5,3),(16,'sqrt',3),(16,0.5,3),(16,'sqrt',3)],
 'XGBoost':[(200,2,.03),(500,4,.03),(500,2,.1),(200,6,.1),(500,2,.1)],
 'LightGBM':[(500,15,.1),(500,31,.1),(200,15,.03),(200,31,.03),(500,7,.03)],
 'CatBoost':[(600,3,.05,10,1.),(600,6,.03,15,2.),(600,6,.03,15,2.),(600,6,.03,15,2.),(600,6,.03,15,2.)]
}
C2={
 'RF':[(16,'sqrt',1),(None,'sqrt',1),(16,'sqrt',3),(None,'sqrt',3),(16,'sqrt',1)],
 'XGBoost':[(500,6,.03),(500,6,.03),(200,2,.03),(200,6,.03),(200,2,.03)],
 'LightGBM':[(500,7,.1),(500,15,.1),(200,31,.03),(200,7,.03),(500,15,.1)],
 'CatBoost':[(600,4,.05,15,1.),(600,4,.05,15,1.),(600,6,.03,15,2.),(600,6,.03,15,2.),(600,6,.03,15,2.)]
}
A={'RF':[0,0,0,0,1.0],'XGBoost':[0,0,0,0,.75],'LightGBM':[.25,0,0,.25,.75],'CatBoost':[.75,0,1.,0,.25]}
XGB_SLOPE=[.5,.25,.25,.25,.25]
CB_DELTA=[2.,.3,.3,.5,2.]
MICE_SEEDS=[42,43,44,45,46]
MODELS=['RF','XGBoost','LightGBM','CatBoost']

def numeric_matrix(Z,positions):
 return np.asarray([[np.nan if Z[i,j] is None or Z[i,j]=='' else float(Z[i,j]) for j in positions] for i in range(len(Z))],dtype=float)
def cat_matrix(Z,positions,mode):
 vals=np.asarray([['__NA__' if Z[i,j] is None or Z[i,j]=='' else str(Z[i,j]) for j in positions] for i in range(len(Z))],dtype=object).reshape(len(Z),len(positions))
 if mode=='Missing': vals[vals=='__NA__']='Missing'
 else: vals[vals=='__NA__']=np.nan
 return vals

def prep(trainidx,validx,feats,model,imputation_seed=None):
 cols=[headers.index(c) for c in feats]
 ca=[j for j,col in enumerate(feats) if col in CATCOLS]
 nu=[j for j,col in enumerate(feats) if col not in CATCOLS]
 tra=Xall[np.ix_(trainidx,cols)];val=Xall[np.ix_(validx,cols)]
 a=numeric_matrix(tra,nu); b=numeric_matrix(val,nu)
 if model=='RF': im=KNNImputer(missing_values=np.nan,n_neighbors=5,weights='distance',metric='nan_euclidean',add_indicator=False,keep_empty_features=False)
 elif model in ['XGBoost','LightGBM']:
  assert imputation_seed is not None
  im=IterativeImputer(estimator=BayesianRidge(),max_iter=10,tol=1e-3,initial_strategy='median',sample_posterior=True,skip_complete=True,random_state=imputation_seed)
 else: im=SimpleImputer(strategy='median')
 an=im.fit_transform(a);bn=im.transform(b)
 sc=StandardScaler(with_mean=True,with_std=True);an=sc.fit_transform(an);bn=sc.transform(bn)
 catmode='Mode' if model=='CatBoost' else 'Missing'
 ac=cat_matrix(tra,ca,catmode);bc=cat_matrix(val,ca,catmode)
 if catmode=='Mode':
  imp=SimpleImputer(strategy='most_frequent');ac=imp.fit_transform(ac);bc=imp.transform(bc)
 enc=OneHotEncoder(categories='auto',drop=None,handle_unknown='ignore',dtype=np.float64,sparse_output=model!='RF')
 ac=enc.fit_transform(ac);bc=enc.transform(bc)
 if model=='RF':return np.hstack([an,ac]),np.hstack([bn,bc])
 return hstack([csr_matrix(an),ac],format='csr'),hstack([csr_matrix(bn),bc],format='csr')

def model_construct(model,stage,fold):
 p=(C1 if stage==1 else C2)[model][fold]
 if model=='RF':
  dp,mf,lf=p
  return RandomForestRegressor(n_estimators=200,max_depth=dp,max_features=mf,min_samples_leaf=lf,random_state=42,n_jobs=2)
 if model=='XGBoost':
  ne,depth,lr=p
  arg={'n_estimators':ne,'max_depth':depth,'learning_rate':lr,'subsample':0.8,'colsample_bytree':0.8,'reg_lambda':1.,'objective':'reg:squarederror','random_state':42,'tree_method':'hist','n_jobs':2,'verbosity':0}
  if stage>=3:arg.update(objective='reg:pseudohubererror',huber_slope=XGB_SLOPE[fold])
  return XGBRegressor(**arg)
 if model=='LightGBM':
  ne,leaves,lr=p
  return LGBMRegressor(n_estimators=ne,num_leaves=leaves,learning_rate=lr,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.,objective='regression',random_state=42,n_jobs=1,verbosity=-1)
 it,dp,lr,l2,rs=p
 cfg=dict(iterations=it,depth=dp,learning_rate=lr,l2_leaf_reg=l2,random_strength=rs,loss_function='RMSE',random_seed=42,verbose=False,allow_writing_files=False,thread_count=2)
 if stage>=3:cfg['loss_function']=f'Huber:delta={CB_DELTA[fold]}'
 return CatBoostRegressor(**cfg)

def model_predict(tr,va,feat,model,stage,fold):
 ytrain=Y[tr]
 kw={}
 if stage>=3:
  alpha=A[model][fold]
  if alpha>0:
   cnt=Counter(refs[tr]);w=np.asarray([cnt[r]**(-alpha) for r in refs[tr]],float);w/=w.mean();kw['sample_weight']=w
 pred=[]
 for seed in (MICE_SEEDS if model in ['XGBoost','LightGBM'] else [None]):
  XA,XB=prep(tr,va,feat,model,seed)
  md=model_construct(model,stage,fold)
  md.fit(XA,ytrain,**kw)
  pred.append(np.asarray(md.predict(XB),float))
 return np.mean(pred,axis=0)

def metrics(yy,p):return (float(r2_score(yy,p)),float(np.sqrt(mean_squared_error(yy,p))),float(mean_absolute_error(yy,p)))

def main():
 args=argparse.ArgumentParser();args.add_argument('--stages',type=int,nargs='+',default=[1,2,3,4]);args.add_argument('--types',type=str,nargs='+',default=['Publication-grouped','Conventional random']);opt=args.parse_args()
 results=[]
 outf=ROOT+'cv_results.json'
 if os.path.isfile(outf):results=json.load(open(outf))
 for stage in opt.stages:
  for typ in opt.types:
   if any(r['Stage']==stage and r['CV_Type']==typ for r in results):print('SKIP',stage,typ,flush=True);continue
   fs=FOLDS[typ];p={m:np.full(n,np.nan) for m in MODELS};foldmet=[]
   for fi,(tr,va) in enumerate(fs):
    t0=time.time()
    for model in MODELS:
     feats=ALL_FEAT[:22] if stage==1 else (subsets[model] if stage==4 else ALL_FEAT)
     pv=model_predict(tr,va,feats,model,stage,fi)
     p[model][va]=pv
     vals=metrics(Y[va],pv)
     foldmet.append({'Stage':stage,'CV_Type':typ,'Model':model,'Fold':fi+1,'n_validation':len(va),'R2':vals[0],'RMSE':vals[1],'MAE':vals[2]})
     print('FIT',stage,typ,fi+1,model,tuple(round(x,4) for x in vals),'elapsed',round(time.time()-t0,1),'s',flush=True)
   p['Equal4']=np.mean(np.column_stack([p[m] for m in MODELS]),axis=1)
   for fi,(tr,va) in enumerate(fs):
    vals=metrics(Y[va],p['Equal4'][va]);foldmet.append({'Stage':stage,'CV_Type':typ,'Model':'Equal4','Fold':fi+1,'n_validation':len(va),'R2':vals[0],'RMSE':vals[1],'MAE':vals[2]})
   for model,ypred in p.items():
    assert np.all(np.isfinite(ypred))
    rr,rm,ma=metrics(Y,ypred)
    results.append({'Stage':stage,'CV_Type':typ,'Model':model,'R2':rr,'RMSE':rm,'MAE':ma,'n':n})
    print('RESULT',stage,typ,model,'R2',round(rr,7),'RMSE',round(rm,7),'MAE',round(ma,7),flush=True)
   json.dump(results,open(outf,'w'),indent=2)
   foldfile=ROOT+'fold_results.json';older=json.load(open(foldfile)) if os.path.isfile(foldfile) else []
   older=[r for r in older if not(r['Stage']==stage and r['CV_Type']==typ)]+foldmet
   json.dump(older,open(foldfile,'w'),indent=2)
   np.savez_compressed(ROOT+f"oof_stage{stage}_{typ.replace(' ','_').replace('-','_')}.npz",y=Y,refs=refs,**p)
 print('FINISHED')
if __name__=='__main__':main()
