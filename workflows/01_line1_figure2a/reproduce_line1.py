import os
os.environ['OPENBLAS_NUM_THREADS']='5'; os.environ['MKL_NUM_THREADS']='5'; os.environ['OMP_NUM_THREADS']='3'
import sys, time, json, warnings, hashlib, argparse
import numpy as np, pandas as pd
from docx import Document
from pathlib import Path
from sklearn.model_selection import KFold,train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from scipy import sparse
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT = Path(__file__).resolve().parent
D = str(ROOT / 'data') + '/'
DOCS = ROOT / 'documents'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
STAGEFILES={
 'S1':['Conventional_RandomSplit_Stage1_Training_22features(3).xlsx','Conventional_RandomSplit_Stage1_Test_22features(3).xlsx'],
 'S2':['Conventional_RandomSplit_Stage2_Training_55features(3).xlsx','Conventional_RandomSplit_Stage2_Test_55features(3).xlsx'],
 'S3':['Conventional_RandomSplit_Stage2_Training_55features(3).xlsx','Conventional_RandomSplit_Stage2_Test_55features(3).xlsx']}
CATS=['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']
ORDER=['Elastic Net','SVR-RBF','Random Forest','XGBoost','LightGBM','CatBoost']
S1PARAMS={
'Elastic Net': dict(alpha=0.0011789424397147285,l1_ratio=0.04833794796157109),
'SVR-RBF':dict(C=78.3645526513463,epsilon=0.05617101982934737,gamma=0.0431392384084347),
'Random Forest':dict(),
'XGBoost':dict(n_estimators=600,max_depth=6,learning_rate=0.025230515223593646,min_child_weight=0.517914384198523,subsample=0.8374709757162002,colsample_bytree=0.8610724800899925,reg_alpha=0.05006233869672168,reg_lambda=0.03616020757636161,gamma=0.004660824731852315),
'LightGBM':dict(n_estimators=600,max_depth=7,num_leaves=18,learning_rate=0.2232935470021013,min_child_samples=19,subsample=0.8317924627337946,colsample_bytree=0.6513355995465208,reg_alpha=0.09693405549979402,reg_lambda=4.513107821713152),
'CatBoost':dict(iterations=500,depth=5,learning_rate=0.1775261891365479,l2_leaf_reg=0.001039619663133687,random_strength=2.025642893508201,subsample=0.9035051710026214,rsm=0.6285276584655902,bootstrap_type='MVS')}
S2PARAMS={
'Elastic Net': S1PARAMS['Elastic Net'],
'SVR-RBF':dict(C=226.11277484216907,epsilon=0.052412492469052004,gamma=0.036363829416398326),
'Random Forest':dict(n_estimators=200,max_depth=20,min_samples_split=2,min_samples_leaf=1,max_features=0.75,bootstrap=True),
'XGBoost':S1PARAMS['XGBoost'],
'LightGBM':dict(n_estimators=600,max_depth=8,num_leaves=24,learning_rate=0.20104161699065715,min_child_samples=23,subsample=0.8760988294276582,colsample_bytree=0.6060576948411027,reg_alpha=0.04799114428365162,reg_lambda=0.047971942352080736),
'CatBoost':dict(iterations=500,depth=5,learning_rate=0.1786861885549884,l2_leaf_reg=0.0010839006279033633,random_strength=1.997407617480316,subsample=0.9866214592657088,rsm=0.6273081880110045,bootstrap_type='MVS')}
TARGET={
'S1': [(0.6591,0.5991),(.8798,.8263),(.8698,.8529),(.8954,.8505),(.8768,.8528),(.8641,.8565)],
'S2':[(.6602,.6000),(.8807,.8101),(.8802,.8586),(.9080,.8662),(.9032,.8760),(.8972,.8680)],
'S3':[(.6678,.5979),(.8807,.8101),(.8834,.8493),(.9108,.8634),(.9049,.8870),(.9023,.8826)]}

def import_sheet(file):
    """Read exact XLSX Data sheet values (verified bitwise-equivalent to original reader)."""
    t = time.monotonic()
    df = pd.read_excel(D + file, sheet_name='Data', engine='openpyxl')
    expected = 995 if 'Training' in file else 249
    width = 57 if '55features' in file else 24
    assert df.shape == (expected, width), (file, df.shape)
    print('read', file, df.shape, 'time', round(time.monotonic()-t, 1), flush=True)
    return df

def stage3_subsets(features):
 D0=Document(str(DOCS/'SI_20261016.docx'))
 subsets={}
 names={'Elastic Net':'Elastic Net','SVR-RBF':'RBF-SVR','Random Forest':'Random Forest','XGBoost':'XGBoost','LightGBM':'LightGBM','CatBoost':'CatBoost'}
 for m,token in names.items():
  p=next((p.text for p in D0.paragraphs if p.text.startswith('Conventional Stage-3 '+token+' RFE selection.')),None)
  if p is None: raise ValueError('Missing Stage3 subset '+m)
  if m=='SVR-RBF': subset=list(features)
  else: subset=[x.strip().rstrip('.') for x in p.split('was: ',1)[1].split(';')]
  expected={'Elastic Net':26,'SVR-RBF':55,'Random Forest':26,'XGBoost':48,'LightGBM':50,'CatBoost':48}[m]
  assert len(subset)==expected,(m,len(subset),subset)
  assert all(x in features for x in subset),(m,[x for x in subset if x not in features])
  subsets[m]=subset
 return subsets

def preprocess(tr,va,features,model):
 cats=[x for x in features if x in CATS]
 nums=[x for x in features if x not in CATS]
 if model=='CatBoost':
  Xtr=tr[features].copy();Xva=va[features].copy()
  for df in (Xtr,Xva):
   for x in cats: df[x]=df[x].fillna('__MISSING__').astype(str)
   for x in nums: df[x]=pd.to_numeric(df[x],errors='coerce')
  return Xtr,Xva,cats
 for df in (tr,va):
  assert (set(features)-set(df.columns))==set()
 numtr=tr[nums].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
 numva=va[nums].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
 cat_tr=tr[cats].fillna('__MISSING__').astype(str)
 cat_va=va[cats].fillna('__MISSING__').astype(str)
 im=SimpleImputer(strategy='median')
 Xt=im.fit_transform(numtr); Xv=im.transform(numva)
 if model in ('Elastic Net','SVR-RBF'):
  sc=StandardScaler(with_mean=True,with_std=True)
  Xt=sc.fit_transform(Xt); Xv=sc.transform(Xv)
 enc=OneHotEncoder(categories='auto',drop=None,handle_unknown='ignore',dtype=np.float64,sparse_output=True)
 catXt=enc.fit_transform(cat_tr);catXv=enc.transform(cat_va)
 Xtr=sparse.hstack([sparse.csr_matrix(Xt),catXt],format='csr',dtype=np.float64)
 Xva=sparse.hstack([sparse.csr_matrix(Xv),catXv],format='csr',dtype=np.float64)
 return Xtr,Xva,cats

def model_instance(m,params):
 if m=='Elastic Net':return ElasticNet(**params,random_state=42,max_iter=10000,tol=1e-4,selection='random')
 if m=='SVR-RBF':return SVR(kernel='rbf',cache_size=1000,**params)
 if m=='Random Forest':return RandomForestRegressor(**params,random_state=42,n_jobs=-1)
 if m=='XGBoost':return XGBRegressor(**params,random_state=42,n_jobs=-1,verbosity=0)
 if m=='LightGBM':return LGBMRegressor(**params,random_state=42,n_jobs=-1,verbose=-1)
 if m=='CatBoost':return CatBoostRegressor(**params,random_seed=42,verbose=False,allow_writing_files=False,thread_count=-1)

def fitpred(m,model,Xt,yt,Xv,cats):
 if m=='CatBoost':
  model.fit(Xt,yt,cat_features=cats)
 else: model.fit(Xt,yt)
 return model.predict(Xv)

def calc(y,p):
 return dict(r2=float(r2_score(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)))

def run(s,m):
 filetr,filete=STAGEFILES[s]
 dftr=import_sheet(filetr);dfte=import_sheet(filete)
 assert len(dftr)==995 and len(dfte)==249
 original=list(dftr.columns[:22]);extra=list(dftr.columns[24:57]) if s!='S1' else []
 assert len(original)==22 and len(extra) in (0,33)
 features=original+extra
 if s=='S3': features=stage3_subsets(features)[m]
 t0=time.monotonic()
 ytr=np.log1p(dftr['DBP concentration (µg/L)'].to_numpy(dtype=float))
 yte=np.log1p(dfte['DBP concentration (µg/L)'].to_numpy(dtype=float))
 assert np.isfinite(ytr).all() and np.isfinite(yte).all()
 params=S1PARAMS[m] if s=='S1' else S2PARAMS[m]
 kf=KFold(n_splits=5,shuffle=True,random_state=42)
 oof=np.full(len(dftr),np.nan,dtype=float)
 folds=[]
 for fold,(i,j) in enumerate(kf.split(dftr),start=1):
  t=time.monotonic()
  Xt,Xv,cats=preprocess(dftr.iloc[i],dftr.iloc[j],features,m)
  mm=model_instance(m,params)
  oof[j]=fitpred(m,mm,Xt,ytr[i],Xv,cats)
  folds.append(calc(ytr[j],oof[j]))
  print(s,m,'fold',fold,folds[-1]['r2'],'time',round(time.monotonic()-t,1),flush=True)
 Xt,Xv,cats=preprocess(dftr,dfte,features,m)
 mm=model_instance(m,params);pred=fitpred(m,mm,Xt,ytr,Xv,cats)
 cm,tm=calc(ytr,oof),calc(yte,pred)
 target=TARGET[s][ORDER.index(m)]
 result={'stage':s,'model':m,'n_features':len(features),'features':features,'cv':cm,'test':tm,'folds':folds,
         'target_cv_r2':target[0],'target_test_r2':target[1],
         'delta_cv_r2':cm['r2']-target[0],'delta_test_r2':tm['r2']-target[1],
         'params':params,'runtime_sec':round(time.monotonic()-t0,2)}
 fn=f'line1_{s}_{m.replace(" ","_")}.json';open(OUT/fn,'w').write(json.dumps(result,indent=2,ensure_ascii=False))
 np.savez_compressed(OUT/f'line1_{s}_{m.replace(" ","_")}_preds.npz',oof=oof,test=pred,y_train=ytr,y_test=yte)
 print('DONE',s,m,'CV',cm,'TEST',tm,'DIFF',result['delta_cv_r2'],result['delta_test_r2'],'sec',result['runtime_sec'],flush=True)
 return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['S1','S2','S3']);ap.add_argument('models',nargs='*');args=ap.parse_args()
 names=args.models or ORDER
 for model in names:
  run(args.stage,model)
