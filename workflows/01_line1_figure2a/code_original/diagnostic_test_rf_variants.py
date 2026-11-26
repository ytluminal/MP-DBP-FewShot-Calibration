import reproduce_line1 as q
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import time
tr=q.import_sheet(q.STAGEFILES['S2'][0]);te=q.import_sheet(q.STAGEFILES['S2'][1]); feats=list(tr.columns[:22])+list(tr.columns[24:57]); ytr=np.log1p(tr['DBP concentration (µg/L)'].to_numpy(dtype=float));yte=np.log1p(te['DBP concentration (µg/L)'].to_numpy(dtype=float))
cs=[x for x in feats if x in q.CATS]; ns=[x for x in feats if x not in q.CATS]

def transform(a,b,scale=False,dens=False,fill='missing'):
 z=SimpleImputer(strategy='median'); X=z.fit_transform(a[ns].to_numpy(dtype=float)); V=z.transform(b[ns].to_numpy(dtype=float))
 if scale: z=StandardScaler();X=z.fit_transform(X);V=z.transform(V)
 C=a[cs].copy();E=b[cs].copy()
 if fill=='missing':
  C=C.fillna('__MISSING__').astype(str);E=E.fillna('__MISSING__').astype(str)
 else:
  im=SimpleImputer(strategy='most_frequent')
  C=im.fit_transform(C);E=im.transform(E)
 z=OneHotEncoder(categories='auto',drop=None,handle_unknown='ignore',dtype=np.float64,sparse_output=True)
 T=z.fit_transform(C); U=z.transform(E)
 X=sparse.hstack([sparse.csr_matrix(X),T],format='csr');V=sparse.hstack([sparse.csr_matrix(V),U],format='csr')
 if dens: X=X.toarray();V=V.toarray()
 return X,V
for scale in (False,True):
 for dens in (False,True):
  for fill in ('missing','mode'):
   oof=np.full(len(ytr),np.nan); 
   for i,j in q.KFold(n_splits=5,shuffle=True,random_state=42).split(tr):
    a,b=transform(tr.iloc[i],tr.iloc[j],scale,dens,fill)
    mod=q.model_instance('Random Forest',q.S2PARAMS['Random Forest']);oof[j]=mod.fit(a,ytr[i]).predict(b)
   a,b=transform(tr,te,scale,dens,fill);mod=q.model_instance('Random Forest',q.S2PARAMS['Random Forest']);v=mod.fit(a,ytr).predict(b)
   print('variant',scale,dens,fill,'CV',round(q.r2_score(ytr,oof),7),'test',round(q.r2_score(yte,v),7),flush=True)
