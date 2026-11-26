"""Independent RF-only validation against latest SI using byte-hash-verified original XLSX.
No changes made to SI or locked model definitions.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','5')
os.environ.setdefault('MKL_NUM_THREADS','5')
os.environ.setdefault('OMP_NUM_THREADS','3')
import hashlib, json, time
from pathlib import Path
from artifact_tool import Blob, SpreadsheetFile
from docx import Document
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

root=Path('/mnt/data'); si_file=root/'SI_Updating(3).docx'
TRAIN=root/'Training(9).xlsx'; TEST=root/'Test(9).xlsx'
sha={TRAIN:'de1e28d1525c97d802e5b3dd261dd7409bea3bbeade02875d3ac36fe76947fb9',TEST:'75e5164aba4c974e54f4974dee86a65d37228e9194ada504f6133552c77fdfc8'}
for f,expected in sha.items():
    actual=hashlib.sha256(f.read_bytes()).hexdigest()
    assert actual==expected,(f,actual,expected)
    print('HASH verified',f.name,actual,flush=True)

def load_xlsx(path,nrows):
    workbook=SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    data=workbook.worksheets.get_item('Data').get_range(f'A1:BE{nrows+1}').values
    assert len(data)==nrows+1,(len(data),nrows)
    hdr=data[0]
    assert len(hdr)==57 and len(data[1])==57,(len(hdr),len(data[1]))
    return hdr,np.asarray(data[1:],dtype=object)
start=time.time(); hdr,train=load_xlsx(TRAIN,995); hdrt,test=load_xlsx(TEST,249)
assert hdr==hdrt
print('loaded in',round(time.time()-start,2),'s',flush=True)
print('headers',[(i,h) for i,h in enumerate(hdr[:26])],flush=True)
print('last headers',[(i,h) for i,h in list(enumerate(hdr))[50:]],flush=True)
ref_idx=hdr.index('ref ID'); y_idx=hdr.index('DBP concentration (ug/L)') if 'DBP concentration (ug/L)' in hdr else next(i for i,h in enumerate(hdr) if str(h).startswith('DBP concentration'))
Y=np.log1p(np.asarray(train[:,y_idx],float)); Yte=np.log1p(np.asarray(test[:,y_idx],float))
refs=np.asarray(train[:,ref_idx],int)
assert np.isfinite(Y).all() and np.isfinite(Yte).all()

si=Document(si_file)
rf_features=[s.strip().rstrip('.') for s in si.tables[21].rows[1].cells[1].text.split(';')]
all_features=hdr[:y_idx]+hdr[y_idx+2:]
assert len(all_features)==55 and len(rf_features)==37 and all(f in all_features for f in rf_features)
catnames=['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']
groups=[{4,8,9,13,18},{5,10,15,23},{2,12,17,19,25,27},{1,11,24},{3,6,20,26}]
folds=[(np.flatnonzero(~np.isin(refs,list(g))),np.flatnonzero(np.isin(refs,list(g)))) for g in groups]
core=[(None,'sqrt',1),(8,'sqrt',3),(8,'sqrt',3),(None,'sqrt',1),(16,0.5,1)]
alphas=[0,.5,0,.5,0]

def prepare(tr,va,features):
    idx=[hdr.index(f) for f in features]
    ctr=[j for j,f in enumerate(features) if f in catnames]
    num=[j for j in range(len(features)) if j not in ctr]
    aa=train[np.ix_(tr,idx)];bb=va[:,idx]
    def numeric(arr):
        return np.array([[np.nan if (v is None or v=='') else float(v) for v in r[num]] for r in arr],dtype=np.float64)
    na,nb=numeric(aa),numeric(bb)
    imp=SimpleImputer(strategy='median')
    na=imp.fit_transform(na);nb=imp.transform(nb)
    scaler=StandardScaler()
    na=scaler.fit_transform(na);nb=scaler.transform(nb)
    if not ctr:return na,nb
    def categorical(arr):
        cc=arr[:,ctr].copy().astype(object)
        for j in range(cc.shape[1]):
            for i in range(cc.shape[0]):
                if cc[i,j] is None or cc[i,j]=='':cc[i,j]=np.nan
                else: cc[i,j]=str(cc[i,j])
        return cc
    ca,cb=categorical(aa),categorical(bb)
    mode=SimpleImputer(strategy='most_frequent')
    ca=mode.fit_transform(ca);cb=mode.transform(cb)
    enc=OneHotEncoder(categories='auto',drop=None,handle_unknown='ignore',dtype=np.float64,sparse_output=False)
    ca=enc.fit_transform(ca);cb=enc.transform(cb)
    return np.hstack([na,ca]),np.hstack([nb,cb])

result={}
for s,features in [(3,all_features),(4,rf_features)]:
    pred=np.full(995,np.nan);tpred=[]
    for i,(tr,va) in enumerate(folds):
        a,b=prepare(tr,train[va],features)
        _,c=prepare(tr,test,features)
        depth,mx,leaf=core[i]
        model=RandomForestRegressor(n_estimators=200,max_depth=depth,max_features=mx,min_samples_leaf=leaf,random_state=42,n_jobs=-1)
        alpha=alphas[i]
        kwargs={}
        if alpha>0:
            rr=refs[tr]; unique, counts=np.unique(rr,return_counts=True)
            cmap=dict(zip(unique,counts));weights=np.array([cmap[v]**(-alpha) for v in rr],float)
            weights/=weights.mean()
            kwargs={'sample_weight':weights}
        model.fit(a,Y[tr],**kwargs)
        pred[va]=model.predict(b)
        tpred.append(model.predict(c))
        print('Stage',s,'fold',i+1,'alpha',alpha,'valR2',round(r2_score(Y[va],pred[va]),6),flush=True)
    tpred=np.mean(np.stack(tpred),axis=0)
    m=(r2_score(Y,pred),np.sqrt(mean_squared_error(Y,pred)),mean_absolute_error(Y,pred))
    mt=(r2_score(Yte,tpred),np.sqrt(mean_squared_error(Yte,tpred)),mean_absolute_error(Yte,tpred))
    result[s]={'CV':list(m),'Test':list(mt),'SI_CV_R2': 0.3393908629 if s==3 else 0.3867462415, 'SI_Test_R2': -0.0529128746 if s==3 else -0.0470013503}
    print('STAGE',s,'CV',m,'Test',mt,flush=True)
    print('CV R2 diff',m[0]-result[s]['SI_CV_R2'],'Test R2 diff',mt[0]-result[s]['SI_Test_R2'],flush=True)
Path('/mnt/data/Line2_CV_Comparison/RF_FromLatestSI_Audit.json').write_text(json.dumps(result,indent=2))
