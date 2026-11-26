"""Verifiable Line 2 LightGBM reconstruction from original frozen inputs.

- 4 stages x (grouped/random) x 5 outer folds
- no new Ref 28-30 data read or used
- exact frozen feature subsets, fold-specific hyperparameters and weighting
- archive per-fold fitted regressors WITH fitted fold preprocessors to use for future zero-shot inference
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='5'
os.environ['MKL_NUM_THREADS']='5'
os.environ['OMP_NUM_THREADS']='3'
from pathlib import Path
import hashlib,json, time, warnings
import numpy as np
import joblib
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'LightGBM_Reproduced_Models'
DEST.mkdir(exist_ok=True)
source=(ROOT/'run.py').read_text().split('STAGES=[1,2,3,4]')[0]
ns={'__file__':str(ROOT/'run.py')}
exec(compile(source,str(ROOT/'run.py'),'exec'),ns)
locks=json.loads((ROOT/'audit_stage1_results.json').read_text())['models']['LightGBM']
ns['C1']['LightGBM']=[tuple(f['selected_params']) for f in locks['folds']]
trainn=int(ns['n'])
rows_test=json.loads((ROOT/'test55.json').read_text())['rows']
assert trainn==995 and len(rows_test)==249
ns['Xall']=np.vstack([ns['Xall'],np.asarray(rows_test,dtype=object)])
Ytest=np.log1p(np.asarray([r[22] for r in rows_test],dtype=float))
Itest=np.arange(trainn,trainn+len(rows_test))
H=ns['h']; REF=ns['refs']; FULL=ns['Xall']; Y=ns['Y']; CATS=ns['CATCOLS']
reference=json.loads((ROOT/'AUDIT_FINAL_All_Stages_Results.json').read_text())
ref={(r['Stage'],r['Splitting']):r for r in reference if r['Model']=='LightGBM'}

from line2_lightgbm_preprocess import FrozenLightGBMPreprocess

records=[]
for stage in [1,2,3,4]:
 for method in ['Publication-grouped','Conventional random']:
    t0=time.perf_counter()
    features=ns['ALL_FEAT'][:22] if stage==1 else (ns['SUBSETS']['LightGBM'] if stage==4 else ns['ALL_FEAT'])
    if len(features) not in [22,55,18]:raise ValueError((stage,len(features)))
    indices=[H.index(f) for f in features]
    train_pred=np.full(trainn,np.nan)
    old_test_by_fold=[]
    fold_summaries=[]
    for i,(tr,va) in enumerate(ns['folds'][method]):
        # Source exactly as written for verification; re-fit preprocessing on the same tr in saved converter
        Xtr,Xva,_=ns['prepare'](tr,va,features,'LightGBM',stage)
        src_test=ns['prepare'](tr,Itest,features,'LightGBM',stage)[1]
        prep=FrozenLightGBMPreprocess(features).fit(FULL[np.ix_(tr,indices)])
        Xtr2=prep.transform(FULL[np.ix_(tr,indices)])
        Xva2=prep.transform(FULL[np.ix_(va,indices)])
        Xte2=prep.transform(FULL[np.ix_(Itest,indices)])
        for a,b,name in [(Xtr,Xtr2,'train'),(Xva,Xva2,'validation'),(src_test,Xte2,'old test')]:
            d=(a-b).tocsr()
            if d.nnz and max(abs(d.data))>1e-10:raise AssertionError(('preprocessor mismatch',stage,method,i,name,max(abs(d.data))))
        # Exact original source model construction and fitting
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted=ns['fit']('LightGBM',Xtr,Y[tr],None,stage,i,tr)
        pred=fitted.predict(Xva2)
        oldpred=fitted.predict(Xte2)
        train_pred[va]=pred
        old_test_by_fold.append(oldpred)
        file=DEST/(f'LightGBM_Line2_S{stage}_{"grouped" if method=="Publication-grouped" else "random"}_F{i+1}.joblib')
        joblib.dump({'model':fitted,'processor':prep,'features':features,'stage':stage,'validation':method,'fold':i+1,
                     'train_ref_ids':sorted(set(map(int,REF[tr]))),'target_transform':'ln(1+DBP ug/L)',
                     'training_rows':len(tr),'seed':42},str(file),compress=3)
        fold_summaries.append({'fold':i+1,'n_train':len(tr),'n_holdout':len(va),'CV_R2':r2_score(Y[va],pred),
                               'setting':list(ns['C1' if stage==1 else 'C2']['LightGBM'][i]),
                               'alpha':ns['A']['LightGBM'][i] if stage>=3 else 0,
                               'model_file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        print('fit',stage,method,i+1,'old_test_R2',round(r2_score(Ytest,oldpred),4),flush=True)
    ptest=np.mean(old_test_by_fold,axis=0)
    metrics={'Stage':stage,'CV type':method,'features':len(features),
        'CV R2':float(r2_score(Y,train_pred)),'CV RMSE':float(mean_squared_error(Y,train_pred)**0.5),'CV MAE':float(mean_absolute_error(Y,train_pred)),
        'Test R2':float(r2_score(Ytest,ptest)),'Test RMSE':float(mean_squared_error(Ytest,ptest)**0.5),'Test MAE':float(mean_absolute_error(Ytest,ptest)),
        'fold_info':fold_summaries}
    target=ref[(stage,method)]
    metrics['reference_metric_differences']={x:float(metrics[y]-target[x]) for x,y in [('CV R2','CV R2'),('CV RMSE','CV RMSE'),('CV MAE','CV MAE'),('Test R2','Test R2'),('Test RMSE','Test RMSE'),('Test MAE','Test MAE')]}
    metrics['exact_to_8_decimal_places']=all(abs(v)<0.000000005 for v in metrics['reference_metric_differences'].values())
    if not metrics['exact_to_8_decimal_places']:
        raise AssertionError(('metrics do not match archived!',stage,method,metrics['reference_metric_differences']))
    records.append(metrics)
    np.savez_compressed(DEST/f'Original_Validation_Predictions_S{stage}_{"grouped" if method=="Publication-grouped" else "random"}.npz',
                       cv_observed=Y,cv_predicted=train_pred,cv_ref=REF,
                       old_test_observed=Ytest,old_test_predicted=ptest,
                       old_test_per_fold=np.asarray(old_test_by_fold))
    (DEST/'reproduction_summary.json').write_text(json.dumps(records,indent=2))
    print('COMPLETED',stage,method,'CV',metrics['CV R2'],'OLD TEST',metrics['Test R2'], 'seconds',round(time.perf_counter()-t0,2),flush=True)
print('SUCCESS: 4 stages x 2 validation regimes = 8 LightGBM configurations, 40 archived models')
