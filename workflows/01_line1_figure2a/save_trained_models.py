"""Re-fit and archive every frozen Line-1 estimator AND its fitted preprocessing.

Outputs 18 full-Training models + 90 5-fold CV models, one joblib payload each.
The frozen hyperparameters/features are imported from reproduce_line1.py. Does NOT
repeat Optuna optimization or RFE feature selection. Running overwrites models/.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='5'
os.environ['MKL_NUM_THREADS']='5'
os.environ['OMP_NUM_THREADS']='3'
import json, hashlib, time, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import reproduce_line1 as locked

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models'
MODELS.mkdir(exist_ok=True)
NUM_TOL=3e-7


def prepare_and_fit(train, selected_columns, algo, params, train_idx, stage, fold=None):
    """Store fitted imputer/scaler/encoder rather than just the trained estimator."""
    cat=[c for c in selected_columns if c in locked.CATS]
    num=[c for c in selected_columns if c not in locked.CATS]
    x=train.loc[:,selected_columns]
    if algo=='CatBoost':
        model_x=x.copy()
        for c in cat: model_x[c]=model_x[c].fillna('__MISSING__').astype(str)
        for c in num: model_x[c]=pd.to_numeric(model_x[c],errors='coerce')
        prep={'categorical_fill':'__MISSING__','numerical_imputer':None,'numerical_scaler':None,'categorical_encoder':None}
    else:
        numeric=x[num].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
        imp=SimpleImputer(strategy='median')
        X_num=imp.fit_transform(numeric)
        scl=None
        if algo in ('Elastic Net','SVR-RBF'):
            scl=StandardScaler(with_mean=True, with_std=True)
            X_num=scl.fit_transform(X_num)
        enc=OneHotEncoder(categories='auto',drop=None,handle_unknown='ignore',dtype=np.float64,sparse_output=True)
        X_cat=enc.fit_transform(x[cat].fillna('__MISSING__').astype(str))
        model_x=sparse.hstack([sparse.csr_matrix(X_num),X_cat],format='csr',dtype=np.float64)
        prep={'categorical_fill':'__MISSING__','numerical_imputer':imp,'numerical_scaler':scl,'categorical_encoder':enc}
    model=locked.model_instance(algo,params)
    y=np.log1p(train['DBP concentration (µg/L)'].to_numpy(dtype=np.float64))
    if algo=='CatBoost':
        model.fit(model_x,y,cat_features=cat)
    else:
        model.fit(model_x,y)
    return {
        'archive_format':'Line1_fitted_model_v1',
        'stage':stage,
        'algorithm':algo,
        'scope':'all_995_training' if fold is None else f'CV_fold_{fold}',
        'fold':fold,
        'train_row_positions':np.asarray(train_idx,dtype=np.int32),
        'feature_names':selected_columns,
        'numeric_features':num,
        'categorical_features':cat,
        'transform':'ln(1 + DBP concentration in micrograms per liter)',
        'preprocessing':prep,
        'model':model,
        'parameters':params,
        'package_versions':{'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':'1.8.0','xgboost':'3.1.3','lightgbm':'4.6.0','catboost':'1.2.8','joblib':joblib.__version__},
    }


def predict_payload(payload, df):
    """Return log1p predictions; accepts a pandas DataFrame containing named inputs."""
    sel=payload['feature_names']
    miss=[c for c in sel if c not in df.columns]
    if miss: raise KeyError(f'Missing required columns: {miss}')
    x=df.loc[:,sel]
    num=payload['numeric_features']; cat=payload['categorical_features']
    if payload['algorithm']=='CatBoost':
        X=x.copy()
        for c in cat: X[c]=X[c].fillna('__MISSING__').astype(str)
        for c in num: X[c]=pd.to_numeric(X[c],errors='coerce')
    else:
        pp=payload['preprocessing']
        N=x[num].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
        N=pp['numerical_imputer'].transform(N)
        if pp['numerical_scaler'] is not None:
            N=pp['numerical_scaler'].transform(N)
        C=pp['categorical_encoder'].transform(x[cat].fillna('__MISSING__').astype(str))
        X=sparse.hstack([sparse.csr_matrix(N),C],format='csr',dtype=np.float64)
    return np.asarray(payload['model'].predict(X),dtype=np.float64).ravel()


def scores(y,p):
    return {'r2':float(r2_score(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'mae':float(mean_absolute_error(y,p))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['S1','S2','S3'],default=None)
    ap.add_argument('--algorithm',choices=locked.ORDER,default=None)
    args=ap.parse_args()
    start=time.time()
    summary=[]
    failures=[]
    kfold=KFold(n_splits=5,shuffle=True,random_state=42)
    stages={}
    for s in ('S1','S2'):
        stages[s]=(locked.import_sheet(locked.STAGEFILES[s][0]), locked.import_sheet(locked.STAGEFILES[s][1]))
    stages['S3']=stages['S2']
    for stage in ('S1','S2','S3'):
        if args.stage and stage!=args.stage: continue
        df_tr, df_test=stages[stage]
        predictors=list(df_tr.columns[:22])+([] if stage=='S1' else list(df_tr.columns[24:57]))
        subset_by_algo=locked.stage3_subsets(predictors) if stage=='S3' else {}
        ytr=np.log1p(df_tr['DBP concentration (µg/L)'].to_numpy(dtype=float))
        yte=np.log1p(df_test['DBP concentration (µg/L)'].to_numpy(dtype=float))
        for algo in locked.ORDER:
            if args.algorithm and algo!=args.algorithm: continue
            t=time.time()
            selected=list(subset_by_algo[algo] if stage=='S3' else predictors)
            params=locked.S1PARAMS[algo] if stage=='S1' else locked.S2PARAMS[algo]
            short=algo.replace(' ','_')
            archived=np.load(ROOT/'results'/f'line1_{stage}_{short}_preds.npz')
            oof=np.full(len(df_tr),np.nan,dtype=float)
            records=[]
            for k,(tridx,validx) in enumerate(kfold.split(df_tr),1):
                print('FITTING',stage,algo,'fold',k,flush=True)
                obj=prepare_and_fit(df_tr.iloc[tridx],selected,algo,params,tridx,stage,k)
                obj['validation_row_positions']=np.asarray(validx,dtype=np.int32)
                fn=f'{stage}_{short}_cv_fold{k}.joblib'
                dest=MODELS/fn
                joblib.dump(obj,dest,compress=3)
                loaded=joblib.load(dest)
                pred=predict_payload(loaded,df_tr.iloc[validx])
                oof[validx]=pred
                records.append({'file':fn,'validation_count':len(validx),'r2':float(r2_score(ytr[validx],pred)),'size_bytes':dest.stat().st_size})
            allidx=np.arange(len(df_tr))
            obj=prepare_and_fit(df_tr,selected,algo,params,allidx,stage,None)
            fn=f'{stage}_{short}_full_training.joblib'
            dest=MODELS/fn
            joblib.dump(obj,dest,compress=3)
            loaded=joblib.load(dest)
            test_pred=predict_payload(loaded,df_test)
            max_oof_error=float(np.max(np.abs(oof-archived['oof'])))
            max_test_error=float(np.max(np.abs(test_pred-archived['test'])))
            if max_oof_error>NUM_TOL or max_test_error>NUM_TOL:
                failures.append([stage,algo,max_oof_error,max_test_error])
            item={'stage':stage,'algorithm':algo,'n_features':len(selected),'full_training_file':fn,'cv_fold_files':records,
                  'cv':scores(ytr,oof),'test':scores(yte,test_pred),'max_oof_difference_vs_original_npz':max_oof_error,
                  'max_test_difference_vs_original_npz':max_test_error,'match_tolerance':NUM_TOL,'sec':round(time.time()-t,2)}
            summary.append(item)
            print('SAVED',stage,algo,'CV',round(item['cv']['r2'],6),'TEST',round(item['test']['r2'],6),
                  'max deviations',format(max_oof_error,'.3g'),format(max_test_error,'.3g'), 'sec',item['sec'],flush=True)
    record={'created':'2026-10-27','model_configurations':18,'full_training_models':18,'cv_fold_models':90,
            'training_observations':995,'test_observations':249,'method':'frozen parameters/subsets; no new search/RFE',
            'numerical_tolerance':NUM_TOL,'time_sec':round(time.time()-start,1),'failures':failures,'results':summary}
    (MODELS/'validation_report.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALL DONE','num_failures=',len(failures),'total sec',round(time.time()-start,1),flush=True)
    if failures:
        print('MISMATCHES',failures,flush=True)
        raise SystemExit(2)

if __name__=='__main__':
    main()
