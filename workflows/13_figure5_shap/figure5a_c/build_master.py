"""Build a canonical OOF SHAP archive; no model fitting or changes to the split."""
import os
os.environ['OPENBLAS_NUM_THREADS']='5'
os.environ['OMP_NUM_THREADS']='5'
os.environ['MKL_NUM_THREADS']='5'
from pathlib import Path
import argparse, json, hashlib, platform, importlib.metadata as md
import numpy as np, pandas as pd, joblib, lightgbm
from scipy import sparse
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits, threadpool_info
import line2_lightgbm_preprocess
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--workbook',default=str(ROOT/'MP-DBPs SI(3).xlsx'));p.add_argument('--outdir',default=str(ROOT));a=p.parse_args();out=Path(a.outdir);out.mkdir(exist_ok=True,parents=True);(out/'fold_matrices').mkdir(exist_ok=True)
models=ROOT/'models';ref='ref ID';target='DBP concentration (µg/L)'
arch=np.load(models/'Original_Validation_Predictions_S3_grouped.npz');raw=pd.read_excel(a.workbook,sheet_name='Ref 1-27');source_ix=np.flatnonzero(raw[ref].isin(np.unique(arch['cv_ref'])).to_numpy());t=raw.iloc[source_ix].reset_index(drop=True).copy();n=len(t);assert n==995 and np.array_equal(t[ref],arch['cv_ref']) and np.allclose(np.log1p(t[target]),arch['cv_observed'],atol=1e-12,rtol=0)
features=list(joblib.load(models/'LightGBM_Line2_S3_grouped_F1.joblib')['features']);assert len(features)==55 and len(set(features))==55
shap=np.full((n,55),np.nan);pred=np.full(n,np.nan);base=pred.copy();fold=np.zeros(n,int);info=[];maps={}
with threadpool_limits(limits=5,user_api='blas'):
 for f in range(1,6):
  path=models/f'LightGBM_Line2_S3_grouped_F{f}.joblib';b=joblib.load(path);pre=b['processor'];model=b['model'];assert b['stage']==3 and b['validation']=='Publication-grouped' and list(b['features'])==features
  train=t[ref].isin(b['train_ref_ids']).to_numpy();ix=np.flatnonzero(~train);assert (fold[ix]==0).all();fold[ix]=f
  x=pre.transform_dataframe(t.iloc[ix]);x=sparse.csr_matrix(x);x.sort_indices();pr=model.predict(x,num_threads=5);s=model.predict(x,pred_contrib=True,num_threads=5);s=s.toarray() if sparse.issparse(s) else np.asarray(s)
  numeric=[features[i] for i in pre.num_local];assert len(pre.numerator.get_feature_names_out())==len(numeric)
  mapping={c:[i] for i,c in enumerate(numeric)};names=numeric.copy();j=len(numeric)
  for i,cats in zip(pre.cat_local,pre.encoder.categories_):
   c=features[i];mapping[c]=list(range(j,j+len(cats)));names.extend([f'{c} = {v}' for v in cats]);j+=len(cats)
  assert j==x.shape[1]==s.shape[1]-1 and set(mapping)==set(features)
  for k,c in enumerate(features):shap[ix,k]=s[:,mapping[c]].sum(axis=1)
  pred[ix]=pr;base[ix]=s[:,-1];maps[str(f)]=mapping
  folder=out/'fold_matrices'/f'fold_{f}';folder.mkdir(exist_ok=True)
  sparse.save_npz(folder/'X_validation_csr.npz',x);np.savez_compressed(folder/'encoded_SHAP_and_predictions.npz',training_row=ix,encoded_SHAP=s[:,:-1],base_value=s[:,-1],prediction=pr)
  pd.DataFrame({'training_row':ix,'ref ID':t.iloc[ix][ref].to_numpy(),'source_excel_row':source_ix[ix]+2}).to_csv(folder/'validation_rows.csv',index=False)
  pd.DataFrame({'encoded_column_index':range(j),'encoded_column_name':names,'original_feature':[next(c for c,v in mapping.items() if k in v) for k in range(j)]}).to_csv(folder/'encoded_feature_map.csv',index=False)
  # Training matrix uses stored fitted preprocessor, not refitting. Validation matrix above is authoritative for SHAP.
  ti=np.flatnonzero(train);xt=pre.transform_dataframe(t.iloc[ti]);sparse.save_npz(folder/'X_training_csr.npz',sparse.csr_matrix(xt));pd.DataFrame({'training_row':ti,'ref ID':t.iloc[ti][ref].to_numpy(),'source_excel_row':source_ix[ti]+2}).to_csv(folder/'training_rows.csv',index=False)
  info.append({'fold':f,'n_validation':len(ix),'n_fit':len(ti),'encoded_features':j,'validation_refs':sorted(map(int,t.iloc[ix][ref].unique())),'fit_refs':sorted(map(int,t.iloc[ti][ref].unique())),'base_value':float(s[0,-1]),'max_native_additivity_error':float(np.max(abs(s.sum(1)-pr))),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
assert (fold>0).all() and np.isfinite(shap).all() and np.isfinite(pred).all() and np.isfinite(base).all()
err=abs(pred-arch['cv_predicted']);add=abs(shap.sum(1)+base-pred);assert err.max()<1e-8,(err.max(),np.sum(err>1e-8));assert add.max()<1e-10
meta=pd.DataFrame({'training_row':range(n),'source_excel_row':source_ix+2,'ref ID':t[ref],'fold':fold,'observed_ln1p':arch['cv_observed'],'OOF_prediction_ln1p':pred,'SHAP_base_ln1p':base,'archived_prediction_ln1p':arch['cv_predicted'],'prediction_delta':pred-arch['cv_predicted'],'SHAP_sum':shap.sum(1),'additivity_error':shap.sum(1)+base-pred})
raw_out=t.copy();raw_out.insert(0,'source_excel_row',source_ix+2);raw_out.insert(0,'training_row',range(n));raw_out.insert(2,'fold',fold);raw_out.to_csv(out/'Original_Training_Records.csv',index=False,float_format='%.17g')
meta.to_csv(out/'OOF_Predictions_and_Fold_Assignments.csv',index=False,float_format='%.17g');pd.DataFrame(shap,columns=features).to_csv(out/'SHAP_Matrix_995x55.csv',index=False,float_format='%.17g');pd.concat([meta,pd.DataFrame(shap,columns=['SHAP | '+c for c in features])],axis=1).to_csv(out/'Master_SHAP_Dataset.csv',index=False,float_format='%.17g');np.savez_compressed(out/'Master_SHAP_Exact.npz',training_row=np.arange(n),features=np.array(features),shap=shap,prediction=pred,base=base,fold=fold,source_excel_row=source_ix+2,ref_id=t[ref].to_numpy(),observed=arch['cv_observed']);pd.DataFrame({'feature_index':range(55),'feature':features,'mean_absolute_SHAP':np.abs(shap).mean(0)}).sort_values('mean_absolute_SHAP',ascending=False).to_csv(out/'Global_SHAP_Importance.csv',index=False,float_format='%.17g')
report={'n_rows':n,'n_publications':int(t[ref].nunique()),'n_features':len(features),'model':'Line 2 Stage 3 publication-grouped LightGBM','method':'native TreeSHAP; sum one-hot contributions to original feature; OOF only','max_abs_archived_prediction_difference':float(err.max()),'n_different_gt_1e-8':int((err>1e-8).sum()),'max_SHAP_additivity_error':float(add.max()),'archived_R2':float(r2_score(arch['cv_observed'],arch['cv_predicted'])),'recomputed_R2':float(r2_score(arch['cv_observed'],pred)),'folds':info,'threadpools':threadpool_info(),'python':platform.python_version(),'packages':{k:md.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib','threadpoolctl','matplotlib','openpyxl']},'workbook_sha256':hashlib.sha256(Path(a.workbook).read_bytes()).hexdigest(),'note':'No models refitted. Exact NPZ is the canonical binary source. CSV uses 17 significant digits. Encoded SHAP and model inputs are saved per fold.'};(out/'Validation_Report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['folds','threadpools']},indent=2));print([(q['fold'],q['n_validation'],q['encoded_features']) for q in info])
