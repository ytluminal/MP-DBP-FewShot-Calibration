"""Update Figure 5b/c data from the canonical SHAP archive, preserving original row selection and trend bins."""
from pathlib import Path
import pandas as pd, numpy as np, json
ROOT=Path(__file__).resolve().parent
src=ROOT/'legacy'/'Source_Data_and_Parameters';out=ROOT/'figures'/'Source_Data_and_Parameters';out.mkdir(exist_ok=True,parents=True)
z=np.load(ROOT/'Master_SHAP_Exact.npz');features=list(z['features']);raw=pd.read_csv(ROOT/'Original_Training_Records.csv',float_precision='round_trip');audit={}
bfile='Figure5b_AllDBPs_DOC_SHAP_Source_Data.csv';bt='Figure5b_AllDBPs_DOC_SHAP_Trend_Summary.csv';cfile='Figure5c_AllDBPs_MPloading_ParticleSize_Source_Data_v2.csv';ct='Figure5c_AllDBPs_MPloading_Trend_Summary_v2.csv'
for letter,file,tr,feature,col in [('b',bfile,bt,'DOC (mg/L C)','SHAP_DOC_on_ln1p_DBP'),('c',cfile,ct,'MP dose (g/L)','SHAP_MP_loading_on_ln1p_DBP')]:
 old=pd.read_csv(src/file,float_precision='round_trip');d=old.copy();ix=d['Original Training row (zero-based)'].to_numpy();assert len(np.unique(ix))==len(ix);assert np.array_equal(d['ref ID'],z['ref_id'][ix]);assert np.array_equal(d['OOF fold'],z['fold'][ix]);assert np.array_equal(d['DBP name'],raw.iloc[ix]['DBP name']);assert np.allclose(d['DBP concentration (µg/L)'],raw.iloc[ix]['DBP concentration (µg/L)'],rtol=1e-8,atol=1e-9)
 d[col]=z['shap'][ix,features.index(feature)];d['OOF predicted ln1p DBP']=z['prediction'][ix];oldtr=pd.read_csv(src/tr);t=oldtr.copy();members=[]
 if letter=='b':
  order=old.sort_values('DOC (mg/L C)',kind='stable').index.to_numpy();start=0
  for _,r in oldtr.iterrows():
   end=start+int(r.n);members.append(order[start:end]);start=end
  assert start==len(d)
 else:
  for _,r in oldtr.iterrows():members.append(np.flatnonzero(np.isclose(old['MP loading (g/L)'],r['MP loading (g/L)'],atol=1e-12,rtol=0)))
 for i,ids in enumerate(members):
  assert len(ids)==int(oldtr.iloc[i].n)
  if letter=='b':assert np.isclose(old.iloc[ids]['DOC (mg/L C)'].median(),oldtr.iloc[i].DOC_center,rtol=1e-10)
  orig=np.quantile(old.iloc[ids][col],[.5,.25,.75]);assert np.allclose(orig,oldtr.iloc[i][['SHAP_median','SHAP_q25','SHAP_q75']].astype(float),atol=1e-9,rtol=0)
  t.loc[i,['SHAP_median','SHAP_q25','SHAP_q75']]=np.quantile(d.iloc[ids][col],[.5,.25,.75])
 d.to_csv(out/file,index=False,float_format='%.17g');t.to_csv(out/tr,index=False,float_format='%.17g')
 audit[letter]={'n':len(d),'changed_predictions_gt_1e-8':int((abs(d['OOF predicted ln1p DBP']-old['OOF predicted ln1p DBP'])>1e-8).sum()),'changed_SHAP_gt_1e-8':int((abs(d[col]-old[col])>1e-8).sum()),'max_SHAP_change':float(abs(d[col]-old[col]).max()),'max_trend_median_change':float(abs(t.SHAP_median-oldtr.SHAP_median).max()),'original_trend_bin_membership_verified':True}
(out/'plot_data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
