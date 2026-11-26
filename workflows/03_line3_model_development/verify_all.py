"""Audits all 80 Line 3 frozen model configurations using repaired RF & Equal4 arrays."""
import sys,csv,hashlib,json,numpy as np
from pathlib import Path
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
root=Path(__file__).resolve().parent; outputs=root/'outputs'; archive=root/'reference';base=root/'verified_base_predictions'
models=['RF','XGBoost','LightGBM','CatBoost','Equal4']
ref={}
for typ in ['CV','Test']:
  for r in csv.DictReader(open(archive/f'Line3_{typ}_Row_Predictions.csv')):
    ref[(typ,int(r['Stage']),r['Validation_Scheme'],int(r['Row_1based']))]=r
results=[];rowvals=[]
for stage in range(1,5):
 for scheme in ['Publication-grouped','Conventional random']:
  key=f'{stage}_{scheme.replace(" ","_").replace("-","_")}'
  with np.load(outputs/f'RF_Stage{key}.npz') as rf,np.load(base/f'CV_Stage{key}.npz') as bc,np.load(base/f'Test_Stage{key}.npz') as bt:
   for typ in ['CV','Test']:
    y=rf['Y_train'] if typ=='CV' else rf['Y_test']
    pmap={'RF':rf[typ],'XGBoost':(bc if typ=='CV' else bt)['XGBoost'], 'LightGBM':(bc if typ=='CV' else bt)['LightGBM'],'CatBoost':(bc if typ=='CV' else bt)['CatBoost'],'Equal4':np.load(outputs/f'Equal4_{typ}_Stage{key}.npy')}
    for model,p in pmap.items():
     historical=np.asarray([float(ref[(typ,stage,scheme,i+1)][f'{model}_pred_ln1p']) for i in range(len(y))]);diff=abs(p-historical)
     scores=[r2_score(y,p),np.sqrt(mean_squared_error(y,p)),mean_absolute_error(y,p)]
     gold=[r2_score(y,historical),np.sqrt(mean_squared_error(y,historical)),mean_absolute_error(y,historical)]
     result=dict(Stage=stage,Scheme=scheme,Dataset=typ,Model=('Preprocessed CatBoost' if model=='CatBoost' else model),R2=float(scores[0]),RMSE=float(scores[1]),MAE=float(scores[2]),Reference_R2=float(gold[0]),Reference_RMSE=float(gold[1]),Reference_MAE=float(gold[2]),R2_Diff=float(scores[0]-gold[0]),RMSE_Diff=float(scores[1]-gold[1]),MAE_Diff=float(scores[2]-gold[2]),Max_Row_Diff=float(diff.max()),N=len(y),Verified=bool(diff.max()<1e-7))
     results.append(result)
     for i in range(len(y)):
      rowvals.append([stage,scheme,typ,model,i+1,float(y[i]),float(p[i]),float(historical[i]),float(p[i]-historical[i])])
assert len(results)==80
with open(outputs/'ALL80_Reproduction_Audit.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
with open(outputs/'ALL80_Row_Level_Predictions.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['Stage','Scheme','Dataset','Model','Row_1based','True_ln1p','Refit_Pred_ln1p','Frozen_Pred_ln1p','Difference']);w.writerows(rowvals)
maxerr=max(z['Max_Row_Diff'] for z in results);passed=sum(z['Verified'] for z in results)
filehash={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'input').iterdir()) if p.is_file()}
manifest={'configs_total':len(results),'configs_verified':passed,'Max_abs_prediction_delta':maxerr,'N_comparisons':len(rowvals),'RF_Equal4_configs':len([r for r in results if r['Model'] in ['RF','Equal4']]),'other_base_configs':len([r for r in results if r['Model'] not in ['RF','Equal4']]),'input_SHA256':filehash,'numpy_version':np.__version__,'python_version':sys.version,'sklearn_version':__import__('sklearn').__version__}
(outputs/'VERIFICATION_SUMMARY.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in manifest.items() if k!='input_SHA256'},indent=2));assert passed==80
