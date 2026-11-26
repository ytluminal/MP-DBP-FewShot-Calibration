"""Rebuild Equal4 using only all four newly refitted base models; audit 80 configurations."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--stages',type=int,nargs='+',choices=[1,2,3,4],default=[1,2,3,4])
parser.add_argument('--schemes',nargs='+',choices=['Publication-grouped','Conventional random'],default=['Publication-grouped','Conventional random'])
args=parser.parse_args()
ref={}
for typ in ['CV','Test']:
 with (root/'reference'/f'Line3_{typ}_Row_Predictions.csv').open(newline='') as f:
  for r in csv.DictReader(f):ref[(typ,int(r['Stage']),r['Validation_Scheme'],int(r['Row_1based']))]=r
models=['RF','XGBoost','LightGBM','CatBoost','Equal4']
rows=[]
for stage in args.stages:
 for scheme in args.schemes:
  key=f'{stage}_{scheme.replace(" ","_").replace("-","_")}'
  with np.load(root/'outputs'/f'RF_Stage{key}.npz') as rf:
   for typ in ['CV','Test']:
    y=np.array(rf['Y_train'] if typ=='CV' else rf['Y_test'])
    predictions={'RF':np.array(rf[typ])}
    for model in models[1:4]:
     with np.load(root/'refitted_components'/f'{model}_Stage{key}.npz') as f:
      predictions[model]=np.array(f[typ])
    predictions['Equal4']=np.mean(np.column_stack([predictions[x] for x in models[:4]]),axis=1)
    (root/'refitted_components').mkdir(exist_ok=True)
    np.save(root/'refitted_components'/f'Equal4_{typ}_Stage{key}.npy',predictions['Equal4'])
    for mod,pred in predictions.items():
     gold=np.asarray([float(ref[(typ,stage,scheme,i+1)][f'{mod}_pred_ln1p']) for i in range(len(y))])
     maxerr=float(np.max(abs(pred-gold)))
     r2=float(r2_score(y,pred)); rmse=float(np.sqrt(mean_squared_error(y,pred)));mae=float(mean_absolute_error(y,pred))
     rows.append(dict(Stage=stage,Scheme=scheme,Dataset=typ,Model=('Preprocessed CatBoost' if mod=='CatBoost' else mod),R2=r2,RMSE=rmse,MAE=mae,Ref_R2=float(r2_score(y,gold)),Ref_RMSE=float(np.sqrt(mean_squared_error(y,gold))),Ref_MAE=float(mean_absolute_error(y,gold)),Max_Row_Diff=maxerr,Passed=bool(maxerr<1e-7),N=len(y)))
     print('CHECK',stage,scheme,typ,mod,'R2',round(r2,8),'maxerr',maxerr,flush=True)
output=root/'refitted_components'/'FULL_ALL_MODELS_AUDIT.csv'
with output.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'configurations':len(rows),'passed':sum(x['Passed'] for x in rows),'maximum_absolute_prediction_difference':max(x['Max_Row_Diff'] for x in rows),'output_file':str(output)}
(root/'refitted_components'/'FULL_REFIT_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
if summary['passed']!=summary['configurations']:raise SystemExit('Full fresh refit did not reproduce all frozen references.')
