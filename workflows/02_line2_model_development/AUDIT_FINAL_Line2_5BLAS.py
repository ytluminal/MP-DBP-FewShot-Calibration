"""Line 2 re-execution from hashed spreadsheets in extracted JSON representation.
No hyperparameter tuning on Test. Stage 1 LightGBM and CatBoost params restored via inner GroupKFold search.
Stages 2–4 from frozen Table S16/S18/S22 and fixed folds.
Uses current run.py helpers, not old NPZ cache predictions.
"""
import os,time,json,warnings,hashlib
os.environ['OPENBLAS_NUM_THREADS']='5';os.environ['OMP_NUM_THREADS']='3';os.environ['MKL_NUM_THREADS']='5'
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from docx import Document
from pathlib import Path
P=Path(__file__).resolve().parent
setup=(P/'run.py').read_text().split('STAGES=[1,2,3,4]')[0].replace("os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='3';os.environ['MKL_NUM_THREADS']='1'","os.environ['OPENBLAS_NUM_THREADS']='5';os.environ['OMP_NUM_THREADS']='3';os.environ['MKL_NUM_THREADS']='5'")
ns={};exec(compile(setup,str(P/'run.py'),'exec'),ns)
locks=json.loads((P/'audit_stage1_results.json').read_text())['models']
for m in ['LightGBM','CatBoost']:
    ns['C1'][m]=[tuple(f['selected_params']) for f in locks[m]['folds']]
expected_doc=Document(P/'SI_Verified.docx')
rename={'Native CatBoost':'CatBoost','RFE Native CatBoost':'CatBoost','RFE RF':'RF','RFE XGBoost':'XGBoost','RFE LightGBM':'LightGBM','RFE Equal4':'Equal4','RFE MoE4':'MoE4'}
SI={}
for row in expected_doc.tables[24].rows[1:]:
    a=[c.text.strip() for c in row.cells]
    if len(a)<6:continue
    try:st=int(a[0].split()[-1]); m=rename.get(a[1],a[1]);
    except (ValueError,IndexError):continue
    if m in ['RF','XGBoost','LightGBM','CatBoost','Equal4']:
        SI[(st,m)]={'CV_R2':float(a[2]),'Test_R2':float(a[3]),'Test_RMSE':float(a[4]),'Test_MAE':float(a[5])}
assert len(SI)==20, len(SI)
nt=ns['n'];rows=json.load(open(P/'test55.json'))['rows']
assert len(rows)==249 and nt==995
ns['Xall']=np.vstack([ns['Xall'],np.array(rows,dtype=object)])
Itest=np.arange(nt,nt+len(rows));Ytest=np.log1p(np.asarray([r[22] for r in rows],float))
trainy=ns['Y']
mods=['RF','XGBoost','LightGBM','CatBoost']
records=[]
for stage in [1,2,3,4]:
  for method in ['Publication-grouped','Conventional random']:
    tstart=time.time()
    p_cv={m:np.full(nt,np.nan) for m in mods}
    p_test={m:[] for m in mods}
    for i,(tr,va) in enumerate(ns['folds'][method]):
      t=time.time()
      for m in mods:
        feats=ns['ALL_FEAT'][:22] if stage==1 else (ns['SUBSETS'][m] if stage==4 else ns['ALL_FEAT'])
        xt,xv,cats=ns['prepare'](tr,va,feats,m,stage)
        _,xte,_=ns['prepare'](tr,Itest,feats,m,stage)
        with warnings.catch_warnings():
          warnings.filterwarnings('ignore')
          fitted=ns['fit'](m,xt,trainy[tr],cats,stage,i,tr)
          p_cv[m][va]=fitted.predict(xv)
          p_test[m].append(fitted.predict(xte))
      print('PROGRESS S',stage,method,'fold',i+1,'seconds',round(time.time()-t,1),flush=True)
    p_cv['Equal4']=np.mean(np.column_stack([p_cv[m] for m in mods]),axis=1)
    p_test={m:np.mean(np.asarray(p_test[m]),axis=0) for m in mods}
    p_test['Equal4']=np.mean(np.column_stack([p_test[m] for m in mods]),axis=1)
    output={'Train_Ref':ns['refs'],'Test_Ref':np.asarray([int(r[23]) for r in rows]),'Actual_Train_ln1p':trainy,'Actual_Test_ln1p':Ytest}
    for m in [*mods,'Equal4']:
      output['CV_pred_'+m]=p_cv[m];output['Test_pred_'+m]=p_test[m]
      rec={'Stage':stage,'Model':m,'Splitting':method,'CV R2':float(r2_score(trainy,p_cv[m])),
           'CV RMSE':float(np.sqrt(mean_squared_error(trainy,p_cv[m]))),'CV MAE':float(mean_absolute_error(trainy,p_cv[m])),
           'Test R2':float(r2_score(Ytest,p_test[m])),'Test RMSE':float(np.sqrt(mean_squared_error(Ytest,p_test[m]))),
           'Test MAE':float(mean_absolute_error(Ytest,p_test[m]))}
      if method=='Publication-grouped':
        rec['SI CV R2']=SI[(stage,m)]['CV_R2'];rec['SI Test R2']=SI[(stage,m)]['Test_R2'];
        rec['CV Diff']=rec['CV R2']-rec['SI CV R2'];rec['Test Diff']=rec['Test R2']-rec['SI Test R2']
      records.append(rec)
      print('RESULT',json.dumps(rec),flush=True)
    np.savez_compressed(P/f'AUDIT_FINAL_Stage{stage}_{method.replace("-","_").replace(" ","_")}.npz',**output)
    (P/'AUDIT_FINAL_All_Stages_Results.json').write_text(json.dumps(records,indent=2))
    print('FINISHED S',stage,method,'sec',round(time.time()-tstart,1),flush=True)
print('AUDIT FINISHED',flush=True)
