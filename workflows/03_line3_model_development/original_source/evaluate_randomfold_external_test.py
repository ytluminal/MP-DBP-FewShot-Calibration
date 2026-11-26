"""Independent held-out Line-3 Test evaluation for frozen paired CV-scheme models.
Uses exactly Line3_CV_Comparison/run.py model, preprocessing, weighting, hyperparameters.
No Test data involved in CV development; row-level true Test response used only for scoring.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='1'
import sys, json, time, hashlib, importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
P=Path('/mnt/data/Line3_CV_Comparison')
spec=importlib.util.spec_from_file_location('line3_frozen_run', P/'run.py')
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
T22=m.T22['rows'];T55=m.T55['rows'];N=len(T55)
assert N==218
assert all(a==b[:24] for a,b in zip(T22,T55))
assert len(m.Xall)==1026 and len(m.Y)==1026
m.Xall=np.concatenate([m.Xall,np.asarray(T55,dtype=object)],axis=0)
TIDX=np.arange(1026,1026+N)
YTEST=np.log1p(np.array([float(r[22]) for r in T55],dtype=float))
REFTEST=np.array([int(r[23]) for r in T55])
assert set(m.refs).isdisjoint(set(REFTEST)) and len(set(REFTEST))==7
assert np.isfinite(YTEST).all()
print('Loaded datasets: Training 1026, Test 218; held-out Ref IDs:', sorted(set(REFTEST)),flush=True)
recs=[]
for stage in range(1,5):
  for scheme in ['Conventional random','Publication-grouped']:
    out={mod:[] for mod in m.MODELS}
    for fi,(tr,va) in enumerate(m.FOLDS[scheme]):
      start=time.time()
      for mod in m.MODELS:
        feat=m.ALL_FEAT[:22] if stage==1 else (m.subsets[mod] if stage==4 else m.ALL_FEAT)
        pred=m.model_predict(tr,TIDX,feat,mod,stage,fi)
        assert len(pred)==N and np.all(np.isfinite(pred))
        out[mod].append(pred)
      print(f'COMPLETED stage={stage} scheme={scheme} fold={fi+1} time={time.time()-start:.1f}s',flush=True)
    means={mod:np.mean(np.stack(v),axis=0) for mod,v in out.items()}
    means['Equal4']=np.mean(np.stack([means[mod] for mod in m.MODELS]),axis=0)
    np.savez_compressed(P/f'Test_Stage{stage}_{scheme.replace(" ","_")}.npz',
       actual_ln1p=YTEST,actual_ugL=np.array([float(r[22]) for r in T55]),Ref_ID=REFTEST,
       **{f'Pred_{k}':v for k,v in means.items()},
       **{f'FoldPred_{k}':np.stack(v) for k,v in out.items()})
    for mod,pred in means.items():
      rr,rm,ma=m.metrics(YTEST,pred)
      record={'Stage':stage,'Scheme':scheme,'Model':mod,'Test_R2':rr,'Test_RMSE':rm,'Test_MAE':ma,'n_test':N}
      recs.append(record)
      print('RESULT',json.dumps(record,ensure_ascii=False),flush=True)
    (P/'test_results_partial.json').write_text(json.dumps(recs,indent=2))
print('DONE ALL',flush=True)
