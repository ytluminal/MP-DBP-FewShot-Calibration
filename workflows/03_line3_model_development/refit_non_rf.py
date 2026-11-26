"""Independent fresh CV and external Test refit of one fixed Line 3 base algorithm.

Each model/stage/CV scheme must run in its OWN subprocess.  The original
frozen recipe in input/run.py is imported and used without altered parameters.
Training data and publication/fold indices are not modified.  No Test label
is used to fit the model.  Saved archived predictions are used only to audit.
"""
import os
for key, value in {'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','BLIS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}.items():
    os.environ[key]=value
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--model', choices=['XGBoost','LightGBM','CatBoost'],required=True)
parser.add_argument('--stage', choices=[1,2,3,4], type=int,required=True)
parser.add_argument('--scheme', choices=['Publication-grouped','Conventional random'],required=True)
args=parser.parse_args()

spec=importlib.util.spec_from_file_location('line3_frozen_recipe', ROOT/'input'/'run.py')
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

ntrain=len(m.Y)
testrows=m.T55['rows']
ytest=np.log1p(np.asarray([float(r[22]) for r in testrows],dtype=float))
m.Xall=np.concatenate([m.Xall, np.asarray(testrows,dtype=object)],axis=0)
testindices=np.arange(ntrain,ntrain+len(testrows))
features=m.ALL_FEAT[:22] if args.stage==1 else (m.subsets[args.model] if args.stage==4 else m.ALL_FEAT)
oof=np.full(ntrain,np.nan,dtype=float)
heldout=[]
start=time.time()
for fi,(tr,va) in enumerate(m.FOLDS[args.scheme]):
    # CV OOF and external Test predictions are independently refitted, as in
    # frozen original CV and Test prediction scripts. DO NOT concatenate
    # validation+Test arrays; posterior MICE draws can then change.
    oof[va]=m.model_predict(tr,va,features,args.model,args.stage,fi)
    heldout.append(m.model_predict(tr,testindices,features,args.model,args.stage,fi))
    print('FOLD',args.model,args.stage,args.scheme,fi+1,'seconds',round(time.time()-start,2),flush=True)
assert np.isfinite(oof).all()
test_pred=np.mean(np.stack(heldout),axis=0)

maxdelta=0.
for typ,pred in [('CV',oof),('Test',test_pred)]:
    with (ROOT/'reference'/f'Line3_{typ}_Row_Predictions.csv').open(newline='',encoding='utf-8') as fh:
        rows=[r for r in csv.DictReader(fh) if int(r['Stage'])==args.stage and r['Validation_Scheme']==args.scheme]
    rows.sort(key=lambda r:int(r['Row_1based']))
    historical=np.asarray([float(r[f'{args.model}_pred_ln1p']) for r in rows],dtype=float)
    assert len(pred)==len(historical)
    delta=float(np.max(np.abs(pred-historical)))
    maxdelta=max(maxdelta,delta)
    print('CHECK',args.model,args.stage,args.scheme,typ,'max_abs_prediction_delta',delta,flush=True)
    if not delta<1e-7:
        raise RuntimeError(f'Mismatch: {args.model} {args.stage} {args.scheme} {typ}; max diff={delta}')

key=f'{args.stage}_{args.scheme.replace(" ","_").replace("-","_")}'
out=ROOT/'refitted_components'
out.mkdir(exist_ok=True)
np.savez_compressed(out/f'{args.model}_Stage{key}.npz',CV=oof,Test=test_pred,Y_train=m.Y,Y_test=ytest)
print('PASS',args.model,args.stage,args.scheme,'MAX',maxdelta,'seconds',round(time.time()-start,2),flush=True)
