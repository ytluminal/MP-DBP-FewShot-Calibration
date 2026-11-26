"""Load and verify all 108 serialized models against pre-existing NPZ predictions.
Run from extracted archive: python validate_saved_models.py
No models are refitted. Original frozen prediction NPZ files are read-only.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='5';os.environ['MKL_NUM_THREADS']='5';os.environ['OMP_NUM_THREADS']='3'
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import KFold
import reproduce_line1 as r
from save_trained_models import predict_payload,scores

ROOT=Path(__file__).resolve().parent
warnings.filterwarnings('ignore',message='X does not have valid feature names')


def main():
    ds={}
    for s in ('S1','S2'):
        ds[s]=(r.import_sheet(r.STAGEFILES[s][0]),r.import_sheet(r.STAGEFILES[s][1]))
    ds['S3']=ds['S2']
    rows=[]; failures=[]
    for s in ('S1','S2','S3'):
        train,test=ds[s]
        for model in r.ORDER:
            alias=model.replace(' ','_')
            archived=np.load(ROOT/'results'/f'line1_{s}_{alias}_preds.npz')
            oof=np.full(len(train),np.nan)
            files=[]
            for k,(itr,iva) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(train),1):
                path=ROOT/'models'/f'{s}_{alias}_cv_fold{k}.joblib'
                obj=joblib.load(path)
                assert obj['scope']==f'CV_fold_{k}'
                assert np.array_equal(obj['validation_row_positions'],iva)
                assert np.array_equal(obj['train_row_positions'],itr)
                oof[iva]=predict_payload(obj,train.iloc[iva])
                files.append(path.name)
            final_path=ROOT/'models'/f'{s}_{alias}_full_training.joblib'
            obj=joblib.load(final_path)
            assert obj['scope']=='all_995_training'
            assert np.array_equal(obj['train_row_positions'],np.arange(995))
            pred=predict_payload(obj,test)
            max_cv=float(np.max(np.abs(oof-archived['oof'])))
            max_test=float(np.max(np.abs(pred-archived['test'])))
            passed=max_cv<1e-8 and max_test<1e-8
            if not passed: failures.append({'stage':s,'algorithm':model,'max_cv_diff':max_cv,'max_test_diff':max_test})
            row={'stage':s,'algorithm':model,'n_features':len(obj['feature_names']),
                 'final_model':final_path.name,'fold_models':files,
                 'cv':scores(archived['y_train'],oof), 'test':scores(archived['y_test'],pred),
                 'max_cv_abs_deviation_from_20261026_npz':max_cv,
                 'max_test_abs_deviation_from_20261026_npz':max_test,'matched':passed}
            rows.append(row)
            print('PASS' if passed else 'FAIL',s,model,'CV',format(row['cv']['r2'],'.6f'),
                  'Test',format(row['test']['r2'],'.6f'), 'max_delta',max_cv,max_test,flush=True)
    report={'created':'2026-10-27','note':'All saved models are newly fitted using frozen previously specified settings and original data; not original 2026-10-26 fitted objects; not full Optuna/RFE re-selection.',
            'archived_fitted_models':108,'full_training_models':18,'five_fold_cv_models':90,
            'no_refit_during_validation':True,'acceptance_abs_difference':1e-8,'failures':failures,'results':rows}
    (ROOT/'models'/'validation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    if failures: raise SystemExit(f'{len(failures)} failed checks')
    print('ALL 108 SERIALIZED MODEL FILES VERIFIED',flush=True)

if __name__=='__main__': main()
