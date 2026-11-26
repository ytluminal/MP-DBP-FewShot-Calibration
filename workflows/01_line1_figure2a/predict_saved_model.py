"""Use a saved final Line-1 model without refitting. Example:
python predict_saved_model.py --stage S3 --algorithm LightGBM --input "data/Conventional_RandomSplit_Stage2_Test_55features(3).xlsx"
This produces predictions in a CSV with both ln(1+C) and C in ug/L.
WARNING: Never load .joblib files from untrusted sources (arbitrary code execution risk).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from save_trained_models import predict_payload

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--stage',required=True,choices=['S1','S2','S3'])
parser.add_argument('--algorithm',required=True,choices=['Elastic Net','SVR-RBF','Random Forest','XGBoost','LightGBM','CatBoost'])
parser.add_argument('--input',required=True,help='Excel with all required feature columns in its Data worksheet')
parser.add_argument('--output',default=None,help='Output CSV; defaults to predictions_<stage>_<algorithm>.csv')
a=parser.parse_args()
name=f"{a.stage}_{a.algorithm.replace(' ','_')}_full_training.joblib"
f=ROOT/'models'/name
p=joblib.load(f)
df=pd.read_excel(a.input,sheet_name='Data')
yhat=predict_payload(p,df)
out=pd.DataFrame({'predicted_ln1p_DBP':yhat,'predicted_DBP_ug_L':np.expm1(yhat)})
output=Path(a.output) if a.output else ROOT/f"predictions_{a.stage}_{a.algorithm.replace(' ','_')}.csv"
out.to_csv(output,index=False)
print('Model:',f, 'Rows:',len(df), 'Saved:',output)
