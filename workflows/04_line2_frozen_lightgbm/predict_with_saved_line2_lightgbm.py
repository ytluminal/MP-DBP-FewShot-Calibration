"""Inference only: loads previously frozen LightGBM models/preprocessors.

Example:
  python predict_with_saved_line2_lightgbm.py --input future_external_55_features.xlsx --output predictions.csv

For each stage x CV route, averages the 5 fold-model predictions in ln(1+C) space.
NEVER fits/refits on the new input, and NEVER inspects new input target to make a prediction.
"""
from pathlib import Path
import argparse,joblib,warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import line2_lightgbm_preprocess  # required for joblib unpickling

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,help='xlsx containing unchanged original feature labels')
    parser.add_argument('--output',required=True,help='output csv with predictions')
    parser.add_argument('--models-dir',default=str(Path(__file__).resolve().parent/'LightGBM_Reproduced_Models'))
    args=parser.parse_args()
    df=pd.read_excel(args.input,sheet_name='Data')
    models_dir=Path(args.models_dir)
    output=pd.DataFrame({'source_row':np.arange(2,len(df)+2)})
    for col in ['ref ID','DBP name','DBP concentration (ug/L)','DBP concentration (μg/L)']:
        if col in df.columns:output[col]=df[col].values
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names')
        for s in [1,2,3,4]:
            for typ in ['grouped','random']:
                allfolds=[]
                for fold in range(1,6):
                    fn=models_dir/f'LightGBM_Line2_S{s}_{typ}_F{fold}.joblib'
                    bundle=joblib.load(fn)
                    missing=sorted(set(bundle['features'])-set(df.columns))
                    if missing:raise ValueError(f'Missing columns for {fn.name}: {missing}')
                    features=bundle['processor'].transform_dataframe(df)
                    allfolds.append(bundle['model'].predict(features))
                p=np.mean(np.array(allfolds),axis=0)
                output[f'Line2_S{s}_{typ}_pred_ln1p']=p
                output[f'Line2_S{s}_{typ}_pred_ugL']=np.expm1(p)
    output.to_csv(args.output,index=False)
    print('Predictions saved:',args.output,'rows',len(output),'models',8)
if __name__=='__main__':main()
