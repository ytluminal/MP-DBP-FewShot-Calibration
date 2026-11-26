"""Secondary analysis: drop Ref14 from scoring (shared Test publication with Line2),
using the original Line3 locked predictors and its exact seven-Ref random permutation stream.
Calibrators and offsets are unchanged for the other six Refs; no model refit.
"""
from pathlib import Path
import csv,math
import numpy as np
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader(open(ROOT/'input'/'Line3_Test_Row_Predictions.csv')))
refs=[4,11,12,13,14,20,26];budgets=[.025,.05,.1,.2];nrep=500
schemes=['Publication-grouped','Conventional random'];models=['RF','XGBoost','LightGBM','CatBoost','Equal4']
fields=['RF_pred_ln1p','XGBoost_pred_ln1p','LightGBM_pred_ln1p','CatBoost_pred_ln1p','Equal4_pred_ln1p']
keys=[(s,st,m) for s in schemes for st in range(1,5) for m in models]
groups={(s,st):sorted((r for r in rows if r['Validation_Scheme']==s and int(r['Stage'])==st),key=lambda r:int(r['Row_1based'])) for s in schemes for st in range(1,5)}
base=groups[('Publication-grouped',1)]
y=np.array([float(a['Actual_ln1p']) for a in base]);ref=np.array([int(a['Ref_ID']) for a in base])
P=np.column_stack([[float(r[f]) for r in groups[(s,st)]] for s,st,m in keys for f in [fields[models.index(m)]]]);assert P.shape==(218,40)
refpos={r:np.flatnonzero(ref==r) for r in refs};rng=np.random.default_rng(42)
A=np.zeros((4,500,40,3));Z=np.zeros_like(A)
for rep in range(500):
    perms={r:rng.permutation(refpos[r]) for r in refs}
    for b,f in enumerate(budgets):
        remain=[];offset={}
        for r in refs:
            nc=math.ceil(f*len(refpos[r]));cal=perms[r][:nc]
            offset[r]=np.mean(y[cal,None]-P[cal],axis=0)
            if r!=14:remain.extend(perms[r][nc:])
        idx=np.array(remain,dtype=int);p=P[idx];pcal=p+np.vstack([offset[r] for r in ref[idx]])
        def metrics(pp):
            e=y[idx,None]-pp;sse=np.sum(e*e,axis=0);sst=np.sum((y[idx]-y[idx].mean())**2)
            return np.stack([1-sse/sst,np.sqrt(np.mean(e*e,axis=0)),np.mean(np.abs(e),axis=0)],axis=1)
        Z[b,rep]=metrics(p);A[b,rep]=metrics(pcal)
result=[]
for j,(s,st,m) in enumerate(keys):
 for b,f in enumerate(budgets):
    d=A[b,:,j]-Z[b,:,j];row={'Scheme':s,'Stage':st,'Model':m,'Budget_pct':f*100,'Ref_excluded':14,'N_ref_evaluated':6,
        'N_eval':sum(len(refpos[r])-math.ceil(f*len(refpos[r])) for r in refs if r!=14),
        'Zero_R2_median':float(np.median(Z[b,:,j,0])),'Cal_R2_median':float(np.median(A[b,:,j,0])),
        'Paired_Delta_R2_median':float(np.median(d[:,0])),'Fraction_improved_R2':float(np.mean(d[:,0]>0)),
        'Zero_RMSE_median':float(np.median(Z[b,:,j,1])),'Cal_RMSE_median':float(np.median(A[b,:,j,1])),
        'Paired_Delta_RMSE_median':float(np.median(d[:,1])),'Fraction_improved_RMSE':float(np.mean(d[:,1]<0))}
    result.append(row)
fn=ROOT/'outputs'/'Line3_ExcludeSharedRef14_Sensitivity_160.csv'
with open(fn,'w',newline='') as fh:
    writer=csv.DictWriter(fh,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
for s in schemes:
 print('\n',s)
 for f in budgets:
    q=[a for a in result if a['Scheme']==s and a['Budget_pct']==f*100];print('budget',f,'configs improving',sum(a['Paired_Delta_R2_median']>0 for a in q),'/20')
 for m in models:
    a=next(a for a in result if a['Scheme']==s and a['Stage']==4 and a['Model']==m and a['Budget_pct']==20.)
    print('20%-Stage4',m,'ZeroRem',round(a['Zero_R2_median'],4),'Cal',round(a['Cal_R2_median'],4),'gain',round(a['Paired_Delta_R2_median'],4),'success',round(a['Fraction_improved_R2'],3))
