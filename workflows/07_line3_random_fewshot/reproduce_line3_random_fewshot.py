#!/usr/bin/env python3
"""Reproducible response-label-aware offset calibration of frozen LINE 3 Figure 2c TEST predictions.
Method matches Line 2 experiment, but Line 3 tests seven *different* held-out Ref IDs.
No ML fitting, tuning or feature selection is executed here.
"""
from pathlib import Path
import os,csv,json,math,hashlib,sys
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
SOURCE=HERE/'input'/'Line3_Test_Row_Predictions.csv'
AUDIT=HERE/'input'/'Line3_80_Metric_Audit.csv'
SEED=42;NREP=500;BUDGETS=[0.025,0.05,0.10,0.20];REFS=[4,11,12,13,14,20,26]
SCHEMES=['Publication-grouped','Conventional random'];MODELS=['RF','XGBoost','LightGBM','CatBoost','Equal4']
PFIELD={'RF':'RF_pred_ln1p','XGBoost':'XGBoost_pred_ln1p','LightGBM':'LightGBM_pred_ln1p','CatBoost':'CatBoost_pred_ln1p','Equal4':'Equal4_pred_ln1p'}
METRICS=['R2','RMSE','MAE']

def readcsv(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p,records):
    if not records:return
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

source=readcsv(SOURCE);audit=readcsv(AUDIT)
assert len(source)==218*8 and len(audit)==80
stages=list(range(1,5))
key_order=[(scheme,stage,model) for scheme in SCHEMES for stage in stages for model in MODELS]
master={(k):sorted([r for r in source if r['Validation_Scheme']==k[0] and int(r['Stage'])==k[1]],key=lambda r:int(r['Row_1based'])) for k in [(s,st) for s in SCHEMES for st in stages]}
for group in master.values():assert len(group)==218 and [int(z['Row_1based']) for z in group]==list(range(1,219))
ref=np.array([int(z['Ref_ID']) for z in master[(SCHEMES[0],1)]],dtype=int)
y=np.array([float(z['Actual_ln1p']) for z in master[(SCHEMES[0],1)]],dtype=float)
for group in master.values():
    assert np.array_equal(ref,np.array([int(z['Ref_ID']) for z in group]))
    assert np.allclose(y,np.array([float(z['Actual_ln1p']) for z in group]),atol=1e-15,rtol=0)
assert set(ref)==set(REFS)
refpos={r:np.flatnonzero(ref==r) for r in REFS}
assert {r:len(refpos[r]) for r in REFS}=={4:16,11:90,12:6,13:4,14:72,20:16,26:14}
P=np.column_stack([[float(z[PFIELD[model]]) for z in master[(scheme,stage)]] for scheme,stage,model in key_order]).astype(float)
assert P.shape==(218,40) and np.all(np.isfinite(P))
for scheme in SCHEMES:
 for stage in stages:
  sub=P[:,[key_order.index((scheme,stage,m)) for m in MODELS[:4]]].mean(axis=1)
  assert np.max(np.abs(sub-P[:,key_order.index((scheme,stage,'Equal4'))]))<1e-12

def vec_metrics(y0,p):
    # (N, K) prediction matrix, returns three metrics (K,)
    e=y0[:,None]-p
    sse=np.sum(e*e,axis=0)
    sst=np.sum((y0-np.mean(y0))**2)
    return np.stack([1-sse/sst,np.sqrt(np.mean(e*e,axis=0)),np.mean(np.abs(e),axis=0)],axis=1)

full=vec_metrics(y,P)
for i,(scheme,stage,model) in enumerate(key_order):
    matches=[r for r in audit if r['Dataset']=='External Test' and int(r['Stage'])==stage and r['Scheme']==scheme and r['Model']==model]
    assert len(matches)==1,(scheme,stage,model)
    for col,metric in enumerate(METRICS):
        target=float(matches[0]['New_'+metric]); actual=full[i,col]
        assert abs(target-actual)<5e-12,(scheme,stage,model,metric,target,actual)
print('Input/Test audit: verified all 40 locked vectors, identical Test rows, 40x3 external Test metrics, Equal4 means.',flush=True)

# To ensure nonadaptive, identical selection across *all* 40 model configurations.
rng=np.random.default_rng(SEED)
perms=[{r:rng.permutation(refpos[r]) for r in REFS} for rep in range(NREP)]
perm_records=[{'Replicate':rep,'Ref_ID':r,'N':len(refpos[r]),'Ordered_Test_Rows_1based':';'.join(str(v+1) for v in pm[r])} for rep,pm in enumerate(perms,1) for r in REFS]
writecsv(OUT/'Random_Permutations_500x7.csv',perm_records)
budget_rows=[]
for frac in BUDGETS:
 for r in REFS:
    n=len(refpos[r]);ncal=math.ceil(frac*n)
    budget_rows.append({'Budget_pct':frac*100,'Ref_ID':r,'N':n,'n_cal':ncal,'n_remaining':n-ncal})
writecsv(OUT/'Per_Publication_Budgets.csv',budget_rows)

# A[budget,rep,model_config,metric] in [R2,RMSE,MAE]
A=np.full((4,NREP,40,3),np.nan);Z=np.full_like(A,np.nan)
REFA=np.full((4,NREP,7,40,3),np.nan);REFZ=np.full_like(REFA,np.nan)
RAW=[]
perref_file=OUT/'Line3_RandomFewShot_PerPublication_Replicates_560000.csv'
with open(perref_file,'w',newline='',encoding='utf-8') as f:
    keys=['Scheme','Stage','Model','Budget_pct','Replicate','Ref_ID','N_cal','N_eval']+[f'{p}_{m}' for p in ('Zero','Cal','Delta') for m in METRICS]
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    for rep,pm in enumerate(perms,1):
      for b,frac in enumerate(BUDGETS):
        counts={r:math.ceil(frac*len(refpos[r])) for r in REFS}
        cal={r:pm[r][:counts[r]] for r in REFS}
        remaining={r:pm[r][counts[r]:] for r in REFS}
        ev=np.concatenate([remaining[r] for r in REFS]);assert len(ev)==218-sum(counts.values())
        offsets={r:np.mean(y[cal[r],None]-P[cal[r],:],axis=0) for r in REFS}
        zero=vec_metrics(y[ev],P[ev]);calpred=P[ev]+np.vstack([offsets[r] for r in ref[ev]])
        after=vec_metrics(y[ev],calpred)
        A[b,rep-1]=after;Z[b,rep-1]=zero
        for j,(scheme,stage,model) in enumerate(key_order):
            d=after[j]-zero[j]
            RAW.append({'Scheme':scheme,'Stage':stage,'Model':model,'Budget_pct':frac*100,'Replicate':rep,'N_cal':sum(counts.values()),'N_eval':len(ev),
                **{f'Zero_{m}':zero[j,i] for i,m in enumerate(METRICS)},**{f'Cal_{m}':after[j,i] for i,m in enumerate(METRICS)},**{f'Delta_{m}':d[i] for i,m in enumerate(METRICS)},
                'Improved_R2':int(d[0]>0),'Improved_RMSE':int(d[1]<0),'Improved_MAE':int(d[2]<0)})
        for ir,r in enumerate(REFS):
            ix=remaining[r];assert len(ix)>=2
            z=vec_metrics(y[ix],P[ix]);a=vec_metrics(y[ix],P[ix]+offsets[r][None,:]);
            REFZ[b,rep-1,ir]=z;REFA[b,rep-1,ir]=a
            for j,(scheme,stage,model) in enumerate(key_order):
                d=a[j]-z[j]
                w.writerow({'Scheme':scheme,'Stage':stage,'Model':model,'Budget_pct':frac*100,'Replicate':rep,'Ref_ID':r,'N_cal':counts[r],'N_eval':len(ix),
                    **{f'Zero_{m}':z[j,i] for i,m in enumerate(METRICS)},**{f'Cal_{m}':a[j,i] for i,m in enumerate(METRICS)},**{f'Delta_{m}':d[i] for i,m in enumerate(METRICS)}})
      if rep in (100,250,500):print('Completed',rep,'of',NREP,'replicates across all models/stages/protocols/budgets',flush=True)
writecsv(OUT/'Line3_RandomFewShot_Replicates_80000.csv',RAW)
SUM=[];R_SUM=[];ZERO=[]
for j,(scheme,stage,model) in enumerate(key_order):
    ZERO.append({'Scheme':scheme,'Stage':stage,'Model':model,'N_Test':218,**{f'Zero_{m}':full[j,k] for k,m in enumerate(METRICS)}})
    for b,frac in enumerate(BUDGETS):
        counts={r:math.ceil(frac*len(refpos[r])) for r in REFS};n_cal=sum(counts.values());n_eval=218-n_cal
        row={'Scheme':scheme,'Stage':stage,'Model':model,'Budget_pct':frac*100,'N_cal':n_cal,'N_eval':n_eval}
        for k,m in enumerate(METRICS):
            row[f'Zero_full_{m}']=full[j,k]
        for pref,arr in [('Zero',Z),('Cal',A),('Delta',A-Z)]:
            for k,m in enumerate(METRICS):
                vals=arr[b,:,j,k]
                row.update({f'{pref}_{m}_{q}':float(np.quantile(vals,p)) for q,p in [('median',.5),('q05',.05),('q25',.25),('q75',.75),('q95',.95)]})
        for k,m in enumerate(METRICS):row[f'Fraction_improved_{m}']=float(np.mean((A-Z)[b,:,j,k]*(1 if m=='R2' else -1)>0))
        SUM.append(row)
        for ir,r in enumerate(REFS):
            rr={'Scheme':scheme,'Stage':stage,'Model':model,'Budget_pct':frac*100,'Ref_ID':r,'N_cal':counts[r],'N_eval':len(refpos[r])-counts[r]}
            for k,m in enumerate(METRICS):
                rr[f'Zero_{m}_median']=float(np.median(REFZ[b,:,ir,j,k]));rr[f'Cal_{m}_median']=float(np.median(REFA[b,:,ir,j,k]));
                rr[f'Delta_{m}_median']=float(np.median(REFA[b,:,ir,j,k]-REFZ[b,:,ir,j,k]));
                rr[f'Fraction_improved_{m}']=float(np.mean((REFA[b,:,ir,j,k]-REFZ[b,:,ir,j,k])*(1 if m=='R2' else -1)>0))
            R_SUM.append(rr)
writecsv(OUT/'Line3_ZeroShot_Original_218.csv',ZERO)
writecsv(OUT/'Line3_RandomFewShot_Summary_160.csv',SUM)
writecsv(OUT/'Line3_RandomFewShot_PerPublication_Summary_1120.csv',R_SUM)
assert len(SUM)==160 and len(R_SUM)==1120 and len(RAW)==80000
# Internal audit: paired difference preserves each exact replicate/row set, and same-rem zero-shot scores don't change across budgets other than removed rows.
for b in range(4):
 for j in range(40):
    zvals=Z[b,:,j,0];calvals=A[b,:,j,0]
    assert np.isfinite(calvals).all() and np.isfinite(zvals).all()
    assert abs(float(np.median(calvals-zvals)) - next(s['Delta_R2_median'] for s in SUM if s['Scheme']==key_order[j][0] and s['Stage']==key_order[j][1] and s['Model']==key_order[j][2] and s['Budget_pct']==BUDGETS[b]*100))<1e-12

manifest={'scope':'Line3 Figure2c: all five models, Stages 1-4, 2 CV-trained external-test prediction schemes, four budgets, 500 random replicates each',
    'input_prediction_sha256':sha(SOURCE),'input_audit_sha256':sha(AUDIT),'rows':218,'references':REFS,'budget_fractions':BUDGETS,'replicates_per_configuration':NREP,'random_seed':SEED,'total_calibration_comparisons':len(RAW),
    'total_per_publication_comparisons':NREP*len(BUDGETS)*len(REFS)*len(key_order),
    'sampling_rule':'single np.random.default_rng(42); for each replicate, permute all 7 Ref ID row indices in ascending Ref ID order; nested prefixes for 2.5/5/10/20%; same draws shared across ALL models/stages/protocols',
    'budget_rule':'per-publication ceil(f*N); report exact counts; minimum one observation per ref',
    'calibration':'per-reference additive mean(y_cal - yhat_cal) on ln(1+DBP concentration); all calibration rows excluded from zero-shot and calibrated scoring',
    'validation':'frozen external Test predictions independently previously reproduced against 80 SI targets; all 40 row vectors validated against source audit before calibration; no new ML training, tuning or selection',
    'uncertainty':'500 selection-resampling quantiles describe variation conditional on seven held-out publications; not cross-publication confidence intervals',
    'outputs':{p.name:sha(p) for p in OUT.glob('*.csv')}}
(OUT/'Experiment_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('COMPLETE: summarized',len(SUM),'model-stage-protocol-budget configs /',len(RAW),'replicates /',len(R_SUM),'per-ref summaries')
for b,frac in enumerate(BUDGETS):
 print('Budget',frac*100,'ncal=',sum(math.ceil(frac*len(refpos[r])) for r in REFS),'n_eval=',218-sum(math.ceil(frac*len(refpos[r])) for r in REFS))
 for model in MODELS:
  j=key_order.index(('Publication-grouped',4,model));cal=float(np.median(A[b,:,j,0]));before=full[j,0];gain=float(np.median(A[b,:,j,0]-Z[b,:,j,0]));succ=float(np.mean((A-Z)[b,:,j,0]>0))
  print(f' {model:9s} zero_all={before:+.4f} calibrated={cal:+.4f} paired_gain={gain:+.4f} frac_gain={succ:.1%}')
