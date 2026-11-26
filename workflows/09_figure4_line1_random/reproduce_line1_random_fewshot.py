from pathlib import Path
import numpy as np, csv, math, json, hashlib
ROOT=Path('/mnt/data');OUT=Path(__file__).resolve().parent;OUT.mkdir(exist_ok=True)
SOURCE=Path(__file__).resolve().parent/'Frozen_Line1_External_Predictions_64.csv'
with SOURCE.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
refs=np.array([int(r['Ref_ID']) for r in rows]); y=np.array([float(r['Observed_ln1p']) for r in rows]); ref_order=(28,29,30)
assert len(y)==64 and {k:int(np.sum(refs==k)) for k in ref_order}=={28:12,29:12,30:40}
pred={stage:np.array([float(r[f'Stage{stage}_pred_ln1p']) for r in rows]) for stage in (1,2,3)}

def met(a,b):
 err=np.asarray(a)-np.asarray(b); denom=np.sum((a-a.mean())**2)
 return dict(R2=float(1-np.sum(err**2)/denom) if denom>1e-14 else float('nan'),RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))))

baseline=[]
for s in (1,2,3):
 for ref in ('Overall',*ref_order):
  idx=np.arange(len(y)) if ref=='Overall' else np.flatnonzero(refs==ref)
  baseline.append(dict(Stage=s,Ref_ID=ref,N=len(idx),**met(y[idx],pred[s][idx])))
  if ref=='Overall':print('Line 1 zero shot stage',s,baseline[-1])

n_reps=500;budgets=(.025,.05,.10,.20)
rng=np.random.default_rng(42)
positions={ref:np.flatnonzero(refs==ref) for ref in ref_order}
permutations=[]
perm_rows=[]
for rep in range(1,n_reps+1):
 per={ref:rng.permutation(positions[ref]) for ref in ref_order}
 permutations.append(per)
 for ref in ref_order:
  perm_rows.append(dict(Replicate=rep,Ref_ID=ref,Ordered_Source_Rows_1based=';'.join(str(int(rows[i]['Source_row'])) for i in per[ref]),Ordered_ZeroBased_Indices=';'.join(str(i) for i in per[ref])))

out=[];out_per=[]
for stage in (1,2,3):
 for budget in budgets:
  count={ref:math.ceil(budget*len(positions[ref])) for ref in ref_order}
  for rep,perm in enumerate(permutations,start=1):
   eval_parts=[]; cal_parts=[]; correction={}
   for ref in ref_order:
    cal=perm[ref][:count[ref]]; evl=perm[ref][count[ref]:]
    eval_parts.append(evl);cal_parts.append(cal)
    correction[ref]=float((y[cal]-pred[stage][cal]).mean())
   idx=np.concatenate(eval_parts)
   pred_cal=pred[stage][idx]+np.array([correction[r] for r in refs[idx]])
   a=met(y[idx],pred[stage][idx]);b=met(y[idx],pred_cal)
   record=dict(Stage=stage,Budget_pct=budget*100,Replicate=rep,N_cal=sum(count.values()),N_eval=len(idx),
      **{f'Zero_{k}':v for k,v in a.items()},**{f'Cal_{k}':v for k,v in b.items()},
      **{f'Delta_{k}':b[k]-a[k] for k in a},
      Improved_R2=int(b['R2']>a['R2']),Improved_RMSE=int(b['RMSE']<a['RMSE']),Improved_MAE=int(b['MAE']<a['MAE']))
   out.append(record)
   for ref,evl in zip(ref_order,eval_parts):
    before=met(y[evl],pred[stage][evl]);after=met(y[evl],pred[stage][evl]+correction[ref]);
    out_per.append(dict(Stage=stage,Budget_pct=budget*100,Replicate=rep,Ref_ID=ref,N_cal=count[ref],N_eval=len(evl),Offset=correction[ref],
        **{f'Zero_{k}':v for k,v in before.items()},**{f'Cal_{k}':v for k,v in after.items()},
        **{f'Delta_{k}':after[k]-before[k] for k in before},
        Improved_R2=int(after['R2']>before['R2']),Improved_RMSE=int(after['RMSE']<before['RMSE']),Improved_MAE=int(after['MAE']<before['MAE'])))

def summarize(rows,byref):
 records=[]
 for stage in (1,2,3):
  for budget in budgets:
   for ref in (ref_order if byref else ('Overall',)):
    vals=[r for r in rows if r['Stage']==stage and r['Budget_pct']==budget*100 and (not byref or r['Ref_ID']==ref)]
    r0=vals[0]
    d=dict(Stage=stage,Budget_pct=budget*100,Ref_ID=ref,N_cal=r0['N_cal'],N_eval=r0['N_eval'],N_replicates=len(vals))
    for key in ('Zero_R2','Cal_R2','Delta_R2','Zero_RMSE','Cal_RMSE','Delta_RMSE','Zero_MAE','Cal_MAE','Delta_MAE'):
     v=np.array([r[key] for r in vals]);
     for nam,p in [('median',.5),('q05',.05),('q25',.25),('q75',.75),('q95',.95)]:d[f'{key}_{nam}']=float(np.quantile(v,p))
    for k in ('R2','RMSE','MAE'):d[f'Fraction_Improved_{k}']=float(np.mean([r[f'Improved_{k}'] for r in vals]))
    records.append(d)
 return records
summary=summarize(out,False);summary_per=summarize(out_per,True)

def savecsv(name,recs):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(recs[0].keys()));writer.writeheader();writer.writerows(recs)
for name,recs in [('Baseline_ZeroShot.csv',baseline),('Overall_Summary.csv',summary),('Publication_Summary.csv',summary_per),('Overall_Replicates.csv',out),('Publication_Replicates.csv',out_per),('Random_Permutations.csv',perm_rows)]: savecsv(name,recs)
manifest=dict(Research='Figure 4 — New Publication Ref28–30 Line1 LightGBM Stage1–3 Random Few Shot',N=64,Paper_Counts={str(k):len(v) for k,v in positions.items()},Random_Seed=42,Replicates=500,Budgets_pct=[100*v for v in budgets],Calibration_Counts={str(int(v*1000)/10):{str(k):math.ceil(v*len(positions[k])) for k in ref_order} for v in budgets},Target='ln(1+C_DBP [ug/L])',Base='Previously audited Line1 full-training zero-shot predictions; no retraining',Calibration='Per-reference mean residual offset on randomly selected calibration rows; evaluation excludes calibration rows',Comparison='same remainder zero-shot vs calibrated',Draws='One RNG seed=42, one per-reference permutation per replicate in Ref order 28,29,30; nested budget prefixes, reused across stages',Caveat='These are test-publication retrospective random calibration simulations, not fully untouched publication-independent external test once calibration labels are used.',Source=str(SOURCE),Source_SHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
(OUT/'Experiment_Manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('\nOVERALL summaries:')
for r in summary:
 print('stage',r['Stage'],'budget',r['Budget_pct'],'zero',round(r['Zero_R2_median'],4),'cal',round(r['Cal_R2_median'],4),'delta',round(r['Delta_R2_median'],4),'prob',round(r['Fraction_Improved_R2'],3),'calrmse',round(r['Cal_RMSE_median'],4),'calmae',round(r['Cal_MAE_median'],4))
print('\nOUTPUT:',OUT)
