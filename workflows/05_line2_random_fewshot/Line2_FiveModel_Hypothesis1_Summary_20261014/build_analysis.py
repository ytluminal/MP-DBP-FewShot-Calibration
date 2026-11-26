"""Source-only aggregation of five audited Line-2 Random Few-Shot archives.
No refitting, data resplitting, feature selection, or independent test use for tuning.
Uses stdlib CSV and NumPy, no spreadsheet parsing libraries.
"""
import csv, pathlib, json, hashlib, collections, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path('/mnt/data')
OUT=ROOT/'Line2_FiveModel_Hypothesis1_20261014'
FILES = [
  ('Random Forest','RF','Line2_RF_RandomFewShot_20261014','Line2_RF_RandomFewShot_Reproduction_20261014.zip'),
  ('XGBoost','XGB','Line2_XGB_RandomFewShot_20261014','Line2_XGBoost_RandomFewShot_Reproduction_20261014.zip'),
  ('LightGBM','LGBM','Line2_LightGBM_RandomFewShot_20261014','Line2_LightGBM_RandomFewShot_Reproduction_20261014.zip'),
  ('Native CatBoost','NCB','Line2_NativeCatBoost_RandomFewShot_20261014','Line2_NativeCatBoost_RandomFewShot_Reproduction_20261014.zip'),
  ('Equal4','EQ4','Line2_Equal4_RandomFewShot_20261014','Line2_Equal4_RandomFewShot_Reproduction_20261014.zip'),
]

def read_csv(p):
  with open(p,encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(p,records,fields):
  with open(p,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields, extrasaction='ignore');w.writeheader();w.writerows(records)

def fl(r,k):return float(r[k])
def intx(r,k):return int(float(r[k]))
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

summ=[]; perpub=[]; rep=[]; hashes={}
for model,prefix,folder,z in FILES:
  folder=ROOT/folder/'outputs'
  a=read_csv(folder/f'{prefix}_RandomFewShot_Summary_32.csv')
  b=read_csv(folder/f'{prefix}_RandomFewShot_PerPublication_Summary_160.csv')
  c=read_csv(folder/f'{prefix}_RandomFewShot_Replicates_16000.csv')
  assert len(a)==32 and len(b)==160 and len(c)==16000
  hashes[z]=sha(ROOT/z)
  summ += [dict(Model=model,**r) for r in a]
  perpub += [dict(Model=model,**r) for r in b]
  rep += [dict(r,Model=model) for r in c]
assert len(summ)==160 and len(perpub)==800 and len(rep)==80000

# Re-audit a core set of stats from the raw 80,000 replicate records.
groups=collections.defaultdict(list)
for r in rep:
 k=(r['Model'],r['Protocol'],intx(r,'Stage'),fl(r,'Budget_pct'))
 groups[k].append(r)
for r in summ:
 k=(r['Model'],r['Protocol'],intx(r,'Stage'),fl(r,'Budget_pct'))
 rr=groups[k]; assert len(rr)==500
 for field,raw in [('Delta_R2_median','Delta_R2'),('Cal_R2_median','Cal_R2'),('Cal_RMSE_median','Cal_RMSE'),('Zero_R2_median','Zero_R2'),('Delta_RMSE_median','Delta_RMSE')]:
  err=abs(fl(r,field)-float(np.median([fl(x,raw) for x in rr])))
  assert err<2e-12,(k,field,err)
 good=sum(fl(x,'Delta_R2')>0 for x in rr)/500
 assert abs(good-fl(r,'Fraction_improved_R2'))<1e-12,(k,good)

# A compact 160-row analysis table.
fields=['Model','Protocol','Stage','Budget_pct','N_cal','N_eval','Zero_full_R2','Zero_R2_median','Cal_R2_median','Cal_R2_q05','Cal_R2_q95','Delta_R2_median','Delta_R2_q05','Delta_R2_q95','Fraction_improved_R2','Zero_RMSE_median','Cal_RMSE_median','Delta_RMSE_median','Fraction_improved_RMSE','Zero_MAE_median','Cal_MAE_median','Delta_MAE_median','Fraction_improved_MAE']
clean=[]
for r in summ:
 clean.append({k:(r[k] if k in ('Model','Protocol') else intx(r,k) if k in ('Stage','N_cal','N_eval') else fl(r,k)) for k in fields})
clean.sort(key=lambda x:(['Grouped','RandomCV'].index(x['Protocol']),int(x['Stage']),x['Model'],x['Budget_pct']))
write_csv(OUT/'All160_Model_Stage_Budget_Summary.csv',clean,fields)

# Reference-level summary for Stage4 and all stages, using saved replicate-aggregated medians.
pfields=['Model','Protocol','Stage','Budget_pct','Ref_ID','N_cal','N_eval','Zero_RMSE_median','Cal_RMSE_median','Delta_RMSE_median','Fraction_RMSE_improved','Zero_MAE_median','Cal_MAE_median','Fraction_MAE_improved','Zero_R2_median','Cal_R2_median']
cleanpub=[]
for r in perpub:
 cleanpub.append({k:(r[k] if k in ('Model','Protocol') else intx(r,k) if k in ('Stage','Ref_ID','N_cal','N_eval') else fl(r,k)) for k in pfields})
cleanpub.sort(key=lambda x:(['Grouped','RandomCV'].index(x['Protocol']),int(x['Stage']),x['Model'],x['Budget_pct'],x['Ref_ID']))
write_csv(OUT/'All800_PerPublication_Summary.csv',cleanpub,pfields)

# Overall comparisons of all 20 configurations per protocol and budget.
allbudget=[]
for protocol in ['Grouped','RandomCV']:
  for budget in [2.5,5,10,20]:
    s=[r for r in clean if r['Protocol']==protocol and r['Budget_pct']==budget]
    wins=np.array([r['Fraction_improved_R2'] for r in s]);d=np.array([r['Delta_R2_median'] for r in s]);
    allbudget.append(dict(Protocol=protocol,Budget_pct=budget,N_model_stage=20,N_median_gain_positive=int(np.sum(d>0)),N_median_gain_negative=int(np.sum(d<0)),N_fraction_improvement_at_least_80pct=int(np.sum(wins>=.8)),Median_of_20_config_median_Delta_R2=float(np.median(d)),Median_of_20_config_improvement_fraction=float(np.median(wins))))
write_csv(OUT/'Budget_Overall_Comparison.csv',allbudget,list(allbudget[0]))

# Publication diagnostic at 20% Stage 4, grouped
pub20=[r for r in cleanpub if r['Protocol']=='Grouped' and r['Stage']==4 and r['Budget_pct']==20]
assert len(pub20)==25
countpos=sum(r['Delta_RMSE_median']<0 for r in pub20)

# Stage4 Grouped snapshot by model and budget
stage4=[r for r in clean if r['Protocol']=='Grouped' and r['Stage']==4]
write_csv(OUT/'Stage4_Grouped_20_Results.csv',stage4,fields)

# Stage4 paired delta R2 learning curves; matplotlib default colors.
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,5.7))
for model,*_ in FILES:
  rr=sorted([r for r in stage4 if r['Model']==model],key=lambda x:x['Budget_pct'])
  x=[r['Budget_pct'] for r in rr];y=[r['Delta_R2_median'] for r in rr]
  lo=[r['Delta_R2_q05'] for r in rr];hi=[r['Delta_R2_q95'] for r in rr]
  ln=ax.plot(x,y,marker='o',linewidth=2.25,label=model)[0]
  ax.fill_between(x,lo,hi,color=ln.get_color(),alpha=.10,linewidth=0)
ax.axhline(0,lw=1,ls='--',c='gray')
ax.set_xticks([2.5,5,10,20],['2.5%','5%','10%','20%'])
ax.set_xlabel('Calibration budget (within each target publication)')
ax.set_ylabel(r'Paired $\Delta R^2$ (calibrated $-$ zero-shot)')
ax.set_title('Line 2 | Stage 4 | Publication-grouped | Random few-shot')
ax.legend(loc='upper left',fontsize=9,ncol=2,frameon=False)
ax.grid(axis='y',alpha=.22)
fig.tight_layout();fig.savefig(OUT/'Stage4_Grouped_Paired_R2_Gains_600dpi.png',dpi=600,bbox_inches='tight');fig.savefig(OUT/'Stage4_Grouped_Paired_R2_Gains.pdf',bbox_inches='tight');plt.close(fig)

# Ref diagnostic: horizontal line across refs for 20% Stage4 grouped, five models spaced around x.
fig,ax=plt.subplots(figsize=(9.6,5.5))
refs=[7,14,16,21,22]
shift=np.linspace(-.24,.24,len(FILES))
for i,(model,*_) in enumerate(FILES):
  values=[next(r for r in pub20 if r['Model']==model and r['Ref_ID']==ref)['Delta_RMSE_median'] for ref in refs]
  ax.scatter(np.arange(5)+shift[i],values,s=62,label=model,zorder=3)
ax.axhline(0,color='gray',ls='--',lw=1)
ax.set_xticks(np.arange(5),[f'Ref {r}' for r in refs]);ax.set_xlabel('Held-out publication')
ax.set_ylabel(r'Median paired $\Delta$RMSE (calibrated $-$ zero-shot)')
ax.set_title('Stage 4 | 20% calibration | Publication-level heterogeneity')
ax.grid(axis='y',alpha=.22);ax.legend(frameon=False,ncol=3,fontsize=9)
fig.tight_layout();fig.savefig(OUT/'Stage4_Grouped_PerPublication_RMSE_20pct_600dpi.png',dpi=600,bbox_inches='tight');fig.savefig(OUT/'Stage4_Grouped_PerPublication_RMSE_20pct.pdf',bbox_inches='tight');plt.close(fig)

with open(OUT/'Reproduction_and_Interpretation.json','w',encoding='utf-8') as f:
 json.dump({'input_zip_SHA256':hashes,'total_model_stage_protocol_budget_configs':len(summ),'total_random_replicates':len(rep),'per_publication_summaries':len(perpub),'counts_by_budget':allbudget,'stage4_grouped_20pct_papers_improved_median_RMSE':countpos,'stage4_grouped_20pct_total_paper_model_comparisons':len(pub20),'notes':['Each replicate uses matched non-calibration rows for zero-shot and calibration.','Replicates share target publications and random selections across models; they are not independent publications.','The former held-out target publications are now used for retrospective calibration; not blinded external performance.','This analyses frozen predictions; no models were refit or tuned.']},f,indent=2)
print('Build successful', len(clean),len(cleanpub),len(rep))
print('Stage 4 publication improve:',countpos,'/',len(pub20))
for r in allbudget:print(r)
