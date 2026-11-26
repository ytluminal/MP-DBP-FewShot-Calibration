from pathlib import Path
import csv,json,zipfile,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_tool import Workbook,SpreadsheetFile
P=Path('/mnt/data/Line3_StageMatched_Active_Rerun_20261016'); R=P/'results';W=P/'work'

def read_csv(p):
 with open(p,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def typed(v):
 if v is None or v=='':return None
 if str(v).lower() in ('nan','inf','-inf'):return None
 try:
  s=str(v)
  if s.isdigit() or (s[0]=='-' and s[1:].isdigit()):return int(s)
  return float(s)
 except: return str(v)

new=read_csv(R/'All_160_Configurations_Summary.csv')
old=read_csv(W/'old_results/results/Line3_ActiveVsRandom_Matched_Summary.csv')
oldm={(r['Protocol'],int(r['Stage']),r['Model'],float(r['Budget_pct'])):r for r in old}
D=[]
for v in new:
 key=(v['Protocol'],int(v['Stage']),v['Model'],float(v['Budget_pct']));pre=oldm[key]
 r={'Protocol':key[0],'Stage':key[1],'Model':key[2],'Budget_pct':key[3],
    'N_features_old':55,'N_features_stage_matched':int(v['Features_raw']),
    'Old_Active_R2_matched':float(pre['common_active_R2_median']),
    'New_Active_R2_matched':float(v['Active_R2_median']),
    'Change_in_Active_R2_median':float(v['Active_R2_median'])-float(pre['common_active_R2_median']),
    'Old_Paired_dR2_median':float(pre['active_minus_random_R2_median']),
    'New_Paired_dR2_median':float(v['Paired_dR2_median']),
    'Old_Frac_active_better':float(pre['fraction_active_better_R2']),
    'New_Frac_active_better':float(v['Fraction_Active_Better_R2']),
    'Interpretation':'Matched sets differ between old/new when selection changes; changes between medians are descriptive, not identical-row paired effects.'}
 D.append(r)
with (R/'Old_vs_New_160_Descriptive_Comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(D[0]));w.writeheader();w.writerows(D)

print('OLD_vs_NEW S1/S4:')
for st in [1,4]:
 for typ in ['Publication-grouped','Random folds']:
  vals=[x for x in D if x['Stage']==st and x['Model']=='LightGBM' and x['Protocol']==typ]
  print(typ,st,[(v['Budget_pct'],round(v['Old_Active_R2_matched'],4),round(v['New_Active_R2_matched'],4)) for v in vals])

# Render polished diagnostic plot for requested Stage-Matched vs Original comparison
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'savefig.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(11.0,7.4),sharex=True,sharey=True)
for j,protocol in enumerate(['Publication-grouped','Random folds']):
 for i,stage in enumerate([1,4]):
  ax=axs[i,j]
  x=np.asarray([2.5,5,10,20])
  z=sorted([d for d in D if d['Stage']==stage and d['Model']=='LightGBM' and d['Protocol']==protocol],key=lambda d:d['Budget_pct'])
  ax.plot(x,[r['Old_Active_R2_matched'] for r in z],marker='o',lw=1.8,label='Original (55 features)',color='#8799a5')
  ax.plot(x,[r['New_Active_R2_matched'] for r in z],marker='s',lw=2,label='Stage-matched',color='#175f8d')
  ax.axhline(0.6,ls='--',lw=0.8,color='#cc7957',alpha=.85)
  ax.set_title(f'S{stage}  |  {protocol}',pad=10,fontweight='semibold')
  ax.set_xticks(x);ax.set_xticklabels(['2.5','5','10','20']);ax.set_ylim(-.03,.75)
  ax.grid(axis='y',alpha=.17);ax.spines[['top','right']].set_visible(False)
  if j==0:ax.set_ylabel('Median Test $R^2$ (matched sets)')
  if i==1:ax.set_xlabel('Calibration budget (%)')
fig.suptitle('Line 3 LightGBM  |  Original vs Stage-matched Centroid + Maximin',fontweight='bold',fontsize=14,y=.99)
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.065),frameon=False)
fig.text(.5,.045,'Old/new methods evaluate different common-row sets; differences are descriptive, not paired gains.',ha='center',fontsize=8,color='#5c6670')
fig.tight_layout(rect=(0,.17,1,.95),h_pad=2.1,w_pad=1.8)
fig.savefig(R/'LightGBM_S1_S4_Old_vs_StageMatched_600dpi.png',dpi=600)
fig.savefig(R/'LightGBM_S1_S4_Old_vs_StageMatched.svg')
plt.close(fig)

# Build workbook via artifact_tool
w=Workbook.create()
NAVY='#14293E';MID='#275E83'; BG='#EDF3F7';LIGHT='#EFF6FA'

def add_sheet(name, headers, data, widths=None):
 sh=w.worksheets.add(name)
 data=[list(row) for row in data]
 if headers:
  sh.get_range_by_indexes(0,0,1,len(headers)).values=[list(headers)]
  if data:sh.get_range_by_indexes(1,0,len(data),len(headers)).values=[[typed(v) for v in row] for row in data]
  h=sh.get_range_by_indexes(0,0,1,len(headers))
  h.format={'fill':NAVY,'font':{'color':'#FFFFFF','bold':True},'row_height':27,'wrap_text':True,'vertical_alignment':'center'}
  sh.freeze_panes.freeze_rows(1)
  for k in range(len(headers)):
   col=sh.get_range_by_indexes(0,k,max(1,len(data)+1),1)
   wd=(widths[k] if widths and k<len(widths) else min(32,max(13,len(headers[k])*0.9)))
   col.format.column_width=wd
  if data:
   sh.get_range_by_indexes(1,0,len(data),len(headers)).format.row_height=20
 return sh

# Summary and instructions: keep one crisp landing sheet
intro=[
 ['Project','Microplastics–DBPs | Line 3 | Stage-matched Active Few-Shot'],
 ['Scope','5 models × 4 stages × 2 CV protocols × 4 budgets = 160 configurations'],
 ['Test observations','218 across 7 held-out publications; Training observations = 1026'],
 ['Algorithm','Centroid initialization + greedy Maximin diversity; mean publication-specific residual offset'],
 ['Change vs historical','Only active sample selection features are stage/model matched. Old random permutations, frozen predictions, data split unchanged.'],
 ['S1','22 model features for RF, XGBoost, LightGBM, CatBoost, Equal4'],
 ['S2 / S3','55 features for all five models; outputs reproduce original Active results exactly'],
 ['S4','RF=14, XGBoost=14, LightGBM=14, Preprocessed CatBoost=28, Equal4=32 (union of four feature sets)'],
 ['Pairing rule','Within each random replicate, evaluate only rows not selected by either Active or Random.'],
 ['Calibration','Mean y_ln1p minus zero-shot prediction on sampled rows, separately for each test publication'],
 ['Replications','500 frozen random permutations × 160 configurations = 80,000 paired runs'],
 ['Outcome scale','ln(1 + DBP concentration in µg/L)'],
 ['Interpretation','Median of replicate-wise paired R² differences differs from difference of R² medians.'],
 ['Reproducibility','S2/S3 80 configuration matched medians reproduce historical values within 1e-12.'],
 ['Prior vs new','Old/New R² change not a paired effect because Active-chosen and common-eval rows differ.'],
]
sh=add_sheet('START_HERE',['Item','Description'],intro,[25,116])
sh.get_range('B1:B16').format.wrap_text=True
sh.get_range('A1:B16').format.row_height=27

cols=['Protocol','Stage','Model','Budget_pct','Features_raw','Zero_Full_R2','N_common_eval_median','Zero_R2_median','Random_R2_median','Active_R2_median','Paired_dR2_median','Paired_dR2_q05','Paired_dR2_q95','Fraction_Active_Better_R2','Random_RMSE_median','Active_RMSE_median','Paired_dRMSE_median','Random_MAE_median','Active_MAE_median','Paired_dMAE_median','Active_Own_R2','Active_Own_RMSE','Active_Own_MAE']
sh=add_sheet('All160_Paired',cols,[[r.get(c) for c in cols] for r in new])
for i,c in enumerate(cols):
 if c.endswith('R2') or 'R2_' in c or 'RMSE' in c or 'MAE' in c:
  sh.get_range_by_indexes(1,i,len(new),1).set_number_format('0.0000')
 if c=='Fraction_Active_Better_R2':sh.get_range_by_indexes(1,i,len(new),1).set_number_format('0.0%')
sh.get_range_by_indexes(1,cols.index('Active_R2_median'),len(new),1).conditional_formats.add_color_scale({'minColor':'#F7EBE8','midColor':'#E0EDF2','maxColor':'#94C4DA'})

cols=list(D[0]);sh=add_sheet('Old_vs_New_160',cols,[[r.get(c) for c in cols] for r in D],[22,12,18,13,14,15,19,19,20,19,19,18,20,20,60])
sh.get_range_by_indexes(1,cols.index('Change_in_Active_R2_median'),len(D),1).conditional_formats.add_color_scale({'minColor':'#EEC6BF','midColor':'#FFFFFF','maxColor':'#BADDD3'})

own=read_csv(R/'All_160_Active_OwnRemainder.csv');cols=list(own[0]);add_sheet('Active_Own160',cols,[[r.get(c) for c in cols] for r in own])
per=read_csv(R/'All_1120_PerPublication_Summary.csv')
cols=['Protocol','Stage','Model','Budget_pct','Ref_ID','N_common_eval_median','Active_R2_median','Random_R2_median','Paired_dR2_median','Active_RMSE_median','Random_RMSE_median','Paired_dRMSE_median','Active_MAE_median','Random_MAE_median','Paired_dMAE_median','Fraction_Active_Better_R2','N_replicates']
add_sheet('PerPublication1120',cols,[[r.get(c) for c in cols] for r in per])
feature=read_csv(R/'Stage_Model_Feature_Manifest.csv');cols=list(feature[0]);add_sheet('Stage_Features',cols,[[r.get(c) for c in cols] for r in feature],[13,18,14,120])

# Concise best results / counts table
short=[]
for b in [2.5,5.,10.,20.]:
 v=[r for r in new if float(r['Budget_pct'])==b]
 best=max(v,key=lambda r:float(r['Active_R2_median']))
 short.append([b,sum(float(r['Active_R2_median'])>=.60 for r in v),sum(float(r['Paired_dR2_median'])>0 for r in v),best['Protocol'],best['Stage'],best['Model'],float(best['Active_R2_median'])])
add_sheet('Budget_Overview',['Budget_pct','N_Active_R2_ge_0.60','N_Paired_dR2_positive','Best_protocol','Best_stage','Best_model','Best_Active_R2'],short,[15,22,22,24,16,17,18])

# compact check and export
out=P/'Line3_StageMatched_Active_AllModels_Results_20261016.xlsx'
SpreadsheetFile.export_xlsx(w).save(str(out))
print('EXPORTED',out,out.stat().st_size)
try:
 q=w.inspect({'kind':'table','range':'Budget_Overview!A1:G5','include':'values','table_max_rows':5,'table_max_cols':7})
 print('VERIFIED overview:',q.ndjson[:1400])
except Exception as e: print('Inspect note:',str(e)[:300])
# note .xlsx has a real xlsx zip structure
with zipfile.ZipFile(out) as z: assert z.testzip() is None

# save README & pack program+source inputs in audit zip
readme=f'''# Line 3 Stage-matched Centroid + Maximin Active Few-Shot rerun (2026-10-16)

## Decision and scope
- Recomputed all 160 configs: 5 models (RF, XGBoost, LightGBM, Preprocessed CatBoost, Equal4) × 4 stages × 2 model CV protocols × 4 calibration budgets.
- Stage 1 selects using 22 original model predictors; Stages 2 and 3 select using 55; Stage 4 uses RF 14, XGBoost 14, LightGBM 14, Preprocessed CatBoost 28, and Equal4 union of 32.
- Exact same frozen Line 3 model predictions, Train/Test membership, 500 saved Random permutations, budget rounding (ceil with min 1 and max n-1), and per-publication **mean-offset** calibration. No WCA-FL.
- Feature encoding for selection: numeric Training-only median+StandardScaler; categorical Training-only most frequent+OneHotEncoder. No Test DBP labels are used in selection, only calibration labels after selection.

## Results and audits
- 160 configs / 80,000 paired random replicates / 560,000 per-publication paired records.
- Exact feature set and selection sequences are in `results/Stage_Model_Feature_Manifest.csv`, `results/Active_Complete_Sequences.csv`, `results/Active_Budget_Selections.csv`.
- Stage 2 and Stage 3 selections exactly match old 55-feature method and their 80 grouped summary configs agree within 1e-12.
- `Active_R2_median` in All_160 summary means median Test R² on the samples neither Active nor Random acquired for each Random repeat. `Active_Own_R2` is Test R² on Active's own unselected remaining observations. These two should never be conflated.
- `Paired_dR2_median` is the median of (Active R² − Random R²) replicate-wise on identical remaining samples. It is NOT the difference between medians.
- `Old_vs_New_160_Descriptive_Comparison.csv` includes old vs new performance on their corresponding matched evaluation subsets. Different active selections change evaluation subsets, so differences there are **descriptive only**, not a strictly same-row paired effect.

## LightGBM Line 3: median paired common evaluation Test R²
'''
for protocol in ['Publication-grouped','Random folds']:
 for st in [1,4]:
  vv=sorted([r for r in new if r['Model']=='LightGBM' and r['Protocol']==protocol and int(r['Stage'])==st],key=lambda r:float(r['Budget_pct']))
  readme+=f'- {protocol}, S{st}: '+', '.join(f"{r['Budget_pct']}%={float(r['Active_R2_median']):.4f}" for r in vv)+'\n'
readme+='''
## Running the analysis
Extract this ZIP, then run in the `work` folder:
```
python rerun_line3_stage_matched.py
```
Requirements: Python with numpy, scikit-learn. The `stage_matched_feature_lists.json` is extracted exactly from the original Figure 2c SI Stage 4 predictor lists. The original `run_active_fewshot.py` is included as an unchanged protocol reference. No model retraining is involved.
'''
(P/'README_CN.md').write_text(readme,encoding='utf-8')
zipout=P/'Line3_StageMatched_Active_Full_Reproducibility_20261016.zip'
with zipfile.ZipFile(zipout,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for file in sorted(P.rglob('*')):
  if not file.is_file() or file==zipout:continue
  if 'old_results' in file.parts and file.name not in ['Line3_ActiveVsRandom_Matched_Summary.csv','Line3_Active_Budget_Selections.csv']:continue
  if file.name=='_line3_source_si_temp.docx':continue
  z.write(file,file.relative_to(P))
print('ZIP',zipout,zipout.stat().st_size)
with zipfile.ZipFile(zipout) as z:assert z.testzip() is None
print('DONE')
