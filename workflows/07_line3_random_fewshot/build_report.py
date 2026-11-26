"""Make polished spreadsheet and reproducible 600-dpi figures from CSV audit only.
Uses artifact_tool for spreadsheet writing; no pandas/openpyxl.
"""
from pathlib import Path
import csv,collections,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_tool import Workbook,SpreadsheetFile
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';LINE2=ROOT/'input'
def read(p):
 with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
S=read(OUT/'Line3_RandomFewShot_Summary_160.csv')
R=read(OUT/'Line3_RandomFewShot_PerPublication_Summary_1120.csv')
N=read(OUT/'Line3_ExcludeSharedRef14_Sensitivity_160.csv')
B=read(OUT/'Per_Publication_Budgets.csv')
L2=read(LINE2/'Line2_All160_Model_Stage_Budget_Summary.csv')
models=['RF','XGBoost','LightGBM','CatBoost','Equal4']
model2={'RF':'Random Forest','XGBoost':'XGBoost','LightGBM':'LightGBM','CatBoost':'Native CatBoost','Equal4':'Equal4'}
def num(v):
 try:return float(v)
 except:return v

def select(rows,**conds):
 return [r for r in rows if all((str(r[k])==str(v) if k not in ('Budget_pct',) else math.isclose(float(r[k]),float(v),abs_tol=1e-7)) for k,v in conds.items())]
rows_budget=[]
for scheme in ['Publication-grouped','Conventional random']:
 for b in [2.5,5,10,20]:
  x=select(S,Scheme=scheme,Budget_pct=b)
  assert len(x)==20
  rows_budget.append({'Scheme':scheme,'Budget_pct':b,'n_cal':int(float(x[0]['N_cal'])),'n_eval':int(float(x[0]['N_eval'])),
   'N_positive_median_Delta_R2':sum(float(r['Delta_R2_median'])>0 for r in x),
   'N_80pct_success':sum(float(r['Fraction_improved_R2'])>=0.8 for r in x),
   'Median_of_20_Delta_R2_medians':float(np.median([float(r['Delta_R2_median']) for r in x]))})
stage4=select(S,Scheme='Publication-grouped',Stage=4)
stage4.sort(key=lambda r:(models.index(r['Model']),float(r['Budget_pct'])))
compare=[]
for m in models:
 for b in [2.5,5,10,20]:
  a=select(S,Scheme='Publication-grouped',Stage=4,Model=m,Budget_pct=b)[0]
  c=select(L2,Protocol='Grouped',Stage=4,Model=model2[m],Budget_pct=b)[0]
  d=select(N,Scheme='Publication-grouped',Stage=4,Model=m,Budget_pct=b)[0]
  compare.append({'Model':m,'Budget_pct':b,
   'Line2_Zero_full_R2':float(c['Zero_full_R2']),'Line2_FewShot_R2_median':float(c['Cal_R2_median']),'Line2_Paired_Delta_R2_median':float(c['Delta_R2_median']),
   'Line3_Zero_full_R2':float(a['Zero_full_R2']),'Line3_FewShot_R2_median':float(a['Cal_R2_median']),'Line3_Paired_Delta_R2_median':float(a['Delta_R2_median']),
   'Line3_ExclRef14_FewShot_R2_median':float(d['Cal_R2_median']),'Line3_ExclRef14_Paired_Delta_R2_median':float(d['Paired_Delta_R2_median'])})

def writecsv(fn,rows):
 with open(OUT/fn,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('Budget_Comparison_8.csv',rows_budget)
writecsv('Line2_Line3_Stage4_Comparison_20.csv',compare)

# Figures
x=np.array([2.5,5,10,20]);fig,ax=plt.subplots(figsize=(9.3,5.1),layout='constrained')
for m in models:
 seq=[select(S,Scheme='Publication-grouped',Stage=4,Model=m,Budget_pct=b)[0] for b in x]
 med=np.array([float(r['Delta_R2_median']) for r in seq]);lo=np.array([float(r['Delta_R2_q05']) for r in seq]);hi=np.array([float(r['Delta_R2_q95']) for r in seq]);
 line,=ax.plot(x,med,marker='o',lw=2.1,label=m)
 ax.fill_between(x,lo,hi,color=line.get_color(),alpha=.10)
ax.axhline(0,color='#666666',lw=.8,ls='--');ax.set_xticks(x,['2.5%\n(n=10)','5%\n(n=14)','10%\n(n=25)','20%\n(n=47)']);ax.set_ylabel(r'Median paired $\Delta R^2$');ax.set_xlabel('Calibration budget (all 7 publications)');ax.set_title('Line 3  |  Stage 4 publication-grouped models',fontweight='bold',loc='left');ax.legend(ncol=3,frameon=False,loc='upper left');ax.grid(axis='y',lw=.6,alpha=.23);ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'Line3_Stage4_Grouped_Paired_R2_Gains_600dpi.png',dpi=600,bbox_inches='tight')
fig.savefig(OUT/'Line3_Stage4_Grouped_Paired_R2_Gains.pdf',bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12.1,4.8),layout='constrained',sharey=False)
for ia,(label,dat,extractor) in enumerate([('Line 2 | 5 Test publications',L2,'line2'),('Line 3 | 7 Test publications',S,'line3')]):
 ax=axs[ia]
 for m in models:
  raw=[select(dat,**({'Protocol':'Grouped','Stage':4,'Model':model2[m],'Budget_pct':b} if extractor=='line2' else {'Scheme':'Publication-grouped','Stage':4,'Model':m,'Budget_pct':b}))[0] for b in x]
  y=[float(r['Cal_R2_median']) for r in raw]
  ax.plot(x,y,marker='o',lw=1.8,label=m)
 ax.set_title(label,loc='left',fontweight='bold');ax.set_xlabel('Calibration budget');ax.set_xticks(x,[f'{b:g}%' for b in x]);ax.set_ylabel('Few-Shot pooled test R² (median)');ax.grid(axis='y',alpha=.24);ax.spines[['top','right']].set_visible(False)
axs[1].legend(loc='upper left',frameon=False,ncol=2)
fig.savefig(OUT/'Line2_vs_Line3_Stage4_R2_600dpi.png',dpi=600,bbox_inches='tight')
fig.savefig(OUT/'Line2_vs_Line3_Stage4_R2.pdf',bbox_inches='tight');plt.close(fig)

refid=[4,11,12,13,14,20,26]
heat=np.array([[float(select(R,Scheme='Publication-grouped',Stage=4,Model=m,Budget_pct=20,Ref_ID=str(r))[0]['Delta_RMSE_median']) for r in refid] for m in models])
fig,ax=plt.subplots(figsize=(9.5,4.3),layout='constrained');vmax=max(abs(heat.min()),abs(heat.max()));im=ax.imshow(heat,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
ax.set_xticks(np.arange(7),[f'Ref {r}' for r in refid]);ax.set_yticks(np.arange(5),models)
for i in range(5):
 for j in range(7):ax.text(j,i,f'{heat[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(heat[i,j])>.65*vmax else '#1c2937')
fig.colorbar(im,ax=ax,label='Median paired ΔRMSE  (negative = improvement)',shrink=.8)
ax.set_title('Line 3  |  Stage 4 grouped  |  20% calibration',loc='left',fontweight='bold')
fig.savefig(OUT/'Line3_Stage4_PerPublication_RMSE_20pct_600dpi.png',dpi=600,bbox_inches='tight');fig.savefig(OUT/'Line3_Stage4_PerPublication_RMSE_20pct.pdf',bbox_inches='tight');plt.close(fig)

# Spreadsheet
wb=Workbook.create()
def style_sheet(name,rows,cols,widths=None):
 sh=wb.worksheets.add(name)
 data=[[num(r.get(c)) if c not in ('Scheme','Model','Protocol','Ref_excluded') else r.get(c) for c in cols] for r in rows]
 sh.get_range_by_indexes(0,0,len(data)+1,len(cols)).values=[cols]+data
 sh.get_range_by_indexes(0,0,1,len(cols)).format={'fill':'#173651','font':{'bold':True,'color':'#FFFFFF'},'row_height':40,'wrap_text':True,'vertical_alignment':'center'}
 sh.get_range_by_indexes(1,0,len(data),len(cols)).format.row_height=20
 for j,c in enumerate(cols):
  letter='';q=j+1
  while q:q,k=divmod(q-1,26);letter=chr(65+k)+letter
  sh.get_range(f'{letter}:{letter}').format.column_width=(widths[c] if widths and c in widths else (23 if c in ('Model','Scheme') else 19))
  if c.startswith('Fraction_'):sh.get_range_by_indexes(1,j,len(data),1).set_number_format('0.0%')
  elif c not in ('Model','Scheme','Protocol','Ref_excluded') and c not in ('Stage','Ref_ID','N_cal','N_eval','n_cal','n_eval','N_positive_median_Delta_R2','N_80pct_success','N_ref_evaluated') and c!='Budget_pct':sh.get_range_by_indexes(1,j,len(data),1).set_number_format('0.0000')
 sh.freeze_panes.freeze_rows(1)
 return sh
intro=wb.worksheets.add('Read Me');intro.merge_cells('A1:F2');intro.get_range('A1').values=[['LINE 3 | FIVE-MODEL RANDOM FEW-SHOT CALIBRATION']]
intro.get_range('A1:F2').format={'fill':'#173651','font':{'bold':True,'color':'#FFFFFF','size':15},'row_height':23,'vertical_alignment':'center'}
notes=[
 ('Analysis','5 models x 4 stages x 2 CV schemes x 4 budgets x 500 selections = 80,000 matched calibration comparisons.'),
 ('Data','218 Test observations from Refs 4, 11, 12, 13, 14, 20, 26; frozen Fig 2c predictions.'),
 ('Calibration','For each held-out publication, additive offset = mean(y_observed - y_predicted) on calibration rows; labels used only for that offset.'),
 ('Budgets','2.5%, 5%, 10%, 20% yield 10, 14, 25, 47 observed labels, and 208, 204, 193, 171 held-out evaluations.'),
 ('Random selection','One numpy default_rng(42) stream; 500 ref-specific permutations; same permutations across model/stage/scheme.'),
 ('Paired improvement','Each replicate compares Zero-Shot and Few-Shot on identical noncalibration observations. Positive Delta_R2 means better.'),
 ('Quantiles','5th–95th percentiles across random draws measure acquisition variability; not confidence intervals across new papers.'),
 ('Shared source','Ref 14 (72 observations) occurs in BOTH Line 2 and Line 3 Test sets. See Exclude Ref14 sheet.'),
 ('Tiny studies','Ref 13 has only 4 rows; n_cal=1 for all budgets. Rows may share experimental settings across DBPs.'),
 ('Interpretation','This is retrospective, label-assisted adjustment within new publications; cannot be called untouched zero-shot validation.'),
 ('Auditing','The 40 row-level zero-shot predictions match Line3_80_Metric_Audit.csv to <5e-12 for all Test R2/RMSE/MAE.'),
 ('Other files','CSV: 80,000 replicate-level rows, 560,000 per-publication rows, random selections; code and hashes in ZIP.')]
intro.get_range('A4:B15').values=[[a,b] for a,b in notes]
intro.get_range('A4:A15').format={'fill':'#E7EFF5','font':{'bold':True,'color':'#173651'},'wrap_text':True}
intro.get_range('B4:B15').format.wrap_text=True;intro.get_range('A:A').format.column_width=25;intro.get_range('B:B').format.column_width=107;intro.get_range('A4:B15').format.row_height=35
style_sheet('Budget Summary',rows_budget,list(rows_budget[0]))
cols4=['Scheme','Stage','Model','Budget_pct','N_cal','N_eval','Zero_full_R2','Zero_R2_median','Cal_R2_median','Delta_R2_median','Delta_R2_q05','Delta_R2_q95','Fraction_improved_R2','Zero_RMSE_median','Cal_RMSE_median','Delta_RMSE_median','Cal_MAE_median','Delta_MAE_median']
style_sheet('Stage4 Grouped',stage4,cols4)
style_sheet('Line2 vs Line3',compare,list(compare[0]))
cols160=cols4[:]+['Fraction_improved_RMSE','Fraction_improved_MAE']
style_sheet('All 160',S,cols160)
r_cols=['Scheme','Stage','Model','Budget_pct','Ref_ID','N_cal','N_eval','Zero_RMSE_median','Cal_RMSE_median','Delta_RMSE_median','Fraction_improved_RMSE','Zero_R2_median','Cal_R2_median','Delta_R2_median','Fraction_improved_R2','Zero_MAE_median','Cal_MAE_median']
style_sheet('PerRef 1120',R,r_cols)
style_sheet('Excl Shared Ref14',N,list(N[0]))
style_sheet('Calibration Counts',B,list(B[0]))
fn=OUT/'Line3_FiveModel_RandomFewShot_Comparison.xlsx'
SpreadsheetFile.export_xlsx(wb).save(str(fn))
print('Created',fn,fn.stat().st_size)
print('Sheets', [wb.worksheets.get_item_at(i).name for i in range(8)])
print('Headline',wb.inspect({'kind':'table','range':'Budget Summary!A1:G9','include':'values,formulas','table_max_rows':9,'table_max_cols':7}).ndjson[:1200])
