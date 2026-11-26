import os,json,hashlib,sys
from docx import Document
from artifact_tool import Workbook,SpreadsheetFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

D='/mnt/data/'
name=['Elastic Net','SVR-RBF','Random Forest','XGBoost','LightGBM','CatBoost']
stages=['S1','S2','S3']
results={(s,m):json.load(open(D+f'line1_{s}_{m.replace(" ","_")}.json')) for m in name for s in stages}
T=Document(D+'SI_20261016.docx').tables[0]
archive={}
for r in T.rows[1:]:
 cells=[c.text.strip() for c in r.cells]
 st='S'+cells[0].split(' ')[1]
 model=cells[0].split('–',1)[1].strip()
 archive[st,model]={'n_features':int(cells[1]),'cv_r2':float(cells[2]),'cv_rmse':float(cells[3]),'cv_mae':float(cells[4]),'test_r2':float(cells[5]),'test_rmse':float(cells[6]),'test_mae':float(cells[7])}

# main figure: original Figure 2a is a 6x3 dumbbell comparison
fig,ax=plt.subplots(figsize=(9.5,9.3))
colors={'cv':'#26867d','test':'#d87655'}
markers={'cv':'o','test':'s'}
Y=[]
for i,m in enumerate(name):
 for k,s in enumerate(stages):
  y=(len(name)-1-i)*3.8+(2-k)*0.92
  v=results[s,m]
  cv=v['cv']['r2'];te=v['test']['r2']
  ax.plot([cv,te],[y,y],color='#a0a8ae',linewidth=2.0,zorder=1)
  ax.scatter([cv],[y],s=40,marker='o',color=colors['cv'],zorder=3)
  ax.scatter([te],[y],s=40,marker='s',color=colors['test'],zorder=3)
  ax.text(.548,y,'S'+str(k+1),ha='right',va='center',fontsize=9.1,color='#555f68')
  # mark model group heading above
  if k==0:
   ax.text(.548,y+.68,m,ha='right',va='bottom',fontsize=12,fontweight='bold',color='#1f2937')
 if i<len(name)-1:
  ax.axhline(y-.65,color='#edf0f2',lw=1.2)
ax.set_xlim(.535,.955)
ax.set_ylim(-1.5,23.0)
ax.set_yticks([])
ax.set_xticks(np.arange(.55,.96,.05))
ax.grid(axis='x',color='#e5e7eb',linewidth=.8)
ax.set_axisbelow(True)
ax.spines[['left','top','right']].set_visible(False)
ax.set_xlabel(r'$R^2$ (ln(1 + DBP concentration) scale)',fontsize=12,labelpad=12)
ax.set_title('Figure 2a — Independently rerun Line 1 models',loc='left',weight='bold',fontsize=15,pad=19)
ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=colors['cv'],label='5-fold pooled OOF CV',markersize=7),Line2D([0],[0],marker='s',ls='',color=colors['test'],label='Random-split Test',markersize=7)],loc='lower right',bbox_to_anchor=(1.0,.985),ncol=2,frameon=False,fontsize=10)
fig.text(.11,.02,'995 Training / 249 Test observations; original 22, augmented 55, and RFE-reduced predictors.',fontsize=9,color='#52525b')
fig.subplots_adjust(left=.27,right=.97,top=.91,bottom=.09)
fig.savefig(D+'Figure_2a_Line1_Reproduced.png',dpi=320,facecolor='white',bbox_inches='tight')
fig.savefig(D+'Figure_2a_Line1_Reproduced.svg',facecolor='white',bbox_inches='tight')
plt.close(fig)

wb=Workbook.create()
sh=wb.worksheets.add('Reproduction Summary')
head=['Stage','Algorithm','Features','CV R2 (SI)','CV R2 (rerun)','Delta CV R2','Test R2 (SI)','Test R2 (rerun)','Delta Test R2','CV RMSE (rerun)','CV MAE (rerun)','Test RMSE (rerun)','Test MAE (rerun)','4-decimal agreement']
rows=[head]
for s in stages:
 for m in name:
  a=archive[s,m]; z=results[s,m];cv=z['cv']['r2'];te=z['test']['r2']
  match=(round(cv,4)==a['cv_r2'] and round(te,4)==a['test_r2'])
  rows.append([s,m,z['n_features'],a['cv_r2'],cv,cv-a['cv_r2'],a['test_r2'],te,te-a['test_r2'],z['cv']['rmse'],z['cv']['mae'],z['test']['rmse'],z['test']['mae'],'Match' if match else 'Review'])
sh.get_range_by_indexes(0,0,len(rows),len(head)).values=rows
sh.get_range('A1:N1').format={'fill':'#0D334A','font':{'bold':True,'color':'#FFFFFF'},'row_height':26,'wrap_text':True,'vertical_alignment':'center'}
sh.get_range('A2:N19').format.row_height=22
sh.get_range('D2:M19').set_number_format('0.000000')
sh.get_range('B1:B19').format.column_width=18
sh.get_range('A1:A19').format.column_width=11
sh.get_range('C1:C19').format.column_width=12
sh.get_range('D1:M19').format.column_width=17
sh.get_range('N1:N19').format.column_width=24
for i,s in enumerate(stages):
 start=2+i*6
 if i%2==1:sh.get_range(f'A{start}:N{start+5}').format.fill='#F1F5F9'
for i,row in enumerate(rows[1:],start=2):
 if row[-1]=='Review': sh.get_range(f'N{i}').format={'fill':'#FFF0E6','font':{'bold':True,'color':'#9A3412'}}
 else:sh.get_range(f'N{i}').format={'font':{'color':'#176F61'}}
sh.freeze_panes.freeze_rows(1)
sh.tables.add('A1:N19',True,'Line1Reproduced')

sf=wb.worksheets.add('Five-fold CV detail')
heads=['Stage','Model','Fold 1 R2','Fold 2 R2','Fold 3 R2','Fold 4 R2','Fold 5 R2','Mean fold R2','Pooled OOF R2']
rws=[heads]
for s in stages:
 for m in name:
  z=results[s,m];fs=[d['r2'] for d in z['folds']]
  rws.append([s,m,*fs,float(np.mean(fs)),z['cv']['r2']])
sf.get_range_by_indexes(0,0,len(rws),9).values=rws
sf.get_range('A1:I1').format={'fill':'#0D334A','font':{'bold':True,'color':'#FFFFFF'},'row_height':24,'wrap_text':True}
sf.get_range('B1:B19').format.column_width=19
sf.get_range('A1:A19').format.column_width=10
sf.get_range('C1:I19').format.column_width=17
sf.get_range('C2:I19').set_number_format('0.000000')
sf.freeze_panes.freeze_rows(1)

feat=wb.worksheets.add('Stage 3 RFE subsets')
feat.get_range('A1:D1').values=[['Algorithm','Original feature rank','Predictor','Number selected']]
feat.get_range('A1:D1').format={'fill':'#0D334A','font':{'bold':True,'color':'#FFFFFF'}}
sub=[['Algorithm','Original feature rank','Predictor','Number selected']]
for m in name:
 z=results['S3',m]
 for i,n in enumerate(z['features'],1):sub.append([m,i,n,z['n_features']])
feat.get_range_by_indexes(0,0,len(sub),4).values=sub
feat.get_range('A1:A250').format.column_width=19
feat.get_range('B1:B250').format.column_width=22
feat.get_range('C1:C250').format.column_width=51
feat.get_range('D1:D250').format.column_width=20
feat.freeze_panes.freeze_rows(1)

notes=wb.worksheets.add('Audit notes')
allfiles={'Source 22 predictors':'Microplastics_DBPs_Combined_Original22 features(5).xlsx','S1 Training':'Conventional_RandomSplit_Stage1_Training_22features(3).xlsx','S1 Test':'Conventional_RandomSplit_Stage1_Test_22features(3).xlsx','S2/S3 Training':'Conventional_RandomSplit_Stage2_Training_55features(3).xlsx','S2/S3 Test':'Conventional_RandomSplit_Stage2_Test_55features(3).xlsx'}
oldhash={'Source 22 predictors':'15eb91ec82798b4fd553efa441e8c3e11f3fca4b7d7ccfa2ea159e713daebd74','S1 Training':'d9796d67314eef25f761eccf46a613b89391ec473c290b7ce7a522b62f23671b','S1 Test':'20d0cb03f75c506f257fcefd9a0624993b46ad487611a127ac5b2d7378ee02e4','S2/S3 Training':'32e9ec4884aca2c613ba03dc8088935bd5257a0c1d2f0c3fb4dd34879f9070ee','S2/S3 Test':'f738ff2d6c73857d5652d5d64e600d136faac0a5b33f1e4d666d40479d15d2e9'}
hashes=[['Input','Filename','SHA256 equals SI lock?','SHA256']]
for label,f in allfiles.items():
 h=hashlib.sha256(open(D+f,'rb').read()).hexdigest(); hashes.append([label,f,'Yes' if h==oldhash[label] else 'NO',h])
notes.get_range('A1:D6').values=hashes
notes.get_range('A1:D1').format={'fill':'#0D334A','font':{'bold':True,'color':'#FFFFFF'}}
notes.get_range('A1:A6').format.column_width=24
notes.get_range('B1:B6').format.column_width=62
notes.get_range('C1:C6').format.column_width=25
notes.get_range('D1:D6').format.column_width=75
comments=[
'Protocol: Frozen final parameter and feature subset refits, not independent repeat of the 20-trial Optuna searches or the RFE elimination paths.',
'Cross validation: KFold(5, shuffle=True, random_state=42), pooled 995 OOF predictions. Test: final model trained on 995 and evaluated on 249.',
'Target: y = ln(1 + DBP concentration [microgram per liter]). Ref ID is excluded from predictors.',
'Preprocessing: fold-fitted numerical median; explicit __MISSING__ categorical one-hot; ElasticNet/SVR numerical z-scoring only; native CatBoost missing support.',
'Note discrepancies: S2 and S3 Random Forest cannot reproduce all archived values with the explicitly recorded parameter/encoding recipe; alternative dense/mode/scaled implementations give other results but were not substituted into this report.',
'S2/S3 SVR Test R2 reproduced 0.8101717 (rounds to 0.8102), whereas archived SI Table S1 reports 0.8101. Difference in full precision is approximately 7.2e-5.',
'All other pairs match CV/Test R2 to the displayed four decimals. Package versions: Python 3.13.5; NumPy 2.3.5; SciPy 1.17.0; sklearn 1.8.0; XGBoost 3.1.3; LightGBM 4.6.0; CatBoost 1.2.8.',
'The original source SI_20261016.docx documents the frozen feature subsets and all hyperparameters (Text S2, Table S1).'
]
notes.get_range('A9:B9').values=[['Audit and interpretation','Details']]
notes.get_range('A9:B9').format={'fill':'#0D334A','font':{'bold':True,'color':'#FFFFFF'}}
notes.get_range_by_indexes(9,0,len(comments),2).values=[[i+1,x] for i,x in enumerate(comments)]
notes.get_range('A10:A17').format.column_width=24
notes.get_range('B10:B17').format.column_width=112
notes.get_range('B10:B17').format.wrap_text=True
notes.get_range('A10:B17').format.row_height=42

out=D+'Figure2a_Line1_Model_Reproduction_Audit.xlsx'
SpreadsheetFile.export_xlsx(wb).save(out)
print('WROTE',out)
print('Figure',D+'Figure_2a_Line1_Reproduced.png')
for s in stages:
 for m in name:
  z=results[s,m];a=archive[s,m]
  print(s,m,'CV',round(z['cv']['r2'],6),'SI',a['cv_r2'],'Test',round(z['test']['r2'],6),'SI',a['test_r2'])
