"""Rebuild Figure 2a and the numerical audit from cached or regenerated 18 model JSON files."""
import json,hashlib
from pathlib import Path
from docx import Document
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font,PatternFill,Alignment
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
DOCS=ROOT/'documents'
FIGS=ROOT/'figures'
FIGS.mkdir(exist_ok=True)
name=['Elastic Net','SVR-RBF','Random Forest','XGBoost','LightGBM','CatBoost']
stages=['S1','S2','S3']
results={(s,m):json.loads((RESULTS/f'line1_{s}_{m.replace(" ","_")}.json').read_text()) for m in name for s in stages}
T=Document(str(DOCS/'SI_20261016.docx')).tables[0]
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
fig.savefig(str(FIGS/'Figure_2a_Line1_Reproduced.png'),dpi=320,facecolor='white',bbox_inches='tight')
fig.savefig(str(FIGS/'Figure_2a_Line1_Reproduced.svg'),facecolor='white',bbox_inches='tight')
plt.close(fig)



summary=[]
folds=[]
subsets=[]
for s in stages:
  for m in name:
    z=results[(s,m)]
    a=archive[(s,m)]
    cv=z['cv']['r2']; test=z['test']['r2']
    match=(round(cv,4)==a['cv_r2'] and round(test,4)==a['test_r2'])
    summary.append({'Stage':s,'Algorithm':m,'Features':z['n_features'],
      'CV R2 (SI)':a['cv_r2'],'CV R2 (rerun)':cv,'Delta CV R2':cv-a['cv_r2'],
      'Test R2 (SI)':a['test_r2'],'Test R2 (rerun)':test,'Delta Test R2':test-a['test_r2'],
      'CV RMSE':z['cv']['rmse'],'CV MAE':z['cv']['mae'],
      'Test RMSE':z['test']['rmse'],'Test MAE':z['test']['mae'],
      '4dp match':'Match' if match else 'Review'})
    folds.append({'Stage':s,'Algorithm':m, **{f'Fold {i+1} R2':f['r2'] for i,f in enumerate(z['folds'])},
      'Mean fold R2':np.mean([f['r2'] for f in z['folds']]), 'Pooled OOF R2':cv})
    if s=='S3':
      for i,f in enumerate(z['features'],1):
        subsets.append({'Algorithm':m,'Feature position':i,'Feature':f,'Number selected':z['n_features']})
files = [ROOT/'data'/p for p in [
  'Microplastics_DBPs_Combined_Original22 features(5).xlsx',
  'Conventional_RandomSplit_Stage1_Training_22features(3).xlsx',
  'Conventional_RandomSplit_Stage1_Test_22features(3).xlsx',
  'Conventional_RandomSplit_Stage2_Training_55features(3).xlsx',
  'Conventional_RandomSplit_Stage2_Test_55features(3).xlsx']]
checks=[{'File':p.name,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
out=RESULTS/'Figure2a_Line1_Model_Reproduction_Audit_regenerated.xlsx'
with pd.ExcelWriter(out,engine='openpyxl') as w:
  pd.DataFrame(summary).to_excel(w,sheet_name='Reproduction Summary',index=False)
  pd.DataFrame(folds).to_excel(w,sheet_name='Five-fold CV detail',index=False)
  pd.DataFrame(subsets).to_excel(w,sheet_name='Stage 3 RFE subsets',index=False)
  pd.DataFrame(checks).to_excel(w,sheet_name='Input SHA256',index=False)
wb=load_workbook(out)
for ws in wb.worksheets:
  ws.freeze_panes='A2'
  ws.auto_filter.ref=ws.dimensions
  for c in ws[1]:
    c.fill=PatternFill('solid',fgColor='133951')
    c.font=Font(color='FFFFFF',bold=True)
    c.alignment=Alignment(wrap_text=True)
  for col in ws.columns:
    col=list(col)
    lengths=[len(str(x.value or '')) for x in col[:100]]
    ws.column_dimensions[col[0].column_letter].width=min(52,max(12,max(lengths)+2))
  for row in ws.iter_rows(min_row=2):
    for cell in row:
      if isinstance(cell.value,float): cell.number_format='0.000000'
wb.save(out)
print('Saved:',out)
print('Saved Figure:',FIGS/'Figure_2a_Line1_Reproduced.png')
