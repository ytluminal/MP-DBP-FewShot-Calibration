from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
rows=list(csv.DictReader(open(OUT/'RF_RandomFewShot_Summary_32.csv',encoding='utf-8')))
per=list(csv.DictReader(open(OUT/'RF_RandomFewShot_PerPublication_Summary_160.csv',encoding='utf-8')))
budgets=np.array([2.5,5,10,20])

for protocol in ('Grouped','RandomCV'):
  fig,ax=plt.subplots(figsize=(7.0,4.8))
  for stage in range(1,5):
    rr=[next(x for x in rows if x['Protocol']==protocol and int(x['Stage'])==stage and float(x['Budget_pct'])==b) for b in budgets]
    med=np.array([float(x['Delta_R2_median']) for x in rr]); l=np.array([float(x['Delta_R2_q05']) for x in rr]); h=np.array([float(x['Delta_R2_q95']) for x in rr])
    line=ax.plot(budgets,med,marker='o',linewidth=2,markersize=5,label=f'Stage {stage}')[0]
    ax.fill_between(budgets,l,h,color=line.get_color(),alpha=.10,linewidth=0)
  ax.axhline(0,linewidth=1,color='#555',linestyle='--')
  ax.set_xticks(budgets,['2.5% (9)','5% (15)','10% (28)','20% (53)'])
  ax.set_xlabel('Calibration budget (% of each publication; total measured rows)')
  ax.set_ylabel(r'Median paired $\Delta R^2$ (few-shot − zero-shot)')
  ax.set_title(f'Random Forest · {"publication-grouped" if protocol=="Grouped" else "conventional random-fold"} training')
  ax.legend(frameon=False,ncol=2)
  ax.grid(axis='y',alpha=.15)
  ax.set_ylim(-.25,.82)
  fig.tight_layout()
  fig.savefig(OUT/f'RF_{protocol}_Paired_R2_Gains_600dpi.png',dpi=600)
  fig.savefig(OUT/f'RF_{protocol}_Paired_R2_Gains.svg')
  plt.close(fig)

# Key caveat visual: Per-publication effects for Stage4 grouped CV, on same remaining rows.
fig,ax=plt.subplots(figsize=(7.2,4.75))
pubs=[7,14,16,21,22]
x=np.arange(len(pubs)); width=.19
for j,b in enumerate(budgets):
 rr=[next(z for z in per if z['Protocol']=='Grouped' and int(z['Stage'])==4 and int(z['Ref_ID'])==r and float(z['Budget_pct'])==b) for r in pubs]
 vals=np.array([float(z['Delta_RMSE_median']) for z in rr])
 ax.bar(x+(j-1.5)*width,vals,width,label=f'{b:g}%')
ax.axhline(0,color='#555',linewidth=1)
ax.set_xticks(x,[f'Ref {r}' for r in pubs]);ax.set_ylabel(r'Median paired $\Delta$RMSE (calibrated − zero-shot)')
ax.set_xlabel('Unseen publication');ax.set_title('Stage 4 RF · Publication-grouped training',pad=54)
ax.legend(title='Budget',frameon=False,ncol=4,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.01))
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT/'RF_Stage4_PerPublication_RMSE_Gains_600dpi.png',dpi=600)
fig.savefig(OUT/'RF_Stage4_PerPublication_RMSE_Gains.svg')
plt.close(fig)
