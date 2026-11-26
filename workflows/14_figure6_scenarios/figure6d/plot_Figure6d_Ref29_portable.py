from pathlib import Path
import zipfile, io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'frozen_figure4_inputs'
OUT=ROOT/'Figure6d_Ref29_5pct'
def csv(p): return pd.read_csv(INPUT/p)
j=json.loads((INPUT/'input/external55.json').read_text(encoding='utf-8'))
e=pd.DataFrame(j['rows'],columns=j['columns'])
p=csv('input/Line1_Line2_LightGBM_Row_Predictions.csv')
s=csv('output/Active_Budget_Selections.csv')
o=csv('output/Active_Offsets.csv')
m=csv('output/Active_Own_Publications.csv')
assert len(e)==len(p)==64
assert np.array_equal(e['ref ID'],p['Ref ID'])
assert np.array_equal(e['DBP name'],p['DBP name'])
assert np.array_equal(e['polymer type'],p['Polymer'])
assert np.allclose(e['DBP concentration (µg/L)'],p.Observed_ugL,rtol=0,atol=1e-10)
sel=s.query('Stage == 3 and Budget_pct == 5 and Ref_ID == 29')
off=o.query('Stage == 3 and Budget_pct == 5 and Ref_ID == 29 and Scheme == "Grouped"').iloc[0]
met=m.query('Stage == 3 and Budget_pct == 5 and Ref_ID == 29 and Scheme == "Grouped"').iloc[0]
assert len(sel)==1 and int(sel.iloc[0].Source_Row)==20
idx=int(sel.iloc[0].External_Index_0based)
assert p.iloc[idx]['Source row']==20 and e.iloc[idx]['DBP name']=='DCAA' and e.iloc[idx]['MP dose (g/L)']==0.5
lncol='Line 2 | Stage 3 | Grouped 5-fold ensemble | predicted ln1p'
q=pd.DataFrame({'source_row':p['Source row'],'Ref_ID':e['ref ID'],'polymer':e['polymer type'],'MP_loading_gL':e['MP dose (g/L)'],'DBP':e['DBP name'],'measured_ugL':p.Observed_ugL,'measured_ln1p':p.Observed_ln1p,'zero_shot_ln1p':p[lncol]})
q=q[q.Ref_ID==29].copy()
q['offset_ln1p']=float(off.Mean_offset_ln1p)
q['zero_shot_ugL']=np.expm1(q.zero_shot_ln1p)
q['few_shot_ln1p']=q.zero_shot_ln1p+q.offset_ln1p
q['few_shot_ugL']=np.expm1(q.few_shot_ln1p)
q['calibration_sample']=q.source_row.isin(sel.Source_Row)
q['nominal_budget_pct']=5
q['actual_budget_pct']=100/12
q['pretreatment']='O3'; q['ozone_mgL']=2; q['disinfectant']='Cl2'; q['chlorine_mgL_as_Cl2']=9; q['contact_time_h']=72
q=q.sort_values(['MP_loading_gL','source_row'])
q.to_csv(str(OUT)+'_source_data.csv',index=False)
d=q[~q.calibration_sample].copy()
assert len(d)==11
rmse=lambda a,b: np.sqrt(np.mean((a-b)**2))
assert np.isclose(rmse(d.measured_ln1p,d.zero_shot_ln1p),met.Zero_RMSE,atol=1e-9)
assert np.isclose(rmse(d.measured_ln1p,d.few_shot_ln1p),met.Active_RMSE,atol=1e-9)
print(d[['MP_loading_gL','DBP','measured_ugL','zero_shot_ugL','few_shot_ugL']].to_string(index=False))
print('RMSE',met.Zero_RMSE,met.Active_RMSE)

# Publication panel, grouped by MP loading. Leave a small gap between groups.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none','axes.linewidth':0.8})
fig,ax=plt.subplots(figsize=(7.2,4.6))
fig.subplots_adjust(left=.29,right=.97,top=.86,bottom=.17)
ypos=[]; groups=[]; y=0
for dose,sub in d.groupby('MP_loading_gL',sort=True):
    ys=np.arange(y,y+len(sub),dtype=float); ypos.extend(ys); groups.append((dose,ys.mean())); y=ys[-1]+1.6
ys=np.array(ypos)
# A common connector color keeps the focus on the three methods.
ax.hlines(ys,d.zero_shot_ugL,d.few_shot_ugL,color='#B3B9BE',lw=1.5,zorder=1)
ax.scatter(d.zero_shot_ugL,ys,marker='o',s=44,c='#D89336',edgecolor='white',linewidth=.5,zorder=3,label='Zero-shot')
ax.scatter(d.few_shot_ugL,ys,marker='o',s=44,c='#258779',edgecolor='white',linewidth=.5,zorder=4,label='Few-shot calibrated')
# Open dark diamond distinguishes measured data even when points overlap.
ax.scatter(d.measured_ugL,ys,marker='D',s=52,facecolor='none',edgecolor='#364150',linewidth=1.25,zorder=5,label='Measured')
ax.set_yticks(ys,labels=d.DBP)
ax.tick_params(axis='y',length=0,pad=9)
ax.set_ylim(ys[-1]+.7,-.7)
ax.set_xlim(0,19)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.set_xlabel('DBP concentration (μg/L)',labelpad=7)
ax.grid(axis='x',color='#E4E7E9',lw=.7)
ax.set_axisbelow(True)
for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
for dose,center in groups:
    ax.text(-.1764705882352941,center,f'{dose:.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontweight='medium',fontsize=10,clip_on=False)
for i in [3,6]:
    ax.axhline((ys[i]+ys[i+1])/2,color='#D7DBDF',lw=.8,zorder=0)
fig.text(.162,.89,'MP loading\n(g/L)',ha='center',va='bottom',fontsize=9)
fig.text(.26,.89,'DBP',ha='right',va='bottom',fontsize=9)
h,l=ax.get_legend_handles_labels(); order=[2,0,1]
fig.legend([h[i] for i in order],[l[i] for i in order],loc='upper center',bbox_to_anchor=(.63,.942),ncol=3,frameon=False,fontsize=9,columnspacing=1.15,handletextpad=.35)
fig.savefig(str(OUT)+'_600dpi.png',dpi=600,facecolor='white')
fig.savefig(str(OUT)+'.svg',facecolor='white')
fig.savefig(str(OUT)+'_preview.png',dpi=150,facecolor='white')
