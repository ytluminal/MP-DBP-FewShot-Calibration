from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
out=Path(__file__).resolve().parent
curves=pd.read_csv(out/'Figure6a_Curves.csv',float_precision='round_trip')
assert len(curves)==174 and curves.groupby(['species','disinfectant']).size().eq(29).all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':16,'axes.titlesize':22,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(9.4,6.8))
styles={('TCM','Cl2'):('#1f77b4','-','o'),('TCM','NH2Cl'):('#ff7f0e','--','o'),('DCAA','Cl2'):('#2ca02c','-','s'),('DCAA','NH2Cl'):('#d62728','--','s'),('DCAN','Cl2'):('#9467bd','-','^'),('DCAN','NH2Cl'):('#9467bd','--','^')}
for (sp,dis),(c,ls,m) in styles.items():
    g=curves[(curves.species==sp)&(curves.disinfectant==dis)].sort_values('dose_mgL_as_Cl2')
    ax.plot(g.dose_mgL_as_Cl2,g.predicted_concentration_ugL,color=c,linestyle=ls,marker=m,markevery=4,linewidth=2.2,markersize=5,label=f'{sp} — '+('Cl₂' if dis=='Cl2' else 'NH₂Cl'))
# Observed-dose rug, identical to the previous Figure 6a bottom-ticks version.
obs = pd.read_csv(out/'Figure6a_Observed_Doses.csv',float_precision='round_trip')
cl2_v = obs.loc[obs['disinfectant']=='Cl2','disinfectant dose (mg/L as Cl2)']
nh2cl_v = obs.loc[obs['disinfectant']=='NH2Cl','disinfectant dose (mg/L as Cl2)']
ax.plot(cl2_v, [0.022]*len(cl2_v), '|', color='#1f77b4', alpha=0.45, ms=8, transform=ax.get_xaxis_transform(), clip_on=True)
ax.plot(nh2cl_v, [0.045]*len(nh2cl_v), '|', color='#ff7f0e', alpha=0.45, ms=8, transform=ax.get_xaxis_transform(), clip_on=True)
ax.set_xlabel('Disinfectant dose (mg/L as Cl$_2$)')
ax.set_ylabel('Predicted DBP concentration (µg/L)')
ax.set_xlim(.8,15.2); ax.set_ylim(0,curves.predicted_concentration_ugL.max()*1.12)
ax.grid(True,axis='y',alpha=.3);ax.legend(loc='upper left',frameon=False,ncol=3,columnspacing=1.1,handletextpad=.6,handlelength=4.0)
fig.suptitle('Disinfectant-dependent DBP formation',x=.12,ha='left',fontweight='bold')
fig.text(.12,.89,'Species-specific precursor scenarios  |  TCM, DCAA and DCAN  |  Frozen LightGBM ensemble',ha='left',fontsize=12)
fig.text(.12,.045,'Only disinfectant type and dose were varied within each species-specific scenario.\nTCM and DCAA: Figure 6a medium conditions; DCAN: Ref. 23 baseline.',ha='left',fontsize=9)
fig.subplots_adjust(left=.12,right=.98,bottom=.17,top=.80)
fig.savefig(out/'Figure6a_LegendUpdated_600dpi.png',dpi=600,facecolor='white')
fig.savefig(out/'Figure6a_LegendUpdated.svg',facecolor='white')
fig.savefig(out/'Figure6a_LegendUpdated_preview.png',dpi=160,facecolor='white')
plt.close(fig)
