import pandas as pd,numpy as np,matplotlib as mpl,matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from pathlib import Path
p=str(Path(__file__).resolve().parent) + '/' 
d=pd.read_csv(p+'Figure6c_integrated_source_data.csv')
rows=d.scenario.drop_duplicates().tolist(); times=[6,12,24,48,72]
a=d.pivot(index='scenario',columns='time_h',values='delta_ln1p').loc[rows,times].to_numpy()
assert a.shape==(13,5) and np.isfinite(a).all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','text.color':'#1d3952','axes.labelcolor':'#1d3952','xtick.color':'#1d3952','ytick.color':'#1d3952'})
# Crop the unused negative portion of the original colormap without altering its numerical color mapping.
lo,hi=-.06,.60
old=mpl.colormaps['RdBu_r']; cmap=LinearSegmentedColormap.from_list('cropped_RdBu_r',old(np.linspace((lo+.6)/1.2,(hi+.6)/1.2,2048)),N=2048)
norm=Normalize(lo,hi)
fig=plt.figure(figsize=(10.2,8.5),facecolor='white')
ax=fig.add_axes([.282,.185,.548,.65]); im=ax.imshow(a,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
labels=[s.replace(' – 10 mg/L – NDMA','–NDMA') for s in rows]
ax.set_xticks(range(5),[str(t) for t in times]);ax.set_yticks(range(13),labels);ax.tick_params(length=0,pad=5);ax.set_xlabel('Reaction time (h)',fontsize=15,labelpad=7)
for s in ax.spines.values():s.set_visible(False)
ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,13,1),minor=True);ax.grid(which='minor',color='white',linewidth=.65);ax.tick_params(which='minor',bottom=False,left=False)
for s in [3,6,9,11]:ax.axhline(s-.5,color='white',lw=2.4)
for i in range(13):
 for j in range(5):
  v=a[i,j]; prec=2 if i<11 else 3
  txt=f'{v:+.{prec}f}' if j else f'{0:.{prec}f}'
  ax.text(j,i,txt,ha='center',va='center',fontsize=11.5,color='white' if abs(v)>.38 else '#1d3952')
cb=fig.colorbar(im,cax=fig.add_axes([.861,.185,.026,.65]));cb.set_ticks([-.06,0,.1,.2,.3,.4,.5,.6]);cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '−0.06' if x<0 else f'{x:.1f}'));cb.ax.tick_params(labelsize=11,length=3);cb.set_label('Change in predicted ln(1 + C) vs. 6 h',fontsize=12,labelpad=10);cb.outline.set_linewidth(.8)
fig.text(.09,.94,'Time-dependent DBP response across scenarios',fontsize=22,weight='bold');fig.text(.09,.90,'Refs. 5 and 27  |  Frozen LightGBM ensemble',fontsize=13)
fig.text(.09,.085,'THMs: chlorination (9.2 mg/L as Cl₂); NDMA: chloramination (10 mg/L as Cl₂).',fontsize=10.5);fig.text(.09,.056,'All cells are model predictions relative to 6 h (C in μg/L); some times lack matched measurements.',fontsize=10.5)
name=p+'Figure6c_integrated_THM_NDMA_adjusted_colorbar'
fig.savefig(name+'_600dpi.png',dpi=600,facecolor='white');fig.savefig(name+'.svg',facecolor='white');plt.close(fig)
im=Image.open(name+'_600dpi.png');print(im.size,im.info['dpi'])
