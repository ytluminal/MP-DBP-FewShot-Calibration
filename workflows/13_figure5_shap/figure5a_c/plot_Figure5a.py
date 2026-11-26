"""Recreate the uploaded Figure 5a visual design using the canonical SHAP values.
Original Figure 5a plotting source was not supplied. Deterministic beeswarm positioning
and raw-value color scaling are reconstructed, not claimed to be identical to the legacy image.
"""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Patch
from PIL import Image
R=Path(__file__).resolve().parent;z=np.load(R/'Master_SHAP_Exact.npz');raw=pd.read_csv(R/'Original_Training_Records.csv',float_precision='round_trip');s=z['shap'];features=list(z['features']);rank=np.argsort(-np.abs(s).mean(axis=0))[:20]
labels=['DBP identity','DBP dipole moment','MP loading','DOC','HOMO–LUMO gap','DBP nitrogen count','Disinfectant dose','MP particle size','Disinfectant type','Polymer aromatic structure','Buffer concentration','Pretreatment type','Pretreatment duration','UV dose','DBP LUMO energy','DBP HOMO energy','Polymer amide/urethane N','Mixing speed','Chemical dose','Bromide added/leached']
expected=['DBP name','Dipole Moment','MP dose (g/L)','DOC (mg/L C)','HOMO-LUMO Gap','N_count','disinfectant dose (mg/L as Cl2)','particle size (mm)','disinfectant','Polymer aromatic structure','buffer conc. (mM)','pretreatment type','duration (h)','UV dose (J/cm2)','LUMO','HOMO','Polymer amide/urethane N','mixing speed (rpm)','chemical dose (mg/L)','bromide added/leached (mg/L)'];assert [features[i] for i in rank]==expected
cats=['DBP identity / chemistry','Plastic information','Water chemistry / MP-DOM','Disinfection conditions','Pretreatment conditions'];palette=['#507aa5','#1b9a88','#80aa43','#b27ba1','#d39331'];groups=[0,0,1,2,0,0,3,1,3,1,3,4,4,4,0,0,1,4,4,2]
fig=plt.figure(figsize=(12.3,7.93),facecolor='white');ax=fig.add_axes([.236,.115,.285,.695]);bx=fig.add_axes([.567,.115,.339,.695]);cax=fig.add_axes([.918,.115,.019,.695]);plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
y=np.arange(20);imp=np.abs(s[:,rank]).mean(axis=0);ax.barh(y,imp,height=.68,color=[palette[g] for g in groups],zorder=3);ax.set_yticks(y,labels,fontsize=10.1,color='#233c55');ax.set_ylim(19.65,-.65);ax.set_xlim(0,.282);ax.set_xticks(np.arange(0,.26,.05));ax.set_xlabel('Mean |SHAP value|',fontsize=12);ax.tick_params(axis='y',length=0,pad=6);ax.tick_params(axis='x',labelsize=10,colors='#496582',length=0);ax.grid(axis='x',color='#d6e0ea',lw=.8);ax.set_axisbelow(True);ax.set_title('Mean absolute contribution',loc='left',pad=24,fontsize=13.2,fontweight='bold',color='#233e59')
for i,v in enumerate(imp):ax.text(v+.002,i,f'{v:.3f}',fontsize=9.9,va='center',color='#233c55')
ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(palette,cats)],loc='lower right',bbox_to_anchor=(1.055,.005),fontsize=11,frameon=False,handlelength=1.35,handleheight=1, labelspacing=.55)
# Beeswarm: stack points by SHAP bin, then scale to maximum half-height 0.34 per feature.
# Color uses raw numerical feature values; categorical and missing values are gray.
cmap=LinearSegmentedColormap.from_list('blue_pink',['#7faec5','#a3a9be','#d68ba5']);rng=np.random.default_rng(42)
for row,k in enumerate(rank):
 vals=s[:,k]; bins=np.floor((vals+0.8)/.019).astype(int);offset=np.zeros(len(vals))
 for b in np.unique(bins):
  inds=np.flatnonzero(bins==b);inds=inds[rng.permutation(len(inds))];n=len(inds);seq=np.arange(n);stack=np.where(seq%2==0,seq//2,-(seq+1)//2);offset[inds]=stack
 maxoff=np.max(abs(offset));offset*=.34/max(maxoff,1)
 val=pd.to_numeric(raw[features[k]],errors='coerce').to_numpy();finite=np.isfinite(val);color=np.tile(np.array([.47,.47,.47,1]),(len(vals),1))
 if finite.any():
  lo,hi=np.nanpercentile(val,[5,95]);lo,hi=(np.nanmin(val),np.nanmax(val)) if lo==hi else (lo,hi);scaled=np.full(len(val),.5) if lo==hi else np.clip((val-lo)/(hi-lo),0,1);color[finite]=cmap(scaled[finite]);
 order=rng.permutation(len(vals));bx.scatter(vals[order],row+offset[order],c=color[order],s=7.5,alpha=.76,linewidths=0,zorder=3)
bx.set_ylim(19.65,-.65);bx.set_xlim(-.70,1.31);bx.set_yticks(y);bx.set_yticklabels([]);bx.set_xticks(np.arange(-.5,1.26,.25));bx.tick_params(axis='both',length=0,labelsize=10,colors='#496582');bx.grid(axis='x',color='#d6e0ea',lw=.8);bx.grid(axis='y',color='#e0e5ea',lw=.65,linestyle=(0,(1,3)));bx.axvline(0,color='#a9b4c0',lw=1.3,zorder=2);bx.set_title('Effect on model prediction',loc='left',pad=24,fontsize=13.2,fontweight='bold',color='#233e59');bx.set_xlabel('SHAP value',fontsize=12)
for a in [ax,bx]:
 for sp in ['top','right','left']:a.spines[sp].set_visible(False)
 a.spines['bottom'].set_color('#b8c6d5')
cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax);cb.set_ticks([0,1],labels=['Low','High']);cb.ax.tick_params(length=0,labelsize=12,pad=7);cb.set_label('Feature value',fontsize=12,labelpad=9);cb.outline.set_visible(False)
out=R/'figures'/'Figure5a_UNIFIED_600dpi';fig.savefig(str(out)+'.png',dpi=600,facecolor='white');fig.savefig(str(out)+'.svg',facecolor='white');plt.close(fig);im=Image.open(str(out)+'.png');im.thumbnail((1536,990));im.save(R/'figures'/'Figure5a_preview.png')
print('Created canonical Figure 5a')
