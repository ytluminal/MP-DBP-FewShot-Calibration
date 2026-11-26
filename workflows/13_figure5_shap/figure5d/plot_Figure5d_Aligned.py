from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
outdir=Path(__file__).resolve().parent
cs=pd.read_csv(outdir/'Figure5d_cell_summary.csv').rename(columns={'polymer type':'polymer','pretreatment type':'pretreat','mean_SHAP':'mean_shap','n_publications':'n_pubs'})
poly_order=['PA','PA6','PA66','PAN','PBAT','PE','PET','PLA','PMMA','PP','PS','PVC','TPU','TWP']
pretreat_order=['N','mixing','UV','Cl2','O3','KMnO4','UV/H2O2','UV/Cl2','UV/O3','O3/H2O2']
pretty={'N':'None','mixing':'Mixing','UV':'UV','Cl2':'Cl$_2$','O3':'O$_3$','KMnO4':'KMnO$_4$','UV/H2O2':'UV/H$_2$O$_2$','UV/Cl2':'UV/Cl$_2$','UV/O3':'UV/O$_3$','O3/H2O2':'O$_3$/H$_2$O$_2$'}
cs['polymer']=pd.Categorical(cs['polymer'],poly_order,ordered=True)
cs['pretreat']=pd.Categorical(cs['pretreat'],pretreat_order,ordered=True)
cs=cs.sort_values(['polymer','pretreat'])
base=cs[cs['pretreat']=='N'][['polymer','mean_shap','n_records','n_pubs']].rename(columns={'mean_shap':'base_shap','n_records':'base_records','n_pubs':'base_pubs'})
delta=cs[cs['pretreat']!='N'].merge(base,on='polymer',how='left'); delta=delta[delta['base_shap'].notna()].copy(); delta['delta_vs_none']=delta['mean_shap']-delta['base_shap']
sel=delta[(delta['n_records']>=10) & (delta['base_records']>=10)].copy(); sel['pretreat_str']=sel['pretreat'].astype(str); sel['label']=sel['polymer'].astype(str)+' – '+sel['pretreat_str'].map(pretty); sel=sel.sort_values('delta_vs_none')
sel.to_csv(outdir/'Figure5d_Option3_Plot_Data.csv',index=False)
pt_colors=dict(zip([p for p in pretreat_order if p!='N'],plt.get_cmap('tab10').colors[:9]))
fig,ax=plt.subplots(figsize=(11.4,8.2)); fig.subplots_adjust(top=0.90,bottom=0.14,left=0.28,right=0.96)
y=np.arange(len(sel)); ax.axvline(0,color='#9099a5',lw=1.2)
for yi,(_,r) in enumerate(sel.iterrows()):
    col=pt_colors.get(r['pretreat_str'],'#4c78a8')
    ax.hlines(yi,0,r['delta_vs_none'],color=col,lw=2.6,alpha=0.9)
    ax.scatter(r['delta_vs_none'],yi,s=35+15*np.sqrt(r['n_records']),color=col,edgecolors='black' if r['mixed_sign'] else 'white',linewidths=0.9,zorder=3)
    ax.text(r['delta_vs_none']+(0.012 if r['delta_vs_none']>=0 else -0.012),yi,f"{r['delta_vs_none']:+.2f}",va='center',ha='left' if r['delta_vs_none']>=0 else 'right',fontsize=13)
ax.set_yticks(y); ax.set_yticklabels(sel['label'],fontsize=15)
ax.tick_params(axis='y', length=0, pad=36)
# Align the center of each en dash with the corresponding SHAP sign.
# Offsets (points), measured from the original 600-dpi rendering, in y-tick order.
from matplotlib.transforms import ScaledTranslation
sign_alignment_pt = [1.50, 1.38, 1.50, 1.38, 1.50, 1.38, 1.38, 1.50,
                     1.98, 2.10, 1.98, 2.10, 1.98, 1.98, 1.98, 1.98, 1.98]
for tick_label, dy in zip(ax.get_yticklabels(), sign_alignment_pt):
    tick_label.set_transform(tick_label.get_transform() +
                             ScaledTranslation(0, dy / 72, fig.dpi_scale_trans))
ax.set_xlabel('Shift in mean pretreatment SHAP relative to untreated (None)',fontsize=17, labelpad=10)
ax.tick_params(axis='x', labelsize=14, pad=5)
ax.grid(axis='x',alpha=0.25)
for s in ax.spines.values(): s.set_visible(False)
handles=[]; labels=[]
for p in ['mixing','UV','Cl2','O3','UV/H2O2','UV/Cl2']:
    if p in sel['pretreat_str'].unique():
        handles.append(plt.Line2D([0],[0],marker='o',color=pt_colors[p],markerfacecolor=pt_colors[p],linestyle='-',markersize=8)); labels.append(pretty[p])
ax.legend(handles,labels,title='Pretreatment',frameon=False,fontsize=13,title_fontsize=15,loc='lower right')
fig.suptitle('Option 3 — ΔSHAP versus untreated baseline',fontsize=21,fontweight='bold',y=0.97)
fig.savefig(outdir/'Figure5d_Aligned_600dpi.png',dpi=600)
plt.rcParams['svg.fonttype']='none'
fig.savefig(outdir/'Figure5d_Aligned.svg')
plt.close(fig)
