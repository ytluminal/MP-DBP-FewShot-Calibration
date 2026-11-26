from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

out=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--master',type=Path,default=out/'Master_SHAP_Dataset.csv')
parser.add_argument('--records',type=Path,default=out/'Original_Training_Records.csv')
args=parser.parse_args()
a=pd.read_csv(args.records); b=pd.read_csv(args.master)
ids=['training_row','source_excel_row','ref ID','fold']
br='bromide added/leached (mg/L)'; sh='SHAP | '+br
species=['TCM','BDCM','DBCM','TBM']
assert len(a)==len(b)==995 and b.filter(regex=r'^SHAP \| ').shape[1]==55
assert a[ids].equals(b[ids])
assert np.allclose(b.OOF_prediction_ln1p,b.archived_prediction_ln1p,atol=1e-12,rtol=0)
assert np.max(np.abs(b.additivity_error))<1e-10
z=a.merge(b[ids+[sh]],on=ids,validate='one_to_one')
keys=['ref ID','polymer type','additives','particle size (mm)','MP dose (g/L)','pretreatment type','duration (h)','mixing speed (rpm)','pretreatment temperature (°C)','chemical dose (mg/L)','UV dose (J/cm2)','sample phase','DOC (mg/L C)','pH','buffer','buffer conc. (mM)',br,'disinfectant','disinfectant dose (mg/L as Cl2)','contact time (h)','disinfection temperature (°C)','quenching agent']
s=z[(z[br]>0)&z['DBP name'].isin(species)].copy()
s=s[s.groupby(keys,dropna=False)['DBP name'].transform('nunique')==4].copy()
assert not s.duplicated(keys+['DBP name']).any()
s['condition_id']=s.groupby(keys,dropna=False,sort=True).ngroup()+1
s=s.sort_values(['ref ID','condition_id','Br_count'])
pub=s.groupby(['ref ID','DBP name'])[sh].mean().unstack().reindex(columns=species)
counts=s.groupby('ref ID').condition_id.nunique()
means=pub.mean(axis=0)
assert s.condition_id.nunique()==17 and len(s)==68 and pub.shape==(5,4)
s[ids+['condition_id','DBP name','Br_count','DBP concentration (µg/L)',sh]+[k for k in keys if k!='ref ID']].to_csv(out/'Figure5e_Matched_Records.csv',index=False)
pub.assign(n_conditions=counts).to_csv(out/'Figure5e_Publication_Means.csv')
means.rename('Mean_SHAP').to_csv(out/'Figure5e_Overall_Means.csv',header=True)
print(pub.to_string()); print(means.to_string()); print(counts.to_string())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#243a50','xtick.color':'#243a50','ytick.color':'#243a50','text.color':'#243a50','axes.edgecolor':'#aab9c6','axes.linewidth':0.9,'savefig.facecolor':'white'})
colors=['#4C78A8','#D39440','#349887','#A67DA5','#8493A2']
x=np.arange(4)
labels=[f'{sp}\n{i}' for i,sp in enumerate(species)]

def base_plot(subtitle):
    fig,ax=plt.subplots(figsize=(8.6,5.9))
    fig.subplots_adjust(left=.15,right=.96,top=.80,bottom=.28)
    fig.text(.15,.935,'Bromide contributions to THM formation',fontsize=21,weight='bold',color='#233e59')
    fig.text(.15,.878,subtitle,fontsize=11,color='#536a80')
    ax.set_xlim(-.48,3.48); ax.set_ylim(-.15,.31)
    ax.set_xticks(x,labels,fontsize=15); ax.tick_params(axis='x',length=0,pad=9)
    ax.set_yticks(np.arange(-.15,.301,.05)); ax.tick_params(axis='y',labelsize=12)
    ax.set_ylabel('SHAP value for bromide',fontsize=16,labelpad=10)
    ax.set_xlabel('Number of bromine atoms',fontsize=15,labelpad=10)
    ax.grid(axis='y',color='#d6dfe6',lw=.8); ax.set_axisbelow(True)
    ax.axhline(0,color='#8393a2',ls='--',lw=1.25,zorder=2)
    return fig,ax

def legend(fig,lines=False):
    handles=[Line2D([],[],marker='o',color=c,linestyle='-' if lines else 'none',markersize=7.5, label=f'Ref. {r}') for r,c in zip(pub.index,colors)]
    handles.append(Line2D([],[],marker='D',color='#202b35',linestyle='-' if lines else 'none',markersize=7.5,label='Mean'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.035),ncol=6,frameon=False,fontsize=10,handletextpad=.35,columnspacing=1.1)

def save(fig,name):
    fig.savefig(out/(name+'_600dpi.png'),dpi=600)
    fig.savefig(out/(name+'.svg'))
    with Image.open(out/(name+'_600dpi.png')) as im:
        print(name,im.size,im.info.get('dpi'))
        im.thumbnail((1400,1000)); im.save(out/(name+'_preview.png'))
    plt.close(fig)

# Option 2: paired profiles; lines link species in the same publication, not a dose-response curve.
fig,ax=base_plot('Publication-level SHAP profiles  |  17 condition groups, 5 publications')
for (ref,row),c in zip(pub.iterrows(),colors):
    ax.plot(x,row.values,'o-',color=c,lw=1.7,ms=7.8,mec='white',mew=.65,zorder=3,alpha=.92)
ax.plot(x,means.values,'D-',color='#202b35',lw=2.6,ms=8.6,mec='white',mew=.8,zorder=5)
# Move the original legend into the lower-right corner of the axes.
handles=[Line2D([],[],marker='o',color=c,linestyle='-',markersize=7.5,label=f'Ref. {r}') for r,c in zip(pub.index,colors)]
handles.append(Line2D([],[],marker='D',color='#202b35',linestyle='-',markersize=7.5,label='Mean'))
ax.legend(handles=handles,loc='lower right',ncol=2,frameon=False,fontsize=10,handletextpad=.35,columnspacing=1.1)
save(fig,'Figure5e_Option2_Profiles')

