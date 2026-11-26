from pathlib import Path
import sys, zipfile, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
out=Path(__file__).resolve().parent
root=out/'work';root.mkdir(exist_ok=True)
with zipfile.ZipFile(out/'Frozen_S3_Grouped_Models.zip') as z:z.extractall(root)
sys.path.insert(0,str(root));import line2_lightgbm_preprocess
warnings.filterwarnings('ignore',message='X does not have valid feature names')
x=pd.read_csv(out/'Original_Training_Records.csv',float_precision='round_trip');assert len(x)==995
models=[joblib.load(root/'LightGBM_Reproduced_Models'/f'LightGBM_Line2_S3_grouped_F{i}.joblib') for i in range(1,6)]
C='DBP concentration (µg/L)'
def predict(a):return np.array([m['model'].predict(m['processor'].transform_dataframe(a),num_threads=2) for m in models])
get=lambda i:x.loc[x.training_row==i].iloc[0].copy()
base={'TCM':get(84),'DCAA':get(92),'DCAN':get(84),'NDMA':get(896)}
# Hypothetical DCAN under the identical Ref. 1 TWP background: replace only DBP identity and molecular descriptors.
fields=['DBP name']+x.columns[x.columns.get_loc('MW'):x.columns.get_loc('Electrophilicity')+1].tolist()
base['DCAN'][fields]=get(884)[fields];base['DCAN'][C]=np.nan
curves=[]; baselines=[]; matches=[]
keys=[k for k in x if k not in ['training_row','source_excel_row','fold','pH',C]]
for sp,b in base.items():
    ph=np.round(np.linspace(6,9,61),2)
    a=pd.DataFrame([b]*len(ph));a['pH']=ph
    pred=predict(a)
    support={'TCM':'Ref. 1 observed pH gradient','DCAA':'Ref. 1 observed pH gradient','DCAN':'hypothetical Ref. 1 conditions; DCAN descriptors from row 884, no matched observations','NDMA':'Ref. 27 pH 8.3 only; exploratory'}[sp]
    d=pd.DataFrame({'species':sp,'pH':ph,'predicted_ln1p':pred.mean(axis=0),'predicted_ugL':np.expm1(pred.mean(axis=0)),'baseline_training_row':b.training_row,'ref_ID':b['ref ID'],'disinfectant':b.disinfectant,'pH_response_support':support})
    for i in range(5):d[f'fold_{i+1}_predicted_ln1p']=pred[i]
    curves.append(d);bb=b.copy();bb['scenario_note']=support;baselines.append(bb)
    if sp=='DCAN':continue
    g=x[((x[keys]==b[keys])|(x[keys].isna()&b[keys].isna())).all(axis=1)].sort_values('pH').copy()
    pp=predict(g);f=g.fold.to_numpy(dtype=int)-1
    g['ensemble_predicted_ugL']=np.expm1(pp.mean(axis=0));g['OOF_predicted_ugL']=np.expm1(pp[f,np.arange(len(g))]);matches.append(g)
d=pd.concat(curves,ignore_index=True);d.to_csv(out/'Figure6b_TWP_Curves.csv',index=False)
pd.DataFrame(baselines).to_csv(out/'Figure6b_TWP_Baseline_Records.csv',index=False)
c=pd.concat(matches,ignore_index=True);c.to_csv(out/'Figure6b_TWP_Matched_Records_and_OOF.csv',index=False)
o=pd.read_csv(out/'OOF_Predictions_and_Fold_Assignments.csv',float_precision='round_trip');a=c.merge(o[['training_row','OOF_prediction_ln1p']],on='training_row');err=np.max(np.abs(np.log1p(a.OOF_predicted_ugL)-a.OOF_prediction_ln1p));assert err<1e-10;print('OOF max difference',err)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':16,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig=plt.figure(figsize=(9.4,6.8),facecolor='white');ax=fig.add_axes([.12,.39,.85,.40]);nd=fig.add_axes([.12,.17,.85,.125],sharex=ax)
colors={'TCM':'#1f77b4','DCAA':'#2ca02c','DCAN':'#9467bd','NDMA':'#D39440'};markers={'TCM':'o','DCAA':'s','DCAN':'^'}
for sp in ['TCM','DCAA','DCAN']:
    g=d[d.species==sp];ax.plot(g.pH,g.predicted_ugL,color=colors[sp],lw=2.2,ls='--' if sp=='DCAN' else '-',marker=markers[sp],markevery=10,ms=5,label=sp+' (exploratory)' if sp=='DCAN' else sp)
for a in [ax,nd]:a.set_xlim(5.9,9.1);a.set_axisbelow(True);a.grid(axis='y',alpha=.3)
y=d.loc[d.species!='NDMA','predicted_ugL'];ax.set_ylim(0,21);ax.tick_params(labelbottom=False);ax.set_ylabel('Predicted concentration (µg/L)');ax.legend(loc='upper left',ncol=3,frameon=False,handlelength=3.0)
for ph in sorted(c.loc[c['DBP name']=='TCM','pH'].unique()):ax.plot(ph,.035,'|',color='#64748b',ms=6,alpha=.7,transform=ax.get_xaxis_transform())
g=d[d.species=='NDMA'];nd.plot(g.pH,g.predicted_ugL,color=colors['NDMA'],lw=2,ls=':',label='NDMA (exploratory)');v=g[np.isclose(g.pH,8.3)].iloc[0];nd.scatter(v.pH,v.predicted_ugL,color=colors['NDMA'],marker='D',s=35,zorder=3)
nd.axvline(8.3,color='#8b949e',ls='--',lw=1);nd.set_ylim(.1,.26);nd.set_yticks([.1,.2]);nd.set_ylabel('NDMA (µg/L)',fontsize=13);nd.set_xticks([6,7,8,9]);nd.set_xlabel('pH',labelpad=4);nd.legend(loc='upper left',frameon=False,fontsize=10,handlelength=3)
fig.text(.12,.95,'pH-dependent DBP formation',fontsize=22,fontweight='bold',va='top');fig.text(.12,.89,'Frozen LightGBM ensemble  |  Reference-based scenarios',fontsize=12,va='top')
fig.text(.12,.815,'TCM, DCAA and DCAN: chlorination of ozonated TWP (Ref. 1)',fontsize=11);fig.text(.12,.312,'NDMA: chloramination of PAN (Ref. 27; training pH = 8.3 only)',fontsize=11)
fig.text(.12,.018,'TWP: 16 mg/L as Cl₂, 24 h, 0.15 g/L MP; PAN: 10 mg/L as Cl₂, 72 h, 0.1 g/L MP. DOC missing.\nTicks: observed TWP pH (TCM/DCAA). DCAN and NDMA curves are exploratory; no matched pH gradients.',fontsize=9,va='bottom')
fig.savefig(out/'Figure6b_TWP_600dpi.png',dpi=600,facecolor='white');fig.savefig(out/'Figure6b_TWP.svg',facecolor='white');plt.close(fig)
im=Image.open(out/'Figure6b_TWP_600dpi.png');print(im.size,im.info['dpi']);im.thumbnail((1500,1100));im.save(out/'Figure6b_TWP_preview.png')
print(d[d.pH.isin([6,7.5,9])].pivot(index='species',columns='pH',values='predicted_ugL'));print(c[['DBP name','pH',C,'OOF_predicted_ugL','ensemble_predicted_ugL']].to_string(index=False))
