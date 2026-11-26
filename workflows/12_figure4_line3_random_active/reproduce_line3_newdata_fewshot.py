#!/usr/bin/env python3
"""Figure 4 Line 3 LightGBM Stage 1–4 external Ref28/29/30 random+active few-shot.
Uses frozen external zero-shot predictions, training-only active feature encoding,
fixed Line1 random permutations, publication-specific mean residual correction.
No LightGBM training or prediction recomputation. Requires numpy, scikit-learn.
Run: python reproduce_line2_newdata_fewshot.py
"""
from pathlib import Path
from collections import Counter
import csv, json, hashlib, math, shutil
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE=Path(__file__).resolve().parent
IN=BASE/'input'; OUT=BASE/'output';OUT.mkdir(exist_ok=True)
BUDGETS=[2.5,5.,10.,20.]
REFS=[28,29,30]
SCHEMES=['Grouped','Random']
STAGES=[1,2,3,4]
NREPS=500
CATS=['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']

def readcsv(path):
 with open(path,'r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(path,rows):
 if not rows:return
 with open(path,'w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction='ignore');w.writeheader();w.writerows(rows)
def metric(y,p):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
 d=y-p;n=len(y);ss=np.sum((y-y.mean())**2)
 return {'R2':float(1.-np.dot(d,d)/ss) if n>1 and ss>1e-14 else float('nan'), 'RMSE':float(np.sqrt(np.mean(d*d))),'MAE':float(np.mean(np.abs(d)))}
def bud_k(n,b): return min(n-1,max(1,math.ceil(n*b/100.)))
def num(x):
 if x is None or isinstance(x,str) and x.strip()=='':return np.nan
 try: return float(x)
 except (ValueError,TypeError):return np.nan

def active_encoder(train,ext,feats,headers):
 cats=[s for s in CATS if s in feats];nums=[s for s in feats if s not in cats]
 idxn=[headers.index(f) for f in nums];idxc=[headers.index(f) for f in cats]
 a=np.array([[num(row[j]) for j in idxn] for row in train],dtype=float)
 b=np.array([[num(row[j]) for j in idxn] for row in ext],dtype=float)
 a[~np.isfinite(a)]=np.nan;b[~np.isfinite(b)]=np.nan
 median=np.array([float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else 0. for v in a.T])
 a=np.where(np.isfinite(a),a,median);b=np.where(np.isfinite(b),b,median)
 z=StandardScaler().fit(a).transform(b)
 if cats:
  def cat(x):return '__MISSING__' if x is None or str(x).strip()=='' else str(x)
  ac=np.array([[cat(row[j]) for j in idxc] for row in train],dtype=object)
  bc=np.array([[cat(row[j]) for j in idxc] for row in ext],dtype=object)
  for j in range(len(cats)):
   valid=[v for v in ac[:,j] if v!='__MISSING__']
   cnt=Counter(valid);mode=sorted(cnt,key=lambda t:(-cnt[t],t))[0] if cnt else '__MISSING__'
   ac[ac[:,j]=='__MISSING__',j]=mode;bc[bc[:,j]=='__MISSING__',j]=mode
  oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=float)
  z=np.hstack([z,oh.fit(ac).transform(bc)])
 assert z.shape[0]==len(ext) and np.isfinite(z).all()
 return z,{'n_raw':len(feats),'n_numeric':len(nums),'n_categorical':len(cats),'n_encoded':int(z.shape[1]),'features':'; '.join(feats)}

def centroid_maximin(z,ix):
 ix=np.asarray(ix,dtype=int)
 zz=z[ix];center=zz.mean(axis=0)
 d=np.einsum('ij,ij->i',zz-center,zz-center)
 first=int(np.argmin(d));order=[first]
 chosen=np.zeros(len(ix),dtype=bool);chosen[first]=True
 neard=np.einsum('ij,ij->i',zz-zz[first],zz-zz[first])
 for _ in range(len(ix)-1):
  ds=neard.copy();ds[chosen]=-1
  nxt=int(np.argmax(ds));chosen[nxt]=True;order.append(nxt)
  neard=np.minimum(neard,np.einsum('ij,ij->i',zz-zz[nxt],zz-zz[nxt]))
 return ix[np.asarray(order)]

def scored(y,z,a,r=None):
 values={'Zero':z,'Active':a}
 if r is not None:values['Random']=r
 out={}
 for k,v in values.items():
  for m,score in metric(y,v).items():out[k+'_'+m]=score
 if r is not None:
  for m in ['R2','RMSE','MAE']:
   out['Paired_d'+m]=out['Active_'+m]-out['Random_'+m]
   out['Random_vs_Zero_d'+m]=out['Random_'+m]-out['Zero_'+m]
 for m in ['R2','RMSE','MAE']:
  out['Active_vs_Zero_d'+m]=out['Active_'+m]-out['Zero_'+m]
 return out

def summarize(rows,by_pub=False):
 cols=['N_eval','N_common_eval']+[name+'_'+m for name in ['Zero','Random','Active'] for m in ['R2','RMSE','MAE']]+['Paired_d'+m for m in ['R2','RMSE','MAE']]+[name+'_vs_Zero_d'+m for name in ['Random','Active'] for m in ['R2','RMSE','MAE']]
 out=[]
 for scheme in SCHEMES:
  for stage in STAGES:
   for budget in BUDGETS:
    for ref in (REFS if by_pub else ['Overall']):
     rws=[r for r in rows if r['Scheme']==scheme and r['Stage']==stage and r['Budget_pct']==budget and r['Ref_ID']==ref]
     assert len(rws)==NREPS,(scheme,stage,budget,ref,len(rws))
     r={'Scheme':scheme,'Stage':stage,'Budget_pct':budget,'Ref_ID':ref,'N_replicates':NREPS,
        'N_cal_pub28':bud_k(12,budget),'N_cal_pub29':bud_k(12,budget),'N_cal_pub30':bud_k(40,budget)}
     for c in cols:
      if c not in rws[0]:continue
      v=np.array([float(x[c]) for x in rws]);v=v[np.isfinite(v)]
      for suffix,q in [('median',.5),('q05',.05),('q95',.95)]:r[c+'_'+suffix]=float(np.quantile(v,q)) if len(v) else float('nan')
     r['Fraction_Active_Better_R2']=float(np.mean([float(x['Paired_dR2'])>0 for x in rws]))
     out.append(r)
 return out


def main():
 train=json.loads((IN/'line3_train55.json').read_text(encoding='utf-8'))
 ext=json.loads((IN/'external55.json').read_text(encoding='utf-8'))
 features=json.loads((IN/'stage_matched_feature_lists.json').read_text(encoding='utf-8'))['stage_features']
 assert train['headers']==ext['columns'] and len(train['rows'])==1026 and len(ext['rows'])==64
 headers=ext['columns'];i_ref=headers.index('ref ID');i_y=headers.index('DBP concentration (µg/L)')
 refs=np.asarray([int(row[i_ref]) for row in ext['rows']],dtype=int)
 y=np.log1p(np.asarray([float(row[i_y]) for row in ext['rows']],dtype=float))
 pi={ref:np.flatnonzero(refs==ref) for ref in REFS}
 assert {ref:len(ix) for ref,ix in pi.items()}=={28:12,29:12,30:40}
 # Preserve the verified Line 3 stage/scheme five-fold ensemble predictions.
 frozen=readcsv(IN/'Line3_Frozen_External_Predictions.csv')
 assert len(frozen)==64*8
 source=[];preds={}
 for scheme in SCHEMES:
  for stage in STAGES:
   styp='Publication-grouped' if scheme=='Grouped' else 'Conventional random'
   rows=sorted((r for r in frozen if int(r['Stage'])==stage and r['Scheme']==styp),key=lambda r:int(r['Excel_Row']))
   assert len(rows)==64,(scheme,stage,len(rows))
   for i,row in enumerate(rows):
    assert int(row['Excel_Row'])==i+2 and int(row['Ref_ID'])==refs[i]
    assert np.isclose(float(row['Observed_ln1p']),y[i],rtol=1e-12,atol=1e-12)
    assert str(row['DBP_name'])==str(ext['rows'][i][headers.index('DBP name')])
    assert np.isclose(np.mean([float(row[f'fold_{j}_pred_ln1p']) for j in range(1,6)]),float(row['Predicted_ln1p']),rtol=1e-11,atol=1e-11)
   preds[(scheme,stage)]=np.array([float(row['Predicted_ln1p']) for row in rows],dtype=float)
   if not source:
    source=[{'Source row':str(i+2),'Ref ID':str(refs[i]),'Observed_ln1p':str(y[i]),'DBP name':rows[i]['DBP_name'],'Polymer':rows[i]['polymer_type']} for i in range(64)]
 assert len(preds)==8
 perms=readcsv(IN/'Original_Line1_Random_Permutations_500.csv');perm_map={}
 for row in perms:
  rep=int(row['Replicate']);ref=int(row['Ref_ID']);ordered=np.array([int(x) for x in row['Ordered_ZeroBased_Indices'].split(';')],dtype=int)
  assert set(ordered.tolist())==set(pi[ref].tolist());perm_map[(rep,ref)]=ordered
 assert len(perm_map)==NREPS*len(REFS)
 orders={};feat_mf=[]
 for stage in STAGES:
  selected_feats=features[str(stage)]['LightGBM'];z,manifest=active_encoder(train['rows'],ext['rows'],selected_feats,headers)
  orders[stage]={ref:centroid_maximin(z,ix) for ref,ix in pi.items()}
  feat_mf.append({'Stage':stage,**manifest})
 assert all(np.array_equal(orders[2][ref],orders[3][ref]) for ref in REFS)
 seq=[];sel=[];calibrations=[]
 for stage in STAGES:
  for ref in REFS:
   for rank,idx in enumerate(orders[stage][ref],1):
    seq.append({'Stage':stage,'Ref_ID':ref,'Rank':rank,'External_Index_0based':int(idx),'Source_Row':int(idx+2),'DBP_Name':source[idx]['DBP name'],'Polymer':source[idx]['Polymer'],'Selection':('Centroid' if rank==1 else 'Maximin')})
    for budget in BUDGETS:
     if rank<=bud_k(len(pi[ref]),budget):
      sel.append({'Stage':stage,'Budget_pct':budget,'Ref_ID':ref,'Rank':rank,'External_Index_0based':int(idx),'Source_Row':int(idx+2),'DBP_Name':source[idx]['DBP name'],'Polymer':source[idx]['Polymer']})
 zero=[];random_reps=[];randpub=[];activeown=[];ownpub=[];paired=[];pairedpub=[]
 for scheme in SCHEMES:
  for stage in STAGES:
   p=preds[(scheme,stage)]
   for ref in ['Overall']+REFS:
    ix=np.arange(64) if ref=='Overall' else pi[ref]
    zero.append({'Scheme':scheme,'Stage':stage,'Ref_ID':ref,'N':len(ix),**metric(y[ix],p[ix])})
   for budget in BUDGETS:
    chosen={ref:orders[stage][ref][:bud_k(len(pi[ref]),budget)] for ref in REFS}
    amask=np.ones(64,dtype=bool);ap=p.copy()
    for ref in REFS:
     ix=chosen[ref];amask[ix]=False;delta=float(np.mean(y[ix]-p[ix]));ap[pi[ref]]+=delta
     calibrations.append({'Scheme':scheme,'Stage':stage,'Budget_pct':budget,'Ref_ID':ref,'N_calibration':len(ix),'Mean_offset_ln1p':delta,'Source_Rows':';'.join(str(i+2) for i in ix)})
    for ref in ['Overall']+REFS:
     ix=np.flatnonzero(amask) if ref=='Overall' else pi[ref][amask[pi[ref]]]
     row={'Scheme':scheme,'Stage':stage,'Budget_pct':budget,'Ref_ID':ref,'N_eval':len(ix),'N_calibration':int(np.sum(~amask)) if ref=='Overall' else len(chosen[ref]),**scored(y[ix],p[ix],ap[ix])}
     (activeown if ref=='Overall' else ownpub).append(row)
    for rep in range(1,NREPS+1):
     mask=np.ones(64,dtype=bool);rp=p.copy();ridx_all=[]
     for ref in REFS:
      ri=perm_map[(rep,ref)][:len(chosen[ref])];ridx_all.extend(ri.tolist())
      mask[ri]=False;rp[pi[ref]]+=float(np.mean(y[ri]-p[ri]))
     rim=np.ones(64,dtype=bool);rim[ridx_all]=False
     for ref in ['Overall']+REFS:
      ix=np.flatnonzero(rim) if ref=='Overall' else pi[ref][rim[pi[ref]]]
      row={'Scheme':scheme,'Stage':stage,'Budget_pct':budget,'Replicate':rep,'Ref_ID':ref,'N_eval':len(ix),**metric(y[ix],rp[ix])}
      (random_reps if ref=='Overall' else randpub).append(row)
     mask[~amask]=False # strict common evaluation: neither active nor random calibration labels
     for ref in ['Overall']+REFS:
      ix=np.flatnonzero(mask) if ref=='Overall' else pi[ref][mask[pi[ref]]]
      row={'Scheme':scheme,'Stage':stage,'Budget_pct':budget,'Replicate':rep,'Ref_ID':ref,'N_common_eval':len(ix),**scored(y[ix],p[ix],ap[ix],rp[ix])}
      (paired if ref=='Overall' else pairedpub).append(row)
   print('Completed',scheme,'stage',stage,flush=True)

 # consistency between pooled paired score and raw metrics
 assert len(paired)==8*4*500 and len(pairedpub)==8*4*500*3
 assert len(random_reps)==len(paired) and len(randpub)==len(pairedpub)
 # construct consistent random-only summary
 def simple_summ(rows):
  out=[]
  for sch in SCHEMES:
   for st in STAGES:
    for bud in BUDGETS:
     for ref in ['Overall']+REFS:
      rs=[z for z in rows if z['Scheme']==sch and z['Stage']==st and z['Budget_pct']==bud and z['Ref_ID']==ref]
      assert len(rs)==500
      obj={'Scheme':sch,'Stage':st,'Budget_pct':bud,'Ref_ID':ref,'N_eval':rs[0]['N_eval']}
      for m in ['R2','RMSE','MAE']:
       arr=np.array([z[m] for z in rs]);obj[m+'_median']=float(np.median(arr));obj[m+'_q05']=float(np.quantile(arr,.05));obj[m+'_q95']=float(np.quantile(arr,.95))
      out.append(obj)
  return out
 manifest={'title':'Figure 4 — Line3 LightGBM, Ref28–30, random and stage-matched active few-shot','n':64,'publications':{str(r):len(ix) for r,ix in pi.items()},'scheme':SCHEMES,'stages':STAGES,'budgets_percent':BUDGETS,'repeat_count_random':500,'target':'ln(1 + DBP concentration [µg/L])','model':'Frozen 5-fold-averaged Line 3 LightGBM external prediction; no retraining','random_sampling':'Original Line 1 500 random permutations, shared across Stage 1–4 and both CV schemes','active_sampling':'Centroid first, greedy Maximin thereafter, using stage-matched descriptors and train-only preprocessing; label blind','selection_features':{str(s):len(features[str(s)]['LightGBM']) for s in STAGES},'mean_offset':'mean(y_true_ln1p - zero_shot_ln1p) within target publication on selected shots','comparators':'Random own remainder vs zero, Active own remainder vs zero; fair Active-vs-Random vs zero on common remainder excluding both calibration sets','data_quality':'Ref30 similarity to original Ref27 requires source-identity confirmation before claim of publication independent external validation','calibration_k':{str(b):{str(r):bud_k(len(pi[r]),b) for r in REFS} for b in BUDGETS}}
 files={'ZeroShot_Baseline.csv':zero,'Feature_Manifest.csv':feat_mf,'Active_Complete_Sequences.csv':seq,'Active_Budget_Selections.csv':sel,'Active_Offsets.csv':calibrations,
        'Random_Own_Replicates.csv':random_reps,'Random_Own_Summary.csv':simple_summ(random_reps+randpub),
        'Active_Own_Overall.csv':activeown,'Active_Own_Publications.csv':ownpub,
        'Matched_Overall_Replicates.csv':paired,'Matched_Publication_Replicates.csv':pairedpub,
        'Matched_Overall_Summary.csv':summarize(paired),'Matched_Publication_Summary.csv':summarize(pairedpub,True)}
 for name,rows in files.items():writecsv(OUT/name,rows)
 manifest['file_checksums_sha256']={k:hashlib.sha256((OUT/k).read_bytes()).hexdigest() for k in files}
 (OUT/'Experiment_Manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print('OUTPUT SUMMARY')
 for row in files['Matched_Overall_Summary.csv']:
  print(row['Scheme'],row['Stage'],row['Budget_pct'], 'zero',round(row['Zero_R2_median'],3),'random',round(row['Random_R2_median'],3),'active',round(row['Active_R2_median'],3),'paired',round(row['Paired_dR2_median'],3))
 print('Finished all models, results in',OUT,flush=True)

if __name__=='__main__':main()
