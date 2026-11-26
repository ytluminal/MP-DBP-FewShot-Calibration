#!/usr/bin/env python3
"""Figure 4 Line 1 LightGBM — stage-matched Centroid+Maximin active few-shot.

Inputs: Line1 Stage2 Training 55-feature XLSX, frozen 64 external predictions,
new external features JSON, and original Line1 500 random permutations.
No refitting, no target label use in sample acquisition, no performance-guided selection.
"""
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json, csv, math, hashlib
from artifact_tool import Blob, SpreadsheetFile

ROOT=Path('/mnt/data')
SRC=ROOT/'line1_fewshot_results'
WORK=ROOT/'line1_active_work'
OUT=WORK/'results'; OUT.mkdir(parents=True,exist_ok=True)
TRAIN=WORK/'Conventional_RandomSplit_Stage2_Training_55features(3).xlsx'
TEST=ROOT/'line12_work/external_55.json'
PRED=SRC/'Frozen_Line1_External_Predictions_64.csv'
PERM=SRC/'Random_Permutations.csv'
S3INFO=WORK/'line1_S3_LightGBM.json'
# When extracted from the self-contained ZIP, automatically read bundled inputs.
BUNDLE=Path(__file__).resolve().parent.parent
if (BUNDLE/'input/Frozen_Line1_External_Predictions_64.csv').exists():
    TRAIN=BUNDLE/'input/Conventional_RandomSplit_Stage2_Training_55features.xlsx'
    TEST=BUNDLE/'input/Ref28_29_30_external_55features.json'
    PRED=BUNDLE/'input/Frozen_Line1_External_Predictions_64.csv'
    PERM=BUNDLE/'input/Original_Line1_Random_Permutations_500.csv'
    S3INFO=BUNDLE/'input/Line1_S3_LightGBM_RFE_50features.json'
    OUT=BUNDLE/'outputs_replay'
    OUT.mkdir(parents=True,exist_ok=True)
CATS=['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']
BUDGETS=[2.5,5.,10.,20.]
REFS=[28,29,30]
NREP=500

def readcsv(p):
 with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p,recs):
 if not recs:return
 with open(p,'w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(recs[0]));w.writeheader();w.writerows(recs)
def metric(a,b):
 a=np.asarray(a);b=np.asarray(b);e=a-b
 ss=float(np.sum((a-a.mean())**2)); return {'R2':float(1-np.sum(e*e)/ss) if len(a)>1 and ss>1e-14 else float('nan'),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e)))}
def score_row(ys,z,a,r):
 d={};
 for k,p in [('Zero',z),('Active',a),('Random',r)]:
  for met,v in metric(ys,p).items():d[k+'_'+met]=v
 for m in ('R2','RMSE','MAE'):
  d['Paired_d'+m]=d['Active_'+m]-d['Random_'+m]
  d['Active_vs_Zero_d'+m]=d['Active_'+m]-d['Zero_'+m]
  d['Random_vs_Zero_d'+m]=d['Random_'+m]-d['Zero_'+m]
 return d

def gettrain():
 w=SpreadsheetFile.import_xlsx(Blob.load(str(TRAIN)))
 rows=w.worksheets.get_item('Data').get_range('A1:BE996').values
 hdr=rows[0]; assert len(rows)==996 and len(hdr)==57, (len(rows),len(hdr));
 assert len(rows[1:])==995 and hdr[22].startswith('DBP concentration') and hdr[23]=='ref ID'
 return hdr,[dict(zip(hdr,row)) for row in rows[1:]]
headers,tr=gettrain()
tejson=json.loads(TEST.read_text()); assert headers==tejson['columns'], ('Header mismatch',[(a,b) for a,b in zip(headers,tejson['columns']) if a!=b])
te=[dict(zip(headers,row)) for row in tejson['rows']]
assert len(te)==64
ps=readcsv(PRED);assert len(ps)==64
references=np.array([int(p['Ref_ID']) for p in ps]);truth=np.array([float(p['Observed_ln1p']) for p in ps]);
for i,p in enumerate(ps):
 assert int(p['Source_row'])==i+2
 assert int(te[i]['ref ID'])==references[i]
 assert te[i]['DBP name']==p['DBP_name']
 assert math.isclose(float(te[i]['DBP concentration (µg/L)']),float(p['Observed_ugL']),rel_tol=1e-10,abs_tol=1e-11)
assert {ref:int((references==ref).sum()) for ref in REFS}=={28:12,29:12,30:40}
pred={s:np.array([float(p[f'Stage{s}_pred_ln1p']) for p in ps]) for s in (1,2,3)}
all55=[x for i,x in enumerate(headers) if i not in (22,23)]
assert len(all55)==55
feature_sets={1:all55[:22],2:all55,3:json.loads(S3INFO.read_text())['features']}
assert [len(feature_sets[x]) for x in (1,2,3)]==[22,55,50]
assert all(set(f).issubset(set(all55)) for f in feature_sets.values())

# Preprocessing mirrors the supplied stage-matched selection protocol, fitted on
# this LINE 1's original 995-row TRAINING only, not on any Ref28–30 covariates.
def encode_features(feats):
 num=[f for f in feats if f not in CATS];cats=[f for f in feats if f in CATS]
 def num_val(x):
  if x is None or (isinstance(x,str) and not x.strip()):return float('nan')
  try:return float(x)
  except (TypeError,ValueError):return float('nan')
 ta=np.array([[num_val(row[f]) for f in num] for row in tr],dtype=float)
 ea=np.array([[num_val(row[f]) for f in num] for row in te],dtype=float)
 ta[~np.isfinite(ta)]=np.nan;ea[~np.isfinite(ea)]=np.nan
 m=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in ta.T])
 ta=np.where(np.isfinite(ta),ta,m);ea=np.where(np.isfinite(ea),ea,m)
 scaler=StandardScaler().fit(ta);numeric=scaler.transform(ea)
 def cv(x):return '__MISSING__' if x is None or (isinstance(x,str) and not x.strip()) else str(x)
 ca=np.array([[cv(row[f]) for f in cats] for row in tr],dtype=object)
 cb=np.array([[cv(row[f]) for f in cats] for row in te],dtype=object)
 for j in range(len(cats)):
  vals=[v for v in ca[:,j] if v!='__MISSING__']
  if vals:
   counter=Counter(vals);mode=sorted(counter,key=lambda t:(-counter[t],t))[0]
  else:mode='__MISSING__'
  ca[ca[:,j]=='__MISSING__',j]=mode;cb[cb[:,j]=='__MISSING__',j]=mode
 enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=float)
 enc.fit(ca); categorical=enc.transform(cb)
 return np.hstack([numeric,categorical]), {'raw_features':len(feats),'numeric_features':len(num),'categorical_features':len(cats),'encoded_features':int(numeric.shape[1]+categorical.shape[1]),'feature_names':feats}

def centroid_maximin(z,idx):
 idx=np.asarray(idx,dtype=int);zz=z[idx];centroid=zz.mean(axis=0)
 d=np.einsum('ij,ij->i',zz-centroid,zz-centroid)
 first=int(np.argmin(d));chosen=[first];used=np.zeros(len(idx),dtype=bool);used[first]=True
 near=np.einsum('ij,ij->i',zz-zz[first],zz-zz[first])
 while len(chosen)<len(idx):
  q=near.copy();q[used]=-1.;nxt=int(np.argmax(q))
  chosen.append(nxt);used[nxt]=True
  new=np.einsum('ij,ij->i',zz-zz[nxt],zz-zz[nxt]);near=np.minimum(near,new)
 return idx[np.array(chosen)]

pub_indices={r:np.flatnonzero(references==r) for r in REFS}
active={};manifests=[];sequences=[];selected_rows=[]
for stage in (1,2,3):
 encoded,mf=encode_features(feature_sets[stage]);mf['Stage']=stage;manifests.append(mf)
 active[stage]={ref:centroid_maximin(encoded,ix) for ref,ix in pub_indices.items()}
 for ref,seq in active[stage].items():
  assert sorted(seq.tolist())==sorted(pub_indices[ref].tolist())
  for rank,ind in enumerate(seq,1):
   sequences.append({'Stage':stage,'Ref_ID':ref,'Acquisition_rank':rank,'Test_row_1based':int(ind)+1,'Source_row':int(ps[ind]['Source_row']),'DBP_name':ps[ind]['DBP_name'],'Polymer':ps[ind]['Polymer'],'Selected_via':'Centroid' if rank==1 else 'Maximin'})
   for pct in BUDGETS:
    n=min(len(seq)-1,max(1,math.ceil(pct*len(seq)/100)))
    if rank<=n:
     selected_rows.append({'Stage':stage,'Budget_pct':pct,'Ref_ID':ref,'Acquisition_rank':rank,'Test_row_1based':int(ind)+1,'Source_row':int(ps[ind]['Source_row']),'DBP_name':ps[ind]['DBP_name'],'Polymer':ps[ind]['Polymer']})

rr=readcsv(PERM);permutations={}
for row in rr:
 rep=int(row['Replicate']);ref=int(row['Ref_ID']);order=np.array([int(x) for x in row['Ordered_ZeroBased_Indices'].split(';')],dtype=int)
 assert sorted(order.tolist())==sorted(pub_indices[ref].tolist())
 permutations[(rep,ref)]=order
assert len(permutations)==1500

baseline=[]
for stage in (1,2,3):
 for ref in ['Overall']+REFS:
  idx=np.arange(64) if ref=='Overall' else pub_indices[ref]
  baseline.append({'Stage':stage,'Ref_ID':ref,'N':len(idx),**metric(truth[idx],pred[stage][idx])})

paired=[];perpub=[];own=[];ownpub=[]
for stage in (1,2,3):
 p=pred[stage]
 for pct in BUDGETS:
  selected={ref:active[stage][ref][:min(len(pub_indices[ref])-1,max(1,math.ceil(pct*len(pub_indices[ref])/100)))] for ref in REFS}
  amask=np.ones(64,dtype=bool)
  activepred=p.copy()
  for ref in REFS:
   v=selected[ref];amask[v]=False
   activepred[pub_indices[ref]]+=np.mean(truth[v]-p[v])
  for ref in ['Overall']+REFS:
   ii=np.flatnonzero(amask) if ref=='Overall' else pub_indices[ref][amask[pub_indices[ref]]]
   cur={'Stage':stage,'Budget_pct':pct,'Ref_ID':ref,'N_active_cal':int((~amask).sum()) if ref=='Overall' else len(selected[ref]),'N_eval':len(ii)}
   for name,pv in [('Zero',p),('Active',activepred)]:
    cur.update({f'{name}_{key}':value for key,value in metric(truth[ii],pv[ii]).items()})
   for m in ['R2','RMSE','MAE']:cur['Active_vs_Zero_d'+m]=cur['Active_'+m]-cur['Zero_'+m]
   if ref=='Overall':own.append(cur)
   else:ownpub.append(cur)
  for rep in range(1,NREP+1):
   randpred=p.copy();rmask=amask.copy()
   for ref in REFS:
    inds=permutations[(rep,ref)][:len(selected[ref])]
    rmask[inds]=False
    randpred[pub_indices[ref]]+=np.mean(truth[inds]-p[inds])
   common=np.flatnonzero(rmask)
   row={'Stage':stage,'Budget_pct':pct,'Replicate':rep,'N_common_eval':len(common),'N_active_cal':int((~amask).sum()),'N_random_cal':int((~amask).sum())}
   row.update(score_row(truth[common],p[common],activepred[common],randpred[common]))
   paired.append(row)
   for ref in REFS:
    ix=pub_indices[ref][rmask[pub_indices[ref]]]
    perpub.append({'Stage':stage,'Budget_pct':pct,'Replicate':rep,'Ref_ID':ref,'N_common_eval':len(ix),**score_row(truth[ix],p[ix],activepred[ix],randpred[ix])})
 print('Completed Stage',stage,flush=True)

measure=['N_common_eval']+[name+'_'+m for name in ['Zero','Active','Random'] for m in ['R2','RMSE','MAE']]+['Paired_d'+m for m in ['R2','RMSE','MAE']]+[f'{typ}_vs_Zero_d{m}' for typ in ['Active','Random'] for m in ['R2','RMSE','MAE']]
def summary(rows,perpublication=False):
 out=[]
 for stage in (1,2,3):
  for pct in BUDGETS:
   for ref in (REFS if perpublication else ['Overall']):
    sel=[row for row in rows if row['Stage']==stage and row['Budget_pct']==pct and (not perpublication or row['Ref_ID']==ref)]
    q={'Stage':stage,'Budget_pct':pct,'Ref_ID':ref,'N_replicates':len(sel),'N_active_cal':min(len(pub_indices[ref])-1,max(1,math.ceil(pct*len(pub_indices[ref])/100))) if perpublication else sum(min(len(pub_indices[r])-1,max(1,math.ceil(pct*len(pub_indices[r])/100))) for r in REFS)}
    for key in measure:
     ar=np.array([float(row[key]) for row in sel]);v=ar[np.isfinite(ar)]
     for lab,per in [('median',.5),('q05',.05),('q25',.25),('q75',.75),('q95',.95)]:q[key+'_'+lab]=float(np.quantile(v,per)) if len(v) else float('nan')
    for met in ['R2','RMSE','MAE']:
     vv=np.array([row['Paired_d'+met] for row in sel])
     q['Fraction_active_better_'+met]=float(np.nanmean(vv>0)) if met=='R2' else float(np.nanmean(vv<0))
    out.append(q)
 return out
sumoverall=summary(paired);sumperpub=summary(perpub,True)
assert len(paired)==6000 and len(perpub)==18000 and len(sumoverall)==12 and len(sumperpub)==36
for name,entries in [('ZeroShot_Baseline.csv',baseline),('Active_Complete_Sequences.csv',sequences),('Active_Budget_Selections.csv',selected_rows),('Active_Own_Remainder.csv',own),('Active_Own_ByPublication.csv',ownpub),('Matched_Overall_Replicates.csv',paired),('Matched_Publication_Replicates.csv',perpub),('Matched_Overall_Summary.csv',sumoverall),('Matched_Publication_Summary.csv',sumperpub)]:writecsv(OUT/name,entries)
manifest={'Research':'Figure4 Line1 LightGBM Stage1–3 on Ref28–30 active vs original random few-shot','n':64,'refs':{str(k):len(v) for k,v in pub_indices.items()},'original_train_count':len(tr),'stages':manifests,'budgets_pct':BUDGETS,'replicates':NREP,'selected_counts':{str(pct):{str(ref):min(len(ix)-1,max(1,math.ceil(pct*len(ix)/100))) for ref,ix in pub_indices.items()} for pct in BUDGETS},'selection_method':'Centroid nearest then greedy Euclidean maximin with first-observation tie breaker; feature preprocess numeric train-median/StandardScaler, categorical train-mode/OneHotEncoder; stage-matched 22/55/50 raw features; Test y not used to select','prediction_source':'frozen Line1 full-training LightGBM Stage 1,2,3 zero-shot predictions','calibration':'each Ref mean log1p target minus zero-shot predicted on acquisition subset; no retraining','comparison':'active/random/zero evaluated on same remaining intersection after excluding both acquisition sets; original 500 random permutations unchanged','target':'ln(1 + DBP concentration [microgram per L])','warning':'within-publication few-shot adaptation, not fully zero-label external validation; publication 30 potentially overlaps 27 in chemistry/source and requires provenance check; candidate points are DBP-specific records and may not correspond to independent lab experiments','SHA256':{str(k.name):hashlib.sha256(k.read_bytes()).hexdigest() for k in [TRAIN,TEST,PRED,PERM,S3INFO]},'csv_row_counts':{'matched_overall':len(paired),'matched_publication':len(perpub),'summary_overall':len(sumoverall),'summary_publication':len(sumperpub)}}
(OUT/'Experiment_Manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print('ENCODED DIMENSIONS',[(v['Stage'],v['encoded_features']) for v in manifests])
print('MAIN RESULT:')
for s in sumoverall:
 print(s['Stage'],s['Budget_pct'],'Own=',round(next(o['Active_R2'] for o in own if o['Stage']==s['Stage'] and o['Budget_pct']==s['Budget_pct']),4),'Matched Active',round(s['Active_R2_median'],4),'Random',round(s['Random_R2_median'],4),'Paired Δ',round(s['Paired_dR2_median'],4),'fracA',round(s['Fraction_active_better_R2'],3),'RMSE',round(s['Active_RMSE_median'],4))
