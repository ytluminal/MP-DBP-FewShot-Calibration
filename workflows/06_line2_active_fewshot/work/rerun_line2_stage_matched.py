#!/usr/bin/env python3
"""Line 2 stage/model-matched Centroid + Maximin active few-shot, frozen baselines.
No model refit. Only acquisition features changed; mean-offset and matched test evaluation unchanged.
"""
from __future__ import annotations
import sys,csv,json,math,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_active_fewshot as origin
BUDGETS=origin.BUDGETS; MODELS=origin.MODELS; PROTOCOLS=origin.PROTOCOLS
CAT_NAMES=origin.CAT_NAMES
INPUT=HERE/'inputs/line2'; OUT=HERE.parent/'results'; OUT.mkdir(parents=True,exist_ok=True)


def preprocess_subset(train,test,feature_names):
    heads=train['headers']; assert heads==test['headers']
    assert len(feature_names)==len(set(feature_names))
    assert set(feature_names) <= set(heads[:22]+heads[24:])
    j_num=[heads.index(h) for h in feature_names if h not in CAT_NAMES]
    j_cat=[heads.index(h) for h in CAT_NAMES if h in feature_names]
    a=np.asarray([[origin.numeric(row[j]) for j in j_num] for row in train['rows']],dtype=float)
    b=np.asarray([[origin.numeric(row[j]) for j in j_num] for row in test['rows']],dtype=float)
    a[~np.isfinite(a)]=np.nan;b[~np.isfinite(b)]=np.nan
    med=np.asarray([np.median(q[np.isfinite(q)]) if np.isfinite(q).any() else 0. for q in a.T])
    a=np.where(np.isfinite(a),a,med);b=np.where(np.isfinite(b),b,med)
    ss=StandardScaler().fit(a);b=ss.transform(b)
    if j_cat:
        def clean(v):return '__MISSING__' if v is None or str(v).strip()=='' else str(v)
        ac=np.asarray([[clean(row[j]) for j in j_cat] for row in train['rows']],dtype=object)
        bc=np.asarray([[clean(row[j]) for j in j_cat] for row in test['rows']],dtype=object)
        for k in range(len(j_cat)):
            valid=[v for v in ac[:,k] if v!='__MISSING__']
            if valid:
                f=Counter(valid); mode=sorted(f,key=lambda x:(-f[x],x))[0]
            else: mode='__MISSING__'
            ac[ac[:,k]=='__MISSING__',k]=mode;bc[bc[:,k]=='__MISSING__',k]=mode
        onehot=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=float)
        onehot.fit(ac); b=np.hstack((b,onehot.transform(bc)))
    assert np.isfinite(b).all()
    return b,{'features_raw':len(feature_names),'features_numeric':len(j_num),'features_categorical':len(j_cat),'features_encoded':b.shape[1], 'feature_names':list(feature_names), 'distance':'Euclidean after Training-median numeric imputation+StandardScaler, categorical Training-most-frequent+OHE'}

def summary(rows, keys, column_names):
    d={k:rows[0][k] for k in keys}
    for c in column_names:
        vv=np.asarray([v[c] for v in rows],dtype=float)
        vv=vv[np.isfinite(vv)]
        d[c+'_median']=float(np.median(vv)) if len(vv) else None
        d[c+'_q05']=float(np.quantile(vv,.05)) if len(vv) else None
        d[c+'_q95']=float(np.quantile(vv,.95)) if len(vv) else None
    if 'Paired_dR2' in rows[0]:
        d['Fraction_Active_Better_R2']=float(np.mean([float(v['Paired_dR2'])>0 for v in rows if np.isfinite(v['Paired_dR2'])]))
    d['N_replicates']=len(rows)
    return d

def writer(path,headers):
    f=path.open('w',newline='',encoding='utf-8-sig'); w=csv.DictWriter(f,fieldnames=headers);w.writeheader()
    return f,w


def execute():
    origin.verify_input_hashes()
    train=origin.load_json(INPUT/'train55.json');test=origin.load_json(INPUT/'test55.json')
    assert len(train['rows'])==995 and len(test['rows'])==249
    y,refs,preds=origin.load_predictions_line2(INPUT,train,test)
    reference_ids=list(map(int,np.unique(refs)))
    ref_idx={ref:np.flatnonzero(refs==ref) for ref in reference_ids}
    perms=origin.permutation_map(INPUT/'Random_Permutations_500.csv',ref_idx,500)
    feature_map=origin.load_json(HERE/'stage_matched_feature_lists.json')['stage_features']

    active_orders={}; selection_feature_manifest=[]
    for stage in range(1,5):
        for model in MODELS:
            feats=feature_map[str(stage)][model]
            z,info=preprocess_subset(train,test,feats)
            selection_feature_manifest.append(dict(Stage=stage,Model=model,**info))
            active_orders[(stage,model)]={ref:origin.centroid_maximin(z,inds) for ref,inds in ref_idx.items()}
    # Both Stage 2 & 3 use the exact same 55 features and are unchanged from the original procedure.
    for mod in MODELS:
        for ref in reference_ids:
            assert np.array_equal(active_orders[(2,mod)][ref],active_orders[(3,mod)][ref])
    old=origin.read_rows(HERE/'old_results/results/Line2_Active_Budget_Selections.csv')
    old_key={(int(r['Ref_ID']),float(r['Budget_pct'])):set() for r in old}
    for r in old:old_key[(int(r['Ref_ID']),float(r['Budget_pct']))].add(int(r['Test_row_1based'])-1)
    for stage in (2,3):
        for model in MODELS:
            for ref in reference_ids:
                for budget in BUDGETS:
                    k=origin.count_budgets(len(ref_idx[ref]))[budget]
                    assert set(active_orders[(stage,model)][ref][:k].tolist())==old_key[(ref,budget)], (stage,model,ref,budget)
    print('AUDIT: Stage 2/3 all selection sets exactly match historical 55-feature Centroid+Maximin.',flush=True)
    ord_hdr=['Stage','Model','Ref_ID','Acquisition_rank','Test_row_1based','Features_raw']
    fsel,wsel=writer(OUT/'Active_Complete_Sequences.csv',ord_hdr)
    bud_hdr=['Stage','Model','Budget_pct','Ref_ID','Acquisition_rank','Test_row_1based','Features_raw']
    fbud,wbud=writer(OUT/'Active_Budget_Selections.csv',bud_hdr)
    for (st,mod),ord_ref in active_orders.items():
        for ref,order in ord_ref.items():
            for rank,row in enumerate(order,1):
                wsel.writerow(dict(Stage=st,Model=mod,Ref_ID=ref,Acquisition_rank=rank,Test_row_1based=int(row)+1,Features_raw=len(feature_map[str(st)][mod])))
                for budget in BUDGETS:
                    if rank<=origin.count_budgets(len(ref_idx[ref]))[budget]:
                        wbud.writerow(dict(Stage=st,Model=mod,Budget_pct=budget,Ref_ID=ref,Acquisition_rank=rank,Test_row_1based=int(row)+1,Features_raw=len(feature_map[str(st)][mod])))
    fsel.close();fbud.close()

    metric_cols=['Zero_R2','Active_R2','Random_R2','Zero_RMSE','Active_RMSE','Random_RMSE',
                 'Zero_MAE','Active_MAE','Random_MAE','Paired_dR2','Paired_dRMSE','Paired_dMAE',
                 'Active_vs_Zero_dR2','Random_vs_Zero_dR2']
    base=['Protocol','Stage','Model','Budget_pct','Replicate','N_common_eval','N_active_cal','N_random_cal']
    fc,wc=writer(OUT/'All_80000_Paired_Replicates.csv',base+metric_cols)
    pc=['Protocol','Stage','Model','Budget_pct','Replicate','Ref_ID','N_common_eval']
    fp,wp=writer(OUT/'All_400000_PerPublication_Replicates.csv',pc+metric_cols)
    summaryrows=[];perpubrows=[];ownrows=[]
    for proto in PROTOCOLS:
        for stage in range(1,5):
            for model in MODELS:
                phat=preds[(proto,stage,model)]
                zero_full=origin.metrics(y,phat)
                for budget in BUDGETS:
                    active=active_orders[(stage,model)]
                    selected={ref:active[ref][:origin.count_budgets(len(ref_idx[ref]))[budget]] for ref in reference_ids}
                    mask_active=np.ones(len(y),dtype=bool)
                    active_predict=np.empty(len(y),dtype=float)
                    for ref,idx in ref_idx.items():
                        cal=selected[ref];mask_active[cal]=False
                        offset=float(np.mean(y[cal]-phat[cal]));active_predict[idx]=phat[idx]+offset
                    am=origin.metrics(y[mask_active],active_predict[mask_active]);az=origin.metrics(y[mask_active],phat[mask_active]);
                    ownrows.append(dict(Protocol=proto,Stage=stage,Model=model,Budget_pct=budget,Features_raw=len(feature_map[str(stage)][model]),N_active_cal=int((~mask_active).sum()),N_eval=int(mask_active.sum()),Zero_Full_R2=zero_full['R2'],Zero_Own_R2=az['R2'],Active_R2=am['R2'],Active_RMSE=am['RMSE'],Active_MAE=am['MAE'],Active_vs_Zero_R2=am['R2']-az['R2']))
                    cases=[];pubcases={ref:[] for ref in reference_ids}
                    for rep in range(1,501):
                        mask=np.ones(len(y),dtype=bool)
                        random_predict=np.empty(len(y),dtype=float)
                        for ref,idx in ref_idx.items():
                            ridx=perms[(rep,ref)][:len(selected[ref])]
                            aidx=selected[ref]
                            mask[np.union1d(aidx,ridx)]=False
                            random_predict[idx]=phat[idx]+float(np.mean(y[ridx]-phat[ridx]))
                        zero=origin.metrics(y[mask],phat[mask])
                        am=origin.metrics(y[mask],active_predict[mask])
                        rm=origin.metrics(y[mask],random_predict[mask])
                        row=dict(Protocol=proto,Stage=stage,Model=model,Budget_pct=budget,Replicate=rep,N_common_eval=int(mask.sum()),N_active_cal=int((~mask_active).sum()),N_random_cal=int((~mask_active).sum()))
                        for prefix,vals in [('Zero_',zero),('Active_',am),('Random_',rm)]:
                            for metric,val in vals.items(): row[prefix+metric]=val
                        row.update(Paired_dR2=am['R2']-rm['R2'],Paired_dRMSE=am['RMSE']-rm['RMSE'],Paired_dMAE=am['MAE']-rm['MAE'],Active_vs_Zero_dR2=am['R2']-zero['R2'],Random_vs_Zero_dR2=rm['R2']-zero['R2'])
                        wc.writerow(row);cases.append(row)
                        for ref,idx in ref_idx.items():
                            ri=idx[mask[idx]]
                            if len(ri)==0: continue
                            zz=origin.metrics(y[ri],phat[ri]); aa=origin.metrics(y[ri],active_predict[ri]); rr=origin.metrics(y[ri],random_predict[ri]);
                            pr=dict(Protocol=proto,Stage=stage,Model=model,Budget_pct=budget,Replicate=rep,Ref_ID=ref,N_common_eval=len(ri))
                            for prefix,vals in [('Zero_',zz),('Active_',aa),('Random_',rr)]:
                                for metric,val in vals.items():pr[prefix+metric]=val
                            pr.update(Paired_dR2=aa['R2']-rr['R2'],Paired_dRMSE=aa['RMSE']-rr['RMSE'],Paired_dMAE=aa['MAE']-rr['MAE'],Active_vs_Zero_dR2=aa['R2']-zz['R2'],Random_vs_Zero_dR2=rr['R2']-zz['R2'])
                            wp.writerow(pr);pubcases[ref].append(pr)
                    summ=summary(cases,['Protocol','Stage','Model','Budget_pct'],['N_common_eval']+metric_cols)
                    summ['Active_Own_R2']=ownrows[-1]['Active_R2'];summ['Active_Own_RMSE']=ownrows[-1]['Active_RMSE'];summ['Active_Own_MAE']=ownrows[-1]['Active_MAE'];summ['Zero_Full_R2']=zero_full['R2'];summ['Features_raw']=len(feature_map[str(stage)][model])
                    summaryrows.append(summ)
                    for ref,rr in pubcases.items():
                        if rr:
                            r=summary(rr,['Protocol','Stage','Model','Budget_pct','Ref_ID'],['N_common_eval']+metric_cols)
                            perpubrows.append(r)
            print('FINISHED',proto,'Stage',stage,'summary so far',len(summaryrows),flush=True)
    fc.close();fp.close()
    assert len(summaryrows)==160 and len(ownrows)==160 and len(perpubrows)==800
    origin.write_rows(OUT/'All_160_Configurations_Summary.csv',summaryrows)
    origin.write_rows(OUT/'All_160_Active_OwnRemainder.csv',ownrows)
    origin.write_rows(OUT/'All_800_PerPublication_Summary.csv',perpubrows)
    origin.write_rows(OUT/'Stage_Model_Feature_Manifest.csv',[dict(Stage=int(st),Model=m,Num_Features=len(q),Feature_Names='; '.join(q)) for st,v in feature_map.items() for m,q in v.items()])
    # rigorous numeric reproduction when Stage2/Stage3 feature sets are unchanged
    original=origin.read_rows(HERE/'old_results/results/Line2_ActiveVsRandom_Matched_Summary.csv')
    omap={(r['Protocol'],int(r['Stage']),r['Model'],float(r['Budget_pct'])):r for r in original}
    for r in summaryrows:
        if r['Stage'] in (2,3):
            ref=omap[(r['Protocol'],r['Stage'],r['Model'],r['Budget_pct'])]
            assert abs(r['Paired_dR2_median']-float(ref['active_minus_random_R2_median']))<1e-12,(r,ref)
            assert abs(r['Active_R2_median']-float(ref['common_active_R2_median']))<1e-12,(r,ref)
    print('AUDIT: all 80 Stage 2/3 configurations exactly reproduced historic matched medians.',flush=True)
    # Double-check random comparator results ALL configurations against old matched (old sample intersection may differ for changed selections).
    audit={'line':2,'train_rows':995,'test_rows':249,'test_publications':reference_ids,'configs':160,'paired_replicates':80000,'perpub_replicates':400000,'active_selection':'Centroid then greedy Maximin, Euclidean in stage+model matched input feature columns','calibration':'Mean residual within each target publication, same as old version','random_permutations':'Frozen 500 sequences per publication, unchanged','stage2_3_historical_reproduction':'Exact 80 configurations (paired R2 medians within 1e-12)','source_feature_map':'SI_Verified.docx RFE-selected stage4 features in original Figure 2b reproduction','test_leakage':'All preprocessors fit on Training only; selection uses only Test covariates (unlabeled) and Test predictions; Test labels used only for calibration samples and held-out scoring','features':selection_feature_manifest,'files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file()}}
    (OUT/'Audit_Manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('ALL COMPLETE',flush=True)

if __name__=='__main__':execute()
