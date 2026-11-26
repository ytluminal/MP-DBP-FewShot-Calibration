#!/usr/bin/env python3
"""Executable, data-contained Centroid + Maximin Active Few-Shot handoff.

Requires: numpy, scikit-learn. Python standard library for all CSV/JSON handling.

Run (full): python run_active_fewshot.py --line both --out outputs
Run (quick verification): python run_active_fewshot.py --line both --smoke --out smoke_outputs

No model training, feature tuning, or Test-label-driven sample selection.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, os
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent
BUDGETS = [2.5, 5.0, 10.0, 20.0]
MODELS = ["RF", "XGBoost", "LightGBM", "CatBoost", "Equal4"]
PROTOCOLS = ["Publication-grouped", "Random folds"]
CAT_NAMES = ["polymer type", "additives", "pretreatment type", "sample phase",
             "buffer", "disinfectant", "quenching agent", "DBP name"]


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not header:
        return
    header = header if header else list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def count_budgets(n):
    return {b: min(n - 1, max(1, int(math.ceil(n * b / 100.0)))) for b in BUDGETS}


def numeric(x):
    if x is None or str(x).strip() == '':
        return float('nan')
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def prepare_features(train, test):
    """55 raw columns, training-only numeric median/scaling and categorical mode/OHE."""
    headers = train['headers']
    assert headers == test['headers'], 'Mismatched 55-feature schema'
    assert headers[22].startswith('DBP concentration') and headers[23] == 'ref ID'
    feats = [j for j in range(len(headers)) if j not in (22, 23)]
    assert len(feats) == 55, len(feats)
    cat_j = [headers.index(x) for x in CAT_NAMES]
    num_j = [j for j in feats if j not in cat_j]
    assert len(cat_j) == 8 and len(num_j) == 47

    tr = train['rows']; te = test['rows']; ntr = len(tr); nte = len(te)
    nt = np.array([[numeric(r[j]) for j in num_j] for r in tr], dtype=float)
    ne = np.array([[numeric(r[j]) for j in num_j] for r in te], dtype=float)
    nt[~np.isfinite(nt)] = np.nan
    ne[~np.isfinite(ne)] = np.nan
    # A feature entirely missing from Training is set to 0, never inferred from Test.
    impute = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
                       for c in nt.T], dtype=float)
    nt = np.where(np.isfinite(nt), nt, impute)
    ne = np.where(np.isfinite(ne), ne, impute)
    scaler = StandardScaler().fit(nt)
    nte_features = scaler.transform(ne)
    ntr_features = scaler.transform(nt)

    def catval(value):
        return '__MISSING__' if value is None or str(value).strip() == '' else str(value)

    ct = np.array([[catval(r[j]) for j in cat_j] for r in tr], dtype=object)
    ce = np.array([[catval(r[j]) for j in cat_j] for r in te], dtype=object)
    for k in range(len(cat_j)):
        vals = [v for v in ct[:, k] if v != '__MISSING__']
        if vals:
            freq = Counter(vals)
            mode = sorted(freq, key=lambda x: (-freq[x], x))[0]
        else:
            mode = '__MISSING__'
        ct[ct[:, k] == '__MISSING__', k] = mode
        ce[ce[:, k] == '__MISSING__', k] = mode
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=float)
    ctr = onehot.fit_transform(ct)
    cte = onehot.transform(ce)
    return np.hstack((ntr_features, ctr)), np.hstack((nte_features, cte)), {
        'n_raw_features': len(feats), 'n_categorical': len(cat_j),
        'n_numeric': len(num_j), 'n_encoded_features': int(nte_features.shape[1] + cte.shape[1]),
        'feature_names': [headers[j] for j in feats], 'categorical_names': CAT_NAMES,
        'selection_feature_rule': 'All 55 raw predictors, same for all model/stage/protocol configs',
        'encoder_rule': 'Numeric median+StandardScaler, categorical most frequent+OneHotEncoder; fit training only',
    }


def centroid_maximin(z, original_indices):
    """All target-ref candidates in their original Test row order; exact argmin/argmax ties.

    Returns full order of candidate indices (zero-based GLOBAL Test row indices).
    """
    original_indices = np.asarray(original_indices, dtype=int)
    zz = z[original_indices]
    centroid = zz.mean(axis=0)
    d2 = np.einsum('ij,ij->i', zz-centroid, zz-centroid)
    first = int(np.argmin(d2))
    chosen = [first]
    picked = np.zeros(len(original_indices), dtype=bool)
    picked[first] = True
    nearest_d2 = np.einsum('ij,ij->i', zz-zz[first], zz-zz[first])
    while len(chosen) < len(original_indices):
        candidates = nearest_d2.copy()
        candidates[picked] = -1.0
        nxt = int(np.argmax(candidates))
        assert not picked[nxt]
        chosen.append(nxt)
        picked[nxt] = True
        new_d2 = np.einsum('ij,ij->i', zz-zz[nxt], zz-zz[nxt])
        nearest_d2 = np.minimum(nearest_d2, new_d2)
    return original_indices[np.asarray(chosen, dtype=int)]


def permutation_map(path, ref_idx, max_replicates):
    ret = {}
    for r in read_rows(path):
        rep = int(r['Replicate']); ref = int(r['Ref_ID'])
        if rep > max_replicates:
            continue
        # 1-based row labels are global Test row IDs in source files.
        order = np.array([int(v)-1 for v in r['Ordered_Test_Rows_1based'].split(';')], dtype=int)
        expected = ref_idx[ref]
        assert len(order) == len(expected) and sorted(order) == sorted(expected.tolist()), (rep,ref)
        ret[(rep,ref)] = order
    assert len(ret) == len(ref_idx) * max_replicates, (len(ret),len(ref_idx),max_replicates)
    return ret


def load_predictions_line2(path, train, test):
    nte = len(test['rows']); ntr=len(train['rows'])
    true = np.log1p(np.array([float(row[22]) for row in test['rows']]))
    refs = np.array([int(row[23]) for row in test['rows']])
    out={}
    for stage in range(1,5):
        for scheme, tag in [('Publication-grouped','Publication_grouped'),('Random folds','Conventional_random')]:
            p = path / f'AUDIT_FINAL_Stage{stage}_{tag}.npz'
            with np.load(p) as data:
                assert np.array_equal(data['Test_Ref'],refs),p
                assert np.allclose(data['Actual_Test_ln1p'],true,atol=1e-10,rtol=1e-10),p
                assert len(data['Train_Ref']) == ntr
                for model in MODELS:
                    pred = np.array(data[f'Test_pred_{model}'], dtype=float)
                    assert len(pred)==nte and np.isfinite(pred).all()
                    out[(scheme,stage,model)]=pred
    assert len(out)==40
    return true,refs,out


def load_predictions_line3(path, train, test):
    nte = len(test['rows']); ntr=len(train['rows'])
    assert ntr==1026
    true = np.log1p(np.array([float(row[22]) for row in test['rows']]))
    refs = np.array([int(row[23]) for row in test['rows']])
    src=read_rows(path/'Line3_Test_Row_Predictions.csv')
    out={}
    for r in src:
        scheme= r['Validation_Scheme']
        scheme= {'Publication-grouped':'Publication-grouped','Conventional random':'Random folds'}[scheme]
        stage=int(r['Stage']); row=int(r['Row_1based'])-1
        assert 0<=row<nte and int(r['Ref_ID'])==refs[row]
        assert math.isclose(float(r['Actual_ln1p']),true[row],rel_tol=1e-9,abs_tol=1e-9)
        for model in MODELS:
            key=(scheme,stage,model)
            if key not in out:
                out[key]=np.full(nte,np.nan)
            assert np.isnan(out[key][row]),(key,row)
            out[key][row]=float(r[f'{model}_pred_ln1p'])
    assert len(out)==40 and all(np.isfinite(x).all() for x in out.values())
    return true,refs,out


def metrics(y,p):
    n=len(y)
    if n==0: return {'R2':np.nan,'RMSE':np.nan,'MAE':np.nan}
    err=y-p
    den=np.sum((y-y.mean())**2)
    r2=(1.0 - float(np.sum(err**2)/den)) if n>=2 and den>1e-15 else np.nan
    return {'R2':r2,'RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.mean(np.abs(err)))}


def gain_metrics(zero, active, random):
    return {'active_gain_R2':active['R2']-zero['R2'],
            'random_gain_R2':random['R2']-zero['R2'],
            'active_minus_random_R2':active['R2']-random['R2'],
            'active_minus_random_RMSE':active['RMSE']-random['RMSE'],
            'active_minus_random_MAE':active['MAE']-random['MAE']}


def summarise_rows(rows, keys):
    # Full replicate distribution quantiles reflect only variations in acquisition, not independent papers.
    grouped={}
    for row in rows:
        k=tuple(row[key] for key in keys)
        grouped.setdefault(k,[]).append(row)
    output=[]
    for k, rs in grouped.items():
        record=dict(zip(keys,k))
        for metric in ['common_zero_R2','common_active_R2','common_random_R2',
                       'active_minus_random_R2','active_gain_R2','random_gain_R2',
                       'active_minus_random_RMSE','active_minus_random_MAE']:
            vals=np.array([float(r[metric]) for r in rs],dtype=float)
            finite=vals[np.isfinite(vals)]
            record[metric+'_median']=float(np.median(finite)) if len(finite) else np.nan
            record[metric+'_q05']=float(np.quantile(finite,.05)) if len(finite) else np.nan
            record[metric+'_q95']=float(np.quantile(finite,.95)) if len(finite) else np.nan
        val=np.array([float(r['active_minus_random_R2']) for r in rs])
        record['fraction_active_better_R2']=float(np.mean(val>0))
        record['replicates']=len(rs)
        output.append(record)
    return sorted(output,key=lambda r:tuple(str(r[x]) for x in keys))


def execute_line(line, args, folder):
    input_dir = BASE/'inputs'/f'line{line}'
    train=load_json(input_dir/'train55.json')
    test=load_json(input_dir/'test55.json')
    expected={(2):(995,249,5),(3):(1026,218,7)}[line]
    assert (len(train['rows']),len(test['rows']))==expected[:2]
    _,z,feat_manifest=prepare_features(train,test)
    if line==2:
        y,refs,preds=load_predictions_line2(input_dir,train,test)
    else:
        y,refs,preds=load_predictions_line3(input_dir,train,test)
    unique=np.unique(refs)
    assert len(unique)==expected[2]
    ref_idx={int(ref):np.flatnonzero(refs==ref) for ref in unique}
    active={ref:centroid_maximin(z,idx) for ref,idx in ref_idx.items()}
    perms=permutation_map(input_dir/'Random_Permutations_500.csv',ref_idx,args.replicates)
    # ensure active full order is deterministic and an exact Test subset
    for ref, order in active.items():
        assert sorted(order.tolist())==sorted(ref_idx[ref].tolist())
    all_cases=[]; active_own=[]; per_pub=[]; active_seq=[]
    for ref, order in active.items():
        for rank, row in enumerate(order,1):
            active_seq.append({'Line':line,'Ref_ID':ref,'Acquisition_rank':rank,
                               'Test_row_1based':int(row)+1,'Is_centroid_initial':int(rank==1)})
    write_rows(folder/f'Line{line}_Active_Complete_Sequence.csv',active_seq)
    designs={}
    for budget in BUDGETS:
        refs_cal={ref:active[ref][:count_budgets(len(ref_idx[ref]))[budget]] for ref in ref_idx}
        allc=np.hstack(list(refs_cal.values()))
        assert len(allc)==len(set(allc))
        designs[budget]=refs_cal
        for ref,sub in refs_cal.items():
            for row in sub:
                active_seq.append({'Line':line,'Ref_ID':ref,'Acquisition_rank':int(np.where(active[ref]==row)[0][0])+1,
                                   'Test_row_1based':int(row)+1,'Is_centroid_initial':int(row==active[ref][0]),'Budget_pct':budget})
    write_rows(folder/f'Line{line}_Active_Budget_Selections.csv', [r for r in active_seq if 'Budget_pct' in r])

    for (scheme,stage,model),phat in preds.items():
        for budget in BUDGETS:
            ac=designs[budget]
            cal_a=np.zeros(len(y),dtype=float)
            mask_a=np.ones(len(y),dtype=bool)
            for ref, aidx in ac.items():
                delta_a=float(np.mean(y[aidx]-phat[aidx]))
                cal_a[ref_idx[ref]]=phat[ref_idx[ref]]+delta_a
                mask_a[aidx]=False
            active_own_ref=metrics(y[mask_a],cal_a[mask_a]); zero_own_ref=metrics(y[mask_a],phat[mask_a])
            own_record={'Line':line,'Protocol':scheme,'Stage':stage,'Model':model,'Budget_pct':budget,
                        'N_cal':int((~mask_a).sum()),'N_eval':int(mask_a.sum())}
            own_record.update({'Zero_'+key:val for key,val in zero_own_ref.items()})
            own_record.update({'Active_'+key:val for key,val in active_own_ref.items()})
            own_record['Active_gain_R2']=active_own_ref['R2']-zero_own_ref['R2']
            active_own.append(own_record)
            for rep in range(1,args.replicates+1):
                mask_union=np.ones(len(y),dtype=bool)
                cal_r=np.zeros(len(y),dtype=float)
                remaining_by_ref={}
                for ref,idx in ref_idx.items():
                    k=count_budgets(len(idx))[budget]
                    aidx=ac[ref];ridx=perms[(rep,ref)][:k]
                    union=np.union1d(aidx,ridx)
                    mask_union[union]=False
                    remaining_by_ref[ref]=np.setdiff1d(idx,union,assume_unique=True)
                    delta_r=float(np.mean(y[ridx]-phat[ridx]))
                    cal_r[idx]=phat[idx]+delta_r
                zero=metrics(y[mask_union],phat[mask_union])
                aeval=metrics(y[mask_union],cal_a[mask_union])
                reval=metrics(y[mask_union],cal_r[mask_union])
                record={'Line':line,'Protocol':scheme,'Stage':stage,'Model':model,'Budget_pct':budget,
                        'Replicate':rep,'N_common_eval':int(mask_union.sum()),
                        'N_active_cal':int((~mask_a).sum()),
                        'N_random_cal':int(sum(count_budgets(len(idx))[budget] for idx in ref_idx.values()))}
                for prefix,m in [('common_zero_',zero),('common_active_',aeval),('common_random_',reval)]:
                    record.update({prefix+key:val for key,val in m.items()})
                record.update(gain_metrics(zero,aeval,reval))
                all_cases.append(record)
                for ref,inds in remaining_by_ref.items():
                    zp=metrics(y[inds],phat[inds]); ap=metrics(y[inds],cal_a[inds]); rp=metrics(y[inds],cal_r[inds])
                    rec={'Line':line,'Protocol':scheme,'Stage':stage,'Model':model,'Budget_pct':budget,
                         'Replicate':rep,'Ref_ID':ref,'N_common_eval':len(inds)}
                    for prefix,m in [('common_zero_',zp),('common_active_',ap),('common_random_',rp)]:
                        rec.update({prefix+key:val for key,val in m.items()})
                    rec.update(gain_metrics(zp,ap,rp))
                    per_pub.append(rec)
    write_rows(folder/f'Line{line}_Active_OwnRemainder_160.csv',active_own)
    write_rows(folder/f'Line{line}_ActiveVsRandom_Matched_Replicates.csv',all_cases)
    write_rows(folder/f'Line{line}_ActiveVsRandom_Matched_Summary.csv',
               summarise_rows(all_cases,['Line','Protocol','Stage','Model','Budget_pct']))
    write_rows(folder/f'Line{line}_ActiveVsRandom_PerPublication_Replicates.csv',per_pub)
    write_rows(folder/f'Line{line}_ActiveVsRandom_PerPublication_Summary.csv',
               summarise_rows(per_pub,['Line','Protocol','Stage','Model','Budget_pct','Ref_ID']))
    manifest={'Line':line,'N_train':len(train['rows']),'N_test':len(test['rows']),
              'Ref_IDs':[int(x) for x in unique], 'N_models_per_scheme_stage':len(preds),
              'replicates':args.replicates,'features':feat_manifest,
              'budgets_percent':BUDGETS,
              'sampling_counts_by_ref':{str(ref):count_budgets(len(idx)) for ref,idx in ref_idx.items()},
              'matched_eval_definition':'Test rows not in union(Active sampled rows, Random sampled rows) for each ref/replicate',
              'target':'ln(1 + DBP concentration in µg/L)',
              'strict_training_test':'No model retraining or refitting; Test outcomes used only after sample selection to estimate each ref offset',
              'output_rows':{'active_own':len(active_own),'replicate_matched':len(all_cases),
                             'perpub_replicate_matched':len(per_pub)},
              'N_encoded_features':feat_manifest['n_encoded_features']}
    (folder/f'Line{line}_Execution_Manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Line{line}: train={len(train["rows"])},test={len(test["rows"])}; '
          f'encoded={feat_manifest["n_encoded_features"]}; configs=40, budgets=4, '
          f'active_own={len(active_own)}, matched={len(all_cases)}, per-ref={len(per_pub)}',flush=True)


def verify_input_hashes():
    manifest=load_json(BASE/'INPUT_SHA256.json')['files']
    for rel,info in manifest.items():
        p=BASE/rel
        assert p.is_file(),f'Missing input: {rel}'
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha==info['sha256'],f'Input hash mismatch: {rel}'
    print(f'Verified {len(manifest)} frozen input asset SHA-256 checksums.',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--line',choices=['2','3','both'],default='both')
    ap.add_argument('--smoke',action='store_true',help='Use 3 random replicates for fast end-to-end input/alignment verification')
    ap.add_argument('--replicates',type=int,default=500)
    ap.add_argument('--out',type=Path,default=BASE/'outputs')
    args=ap.parse_args()
    if args.smoke: args.replicates=3
    if not (1<=args.replicates<=500): ap.error('replicates must be 1..500: the locked permutations file has 500')
    verify_input_hashes()
    args.out.mkdir(parents=True,exist_ok=True)
    for line in ([2,3] if args.line=='both' else [int(args.line)]):
        execute_line(line,args,args.out)
    print('All requested lines completed.',flush=True)

if __name__ == '__main__':
    main()
