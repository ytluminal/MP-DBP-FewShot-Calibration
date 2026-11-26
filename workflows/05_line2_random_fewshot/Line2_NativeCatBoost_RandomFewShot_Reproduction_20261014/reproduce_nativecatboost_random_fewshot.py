#!/usr/bin/env python3
"""Exact random-publication-offset few-shot evaluation for the frozen Figure 2b Native CatBoost predictions.

Uses frozen 249-row Test predictions from the successfully reproduced Figure 2b audit.
One global RNG with seed 42, five pubs permuted in order at each replicate, nested budgets.
Never fits Native CatBoost, revises parameters/features, or selects calibrators with response values.
"""
from pathlib import Path
import csv, json, hashlib, math
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source' / 'Line2_predictions'
OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
REF_ORDER = (7, 14, 16, 21, 22)
BUDGETS = (0.025, 0.05, 0.10, 0.20)
REPS = 500
SEED = 42


def metrics(y, yh):
    residual = np.asarray(y) - np.asarray(yh)
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {'R2': float(1 - sse/sst) if sst > 0 else float('nan'),
            'RMSE': float(np.sqrt(np.mean(residual**2))),
            'MAE': float(np.mean(np.abs(residual)))}


def record_csv(path, records, header=None):
    rows = list(records)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        keys = header if header is not None else list(rows[0])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


sources = {}
for stage in range(1, 5):
    for protocol, suffix in [('Grouped', 'Publication_grouped'), ('RandomCV', 'Conventional_random')]:
        p = SOURCE / f'AUDIT_FINAL_Stage{stage}_{suffix}.npz'
        d = np.load(p, allow_pickle=False)
        sources[(stage, protocol)] = {
            'y': d['Actual_Test_ln1p'].astype(float),
            'pred': d['Test_pred_CatBoost'].astype(float),
            'ref': d['Test_Ref'].astype(int),
            'source': p.name,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
        }

# Strict cross-file consistency: same y, refs and test-row ordering in all 8 frozen vectors.
master = sources[(1, 'Grouped')]
y, ref = master['y'], master['ref']
assert len(y) == 249 and set(ref) == set(REF_ORDER)
for a in sources.values():
    assert np.array_equal(a['ref'], ref)
    assert np.array_equal(a['y'], y)
refpos = {r: np.flatnonzero(ref == r) for r in REF_ORDER}
assert {r: len(pos) for r,pos in refpos.items()} == {7:108,14:72,16:41,21:12,22:16}

# Cache identical, response-independent random selections across ALL model/stages/protocols.
rng = np.random.default_rng(SEED)
permutations = []
selection_records=[]
for replicate in range(1, REPS + 1):
    perms = {r: rng.permutation(refpos[r]) for r in REF_ORDER}
    permutations.append(perms)
    for r in REF_ORDER:
        selection_records.append({'Replicate':replicate, 'Ref_ID':r, 'Ordered_Test_Rows_1based':';'.join(str(i+1) for i in perms[r]),
                                  'N':len(perms[r])})
record_csv(OUT / 'Random_Permutations_500x5.csv', selection_records)

count_records=[]
for budget in BUDGETS:
    counts={r:int(math.ceil(budget*len(refpos[r]))) for r in REF_ORDER}
    for r in REF_ORDER:
        count_records.append({'Budget_pct': budget*100, 'Ref_ID':r, 'N':len(refpos[r]),'n_cal':counts[r], 'n_remaining':len(refpos[r])-counts[r]})
record_csv(OUT / 'Per_Publication_Budgets.csv', count_records)

all_raw=[]
all_ref_raw=[]
zero_records=[]
summary=[]
ref_summary=[]
for protocol in ('Grouped', 'RandomCV'):
  for stage in (1,2,3,4):
    info = sources[(stage,protocol)]
    yhat = info['pred']
    baseline = metrics(y,yhat)
    zero_records.append({'Protocol':protocol,'Stage':stage,'Model':'Native CatBoost', 'N_Test':len(y),**{f'Zero_{k}':v for k,v in baseline.items()},'Source_sha256':info['sha256']})
    for budget in BUDGETS:
      counts = {r:int(math.ceil(budget * len(refpos[r]))) for r in REF_ORDER}
      evalcount = len(y) - sum(counts.values())
      br=[]; ref_br={r:[] for r in REF_ORDER}
      for replicate, perms in enumerate(permutations, start=1):
        eval_indices = []
        cal_indices = []
        deltas = {}
        for r in REF_ORDER:
            cal = perms[r][:counts[r]]
            evl = perms[r][counts[r]:]
            cal_indices.append(cal)
            eval_indices.append(evl)
            deltas[r] = float(np.mean(y[cal] - yhat[cal]))
        eval_idx = np.concatenate(eval_indices)
        assert len(eval_idx) == evalcount
        pred_new = yhat[eval_idx] + np.asarray([deltas[r] for r in ref[eval_idx]])
        before=metrics(y[eval_idx],yhat[eval_idx])
        after=metrics(y[eval_idx],pred_new)
        diff={k:after[k]-before[k] for k in ('R2','RMSE','MAE')}
        row={'Protocol':protocol,'Stage':stage,'Model':'Native CatBoost','Budget_pct':budget*100,'Replicate':replicate,
             'N_cal':sum(counts.values()),'N_eval':len(eval_idx),
             **{f'Zero_{k}':v for k,v in before.items()},
             **{f'Cal_{k}':v for k,v in after.items()},
             **{f'Delta_{k}':v for k,v in diff.items()},
             'Improved_R2':int(after['R2'] > before['R2']),
             'Improved_RMSE':int(after['RMSE'] < before['RMSE'])}
        all_raw.append(row); br.append(row)
        for rr, evl in zip(REF_ORDER, eval_indices):
            if len(evl)<2:continue
            pred_rr = yhat[evl]+deltas[rr]
            met0=metrics(y[evl],yhat[evl]); met1=metrics(y[evl],pred_rr)
            rrrow={'Protocol':protocol,'Stage':stage,'Budget_pct':budget*100,'Replicate':replicate,'Ref_ID':rr,
                   'N_cal':counts[rr],'N_eval':len(evl),
                   **{f'Zero_{k}':v for k,v in met0.items()},**{f'Cal_{k}':v for k,v in met1.items()},
                   **{f'Delta_{k}':met1[k]-met0[k] for k in met1}}
            all_ref_raw.append(rrrow)
            ref_br[rr].append(rrrow)
      arr=lambda k:np.asarray([z[k] for z in br],float)
      sq=lambda key,p:float(np.quantile(arr(key),p))
      row={'Protocol':protocol,'Stage':stage,'Budget_pct':budget*100,'N_cal':sum(counts.values()),'N_eval':evalcount,
           'Zero_full_R2':baseline['R2'],'Zero_full_RMSE':baseline['RMSE'],'Zero_full_MAE':baseline['MAE']}
      for prefix in ['Zero','Cal','Delta']:
        for metric in ['R2','RMSE','MAE']:
          key=f'{prefix}_{metric}'
          row[f'{key}_median']=sq(key,.5)
          row[f'{key}_q05']=sq(key,.05)
          row[f'{key}_q25']=sq(key,.25)
          row[f'{key}_q75']=sq(key,.75)
          row[f'{key}_q95']=sq(key,.95)
      row['Fraction_improved_R2']=float(np.mean(arr('Improved_R2')))
      row['Fraction_improved_RMSE']=float(np.mean(arr('Improved_RMSE')))
      row['Fraction_improved_MAE']=float(np.mean(arr('Delta_MAE')<0))
      summary.append(row)
      for r in REF_ORDER:
        seq=ref_br[r]
        rarr=lambda k:np.asarray([z[k] for z in seq],float)
        ref_summary.append({'Protocol':protocol,'Stage':stage,'Budget_pct':budget*100,'Ref_ID':r,
                            'N_cal':counts[r],'N_eval':len(refpos[r])-counts[r],
                            'Zero_RMSE_median':float(np.median(rarr('Zero_RMSE'))),
                            'Cal_RMSE_median':float(np.median(rarr('Cal_RMSE'))),
                            'Delta_RMSE_median':float(np.median(rarr('Delta_RMSE'))),
                            'Fraction_RMSE_improved':float(np.mean(rarr('Delta_RMSE')<0)),
                            'Zero_MAE_median':float(np.median(rarr('Zero_MAE'))),
                            'Cal_MAE_median':float(np.median(rarr('Cal_MAE'))),
                            'Fraction_MAE_improved':float(np.mean(rarr('Delta_MAE')<0)),
                            'Zero_R2_median':float(np.median(rarr('Zero_R2'))),
                            'Cal_R2_median':float(np.median(rarr('Cal_R2')))})
      print(f'{protocol:8s} Stage{stage} B{budget*100:g}% zero~{row["Zero_R2_median"]:+.4f}, cal~{row["Cal_R2_median"]:+.4f}, improvement likelihood {row["Fraction_improved_R2"]:.1%}',flush=True)

record_csv(OUT / 'ZeroShot_Original_249.csv',zero_records)
record_csv(OUT / 'NCB_RandomFewShot_Summary_32.csv',summary)
record_csv(OUT / 'NCB_RandomFewShot_Replicates_16000.csv',all_raw)
record_csv(OUT / 'NCB_RandomFewShot_PerPublication_Summary_160.csv',ref_summary)
record_csv(OUT / 'NCB_RandomFewShot_PerPublication_Replicates_80000.csv',all_ref_raw)
manifest={'experiment':'Line2 Figure2b Native CatBoost Stages1-4; Grouped and Random-CV fold-trained test predictions',
   'all_runs':len(all_raw),'random_replicates':REPS,'seed':SEED,'protocols':['Grouped','RandomCV'],
   'budgets':BUDGETS,'test_rows':249,'references':REF_ORDER,
   'n_cal_round':'ceil(fraction * N_ref) independently per target reference',
   'random_sampling':'single numpy default_rng(42), for each replicate permutation of ref test rows in order 7,14,16,21,22; identical permutations reused across stage and training protocol; budget samples nested prefixes',
   'calibration':'per-reference mean(y_true_cal - y_pred_cal) added to frozen zero-shot ln(1+C_DBP) Test predictions; calibration rows excluded from scoring',
   'comparison':'paired same-remainder zero-shot and calibrated metrics for each replicate; pooled across remaining rows',
   'quantiles':'numpy quantile linear interpolation; median and 5/25/75/95 percentiles',
   'input_archives':{f'Stage{stage}_{protocol}':{'filename':a['source'],'sha256':a['sha256']} for (stage,protocol),a in sources.items()},
   'output_files':{}
}
for f in OUT.glob('*.csv'):
    manifest['output_files'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(OUT/'Experiment_Manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print('FINISHED',json.dumps({'n_summary':len(summary),'raw':len(all_raw),'ref_raw':len(all_ref_raw),'total_cal_counts':{str(v):sum(int(math.ceil(v*len(refpos[r]))) for r in REF_ORDER) for v in BUDGETS}}))
