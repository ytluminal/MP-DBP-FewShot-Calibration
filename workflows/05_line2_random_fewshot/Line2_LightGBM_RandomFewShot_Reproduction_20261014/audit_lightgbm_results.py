#!/usr/bin/env python3
"""Reproduction checks for the LightGBM frozen-prediction calibration."""
from pathlib import Path
from collections import Counter
import numpy as np, csv, hashlib, json, zipfile
root=Path(__file__).resolve().parent
out=root/'outputs'
src=root/'source/Line2_predictions'
rows=list(csv.DictReader(open(out/'LGBM_RandomFewShot_Replicates_16000.csv',newline='')))
summ=list(csv.DictReader(open(out/'LGBM_RandomFewShot_Summary_32.csv',newline='')))
per=list(csv.DictReader(open(out/'LGBM_RandomFewShot_PerPublication_Replicates_80000.csv',newline='')))
assert len(rows)==16000 and len(summ)==32 and len(per)==80000
assert len(set((r['Protocol'],r['Stage'],r['Budget_pct'],r['Replicate']) for r in rows))==16000
assert set(Counter((r['Protocol'],r['Stage'],r['Budget_pct']) for r in rows).values())=={500}
assert set(Counter((r['Protocol'],r['Stage'],r['Budget_pct'],r['Ref_ID']) for r in per).values())=={500}
assert set((int(r['N_cal']),int(r['N_eval'])) for r in rows)=={(9,240),(15,234),(28,221),(53,196)}
# Identical independently frozen random draw sequence used by RF and XGBoost.
expected_selection_hash = '8dc6aaeeb57a52e49c8ef8baf36b901e069cf9dd415aad3e794353fbcffd5651'
assert hashlib.sha256((out/'Random_Permutations_500x5.csv').read_bytes()).hexdigest()==expected_selection_hash
for previous in ['Line2_XGB_RandomFewShot_20261014','Line2_RF_RandomFewShot_20261014']:
    prior_file=Path('/mnt/data')/previous/'outputs/Random_Permutations_500x5.csv'
    if prior_file.exists():
        assert (out/'Random_Permutations_500x5.csv').read_bytes()==prior_file.read_bytes()
si={('3',5.0):0.5157,('3',10.0):0.5615,('3',20.0):0.5847,('4',5.0):0.5170,('4',10.0):0.5657,('4',20.0):0.5919}
for (stage,budget),target in si.items():
    r=next(r for r in summ if r['Protocol']=='Grouped' and r['Stage']==stage and float(r['Budget_pct'])==budget)
    assert round(float(r['Cal_R2_median']),4)==target,(stage,budget,r['Cal_R2_median'])
# Independent numerical spot-check (first replicate; grouped stage4; 2.5%).
z=np.load(src/'AUDIT_FINAL_Stage4_Publication_grouped.npz')
y=z['Actual_Test_ln1p']; yh=z['Test_pred_LightGBM']; ref=z['Test_Ref'].astype(int)
perm_rows=list(csv.DictReader(open(out/'Random_Permutations_500x5.csv',newline='')))
cal=[]; eval_idx=[]; offsets={}
for r in [7,14,16,21,22]:
    order=next(q for q in perm_rows if q['Replicate']=='1' and q['Ref_ID']==str(r))['Ordered_Test_Rows_1based']
    ids=np.array([int(x)-1 for x in order.split(';')]); n=int(np.ceil(.025*len(ids)))
    assert len(ids)==sum(ref==r) and np.all(ref[ids]==r)
    offsets[r]=np.mean(y[ids[:n]]-yh[ids[:n]])
    cal.extend(ids[:n]); eval_idx.extend(ids[n:])
assert len(cal)==9 and len(eval_idx)==240 and not (set(cal)&set(eval_idx))
e=np.array(eval_idx)
yp=yh[e]+np.array([offsets[r] for r in ref[e]])
r2=lambda y,p: 1-np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
r=next(r for r in rows if r['Protocol']=='Grouped' and r['Stage']=='4' and r['Budget_pct']=='2.5' and r['Replicate']=='1')
assert np.isclose(r2(y[e],yp),float(r['Cal_R2']),rtol=0,atol=1e-12)
assert np.isclose(r2(y[e],yh[e]),float(r['Zero_R2']),rtol=0,atol=1e-12)
zero=list(csv.DictReader(open(out/'ZeroShot_Original_249.csv',newline='')))
assert all(np.isclose(r2(np.load(src/f'AUDIT_FINAL_Stage{r["Stage"]}_{"Publication_grouped" if r["Protocol"]=="Grouped" else "Conventional_random"}.npz')['Actual_Test_ln1p'],np.load(src/f'AUDIT_FINAL_Stage{r["Stage"]}_{"Publication_grouped" if r["Protocol"]=="Grouped" else "Conventional_random"}.npz')['Test_pred_LightGBM']),float(r['Zero_R2']),atol=1e-12) for r in zero)
report={'status':'ALL CHECKS PASS','model':'LightGBM', 'line':'Figure 2b (Line 2)', 'random_draws_same_as_RF_XGBoost':True,
        'replicate_rows':len(rows),'publication_replicate_rows':len(per),'summary_rows':len(summ),
        'acceptance_si_legacy_6_results':si if False else [{'Stage':int(s),'Budget_pct':b,'R2':v} for (s,b),v in si.items()],
        'frozen_test_rows':249,'calibration_total':[9,15,28,53],
        'spotcheck':'Stage4 Grouped 2.5% replicate1 recalculated with independent numpy formula matches stored R2',
        'training_workbook_sha256':'de1e28d1525c97d802e5b3dd261dd7409bea3bbeade02875d3ac36fe76947fb9',
        'test_workbook_sha256':'75e5164aba4c974e54f4974dee86a65d37228e9194ada504f6133552c77fdfc8',
        'software':{'numpy':np.__version__}}
(out/'Audit_Checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,indent=2))
