"""Independently audit the frozen Equal4 random-few-shot result archive."""
from pathlib import Path
import csv, numpy as np, json, hashlib
from collections import Counter
root=Path(__file__).resolve().parent
out=root/'outputs'; src=root/'source/Line2_predictions'
read=lambda n:list(csv.DictReader(open(out/n,newline='',encoding='utf-8')))
raw=read('EQ4_RandomFewShot_Replicates_16000.csv')
per=read('EQ4_RandomFewShot_PerPublication_Replicates_80000.csv')
summ=read('EQ4_RandomFewShot_Summary_32.csv')
assert len(raw)==16000 and len(per)==80000 and len(summ)==32
assert len(set((a['Protocol'],a['Stage'],a['Budget_pct'],a['Replicate']) for a in raw))==16000
assert set(Counter((a['Protocol'],a['Stage'],a['Budget_pct']) for a in raw).values())=={500}
assert set(Counter((a['Protocol'],a['Stage'],a['Budget_pct'],a['Ref_ID']) for a in per).values())=={500}
assert set((int(a['N_cal']),int(a['N_eval'])) for a in raw)=={(9,240),(15,234),(28,221),(53,196)}
ref=np.load(src/'AUDIT_FINAL_Stage1_Publication_grouped.npz')['Test_Ref'].astype(int)
for stage in range(1,5):
 for method in ('Publication_grouped','Conventional_random'):
  a=np.load(src/f'AUDIT_FINAL_Stage{stage}_{method}.npz')
  assert np.array_equal(a['Test_Ref'].astype(int),ref)
  assert a['Test_pred_Equal4'].shape==(249,)
assert hashlib.sha256((out/'Random_Permutations_500x5.csv').read_bytes()).hexdigest()=='8dc6aaeeb57a52e49c8ef8baf36b901e069cf9dd415aad3e794353fbcffd5651'
priorfiles={}
for prior in ['Line2_RF_RandomFewShot_20261014','Line2_XGB_RandomFewShot_20261014','Line2_LightGBM_RandomFewShot_20261014']:
 p=Path('/mnt/data')/prior/'outputs/Random_Permutations_500x5.csv'
 if p.exists():
  assert p.read_bytes()==(out/'Random_Permutations_500x5.csv').read_bytes()
  priorfiles[prior]='identical'
# Numeric zeroshot acceptance targets from latest SI Table S25 for publication-grouped folds.
si_expected={1:.3531282978,2:.1893089887,3:.1752798934,4:.1244381065}
for st,expected in si_expected.items():
    z=next(t for t in read('ZeroShot_Original_249.csv') if t['Protocol']=='Grouped' and int(t['Stage'])==st)
    assert np.isclose(float(z['Zero_R2']),expected,atol=5.1e-11,rtol=0)
# Ensemble check: Equal4 must be the mean of all four frozen base learners.
for stage in range(1,5):
 for method in ('Publication_grouped','Conventional_random'):
  d=np.load(src/f'AUDIT_FINAL_Stage{stage}_{method}.npz')
  four=np.mean(np.column_stack([d[f'Test_pred_{name}'].astype(float) for name in ('RF','XGBoost','LightGBM','CatBoost')]),axis=1)
  assert np.max(np.abs(four-d['Test_pred_Equal4']))<2e-7, (stage,method,np.max(np.abs(four-d['Test_pred_Equal4'])))
# Completely independent recomputation from saved permutation, frozen predictions and actual response, for one replicate.
p=src/'AUDIT_FINAL_Stage4_Publication_grouped.npz'; d=np.load(p)
y=d['Actual_Test_ln1p']; yhat=d['Test_pred_Equal4']; reference=d['Test_Ref'].astype(int)
perms=read('Random_Permutations_500x5.csv')
choices=[]; rem=[]; offsets={}
for r in (7,14,16,21,22):
 rr=next(t for t in perms if t['Replicate']=='1' and t['Ref_ID']==str(r))
 ids=np.array([int(v)-1 for v in rr['Ordered_Test_Rows_1based'].split(';')],dtype=int)
 assert np.all(reference[ids]==r)
 n=int(np.ceil(.025*len(ids)))
 choices.extend(ids[:n]);rem.extend(ids[n:]);offsets[r]=float(np.mean(y[ids[:n]]-yhat[ids[:n]]))
assert len(set(choices))==9 and len(set(rem))==240 and set(choices).isdisjoint(rem)
rem=np.array(rem,dtype=int)
yhcal=yhat[rem]+np.array([offsets[r] for r in reference[rem]])
calc=lambda actual,pred:float(1-np.sum((actual-pred)**2)/np.sum((actual-np.mean(actual))**2))
z=next(t for t in raw if t['Protocol']=='Grouped' and t['Stage']=='4' and t['Budget_pct']=='2.5' and t['Replicate']=='1')
assert np.isclose(calc(y[rem],yhcal),float(z['Cal_R2']),rtol=0,atol=1e-12)
assert np.isclose(calc(y[rem],yhat[rem]),float(z['Zero_R2']),rtol=0,atol=1e-12)
# Mean/median pair rules; ratio of fraction improved.
for d in summ:
 relevant=[r for r in raw if r['Protocol']==d['Protocol'] and r['Stage']==d['Stage'] and r['Budget_pct']==d['Budget_pct']]
 assert len(relevant)==500
 assert np.isclose(np.median([float(a['Delta_R2']) for a in relevant]),float(d['Delta_R2_median']),atol=1e-12)
 assert np.isclose(np.mean([int(a['Improved_R2']) for a in relevant]),float(d['Fraction_improved_R2']),atol=1e-12)
report={'status':'ALL CHECKS PASS','line':'Figure 2b (Line 2)','model':'Equal4', 'n_runs':len(raw),'n_per_publication_runs':len(per),'n_summaries':len(summ),
        'sampling_shared_with_prior_models':priorfiles, 'permutations_sha256':hashlib.sha256((out/'Random_Permutations_500x5.csv').read_bytes()).hexdigest(),
        'SI_ZeroShot_R2_Grouped_Stages1_4':si_expected, 'Equal4_prediction':'independently verified against arithmetic mean of four frozen experts across all 8 Stage/protocol combinations',
        'independent_recomputation':'Stage4, grouped, first 2.5% replicate: exact same-remainder zero-shot and calibrated R2',
        'source_file_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.npz')},
        'original_workbook_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').glob('*.xlsx')},
        'numpy':np.__version__}
(out/'Audit_Checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
