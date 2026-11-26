import csv,json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'output/Experiment_Manifest.json').read_text())
for name,digest in M['file_checksums_sha256'].items():
    actual=hashlib.sha256((ROOT/'output'/name).read_bytes()).hexdigest()
    assert actual==digest,(name,'SHA256 mismatch')
with open(ROOT/'output/ZeroShot_Baseline.csv',encoding='utf-8-sig') as f:z=list(csv.DictReader(f))
expected={('Grouped',1):.011995752846447827,('Grouped',2):.3581130522548792,('Grouped',3):.2681071587449786,('Grouped',4):.2809046547020907,('Random',1):.423004139234895,('Random',2):.4798813554196114,('Random',3):.39219755487504737,('Random',4):.24104531402707619}
for r in z:
 if r['Ref_ID']=='Overall':assert math.isclose(float(r['R2']),expected[(r['Scheme'],int(r['Stage']))],abs_tol=1e-11)
with open(ROOT/'output/Matched_Overall_Replicates.csv',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==16000
for r in rows:
 assert math.isclose(float(r['Active_R2'])-float(r['Random_R2']),float(r['Paired_dR2']),abs_tol=1e-11)
 assert math.isclose(float(r['Active_RMSE'])-float(r['Random_RMSE']),float(r['Paired_dRMSE']),abs_tol=1e-11)
 assert float(r['N_common_eval'])>=36
with open(ROOT/'output/Matched_Publication_Replicates.csv',encoding='utf-8-sig') as f: n=sum(1 for _ in f)-1
assert n==48000
with open(ROOT/'output/Active_Offsets.csv',encoding='utf-8-sig') as f:offs=list(csv.DictReader(f))
assert len(offs)==96
print('AUDIT PASSED: all output hashes, locked 8 zero-shot scores, 16000 paired deltas, 48000 per-publication records, 96 offsets.')
