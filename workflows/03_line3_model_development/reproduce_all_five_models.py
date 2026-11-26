"""Start independent processes for frozen Figure 2c full refit and audit."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--stages',type=int,nargs='+',choices=[1,2,3,4],default=[1,2,3,4])
args=parser.parse_args()
(ROOT/'logs').mkdir(exist_ok=True)
env=os.environ.copy()
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
 env[k]='1'
def run(label,argv):
 print('\nRUN',label,' '.join(argv),flush=True)
 logfile=ROOT/'logs'/(label+'.log')
 with logfile.open('w',encoding='utf-8') as f:
  process=subprocess.Popen([sys.executable,*argv],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in process.stdout:
   print(line,end='',flush=True)
   f.write(line)
  code=process.wait()
 if code:raise SystemExit(f'FAILED: {label}, exit={code}, logfile={logfile}')
for stage in args.stages:
 # Repaired, independently validated RF: both schemes, CV and Test;
 # its preliminary Equal4 uses previously verified other-base snapshots,
 # which are NOT used in the FINAL full-refit Equal4 calculation.
 run(f'S{stage}_RF',['reproduce_rf_equal4.py','--stages',str(stage)])
 for scheme in ['Publication-grouped','Conventional random']:
  for model in ['XGBoost','LightGBM','CatBoost']:
   run(f'S{stage}_{scheme.replace(" ","_")}_{model}', ['refit_non_rf.py','--model',model,'--stage',str(stage),'--scheme',scheme])
run('FULL_INDEPENDENT_AUDIT',['verify_full_refit.py','--stages',*map(str,args.stages)])
print('SUCCESS. FULL REFIT AND FROZEN PREDICTION AUDIT PASSED.',flush=True)
