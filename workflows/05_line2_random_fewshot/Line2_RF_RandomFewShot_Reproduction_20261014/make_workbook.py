"""Create human-readable results workbook using artifact_tool only."""
import csv
from pathlib import Path
from artifact_tool import Workbook, SpreadsheetFile
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
def read(name):return list(csv.DictReader(open(OUT/name,encoding='utf-8')))
S=read('RF_RandomFewShot_Summary_32.csv')
R=read('RF_RandomFewShot_PerPublication_Summary_160.csv')
B=read('Per_Publication_Budgets.csv')
Z=read('ZeroShot_Original_249.csv')
w=Workbook.create()

NUMCOL = ['Budget (%)','n cal','n eval','zero-shot full R²','same remainder zero R² (median)',
          'few-shot R² (median)','few-shot R² (Q05)','few-shot R² (Q95)',
          'paired ΔR² (median)','paired ΔR² (Q05)','paired ΔR² (Q95)',
          'P(R² improved)','Zero RMSE (med)','Few-shot RMSE (med)','paired ΔRMSE (med)',
          'Zero MAE (med)','Few-shot MAE (med)','P(MAE improved)']
cols=['Protocol','Stage']+NUMCOL
s=w.worksheets.add('Main Results')
A=[]
for r in S:
 A.append([r['Protocol'],int(r['Stage']),float(r['Budget_pct']),int(r['N_cal']),int(r['N_eval']),
    float(r['Zero_full_R2']),float(r['Zero_R2_median']),float(r['Cal_R2_median']),float(r['Cal_R2_q05']),float(r['Cal_R2_q95']),
    float(r['Delta_R2_median']),float(r['Delta_R2_q05']),float(r['Delta_R2_q95']),float(r['Fraction_improved_R2']),
    float(r['Zero_RMSE_median']),float(r['Cal_RMSE_median']),float(r['Delta_RMSE_median']),
    float(r['Zero_MAE_median']),float(r['Cal_MAE_median']),float(r['Fraction_improved_MAE'])])
s.get_range_by_indexes(0,0,1+len(A),len(cols)).values=[cols]+A
s.get_range('A1:T1').format={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'row_height':41,'wrap_text':True,'vertical_alignment':'center'}
s.get_range('A2:T33').format.row_height=20
s.get_range('A:A').format.column_width=14
s.get_range('B:E').format.column_width=10
s.get_range('F:M').format.column_width=17
s.get_range('N:N').format.column_width=13
s.get_range('O:S').format.column_width=17
s.get_range('T:T').format.column_width=13
s.get_range('F2:M33').set_number_format('0.0000;[Red]-0.0000')
s.get_range('N2:N33').set_number_format('0.0%')
s.get_range('O2:S33').set_number_format('0.0000;[Red]-0.0000')
s.get_range('T2:T33').set_number_format('0.0%')
s.freeze_panes.freeze_rows(1)

s=w.worksheets.add('Per Publication')
headers=['Protocol','Stage','Budget (%)','Ref ID','n cal','n eval','Zero RMSE med','Cal RMSE med','paired ΔRMSE med','P(RMSE improved)','Zero MAE med','Cal MAE med','P(MAE improved)','Zero R² med','Cal R² med']
rows=[]
for r in R:
 rows.append([r['Protocol'],int(r['Stage']),float(r['Budget_pct']),int(r['Ref_ID']),int(r['N_cal']),int(r['N_eval']),
              float(r['Zero_RMSE_median']),float(r['Cal_RMSE_median']),float(r['Delta_RMSE_median']),float(r['Fraction_RMSE_improved']),
              float(r['Zero_MAE_median']),float(r['Cal_MAE_median']),float(r['Fraction_MAE_improved']),
              float(r['Zero_R2_median']),float(r['Cal_R2_median'])])
s.get_range_by_indexes(0,0,len(rows)+1,len(headers)).values=[headers]+rows
s.get_range('A1:O1').format={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'row_height':40,'wrap_text':True}
s.get_range('A:A').format.column_width=15
s.get_range('B:F').format.column_width=12
s.get_range('G:O').format.column_width=17
s.get_range('G2:I161').set_number_format('0.0000;[Red]-0.0000')
s.get_range('K2:L161').set_number_format('0.0000;[Red]-0.0000')
s.get_range('N2:O161').set_number_format('0.0000;[Red]-0.0000')
for c in ['J','M']:s.get_range(f'{c}2:{c}161').set_number_format('0.0%')
s.freeze_panes.freeze_rows(1)

s=w.worksheets.add('Budgets')
headers=['Budget (%)','Ref ID','N total','n calibration','n remaining']
rows=[[float(r['Budget_pct']),int(r['Ref_ID']),int(r['N']),int(r['n_cal']),int(r['n_remaining'])] for r in B]
s.get_range_by_indexes(0,0,len(rows)+1,len(headers)).values=[headers]+rows
s.get_range('A1:E1').format={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'row_height':30,'wrap_text':True}
s.get_range('A:E').format.column_width=20
s.freeze_panes.freeze_rows(1)

s=w.worksheets.add('Zero Shot')
headers=['Protocol','Stage','Model','N Test','R²','RMSE','MAE','Frozen NPZ SHA-256']
rows=[[r['Protocol'],int(r['Stage']),r['Model'],int(r['N_Test']),float(r['Zero_R2']),float(r['Zero_RMSE']),float(r['Zero_MAE']),r['Source_sha256']] for r in Z]
s.get_range_by_indexes(0,0,len(rows)+1,len(headers)).values=[headers]+rows
s.get_range('A1:H1').format={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'row_height':30,'wrap_text':True}
s.get_range('A:A').format.column_width=16
s.get_range('B:G').format.column_width=16
s.get_range('H:H').format.column_width=65
s.get_range('E2:G9').set_number_format('0.000000')
s.freeze_panes.freeze_rows(1)

s=w.worksheets.add('Method & Notes')
notes=[['Line 2: Random Forest, Stage 1–4 — Random Few-Shot Calibration'],
 ['Frozen zero-shot','Figure 2b audited row-level predictions; do not retrain RF'],
 ['Evaluation protocols','Publication-grouped fold-trained Test predictions and conventional random-fold-trained Test predictions, separately'],
 ['Scale','ln(1 + DBP concentration in µg/L)'],
 ['Test publications','Ref 7 (108), Ref 14 (72), Ref 16 (41), Ref 21 (12), Ref 22 (16); 249 total'],
 ['Budgets','2.5%, 5%, 10%, 20% per publication, with ceil(f * N)'],
 ['Actual n calibration','9, 15, 28, 53 pooled, respectively'],
 ['Sampling','500 random permutations generated from one np.random.default_rng(42); fixed Ref order 7, 14, 16, 21, 22; nested budget prefixes'],
 ['Identical draws','Same calibration indices reused for all Stage and CV strategy configurations'],
 ['Offset','Mean of y_true − y_pred over selected calibration rows independently per publication'],
 ['Prediction','Frozen y_pred + publication-specific offset on non-calibration rows'],
 ['Paired comparison','Zero-shot and calibrated evaluated on precisely same remaining rows in EACH replicate'],
 ['Summary statistics','Median and Q05/Q25/Q75/Q95 across 500 repetitions; Δ values computed replicate-by-replicate before taking median'],
 ['Caveat','Calibration samples are Test responses used for adaptation: this is retrospective simulated adaptation, not new blinded external validation'],
 ['Caveat','At least one calibration row per publication; actual budget 2.5% corresponds to 9/249=3.6% of all Test observations'],
 ['Caveat','Pooled metrics can hide study heterogeneity; check Per Publication sheet, especially Ref 16 and Ref 22'],
 ['Caveat','Rows represent DBP measurements; multiple rows can share an experimental setting'],
 ['Reproducibility','See reproduce_rf_random_fewshot.py, Random_Permutations_500x5.csv, Experiment_Manifest.json and full replicate tables in ZIP']]
for i,row in enumerate(notes):
 s.get_range_by_indexes(i,0,1,len(row)).values=[row]
s.get_range('A1:B1').format={'fill':'#17324D','font':{'bold':True,'color':'#FFFFFF'},'row_height':29}
s.get_range('A:A').format.column_width=32
s.get_range('B:B').format.column_width=110
s.get_range('A2:A18').format.font.bold=True
s.get_range('B2:B18').format.wrap_text=True
s.get_range('A2:B18').format.row_height=30

outfile=OUT/'RF_Line2_RandomFewShot_Results.xlsx'
SpreadsheetFile.export_xlsx(w).save(str(outfile))
print('EXPORTED',outfile, outfile.stat().st_size)
