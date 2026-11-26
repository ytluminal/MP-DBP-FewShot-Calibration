"""Create an accessible summary workbook from the source-only audit CSVs.
No pandas/openpyxl: artifact_tool only for workbook operations."""
import csv
from pathlib import Path
from artifact_tool import Workbook,SpreadsheetFile
P=Path('/mnt/data/Line2_FiveModel_Hypothesis1_20261014')
wb=Workbook.create()

def load(p):
 with open(P/p,encoding='utf-8',newline='') as f:
  r=csv.DictReader(f);rows=list(r);cols=r.fieldnames
 for row in rows:
  for c in cols:
   if c in ('Model','Protocol'):continue
   v=row[c]
   try:row[c]=float(v) if v!='' else None
   except (ValueError,TypeError):pass
 return cols,rows

def make(name,file,columns=None,col_sizes=None):
 cols,rows=load(file)
 if columns:cols=columns
 data=[[r.get(c) for c in cols] for r in rows]
 sh=wb.worksheets.add(name)
 sh.get_range_by_indexes(0,0,len(data)+1,len(cols)).values=[cols]+data
 sh.get_range_by_indexes(0,0,1,len(cols)).format={"fill":"#173651","font":{"bold":True,"color":"#FFFFFF"},"row_height":32,"wrap_text":True,"vertical_alignment":"center"}
 sh.get_range_by_indexes(1,0,len(rows),len(cols)).format.row_height=20
 sh.get_range_by_indexes(0,0,len(data)+1,len(cols)).format.vertical_alignment='center'
 for i,col in enumerate(cols):
  letter=''; x=i+1
  while x: x,rem=divmod(x-1,26);letter=chr(rem+65)+letter
  sh.get_range(f'{letter}:{letter}').format.column_width=(col_sizes.get(col) if col_sizes and col in col_sizes else (23 if col in ('Model','Protocol') else 16))
  if col in ('Fraction_improved_R2','Fraction_RMSE_improved','Fraction_improved_RMSE','Fraction_improved_MAE','Fraction_MAE_improved'):
   sh.get_range_by_indexes(1,i,len(rows),1).set_number_format('0.0%')
  elif col not in ('Model','Protocol') and col not in ('Stage','N_cal','N_eval','Ref_ID','N_model_stage','N_median_gain_positive','N_median_gain_negative','N_fraction_improvement_at_least_80pct'):
   sh.get_range_by_indexes(1,i,len(rows),1).set_number_format('0.0000')
 sh.freeze_panes.freeze_rows(1)
 return sh

info=wb.worksheets.add('Start Here')
info.merge_cells('A1:G2');info.get_range('A1').values=[['LINE 2  |  FIVE-MODEL RANDOM FEW-SHOT CALIBRATION']]
info.get_range('A1:G2').format={'fill':'#173651','font':{'bold':True,'color':'#FFFFFF','size':16},'row_height':22,'vertical_alignment':'center'}
rows=[
 ('Scope','5 models x 4 stages x 2 fold-training protocols x 4 budgets x 500 draws = 80,000 pooled evaluations.'),
 ('Methods','No ML refitting. One additive offset per held-out publication estimated from calibration labels.'),
 ('Budgets','2.5%, 5%, 10%, 20%; actual n_cal = 9, 15, 28, 53 in 249-row test.'),
 ('Primary contrast','Paired Delta R2: calibrated R2 minus zero-shot R2, both on identical remaining observations.'),
 ('Median R2 caveat','Zero_full_R2 scores all 249 rows; Cal_R2_median scores only the noncalibration remainder.'),
 ('Uncertainty','Q05–Q95 are percentiles across 500 random subsets, NOT confidence intervals for new publications.'),
 ('What is proven','Retrospective within-publication adaptation; NOT prospective external prediction with zero target labels.'),
 ('Dependence','The 500 draws share the same five target publications and are nested/reused across models.'),
 ('Rows overlap','Different DBP outcomes may share an experimental condition; separate condition-level evaluation needed.'),
 ('Source','Original five reproducibility ZIP files in /mnt/data; SHA-256s in Reproduction_and_Interpretation.json.'),
]
info.get_range('A4:B13').values=[[a,b] for a,b in rows]
info.get_range('A4:A13').format={'fill':'#E7EFF5','font':{'bold':True,'color':'#173651'},'wrap_text':True}
info.get_range('B4:B13').format.wrap_text=True
info.get_range('A4:A13').format.column_width=25
info.get_range('B4:B13').format.column_width=106
info.get_range('A4:B13').format.row_height=35
info.get_range('A15').values=[['Files: Budget Summary | Stage4 Grouped | All 160 | Per-Publication 800']]
info.get_range('A15').format.font.bold=True

budget_columns=['Protocol','Budget_pct','N_model_stage','N_median_gain_positive','N_median_gain_negative','N_fraction_improvement_at_least_80pct','Median_of_20_config_median_Delta_R2','Median_of_20_config_improvement_fraction']
make('Budget Summary','Budget_Overall_Comparison.csv',budget_columns,{k:27 for k in budget_columns})
col1=['Model','Stage','Budget_pct','N_cal','N_eval','Zero_full_R2','Zero_R2_median','Cal_R2_median','Delta_R2_median','Delta_R2_q05','Delta_R2_q95','Fraction_improved_R2','Zero_RMSE_median','Cal_RMSE_median','Delta_RMSE_median','Cal_MAE_median','Delta_MAE_median']
make('Stage4 Grouped','Stage4_Grouped_20_Results.csv',col1)
make('All 160','All160_Model_Stage_Budget_Summary.csv')
make('Per-Publication 800','All800_PerPublication_Summary.csv')
output=P/'Line2_FiveModel_Hypothesis1_Comparison.xlsx'
SpreadsheetFile.export_xlsx(wb).save(str(output))
print('Workbook:',output,output.stat().st_size)
print('Sheets:',[wb.worksheets.get_item_at(i).name for i in range(5)])
