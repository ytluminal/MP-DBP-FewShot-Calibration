Figure 2b (Line 2) Native CatBoost Stage 1–4 Random Few-Shot Calibration
2026-10-14

Methods:
- Input: Frozen zero-shot predictions from Figure 2b successful full 40-combination reproduction, eight NPZ archives in source/Line2_predictions.
- Both publication-grouped and random-fold trained predictions, four stages each.
- 249 external Test observations from Refs 7 (108), 14 (72), 16 (41), 21 (12), 22 (16).
- Calibration budgets per publication: 2.5%, 5%, 10%, 20%, with ceil(fraction*N_ref); total n_cal 9,15,28,53 and n_evaluation 240,234,221,196.
- A single numpy.default_rng(42) generates 500 ordered permutations across the five publications, same indices as the RF/XGB/LGBM preceding studies, nested across budgets.
- One publication-specific offset: mean(y_observed - y_prediction) on selected calibration rows on ln(1+DBP concentration) scale; correction applied to only the unselected rows.
- Matched zero-shot and calibrated performance computed on same remaining rows for every single replicate.
- Median and 5/25/75/95 percentile results, plus improvement probability, full raw replicate results, per-reference results, and supplementary plots.
- This is retrospective few-shot adaptation on the held-out publication dataset, NOT an unadapted prospective test. The row-level calibration and residual evaluation may include DBP analytes sharing the same experimental condition.
- No base model retraining, model tuning, or active selection.

Files:
reproduce_nativecatboost_random_fewshot.py — full simulation code.
audit_nativecatboost_results.py — numerical acceptance tests, original SI benchmarks and independent spot check.
make_figures.py — two grouped/random CV paired-gain plots and publication RMSE gains at 600 dpi.
make_workbook.py — export Excel summary using artifact_tool.
source/Line2_predictions/*.npz — complete frozen Stage1–4 × 2 protocols archive.
source/*xlsx — original immutable data workbooks.
outputs/NCB_Line2_RandomFewShot_Results.xlsx — readable main/per-publication tables and methods.
outputs/NCB_RandomFewShot_Replicates_16000.csv — 16,000 matched replicate metrics.
outputs/NCB_RandomFewShot_PerPublication_Replicates_80000.csv — 80,000 study-specific replicate records.
outputs/NCB_RandomFewShot_Summary_32.csv — 32 model × stage × protocol × budget summaries.
outputs/NCB_RandomFewShot_PerPublication_Summary_160.csv — 160 publication-level summaries.
outputs/Random_Permutations_500x5.csv — immutable common selection sequence.
outputs/Experiment_Manifest.json, outputs/Audit_Checks.json — hashes & acceptance checks.

To reproduce: python reproduce_nativecatboost_random_fewshot.py; python audit_nativecatboost_results.py; python make_figures.py; python make_workbook.py
