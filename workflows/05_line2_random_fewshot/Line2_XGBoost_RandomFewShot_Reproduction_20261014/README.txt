LINE 2 — FIGURE 2b — XGBOOST RANDOM FEW-SHOT CALIBRATION
Generated 2026-10-14

SCOPE
Frozen XGBoost predictions, Stage 1–4, two model-training validation protocols:
(1) Publication_grouped folds (Grouped), (2) conventional random folds (RandomCV).
For each of 8 frozen prediction vectors: 2.5%, 5%, 10%, 20% per-target-publication
calibration fractions; 500 draws per fraction. Total 16,000 simulated calibrations.
No XGBoost model fitting, feature selection or model tuning was performed here.

INPUTS
source/Line2_predictions/AUDIT_FINAL_Stage{1..4}_{Publication_grouped|Conventional_random}.npz
Use Actual_Test_ln1p, Test_Ref, Test_pred_XGBoost. All 8 inputs have same
249-row Test outcome and Ref ID ordering. Source NPZ SHA256 values are in
outputs/Experiment_Manifest.json. These NPZ were recovered from the earlier
independently verified Figure 2b model-reproduction archive.

SAMPLING (IDENTICAL TO PREVIOUS RANDOM FOREST EXPERIMENT)
Target publications: Ref 7 (108), 14 (72), 16 (41), 21 (12), 22 (16).
For each random replicate, one np.random.default_rng(42) state is advanced by
sampling permutations in reference order 7,14,16,21,22. Calibration budget
n_ref = ceil(fraction * reference count); the 4 budgets use nested prefixes
of each reference permutation. Same 500 permutations reused across all
Stages and both training protocols; exact RF-vs-XGB selections byte-identical.
Total calibration rows (2.5%, 5%, 10%, 20%): 9, 15, 28, 53.
Remaining Test scoring rows: 240, 234, 221, 196.

CALIBRATION
On transformed outcome y = ln(1 + C_DBP [ug/L]), compute offset separately
in each publication: delta_ref = mean(y_true_cal - yhat_zero_cal).
Predict each remaining row with yhat_new = yhat_zero + delta_ref.
No calibration response is scored on the final remaining-row Test subset.
For each replicate compare zero-shot vs calibration on exactly identical
remaining rows. Evaluate pooled R2, RMSE, MAE; also calculate per-publication
scores, all without training/predictor modifications.

OUTPUTS
outputs/XGB_RandomFewShot_Summary_32.csv: 8 baseline models x 4 budgets;
    median, Q05/Q25/Q75/Q95 of R2/RMSE/MAE and paired improvements.
outputs/XGB_RandomFewShot_Replicates_16000.csv: complete pooled metrics per draw.
outputs/XGB_RandomFewShot_PerPublication_Summary_160.csv: per-Ref diagnostics.
outputs/XGB_RandomFewShot_PerPublication_Replicates_80000.csv: per-Ref metrics per draw.
outputs/Random_Permutations_500x5.csv: all 2500 exact ordered selections.
outputs/ZeroShot_Original_249.csv: uncalibrated full-249 Test scores.
outputs/Per_Publication_Budgets.csv: reference-specific number of calibrators.
outputs/Experiment_Manifest.json: frozen inputs and output hashes.
outputs/XGB_Line2_RandomFewShot_Results.xlsx: formatted 4-sheet report.
outputs/XGB_*_600dpi.png and *.svg: paired R2 gains and per-ref RMSE effects.

REPRODUCE
From the ZIP's extracted root:
  python reproduce_xgb_random_fewshot.py
  python make_figures.py
  python make_workbook.py
Use Python with numpy and matplotlib, plus artifact_tool for the .xlsx report.
The result already cross-checks the previous RF experiment's sampling sequence
and the existing SI historical XGBoost Stage3/Stage4 5%,10%,20% random medians.

INTERPRETATION AND LIMITATIONS
The original zero-shot Test score is on all 249 rows; few-shot scores use only
remaining rows. Judge gains from *paired same-remainder* metrics, not direct
subtraction of the all-row score. This is a retrospective simulated test-time
calibration analysis, not a new prospectively untouched external validation.
Target publication rows may be DBP-species siblings from shared experimental
settings; a group-by-experiment sensitivity analysis is recommended later.
Pooled calibration benefits do not mean every publication or every budget
benefits. In particular small budgets are unstable for some publications.
