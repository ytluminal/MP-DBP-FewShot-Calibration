LINE 2 LIGHTGBM: REPRODUCED AND VERIFIED 2026-10-27

This package contains newly FITTED LightGBM models reconstructed from the frozen Line 2 original training dataset and recipe (not original saved model binary files).

Four stages x TWO evaluation / training-fold designs x FIVE folds = 40 fitted LightGBM models. Each .joblib stores BOTH its fitted LightGBM regressor AND all fold-fitted preprocessing objects.

Model development uses only original 995 Training rows; all 249 original Test rows were used only for verification. Neither Ref 28, 29, nor 30 nor any newly collected external test data were read for this reconstruction.

Original Train workbook SHA256: de1e28d1525c97d802e5b3dd261dd7409bea3bbeade02875d3ac36fe76947fb9
Original Test workbook SHA256: 75e5164aba4c974e54f4974dee86a65d37228e9194ada504f6133552c77fdfc8

Stages:
 S1: 22 original experimental features; hyperparameters from recovered nested 3-fold GroupKFold search.
 S2: 55 original features; frozen per-fold hyperparameters.
 S3: 55 features; publication-size weighting with locked fold-specific alphas.
 S4: 18 locked, original RFE-selected predictors, plus weighting.
Both "grouped" and "random" use identical original training observations and frozen stage-matched settings. Per-fold models differ in training partition, not feature schema.

To predict *unseen data*: keep 'line2_lightgbm_preprocess.py' importable and run:
 python predict_with_saved_line2_lightgbm.py --input external_data.xlsx --output new_predictions.csv
This reads only feature columns from Data worksheet, produces 5-fold ensemble-mean prediction in ln(1+C) and micrograms/L for each stage and CV design, and NEVER retrains. Do not use reference 28-30 labels to select, fit, or tune a model. Labels may be used only later for evaluation.

To re-run the same reconstruction again, place the ORIGINAL Line2_Reproducibility_Code_Data_20261013(8).zip contents (run.py, train55.json, test55.json, SI_Verified.docx, audit_stage1_results.json, AUDIT_FINAL_All_Stages_Results.json) alongside 'reproduce_lightgbm_only.py'.

Validation is in LightGBM_Reproduced_Models/reproduction_summary.json and .npz files. Exact reproduced stage/split performance matches the reference archived JSON for CV R2, RMSE, MAE and original Test R2, RMSE, MAE to floating point round-off. Reloading saved model files + preprocessors reproduces all previously reproduced 249-row old-test predictions exactly.

These matches verify the frozen computational recipe and published numerical benchmarks, not bytewise identity to original trained model files (none were present in the user-supplied ZIP). No additional tuning or model selection was performed.

Environment: Python 3.13.5, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, scikit-learn 1.8.0, LightGBM 4.6.0. joblib files should be loaded in a compatible environment; never unpickle files from an untrusted source.
