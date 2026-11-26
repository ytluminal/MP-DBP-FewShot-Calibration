# Fitted model binaries for Figure 2a / Line 1

**Model files: 108 `.joblib` bundles = 18 final full-Training fits + 90 five-fold CV fits.**

These are actual fitted estimators, NOT hyperparameter-only config files. Each joblib file
stores the fitted model plus its own fitted numerical median imputer, optional numerical
StandardScaler, fitted OneHotEncoder (when used), feature order, categorical/numerical
column identities, original Training-row positions, and CV validation indices where
applicable. Native CatBoost stores its native categorical-missing representation and
uses native numerical-NaN handling.

## File naming conventions

* `<stage>_<algorithm>_full_training.joblib`: model refit on all 995 frozen Training
  observations. Use to predict the 249 frozen Test observations or new observations.
* `<stage>_<algorithm>_cv_fold<k>.joblib`, k=1..5: one independent model fit using
  the four non-validation folds. Use the saved fold's `validation_row_positions`
  to regenerate the true out-of-fold CV predictions. **Do not use the full-Training
  model to reconstruct CV scores.**

Stages S1, S2, S3 and algorithms Elastic Net, SVR-RBF, Random Forest, XGBoost,
LightGBM, CatBoost are all present. S3 uses previously frozen RFE subsets. Every
binary was freshly trained on 2026-10-27, using the same frozen model recipe as
`reproduce_line1.py`. Original historical fitted objects had not been archived.

## Reuse without any model retraining

Run from the extracted archive root:

```bash
python predict_saved_model.py --stage S3 --algorithm LightGBM --input "data/Conventional_RandomSplit_Stage2_Test_55features(3).xlsx" --output saved_model_predictions.csv
python validate_saved_models.py
```

The first command loads the stored final LightGBM and writes both `ln(1+C)` and
back-transformed concentration `C` in µg/L. The validation script loads all 108
objects and compares their predictions with original `results/*_preds.npz` arrays.
It **does not refit**. All 18 stage × algorithm combinations passed: maximum
prediction deviation was only floating-point roundoff (~4e-15).

### Security and portability

**Only load these joblib/pickle files from trusted sources.** Joblib loading can
execute arbitrary Python code. Keep the saved checksum manifest and pin the
versions in `requirements.txt`; model binary compatibility is not guaranteed
across future Python/library versions or operating systems. The archive includes
all raw data and code to retrain if needed.

### Scientific caveat

The S2/S3 Random Forest predictions and S2/S3 SVR Test R² have the previously
reported minor discrepancies versus SI Table S1. These persisted in the archived
predictions and are **not** hidden or "corrected" by the serialized models.
The files reproduce the frozen-parameter training/refitting procedure, **not**
the original 20-trial Optuna tuning history or complete RFE search path.
