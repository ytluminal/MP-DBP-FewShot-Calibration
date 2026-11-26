MASTER SHAP DATASET — Line 2, Stage 3, Publication-grouped LightGBM
Canonical analysis: 995 OOF rows from 22 publications, 55 original features. No model fitting or data repartitioning.

FILES
Master_SHAP_Exact.npz: canonical full-precision binary arrays; 995 x 55 SHAP, features, predictions, base values, fold IDs, original record IDs and observed target.
Master_SHAP_Dataset.csv: original row keys + observed/predicted log target + base and 55 original-feature SHAP columns.
SHAP_Matrix_995x55.csv: pure 995 x 55 matrix, in zero-based training_row order.
Original_Training_Records.csv: original experimental records and source Excel row numbers, with fold IDs. Original workbook also included.
OOF_Predictions_and_Fold_Assignments.csv: archived/recomputed predictions, base values, observed target, SHAP sums and errors.
Global_SHAP_Importance.csv: mean absolute contributions for all 55 features.
fold_matrices/fold_1 to fold_5: exact CSR matrices for validation rows and recomputed training matrices using frozen fitted preprocessors; row mapping, transformed feature mapping and native encoded SHAP arrays with base and predictions. Only validation matrices were used for the OOF SHAP dataset.
models/: five saved LightGBM models and their fitted preprocessors; original archived OOF prediction file. Original model reproduction/inference and preprocessing scripts are retained. Original model reproduction script depends on additional original run artifacts; not necessary for inference or plotting.
Validation_Report.json: dimensions, exact agreement with archive, additivity, fold composition, package versions, model/workbook hashes and thread configuration.
Figure5d_Consistency_Check.json: previously saved Figure 5d values agree with canonical archive within numerical tolerance.
legacy/: archived original b/c plotting data and plotting code; kept for audit, not canonical interpretation data.
figures/: regenerated b/c source data and plots (identical plotting code and original row selections and trend bins), plus recreated Figure 5a. Original 5a plotting script was not supplied; design is reconstructed, with deterministic beeswarm offsets and raw numerical feature-value colors; gray denotes categorical or missing raw values. Exact legacy beeswarm locations/colors are not claimed. All three use canonical SHAP.

REPRODUCE
1. In the package directory, install versions listed in requirements.txt. Run: python build_master.py
2. Run: python prepare_plot_data.py
3. Run: python figures/plot_Figure5b_c.py
4. Run: python plot_Figure5a.py
For future analyses, load Master_SHAP_Exact.npz and the stored fold matrices; do not rerun KNN imputation unless rebuilding deliberately. Fold-specific encoded feature dimensions differ; the unified 55-column matrix sums one-hot SHAP contributions into the corresponding original categorical feature. All SHAP values and predictions are on ln(1+C_DBP) scale. Additivity: prediction = base + sum(SHAP).
