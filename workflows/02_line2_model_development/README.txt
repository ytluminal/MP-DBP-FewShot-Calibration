LINE 2 REPRODUCIBILITY PACKAGE (13 October 2026)
==============================================
Includes the exact, SHA256-verified source Training/Test XLSX datasets,
updated SI, stage-1 recovered fold-specific settings, source code and audited
40-row grouped/random CV and external Test metrics.

Important scientific scope:
* Stage1-4 RF, XGBoost, LightGBM, native CatBoost and Equal4 results are
  fixed-recipe reconstructions, not newly tuned models.
* Stage4 predictor subsets are kept fixed from the original RFE study.
  The complete feature-elimination trajectory was not independently repeated.
* Original Stage3/4 MoE4 router predictions and affected RF/Equal4/MoE4
  few-shot results have been excluded from the updated SI; rerun them
  separately before reporting.
* External Test predictions reuse the same previously exposed 249 rows.

Tested environment in this session:
Python 3.13.5, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0,
scikit-learn 1.8.0, XGBoost 3.1.3, LightGBM 4.6.0, CatBoost 1.2.8.
For reproducibility the script sets OPENBLAS_NUM_THREADS=5,
MKL_NUM_THREADS=5 and OMP_NUM_THREADS=3 before importing numpy.

Run in this folder:
  python prepare_datasets.py
  python AUDIT_FINAL_Line2_5BLAS.py
Compare the resulting AUDIT_FINAL_All_Stages_Results.json to the shipped
reference JSON. Output may depend on operating system/BLAS build.

The standalone RF-only file rf_reproduction_from_si.py references absolute
paths from the original session and is provided as an additional audit trail,
not as the primary standalone entry point.

This package is a frozen-configuration rerun. Rebuilding Stage-1 original
hyperparameter selection from scratch is documented in
'audit_stage1_original_parameter_selection.py' and its result JSON.
