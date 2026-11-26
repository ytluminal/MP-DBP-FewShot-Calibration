Figure 2b — LightGBM Stage 1–4 Random Few-Shot Calibration
2026-10-14

CONTENTS
- reproduce_lightgbm_random_fewshot.py : Frozen Test prediction-based simulations.
- make_figures.py : 600-dpi publication-grouped/random-fold paired-gain plots and per-publication Stage4 RMSE plot.
- make_workbook.py : Excel table creation using artifact_tool.
- audit_lightgbm_results.py : numerical integrity, SI consistency, random sampling, and single-replicate independent audit.
- source/Line2_predictions/*.npz : all 8 locked Figure 2b zero-shot prediction files. These are inputs to the simulation; no LightGBM refitting is needed.
- source/Training_995_Locked.xlsx, source/Test_249_Locked.xlsx : immutable Line2 database for provenance.
- outputs/LGBM_Line2_RandomFewShot_Results.xlsx : 32 result rows; 160 per-study rows; budget counts; zero-shot reference values; protocol details.
- outputs/LGBM_RandomFewShot_Replicates_16000.csv : complete per-replicate paired metrics.
- outputs/LGBM_RandomFewShot_PerPublication_Replicates_80000.csv : complete per-study per-replicate metrics.
- outputs/Random_Permutations_500x5.csv : 500 fully reproducible sample sequences for 5 papers.
- outputs/Experiment_Manifest.json and Audit_Checks.json : identity/acceptance audits.

DESIGN
- Five independent Test publications: Refs 7,14,16,21,22; N=108,72,41,12,16.
- Budget fractions 2.5%,5%,10%,20%; per-publication ceil(f*N); total 9,15,28,53; eval 240,234,221,196.
- One np.random.default_rng(42), for each replicate permute rows in fixed reference order 7,14,16,21,22.
- 500 replicates; nested budget prefixes; identical random draws reused for RF/XGBoost/LightGBM, all stages and protocols.
- y and predictions on ln(1 + DBP concentration [ug/L]) scale.
- Per-publication calibration offset = mean(y_observed - y_frozen_prediction) on random calibration rows.
- Add offset to non-calibration rows only; no base model retraining, feature changes, or active selection.
- Pair calibrated and zero-shot metrics strictly on the SAME remaining Test rows per replicate.
- Report medians and Q05/Q25/Q75/Q95 for 500 replicates. Difference-median is computed per replicate (not difference of separate medians).
- Performance measured on retrospectively partitioned TEST response data; calibration measurements are not a new unseen prospective independent test.
- Observations are DBP analyte rows and are not necessarily distinct experimental-condition groups.

REPRODUCTION
Unzip archive; run `python reproduce_lightgbm_random_fewshot.py` from any current working directory.
Run `python audit_lightgbm_results.py` to assert consistency; `python make_figures.py` for plots;
`python make_workbook.py` (requires artifact_tool) to export the Excel workbook.
Raw original Line2 model training/reproduction code is in the separate Figure2_Reproduction_Audit_20261014.zip package.
