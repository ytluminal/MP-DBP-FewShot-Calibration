# Figure 2c / Line 3 — Five-model Random Few-Shot Calibration

## What was independently executed

Five model systems (Random Forest, XGBoost, LightGBM, **preprocessed** CatBoost, Equal4), Stages 1–4, two training-validation fold schemes (publication-grouped and conventional random), four nominal within-publication calibration budgets (2.5%, 5%, 10%, 20%), and 500 fixed random-sampling replicates per configuration. 80,000 pooled calibration evaluations were generated, plus 560,000 study-specific replicate evaluations.

We use the independently reproduced **frozen** per-row *external Test* predictions from Figure 2c (`input/Line3_Test_Row_Predictions.csv`) and audit (`input/Line3_80_Metric_Audit.csv`), never fit/retrain/tune the ML models or change feature sets. Those source prediction vectors are matched against 40 Test R²/RMSE/MAE audit results and tested for consistent row ordering and Equal4 prediction means before running.

## Locked method

- Line 3 external test set: 218 DBP observations, 7 publications: Ref 4 = 16 rows; Ref 11 = 90; Ref 12 = 6; Ref 13 = 4; Ref 14 = 72; Ref 20 = 16; Ref 26 = 14.
- Calibration per publication `n_cal = ceil(f * N_ref)`; the actual total counts are **10, 14, 25, 47** rows for budgets **2.5%, 5%, 10%, 20%**, leaving **208, 204, 193, 171** evaluation rows. Every publication is sampled and calibrated separately. Very small Ref 13 has 1 calibration row at all budgets.
- One `np.random.default_rng(42)` generator; 500 replicates. At each replicate, permute each publication's row indices in fixed Ref ID order `4, 11, 12, 13, 14, 20, 26`; calibration budgets are increasing prefixes of the *same permutation*. These permutations are reused for all 40 locked model configurations.
- For publication `r`, compute `delta_r = mean(y_true_cal - yhat_zero_cal)` on the frozen transformed response `y = ln(1+C_DBP)`; predict each **unselected** evaluation observation as `yhat_zero + delta_r`.
- Compute pooled R², RMSE, MAE on **identical noncalibration rows** before and after offset calibration (same-remainder pairing). Report the median paired deltas; do **not** subtract separate median scores to calculate paired gains. Random 5/25/50/75/95% quantiles summarize resampling variability, not independence-based confidence intervals.

## Reproduce the analysis

In a Python environment with NumPy:

```bash
python reproduce_line3_random_fewshot.py
python audit_nonoverlap.py
```

The first script regenerates all seven core CSV files in `outputs/`. The second regenerates the additional sensitivity CSV excluding Ref 14 from **evaluation only**. With `artifact_tool` and matplotlib installed, `python build_report.py` regenerates the formatted Excel workbook and three print-resolution figures (both scripts and data necessary are archived). Figure 2c raw model refitting is a separate archived reproducibility chain, not necessary to rerun calibration from frozen predictions.

`outputs/Independent_Rerun_SHA256_Audit.json` records **seven byte-identical core CSV files** from an independent clean-folder rerun using the two raw input CSVs and standalone analysis script.

## Important scientific limitations

1. **Line 2 and Line 3 test partitions share Ref 14, comprising the exact same 72 DBP observations**. They should not be described as independent or non-overlapping cross-publication test datasets. The sensitivity analysis `Line3_ExcludeSharedRef14_Sensitivity_160.csv` removes these 72 rows from scoring and recalculates matched zero-shot and calibrated metrics across the remaining six held-out Line 3 publications.
2. At 2.5% budget, the six-nonoverlap-publication grouped analysis has positive median paired R² in only 6 of 20 model-stage configurations, versus 19/20 including Ref 14. At 5/10/20%, all 20 grouped configurations have a positive median paired R² after excluding Ref 14. Absolute few-shot pooled R² nevertheless remains low for the nonoverlap-only subgroup.
3. The test data are retrospective, and there are only 7 held-out publications; 500 permutations over those fixed publications are **not 500 statistically independent publications**. Data sharing experimental conditions but varying by measured DBP analyte may be present; row-level withholding is not necessarily experimental-condition-level generalization.
4. Nominal budget percent differs from the **actual** proportion selected because `ceil` is applied independently to each publication.
5. Pooled metrics weight publications proportionally to their remaining observations (Refs 11 and 14 comprise 162/218=74.3% of Test rows), so additionally inspect the per-reference diagnostics. Stage4 at 20% budget had reduced median paired RMSE in 22/35 model×publication combinations; Ref11 did not improve for any of the five models.
6. This evaluates **Random Few-Shot Calibration only**. No Active Selection claim should be inferred.

## Main artifacts

- `outputs/Line3_FiveModel_RandomFewShot_Comparison.xlsx` — formatted dashboard plus all 160 configuration rows, 1,120 per-publication summaries, Ref14-excluded sensitivity, and Line2-vs-Line3 contrasts.
- `outputs/Line3_RandomFewShot_Summary_160.csv` — full median and quantile scores per configuration.
- `outputs/Line3_RandomFewShot_Replicates_80000.csv` — all pooled matched calibration comparisons.
- `outputs/Line3_RandomFewShot_PerPublication_Summary_1120.csv` — per-publication summaries.
- `outputs/Line3_RandomFewShot_PerPublication_Replicates_560000.csv` — all matched per-publication comparisons.
- `outputs/Random_Permutations_500x7.csv` — exact random selection orders.
- `outputs/Line3_ExcludeSharedRef14_Sensitivity_160.csv` — non-overlap sensitivity.
- `outputs/Line3_Stage4_Grouped_Paired_R2_Gains_600dpi.png`, `outputs/Line2_vs_Line3_Stage4_R2_600dpi.png`, `outputs/Line3_Stage4_PerPublication_RMSE_20pct_600dpi.png` — figures (also PDFs).
