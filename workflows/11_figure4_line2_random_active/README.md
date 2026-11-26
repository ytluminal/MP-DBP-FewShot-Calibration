# Figure 4 — Line 2 LightGBM external Random + Active few-shot calibration

## Scope
Ref 28, Ref 29 and Ref 30 (12, 12, 40 measured DBP concentrations, respectively); 64 rows in total. Line 2 Stage 1–4, Grouped / Random 5-fold-ensemble LightGBM (8 locked prediction configurations). 2.5%, 5%, 10%, 20% calibration budgets per publication. Reuse exact 500 random permutations previously used on these 64 records for Line 1.

## How to reproduce
From this extracted folder, with Python + NumPy + scikit-learn installed:

```
python reproduce_line2_newdata_fewshot.py
```

The script writes all results into `output/`; it reads only the provided `input/` files and does not fit or fine-tune any LightGBM estimator. Inference inputs are **frozen LightGBM predictions**, computed in the prior Line 2 reproduction. Original trained models are separately archived in `Line2_LightGBM_Reproduced_Models_20261027(1).zip` (not duplicated here). To reproduce the complete pipeline *starting from model artifacts*, consult that separate model ZIP and its exact preprocessing scripts first.

## Method
- **Outcome scale:** ln(1 + DBP concentration [µg/L]). All reported R², RMSE, and MAE are on this scale.
- **Random acquisition:** 500 fixed, repeated, per-publication permutations, shared across all stages and Grouped / Random CV schemes. Larger budgets use nested prefixes of each permutation.
- **Active acquisition:** Stage 1 / 2 / 3 / 4 LightGBM predictor sets with 22 / 55 / 55 / 18 raw features; numeric median imputation + StandardScaler on original Training only, categorical mode imputation + OneHotEncoder fit on Training only. In each publication choose one observation nearest the feature centroid, then maximin diversity (Euclidean, original row-order tie breaking).
- **Calibration:** Within each publication, compute mean offset Δ = mean[actual ln1p - frozen predicted ln1p] on selected shots. Add Δ to all frozen predictions from that publication. Selected shots never enter evaluation.
- **Random Own / Active Own:** evaluate on each method's distinct remainder. Random results are 500-draw medians; Active results are deterministic values.
- **Paired:** for each random draw, exclude the *union* of Active and Random calibration rows from the evaluation and report zero-shot, Random, and Active on the *same* remaining records. The `Paired_dR2_median` is the median of per-draw Active−Random differences, not the difference of the separate medians.
- **Critical caveat:** These are retrospective few-shot adaptations that use true labels on selected calibration rows and are therefore not fully zero-shot external predictions. No LightGBM model weights were updated.

## Files
- `input/line2_train55.json` — 995 original Line 2 training records for fitting acquisition encoders only.
- `input/external55.json` — 64 external rows (including measured responses used only for selected calibration shots and honest scoring).
- `input/stage_matched_feature_lists.json` — original Stage 1–4 LightGBM active feature sets.
- `input/Line1_Line2_LightGBM_Row_Predictions.csv` — original audited frozen Line 2 prediction values for all 64 rows and all eight model configurations.
- `input/Original_Line1_Random_Permutations_500.csv` — fixed, reusable random row orders.
- `output/Matched_Overall_Summary.csv`, `output/Matched_Publication_Summary.csv` — paired 500-replicate summaries (R², RMSE, MAE + gains and quantiles).
- `output/Matched_Overall_Replicates.csv`, `output/Matched_Publication_Replicates.csv` — all 16,000 / 48,000 paired repeated scores.
- `output/Random_Own_Summary.csv`, `output/Random_Own_Replicates.csv` — random performance on own unselected rows.
- `output/Active_Own_Overall.csv`, `output/Active_Own_Publications.csv` — active performance on own unselected rows.
- `output/Active_Budget_Selections.csv`, `output/Active_Complete_Sequences.csv`, `output/Active_Offsets.csv` — selected shot provenance and applied offsets.
- `output/ZeroShot_Baseline.csv`, `output/Feature_Manifest.csv`, `output/Experiment_Manifest.json` — anchors, exact features and checksums.

## Important independence check
Before claiming that Ref 30 is completely publication-independent, independently check the original paper's identity against Training Ref 27. Model-source novelty is not established simply by assigning a new Ref ID.
