# Line 3 Stage-matched Centroid + Maximin Active Few-Shot rerun (2026-10-16)

## Decision and scope
- Recomputed all 160 configs: 5 models (RF, XGBoost, LightGBM, Preprocessed CatBoost, Equal4) × 4 stages × 2 model CV protocols × 4 calibration budgets.
- Stage 1 selects using 22 original model predictors; Stages 2 and 3 select using 55; Stage 4 uses RF 14, XGBoost 14, LightGBM 14, Preprocessed CatBoost 28, and Equal4 union of 32.
- Exact same frozen Line 3 model predictions, Train/Test membership, 500 saved Random permutations, budget rounding (ceil with min 1 and max n-1), and per-publication **mean-offset** calibration. No WCA-FL.
- Feature encoding for selection: numeric Training-only median+StandardScaler; categorical Training-only most frequent+OneHotEncoder. No Test DBP labels are used in selection, only calibration labels after selection.

## Results and audits
- 160 configs / 80,000 paired random replicates / 560,000 per-publication paired records.
- Exact feature set and selection sequences are in `results/Stage_Model_Feature_Manifest.csv`, `results/Active_Complete_Sequences.csv`, `results/Active_Budget_Selections.csv`.
- Stage 2 and Stage 3 selections exactly match old 55-feature method and their 80 grouped summary configs agree within 1e-12.
- `Active_R2_median` in All_160 summary means median Test R² on the samples neither Active nor Random acquired for each Random repeat. `Active_Own_R2` is Test R² on Active's own unselected remaining observations. These two should never be conflated.
- `Paired_dR2_median` is the median of (Active R² − Random R²) replicate-wise on identical remaining samples. It is NOT the difference between medians.
- `Old_vs_New_160_Descriptive_Comparison.csv` includes old vs new performance on their corresponding matched evaluation subsets. Different active selections change evaluation subsets, so differences there are **descriptive only**, not a strictly same-row paired effect.

## LightGBM Line 3: median paired common evaluation Test R²
- Publication-grouped, S1: 2.5%=0.3777, 5.0%=0.4133, 10.0%=0.2326, 20.0%=0.4335
- Publication-grouped, S4: 2.5%=0.3695, 5.0%=0.3618, 10.0%=0.0816, 20.0%=0.3031
- Random folds, S1: 2.5%=0.3429, 5.0%=0.3576, 10.0%=0.2184, 20.0%=0.4119
- Random folds, S4: 2.5%=0.3785, 5.0%=0.3819, 10.0%=0.0811, 20.0%=0.3245

## Running the analysis
Extract this ZIP, then run in the `work` folder:
```
python rerun_line3_stage_matched.py
```
Requirements: Python with numpy, scikit-learn. The `stage_matched_feature_lists.json` is extracted exactly from the original Figure 2c SI Stage 4 predictor lists. The original `run_active_fewshot.py` is included as an unchanged protocol reference. No model retraining is involved.
