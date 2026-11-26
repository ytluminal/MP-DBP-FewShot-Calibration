# Figure 4 — Line 3 LightGBM active and random few-shot calibration on Refs 28–30

## Data and prediction inputs
- 64 new MP–DBP observations: Ref 28: 12; Ref 29: 12; Ref 30: 40.
- 8 frozen zero-shot models: Stage 1–4 × (Publication-grouped or Conventional random five-fold CV). Original row-level predictions and each of five fold outputs are stored in `input/Line3_Frozen_External_Predictions.csv`.
- The Line 3 Training set (`input/line3_train55.json`, 1,026 rows) is used ONLY to fit active-selection numeric encoding (median imputation and scaling) and categorical mode imputation / one-hot encoding. The predictive models are not retrained.
- New data for features and labels: `input/external55.json`.
- Random calibration indices: the same 500 ordered random permutations already used for the Line 1 and Line 2 new-data few-shot comparisons.
- Stage matched feature subsets: 22, 55, 55, 14 raw features. Stage 4 derives from `SI_Conventional_RandomSplit_Line3_Stage1-4.docx`, Table 53 (one-based), LightGBM selected features.

## Method
1. Predict zero-shot with frozen LightGBM five-fold ensembles, average on ln(1+C) scale. These predictions are never refit.
2. For each publication and budget (2.5, 5, 10, 20%), choose ceil(n_pub * fraction) calibration records, at least one and leaving at least one evaluation record.
3. Random selection: take the same number of first elements of the 500 saved random permutations.
4. Active selection: select the nearest-to-centroid point in train-encoded feature space, then maximin diversity greedily; **no target labels used for sample selection**.
5. Publication-specific mean-offset is the mean of (true ln1p concentration − zero-shot prediction) over selected calibration samples. Add offset to all predictions for that publication. **No LightGBM parameter fitting occurs**.
6. Random-own and Active-own evaluation exclude own calibration samples. Strict matched comparisons exclude the union of Active and Random calibration indices, using exactly the same remaining observations for both methods and zero-shot.
7. R², RMSE and MAE are on ln(1+C [µg/L]); paired summary uses median across 500 paired random replicates, with q05/q95 uncertainty percentiles.

## Reproduce
Requires Python with numpy and scikit-learn (script also uses only standard-library csv/json).
Run in this folder: `python reproduce_line3_newdata_fewshot.py`
Output CSV files are regenerated in `output/` with fixed random permutations. The `output/Experiment_Manifest.json` stores SHA256 hashes of result files.

## Contents
- `Figure4_Line3_LightGBM_Random_Active_FewShot_Ref28_29_30.xlsx`: polished comparison summary.
- `output/ZeroShot_Baseline.csv`: metrics on all 64 new observations per scheme/stage and per publication.
- `output/Random_Own_Replicates.csv`, `Random_Own_Summary.csv`: 500 independent random calibration repeats, own remainder.
- `output/Active_Own_Overall.csv`, `Active_Own_Publications.csv`: active calibration on its own held-out remainder.
- `output/Matched_Overall_Replicates.csv`, `Matched_Publication_Replicates.csv`: strict paired metrics from common evaluation rows, including paired deltas.
- `output/Matched_Overall_Summary.csv`, `Matched_Publication_Summary.csv`: median/quantiles of matched metrics.
- `output/Active_Complete_Sequences.csv`, `Active_Budget_Selections.csv`, `Active_Offsets.csv`: selected points and offsets.
- `input/stage_matched_feature_lists.json`: Stage-matched raw features.

## Interpretation warning
The calibration stage uses observed DBP concentrations on a small labeled subset; those labeled rows are never scored. These results are **few-shot adapted predictions**, not independent zero-shot external performance. Separate confirmation that Ref 30 is an independent publication distinct from the original Ref 27 is still needed before any claim of strict publication-held-out validation.
