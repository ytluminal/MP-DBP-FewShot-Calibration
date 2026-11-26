# Line 2 — Five-model Random Few-Shot calibration: Hypothesis 1 audit (2026-10-14)

## Scope

Five locked prediction systems: Random Forest, XGBoost, LightGBM, Native CatBoost, Equal4. Four Stages and two fold-training protocols (publication-grouped and conventional random) were evaluated using the previously frozen 249-row external Test predictions. Four within-publication calibration budgets (2.5%, 5%, 10%, 20%; ceil per publication) and 500 random replicates per configuration produced 80,000 model-stage-protocol-budget-replicate pooled comparisons.

The analysis is retrospective: 249 observations in five held-out publications (Refs 7, 14, 16, 21, 22), not blind new data. The methods did not retrain models: selected labels from each target publication set one additive offset per publication, model, budget, and draw; the calibration observations were excluded from all evaluation scores.

## Outcome definitions

- `Zero_full_R2`: original zero-shot predictions evaluated across all 249 observations.
- `Zero_R2_median`: median over 500 zero-shot scores recalculated on each draw's remaining noncalibration observations.
- `Cal_R2_median`: median over 500 offset-calibrated predictions on those remaining observations.
- `Delta_R2_median`: median of within-draw differences `Cal_R2 - Zero_R2`, **not** `Cal_R2_median - Zero_R2_median`.
- `Fraction_improved_R2`: fraction of 500 draws with positive paired delta R²; this is an empirical sampling success fraction, **not** the probability of generalizing to a new, independent publication.
- For RMSE and MAE, negative paired delta signifies calibration improvement.
- Percentile bands Q05–Q95 describe **random-calibration-set variability conditional on these five publications**, not inferential confidence intervals from independent publications.

## Principal results

For publication-grouped CV configurations, the numbers of model x stage configurations whose median paired delta R² was positive were 16/20 at 2.5%, 19/20 at 5%, 20/20 at 10% and 20/20 at 20%. Corresponding conventional-random-fold figures were 17/20, 18/20, 20/20 and 20/20. All five Stage 4 grouped models had positive median paired gains at 20%, ranging from LightGBM +0.0476 to native CatBoost +0.8319. Native CatBoost's larger *gain* need not signify better absolute calibrated predictions: LightGBM Stage 4 still had the highest calibrated median test R² among Stage 4 models at 20% (0.5919).

Stage 4 grouped LightGBM median paired delta R² was **negative** at 2.5% (-0.0743) and 5% (-0.0243), with improvement in only 20.2% and 36.2% of simulations; improvement rose to 66.2% and 91.8% at 10% and 20%. More labels do not mechanically guarantee better predictions for every target publication or replicate.

At 20% Stage 4 grouped, the per-publication paired RMSE median decreased in 20 of 25 model x publication comparisons. Ref 16 and Ref 21 often failed to benefit. Refs 7 and 14 together contributed 180 of the 249 test observations (72.3%; at 20% budget, 143 of 196 remaining observations, 73.0%), so the pooled metric is weighted toward these two large publications.

## Scientific interpretation and scope

The evidence supports **conditional, within-target-publication label adaptation** as a method to reduce systematic prediction offsets, but not a universal guarantee for new studies. Gain size may correlate descriptively with worse zero-shot performance, but comparisons across models share data and mathematical R² effects, so the summary does not establish causality or general study-level significance. Additional replication on the independent Line 3 publication split and experiment-condition-disjoint calibration/evaluation would strengthen generalizability claims. Some Test observations share experimental conditions but differ by DBP analyte; holding out sibling analytes is not equivalent to holding out entirely new experiments.

The current analysis tests *Random Few-Shot*, not Active Selection. The two must be compared at equal budgets on suitable evaluation data before claiming an active acquisition advantage.

## Included files

- `Line2_FiveModel_Hypothesis1_Comparison.xlsx`: overview, all 160 configurations, and 800 per-publication summaries.
- `All160_Model_Stage_Budget_Summary.csv`: full 160-row metric summary.
- `All800_PerPublication_Summary.csv`: all 800 model/protocol/stage/budget/publication summaries.
- `Stage4_Grouped_20_Results.csv`: Stage4 grouped 20-row subset (4 budgets x 5 models).
- `Budget_Overall_Comparison.csv`: 8 budget/protocol aggregate lines.
- Two 600 dpi figures and vector PDFs.
- `build_analysis.py` and `create_summary_workbook.py`: rebuild outputs from the five previously archived experiment folders.
- `Reproduction_and_Interpretation.json`: source ZIP SHA-256 identities and core audit counts.

The five original full reproducibility ZIPs are intentionally **not duplicated** in this small summary package. Keep those ZIPs with this package to reproduce the entire chain.
