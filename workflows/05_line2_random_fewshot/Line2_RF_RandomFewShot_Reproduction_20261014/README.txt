LINE 2 RANDOM FOREST STAGE 1–4 — RANDOM FEW-SHOT REPRODUCTION
Generated 2026-10-14 from the audited Figure 2b NPZ prediction archives.

SCOPE
- Frozen Figure 2b Random Forest S1–S4; two test prediction types:
  Publication-grouped fold-trained and conventional random-fold-trained.
- Original independent held-out Test database: 249 DBP records from Refs 7,14,16,21,22.
- The predictions are reused; no RF refitting, feature selection or hyperparameter search.
- Response = ln(1 + DBP concentration [µg/L]).

RANDOM FEW-SHOT PROCEDURE
- 500 random replicates from one np.random.default_rng(42) instance.
- In each replicate, independently permute within Ref 7, then Ref 14,16,21,22.
- Nested prefixes define 2.5%,5%,10%,20% calibration budgets, using ceil(f*N_ref).
- Total selected calibration records are respectively 9, 15, 28, 53;
  evaluation records respectively 240, 234, 221, 196.
- All four stages and both training protocols use identical calibration records.
- Publication-specific offset = mean(observed transformed y - frozen predicted transformed y)
  on calibrators from that publication alone.
- Add that offset to frozen test predictions of the remaining records in that publication.
- Evaluate calibrated and uncalibrated on EXACT same remainder for each replicate.
- Summarize median, 5th,25th,75th,95th quantiles and fraction of replicates improved.

CORE RESULTS (publication-grouped fold-trained RF predictions)
Stage; original 249-row zero-shot R2; median calibrated R2 at budgets 2.5/5/10/20%
S1; 0.204644; 0.4456 / 0.4841 / 0.5360 / 0.5671
S2; -0.088660; 0.4161 / 0.4549 / 0.5032 / 0.5315
S3; -0.054448; 0.4144 / 0.4535 / 0.5024 / 0.5317
S4; -0.049307; 0.4161 / 0.4559 / 0.5036 / 0.5315
IMPORTANT: These zero-shot baselines are for all 249 rows; use the paired Delta_R2 columns
for apples-to-apples performance gains on identical remaining rows.

IMPORTANT SCIENTIFIC QUALIFICATIONS
- Stage 4 grouped-CV pooled gains are primarily driven by Refs 7 and 14.
  Ref 16 typically shows worsening RMSE after offset calibration, and Ref 22 often does too.
- 2.5% corresponds to nine calibrators in all (3.6% of 249), with only one calibrator
  in the two smallest target publications; some random draws worsen predictions.
- The recorded Test response is retrospectively used for few-shot adaptation; this is
  not an additional blinded external Test experiment.
- A test row is a reported DBP measurement, not necessarily an independent experimental
  condition; other DBP records can share the same experimental setting.
- 500 random repetitions reflect selection sensitivity, NOT 500 independent publications.

REPRODUCTION
1. Extract the ZIP with this directory structure preserved.
2. Run: python reproduce_rf_random_fewshot.py
3. Optional visualizations: python make_figures.py
4. Optional Excel report: python make_workbook.py (requires artifact_tool)
5. Compare output CSV SHA-256 identities to Experiment_Manifest.json.

OUTPUTS
- outputs/RF_Line2_RandomFewShot_Results.xlsx: summarized human-readable workbook.
- outputs/RF_RandomFewShot_Summary_32.csv: main 32 comparisons (2 protocols x 4 stages x 4 budgets).
- outputs/RF_RandomFewShot_Replicates_16000.csv: full paired pooled metrics.
- outputs/RF_RandomFewShot_PerPublication_Summary_160.csv: all study-level summaries.
- outputs/RF_RandomFewShot_PerPublication_Replicates_80000.csv: detailed study-level replicates.
- outputs/Random_Permutations_500x5.csv: exact random selection rows; budgets as nested prefixes.
- outputs/Per_Publication_Budgets.csv: sample counts.
- outputs/ZeroShot_Original_249.csv: full-Test zero-shot targets.
- outputs/*.png and *.svg: 600-dpi exploratory figures and vector graphics.
- outputs/Experiment_Manifest.json: settings and cryptographic hashes.
