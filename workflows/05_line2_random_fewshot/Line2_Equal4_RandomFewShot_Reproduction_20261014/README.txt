Figure 2b (Line 2) — Equal4 Random Few-Shot Calibration, stages 1–4
2026-10-14

WHAT THIS ARCHIVE CONTAINS
Frozen original dataset workbooks, 8 independently audited Figure 2b prediction arrays (.npz),
complete random selection indices, 16,000 replicate-level paired scores, 80,000 publication-level
paired scores, 32 pooled summaries, 160 publication summaries, formatted Excel workbook, and
600 dpi / SVG scientific plots.

PROTOCOL
Test=249 DBP measurement rows from Ref 7 (108), Ref 14 (72), Ref 16 (41),
Ref 21 (12), Ref 22 (16). Two model-fitting/CV regimes represented by frozen
Test predictions: publication-grouped folds and conventional random folds.
For each stage/protocol, Equal4 is the unweighted average of frozen RF,
XGBoost, LightGBM, native CatBoost predictions.

Target = ln(1 + DBP concentration in micrograms/L).
Budgets = 2.5%, 5%, 10%, 20%, separately rounded up per publication.
Calibration n=9,15,28,53 (evaluation n=240,234,221,196).
For 500 replicates, a single numpy.random.default_rng(42) draws an independent
permutation for each of five references, in order 7,14,16,21,22. The same
random sequences are used across every stage and fitting regime and match
RF, XGBoost, LightGBM, CatBoost previous experiments byte-for-byte.
Budgets use nested prefixes of these random permutations.
Each publication-specific correction is mean(y - zero-shot prediction) over
selected calibration rows of the publication. The correction is added only to
noncalibration rows in that same publication. All scores are evaluated
strictly on remaining rows. Original zero-shot on all 249 observations is
also reported separately, but proper comparison uses per-replicate same-remainder
zero-shot and calibrated values.

REPRODUCTION (Python, numpy only for evaluation):
python reproduce_equal4_random_fewshot.py
python audit_equal4_results.py
python make_workbook.py           [artifact_tool required]
python make_figures.py            [matplotlib required]

STATUS
Audit_Checks.json = ALL CHECKS PASS. Zero-shot R² matches SI Table S25
for all grouped stages to 10 decimal places. The historical SI few-shot
results for Equal4 had been removed after correction of its RF component;
therefore, NO historical Equal4 few-shot target is asserted or compared.

IMPORTANT SCIENTIFIC CAVEATS
This is a retrospective few-shot adaptation analysis using some true labels
from previously held-out studies. It is not zero-shot external validation
or new prospective evidence. Improvement on pooled metrics does not mean
improvement for each publication. Multiple DBP response rows may come from
the same experimental setting; publication-level separation is retained,
but completely novel setting transfer is a different evaluation question.
