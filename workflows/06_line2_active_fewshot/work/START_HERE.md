# START HERE — Centroid + Maximin Active Few-Shot Calibration

**Purpose:** Copy this self-contained ZIP into a fresh ChatGPT conversation, upload it, and request execution of the included Python script. It includes the frozen inputs and the complete scientific protocol for BOTH Line 2 and Line 3. No model retraining or source publication data recollection is required.

## One-sentence instruction for the new chat

> Please unzip this package, read `METHOD_FULL_CN.md` before running anything, execute `python run_active_fewshot.py --line both --smoke --out smoke_outputs` and check alignment, then execute `python run_active_fewshot.py --line both --out outputs`. Report Active vs Random Few-Shot on exactly the same common held-out observations, showing final absolute Test R² and paired ΔR², pooled and per publication. Do not resplit data, retrain models, retune hyperparameters, alter 55-feature acquisition space, change sample budgets, or use Test labels during selection. Save all results, selected global Test row IDs, hashes, and scripts in a reproducibility ZIP.

## Install

Python 3.10+; `numpy` and `scikit-learn` (ideally use the already installed scientific Python environment):

```bash
python -m pip install numpy scikit-learn
```

## Run

Unzip so `run_active_fewshot.py` and the `inputs/` folder remain side by side.

```bash
python run_active_fewshot.py --line both --smoke --out smoke_outputs
python run_active_fewshot.py --line both --out outputs
```

The smoke test runs three pre-recorded Random draws per configuration. The full run uses **exactly the original 500 draws** and does not call a new random number generator.

## What is inside

- `METHOD_FULL_CN.md`: comprehensive Chinese protocol, equations, budgets, row alignment, scientific comparisons, limitations and acceptance checks.
- `run_active_fewshot.py`: independently runnable reference implementation using frozen Test predictions and all 55 raw predictors. No model retraining.
- `inputs/line2/`: training/test 55-feature JSON, eight frozen stage/CV `.npz` prediction sets, original 500 Random permutations and per-ref budgets.
- `inputs/line3/`: training/test 55-feature JSON, 40 model/stage/CV predictions in CSV, original 500 Random permutations and per-ref budgets.
- `INPUT_SHA256.json`: checksums and sizes for package input assets.
- `smoke_test/`: small example outputs from successful self-contained end-to-end execution.

**Important terminology:** the 55 predictors are *raw input columns* (47 numeric, 8 categorical). Once encoded from Training only, Line 2 has 120 and Line 3 has 116 columns. Do not confuse these numbers or reuse the classification model's own feature subset for acquisition.

**Output scope:** for each Line: 40 models (5 algorithms × 4 stages × 2 CV regimes) × 4 budgets × 500 random trials = **80,000 matched Active–Random replicate comparisons**. Active itself is deterministic, but performance on its own remainder and its selected sequence are saved.

**Scientific priority:** compare final Test R², RMSE, MAE for Active vs Random on the **same common held-out rows**, not on two different remainders; do not overclaim independence across the 500 resamples or across Lines 2/3 (they share Ref 14).
