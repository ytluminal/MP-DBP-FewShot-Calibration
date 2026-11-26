Line 3 Figure 2c full reproducibility package (2026-10-15).
Start with START_HERE_CN.md for full instructions.
Python 3.13.5, dependencies pinned in requirements_frozen.txt.
Run python verify_package.py to hash-check immutable source files and verify the saved 80 CV/Test configurations.
Run python reproduce_all_five_models.py to independently refit RF, XGBoost, LightGBM and Preprocessed CatBoost across stages 1-4 and both CV schemes, and reconstruct Equal4 from the newly refitted components. Fails if row-level deviations exceed 1e-7.
All 80 reference configurations passed prior controlled audits. The new full-run orchestration has been spot-tested for Stage 1 grouped, but not executed for all 80 in the packaging turn.
Do not change the data split, model parameters, selected predictors, fold definitions or algorithms to force a match.
