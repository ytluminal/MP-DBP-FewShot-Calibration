# Machine Learning with Active Few-Shot Calibration Improves the Understanding and Prediction of Microplastic-Associated Disinfection Byproduct Formation

**Research-code repository (original curation: 11 November 2026; Figure 5–6 update: 26 November 2026).**

This repository accompanies the manuscript *Machine Learning with Active Few-Shot Calibration Improves the Understanding and Prediction of Microplastic-Associated Disinfection Byproduct Formation*. The study uses **1,244 individual DBP records from 27 publications**, plus **64 external records from publications 28–30**. Model outputs use `ln(1 + DBP concentration [µg/L])`.

## Scientific scope

The study compares observation-level random validation and publication-grouped validation (Lines 1–3); examines zero-shot prediction on held-out publications; and evaluates study-specific constant-offset few-shot calibration with random and **response-blind centroid–maximin** sample selection. A calibration observation is one experimental-condition × DBP-species concentration record; it need not equal one water sample or one analytical run. Selected records are excluded from post-calibration evaluation. Reported paired comparisons evaluate zero-shot, random and active predictions on matched noncalibration records. **The underlying predictive models are not retrained during calibration.**

## Start here

1. Use **Python 3.13.5** where feasible, and create a clean virtual environment.
2. Run `python -m pip install -r requirements.txt`. Some workflows include more precise legacy requirements in their own folders. GFN2-xTB and RDKit were used to calculate quantum/chemical descriptors; the published Excel contains already-computed descriptors and calibration does not require regenerating them.
3. The final submitted workbook is `data/MP-DBPs_data_SM.xlsx`. The historical workbook `data/MP-DBPs_SI.xlsx` remains available for frozen model reproduction: `Ref 1-27` (original database), `Ref 28-30` (independent validation), and `Ref list` (source identifiers). The `Ref ID` is **not a model predictor**.
4. **Fastest end-to-end calibration example with frozen predictions:**

   ```bash
   cd workflows/11_figure4_line2_random_active
   python reproduce_line2_newdata_fewshot.py
   ```

   For Line 3, use `workflows/12_figure4_line3_random_active` and run `python reproduce_line3_newdata_fewshot.py`.
5. The workflow folders each preserve the original script and input paths so commands from their archived README files can be run *from that folder*, not from the repository root.
6. Run `python scripts/verify_repository.py` to check curated file integrity and Python syntax. This **is a packaging check**, not scientific replication.

## Where to find code / results

| Folder | What it contains | Entrypoint |
|---|---|---|
| `workflows/01_line1_figure2a` | Figure 2a, Line 1 algorithms, stages, saved prediction arrays | `reproduce_line1.py` and `make_figure2a.py` |
| `workflows/02_line2_model_development` | Line 2 model-reproduction audit, Training/Test inputs | `prepare_datasets.py` then `AUDIT_FINAL_Line2_5BLAS.py` |
| `workflows/03_line3_model_development` | Line 3 model-refitting scripts and saved predictions | `reproduce_all_five_models.py` (see original README) |
| `workflows/04_line2_frozen_lightgbm` | Forty fitted LightGBM fold-models and preprocessing objects | `predict_with_saved_line2_lightgbm.py` |
| `workflows/05_line2_random_fewshot` | Per-model Line 2 random few-shot analyses | Each nested package's `reproduce_*.py` |
| `workflows/06_line2_active_fewshot` | Line 2 stage-matched active selection and matched comparisons | `work/rerun_line2_stage_matched.py` |
| `workflows/07_line3_random_fewshot` | Line 3 random few-shot calibration | `reproduce_line3_random_fewshot.py` |
| `workflows/08_line3_active_fewshot` | Line 3 stage-matched active selection | `work/rerun_line3_stage_matched.py` |
| `workflows/09_figure4_line1_random` | New-publication random calibration, Line 1 | `Line1_FewShot/reproduce_line1_random_fewshot.py` |
| `workflows/10_figure4_line1_active` | New-publication active calibration, Line 1 | `code/reproduce_line1_active_fewshot.py` |
| `workflows/11_figure4_line2_random_active` | New-publication random and active calibration, Line 2 | `reproduce_line2_newdata_fewshot.py` |
| `workflows/12_figure4_line3_random_active` | New-publication random and active calibration, Line 3 | `reproduce_line3_newdata_fewshot.py` |
| `workflows/13_figure5_shap` | Figure 5a–e; canonical OOF SHAP, saved fold models, plotting code; Figure S16 artwork | See [Figure 5 README](workflows/13_figure5_shap/README.md) |
| `workflows/14_figure6_scenarios` | Figure 6a–d; saved scenario data and plotting code; selected frozen-model inference and calibration example | See [Figure 6 README](workflows/14_figure6_scenarios/README.md) |

## Reproducibility limitations — please read before citing results

- This is a **curated collection of independently prepared reproduction packages**, *not* a single refactored end-to-end ML pipeline. The original subpackage names and relative input layout are preserved. Some historical scripts may still contain hardcoded local paths; prefer each package's documented portable entrypoint.
- **Line 1:** the shipped reproduction README documents small discrepancies for some Stage 2/3 Random Forest and SVR results, and acknowledges that original Optuna search and full RFE elimination paths were **not independently rerun**. Do not claim exact full-pipeline numerical replication of all Line 1 models.
- **Line 2/Line 3:** some reproduced archives refit **locked configurations** rather than reconstructing all hyperparameter searches or every RFE step. Line 3 Test-split selection was originally performance-informed and should be interpreted as a sensitivity analysis, not an entirely response-blind external holdout.
- **Frozen predictions:** Figure 4 calibration workflows operate on supplied, previously saved zero-shot predictions. Those calibration scripts do *not by themselves* establish that all frozen prediction files can be independently regenerated from original trained models. The Line 2 LightGBM folder supplies re-fitted fold models and their preprocessing objects.
- **Some historical SI `.docx` files are required by the original code** to retrieve locked feature lists or hyperparameters and are included as computation inputs, not as the current manuscript SI. Replace them with machine-readable parameter configs before final public release, or clearly distinguish their frozen computational role.
- **Large generated replicate tables** (more than 4 MB each) are intentionally omitted from this GitHub edition, as are copyrighted article PDFs, duplicate draft manuscripts, logs and caches. Run the relevant scripts to regenerate missing output tables where supported. Original package checksums that included removed files will **not** match this curated subset; use the new top-level release integrity manifest instead.
- Trained Line 1 models were placed into a **separate optional binary archive** to avoid bloating the code repository. The code and row-level predictions are present without that archive, but workflows that load saved `models/*.joblib` require installing the optional model files in their original locations.
- The SI Excel and archived model inputs may reflect **different dataset revisions**. A byte/descriptor-level provenance check against the exact model training inputs, including selected polymer chemistry descriptors and molecular volumes, is needed before claiming the latest SI workbook reproduces every previously trained model.
- Figure 5 and Figure 6 now include archived plotting code and source data, with exact manuscript composite images. Figure 5a layout was reconstructed; Figure 5d starts from saved cell summaries; Figures 6a and 6c replot stored predictions; Figure S16 includes artwork and a source-data reference, not its original standalone script. The full workflow for every figure (including Figure 1) is not yet independently reproduced. See the new workflow READMEs for the exact scope.
- Some archive README files refer to superseded SI drafts, historical tables or local output paths. Use the newest manuscript/SI for scientific interpretation.

## Data and measurement interpretation

**Data:** final submitted workbook `data/MP-DBPs_data_SM.xlsx`; historical workbook `data/MP-DBPs_SI.xlsx` is preserved for reproducibility. Both use source identifiers in the `Ref list` sheet. The final workbook has not been independently reconciled against every historical model input. For external refs 28–30, retained measurements should comply with the manuscript's MP-only suspension / isolated MP-DOM inclusion rules; verify the source-specific screening independently. Nominal `2.5%` budget is rounded up **within each of three external publications**, giving **three** DBP concentration records out of 64 (4.7% overall), with **61** noncalibration records remaining for method-specific scoring. Direct active-vs-random paired comparisons can use **fewer** common evaluation records because the union of selected calibration sets is excluded.

**Predictions:** all R², RMSE, and MAE values are calculated on `ln(1 + C_DBP)` with C in µg/L. The quoted zero-shot R² of 0.48 uses all 64 records; a calibrated R² near 0.90 is computed only on records that are not used for calibration. **Do not subtract these two R² values to claim a paired improvement.**

## Repository integrity

`MANIFEST_SHA256.json` lists every shipped file and its exact SHA-256 digest. `PROVENANCE.json` records which supplied archive each file came from. `OMITTED_LARGE_OUTPUTS.csv` identifies intentionally excluded contents. Run `python scripts/verify_repository.py` to check.

## Citation, licensing, and release

See `CITATION.cff` for provisional citation metadata. **The authors must choose an explicit open-source license** before releasing code publicly; no license is granted by default. Confirm permissions for shared datasets, model binaries, and any images. Publish a frozen tagged release (and, ideally, an archival DOI such as Zenodo) and place the permanent link in the manuscript's Data Availability Statement.
