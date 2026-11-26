# Figure 2a / Line 1: Independent Numerical Reproduction Archive

**Saved on:** 2026-10-26. This archive is intended to preserve the exact source data, published numerical targets, model settings, per-row predictions, and code used for the reproduced figure. It also includes the uploaded literature PDFs to audit original study data if needed.

## 中文使用说明

这是 Microplastics–DBPs 论文 **Figure 2a / Line 1** 的完整备份。包含 5 个原始数据及 Training/Test XLSX、SI、主文稿、18 组模型各自的 JSON 指标和 NPZ 逐条预测、核对 Excel、PNG/SVG、训练与绘图代码，以及 25 篇已上传原始文献 PDF。

**重要：**本次只是使用 SI 中锁定的超参数和 Stage-3 RFE 特征集合重新训练、计算 5 折 OOF 和 249 个 Test 结果。**没有重新运行原来的 Optuna 20-trial 搜索、也没有重新执行整条 RFE 消除路径**。18 组模型中 14 组的 CV R² 与 Test R² 同时一致到四位小数；S2/S3 Random Forest 和 SVR 有差异，详见 `results/Figure2a_Line1_Model_Reproduction_Audit.xlsx`。不要把未完全核对的值当成精确复制。

### 未来重跑（Windows/macOS/Linux）

1. 完整解压 ZIP。建议 Python 3.13.5，并在虚拟环境中运行 `python -m pip install -r requirements.txt`。
2. 在解压目录的根目录运行以下命令（也可从其他目录用脚本完整路径运行）：

```bash
python reproduce_line1.py S1
python reproduce_line1.py S2
python reproduce_line1.py S3
python make_figure2a.py
```

可单独重跑某个模型：

```bash
python reproduce_line1.py S2 "Random Forest"
python reproduce_line1.py S3 "SVR-RBF"
```

`reproduce_line1.py` 使用 `openpyxl` 从 XLSX 的 `Data` sheet 精确读取数据。四份 Stage-1/Stage-2 Training/Test 文件经逐单元格比对，与本次原始 `artifact_tool` 读取的数值完全一致。未来运行前**建议先把整个文件夹备份一份**，因为重跑会覆盖 `results/` 下对应的 JSON/NPZ，绘图会覆盖 `figures/`。

## Folders

- `data/`: the five identical input XLSX files; preserve row order and exact values.
- `documents/`: manuscript, SI; **SI Text S2 and Table S1** define the exact Line-1 methods and expected outputs.
- `results/`: precomputed `line1_S{1,2,3}_<model>.json` fold metrics/parameters and `*_preds.npz` with `oof`, `test`, `y_train`, `y_test`; reproduction Excel.
- `figures/`: the reproduced 2a plot as PNG and editable vector SVG.
- `code_original/`: exact historical scripts and supplementary troubleshooting scripts. The original code depends on a preinstalled `artifact_tool` and absolute `/mnt/data/` paths; **use root portable scripts to rerun**.
- `literature/`: uploaded reference publications for source audit (not used as training input during model refits).
- `SHA256SUMS.txt`/`manifest.json`: cryptographic byte hashes of archived files. Verify these after archive extraction if desired.

## Frozen modeling protocol

- Response: `ln(1 + DBP concentration [µg/L])`.
- S1: 22 experimental predictors; S2: 55 predictors (22 + 33 chemical); S3: fixed RFE-selected subsets by model.
- Split: 995 Training / 249 Test observations, random observation-level split, fixed KFold(n_splits=5, shuffle=True, random_state=42) on Training. `Ref ID` is provenance only, not a predictor.
- Metrics: pooled 995-row out-of-fold CV R²/RMSE/MAE and 249-row Test R²/RMSE/MAE on the log-transformed target scale.
- No new hyperparameter search is run by this archive's root reproduction script. Existing model settings are held fixed from SI Text S2.
- Package versions are pinned in `requirements.txt`; exact floating-point reproducibility may vary with OS/BLAS/thread scheduling.

## Important observed deviations from SI Table S1

- Stage 2 RBF-SVR Test R² = 0.8101717 -> rounded 0.8102, whereas SI prints 0.8101. Identical in Stage 3.
- Stage 2 Random Forest: rerun CV R² = 0.8808908 (SI 0.8802), Test R² = 0.8553608 (SI 0.8586).
- Stage 3 Random Forest: rerun CV R² ≈ 0.8836 (SI 0.8834), Test R² ≈ 0.8471 (SI 0.8493).
- 14 of 18 Stage × algorithm configurations reproduce both displayed CV and Test R² to four decimal places. The known discrepancies are preserved for scientific transparency; do not silently overwrite them with SI targets.

### Highest recorded reproducible scores

- **S3 XGBoost CV R² = 0.9108**.
- **S3 LightGBM Test R² = 0.8870**.

These random observation-level scores do *not* demonstrate cross-publication transferability; Line 2/3 publication-held-out protocols address that separately.

## Exclusions

This archive is specifically for **Figure 2a Line 1**. It does **not** include the complete independently reproduced Line-2/Line-3 modeling pipelines, full Optuna trial archives, or full RFE elimination trajectories. Reference PDFs are included as optional provenance, not required for a simple model rerun.

## Updated 2026-10-27: trained model binaries (new)

The upgraded archive **also includes 108 real fitted `.joblib` model files** in
`models/`: all **18 full-995-Training** fits and **90 corresponding five-fold CV**
fits. Each binary includes its own trained numerical/categorical preprocessing
objects where used, feature order and fold metadata. See `models/README.md`.

To predict immediately without refitting:

```bash
python predict_saved_model.py --stage S3 --algorithm LightGBM --input "data/Conventional_RandomSplit_Stage2_Test_55features(3).xlsx"
```

To reproduce CV and Test predictions **from the saved model binaries only**, run:

```bash
python validate_saved_models.py
```

All 108 objects were deserialized and validated against the corresponding saved
`results/*_preds.npz`: across all 18 model configurations, the maximum absolute
prediction difference was less than 1e-8 (in fact floating-point roundoff only).
To refit and regenerate the saved `.joblib` binaries, use
`python save_trained_models.py`. This overwrites `models/` and rechecks predictions.

**Provenance notice:** these binaries were generated by a new frozen-recipe refit
on 2026-10-27. The previously trained historical Python objects did not survive
in the 2026-10-26 ZIP. No Optuna search or new RFE feature selection was run.
The original SI-versus-RF and SVR discrepancies remain documented and unchanged.

**Security:** `.joblib` uses Python pickle under the hood; loading untrusted
copies is unsafe. Maintain a verified checksum and matching Python/package versions.
