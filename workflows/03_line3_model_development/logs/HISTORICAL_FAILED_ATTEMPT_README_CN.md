# Line 3 / Figure 2c 独立重新拟合与复现审计（2026-10-15）

## 数据及方案
- 你本次上传的四份 `YangRen_DLM_seed70_*.xlsx` 与 2026-10-14 复现包中记录的四份原始文件 SHA-256 完全相同。
- Training 1,026 条（20 篇 publication），External Test 218 条（7 篇 publication，Ref IDs 4, 11, 12, 13, 14, 20, 26）。Train/Test 没有 publication 交叉。
- 研究的 5 个 model systems：RF、XGBoost、LightGBM、**Preprocessed CatBoost**、Equal4。
- S1 原始 22 predictors，S2/S3 原始 55 predictors，S4 冻结的 RFE subset 大小分别为 RF 14、XGB 14、LGBM 14、CatBoost 28。
- CV protocols: Conventional random 5-fold / Publication-grouped 5-fold。使用冻结参数，不调整原始折、样本、训练参数、RFE 子集、软件算法。
- Equal4 = 4 个基础模型的对应预测均值。指标在 ln(1 + DBP concentration [µg/L]) 空间。
- 实际运行代码从上传 ZIP `Original_Source/run.py` 和 `evaluate_randomfold_external_test.py` 复制；仅将 ROOT 和 SI 的绝对路径调整为当前工作目录。

## 独立运行状态
- 40 个 CV 和 40 个 Test 配置均成功实际训练/评价。每组模型有 R²、RMSE、MAE，因此共核验 240 个指标，并比对 9,952 个逐行预测位置。
- **56/80 个配置**全部指标及逐行预测与给定的参考结果在 1e-7 以内匹配；**24/80 个配置**存在偏差。
- 偏差完全发生于 RF（12 组配置）及受 RF 影响的 Equal4（12 组）。XGB、LGBM、Preprocessed CatBoost 的 48/48 组完全匹配。
- 最大 CV R² 偏差：0.003906（S4 Publication-grouped RF）。
- 最大 External Test R² 偏差：0.018534（S4 Publication-grouped RF，fresh R²=0.042494 vs archived R²=0.061028）。
- 最大预测偏差（ln1p 空间）：0.266017，需以明细 CSV 为准。
- 本次运行的 RF 受影响折重复执行时是稳定的。**尚未证实**参考运行与本次运行不一致的根因；不能仅因为差异集中于 RF 就断言某一特定软件/环境因素为根因。
- 上传包中已有的 2026-10-14 独立审计报告显示，当时全部 80 个组合与历史基准一致，最大指标差异为 2.22e-16。这是一份独立已有的记录，**不是**本次 2026-10-15 重跑的结论。

## Active Few-Shot 前的冻结建议
- 此次 freshly refitted 的部分 RF/Equal4 predictions **不能直接覆盖**上传的原始冻结 Test predictions。
- 后续实验继续以已通过文件哈希核查、且和历史 Figure 2c 复现包完全相同的 `Line3_Test_Row_Predictions.csv` 为依据；其与之前 Active Few-Shot 完整交接包的同名文件 SHA-256 一致。
- 不需要根据本次 Test R² 调参或改变 split。若要求“当前软件环境也要实现每个模型逐位精确复现”，需要先对 RF 的预测差异做独立版本/数值稳定性诊断。

## 文件与复现
- `Line3_Fresh_Independent_Reproduction_Verification_20261015.xlsx`：独立拟合与历史结果的完整对比，包括 S4 数值表、80 组结果、输入文件 SHA。
- `Line3_Fresh_Independent_Reproduction_All80.csv`：R²、RMSE、MAE、逐行预测的比较偏差；布尔标志是否在容差内一致。
- `Line3_Fresh_Reproduction_Summary.json`：全部核查汇总。
- `fresh_run/run.py`, `fresh_run/evaluate_randomfold_external_test.py`：代码仅调整路径；保留全部训练 recipe。
- `fresh_run/FRESH_CV_RUN.log`, `fresh_run/FRESH_TEST_RUN.log`：完整实跑日志。
- `fresh_run/oof_stage*.npz`, `fresh_run/Test_Stage*.npz`：本次实跑的全部 OOF / Test 预测，可用于逐样本审计（**不等于**冻结历史预测）。
- `Original_Reference_Results/`：上传 ZIP 的原始参考报告及冻结预测 CSV，可校验与历史 Figure 2c 一致。

运行环境：scikit-learn 1.8.0、xgboost 3.1.3、lightgbm 4.6.0、catboost 1.2.8。
