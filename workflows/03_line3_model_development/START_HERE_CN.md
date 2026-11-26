# Line 3 / Figure 2c — 完整模型复现包（2026-10-15）

## 建议永久保存本 ZIP（不要只保留 Excel）

本包包含 Figure 2c 的 **全部五个 model systems**：Random Forest、XGBoost、LightGBM、**Preprocessed CatBoost**（不是 Line 2 的 Native CatBoost）和 Equal4；S1–S4 四个 Stage；**Conventional random 5-fold CV** 与 **Publication-grouped 5-fold CV**；以及对应的外部 Test。总计 **40 个 CV + 40 个 Test = 80 个模型评价配置**。研究对象为 1,026 条 Training 记录与 218 条 External Test 记录，20/7 篇 publication，真实 Test label 只用于最终评分，不参与任何 fit。

### 1. 新电脑（Windows）建议操作顺序

1. 安装 **64 位 Python 3.13.5**；确保 `python --version` 输出正确版本。建议新建专用 virtual environment，避免与其他研究项目混用。
2. 把 ZIP 完整解压到**不包含中文特殊字符且路径不太长**的目录，例如 `C:\Research\Line3_Figure2c\`。
3. 双击 `03_INSTALL_DEPENDENCIES_WINDOWS.bat` 安装本次验证的软件依赖（联网安装依赖）。也可运行 `python -m pip install -r requirements_frozen.txt`。
4. 双击 `01_VERIFY_ARCHIVE_WINDOWS.bat`：检查封包关键源文件 SHA-256、现有冻结预测对应的 **80/80 CV/Test 配置**；通常只需要短时间。
5. 需要**真正从原始 Training 数据重拟合所有四个基础算法**并计算 Equal4 时，双击 `02_REFIT_ALL_FIGURE2C_WINDOWS.bat`，或在目录内运行 `python reproduce_all_five_models.py`。这是完整重算，计算时间显著长于步骤 4。每个 Stage、CV scheme、模型使用独立 Python 进程，生成日志；一旦逐行预测偏离冻结参考超过 `1e-7` 就停止并报错。
6. 如果只需重新验证本次修复的 RF / Equal4，双击 `04_REFIT_RF_EQUAL4_ONLY_WINDOWS.bat`。

### 2. 两种“复现”的区别（非常重要）

- **完整存档核查（已测试）**：执行 `python verify_package.py`，逐项验证已保存的 80 组 CV/Test 预测。现存审计基于此前已独立实际运行的 RF / Equal4，以及先前独立运行且逐行核对成功的 XGBoost、LightGBM、Preprocessed CatBoost；**80/80 通过**，历史最大逐行误差 **2.6645352591003757e-15**。
- **完整从头重拟合（已提供可执行程序）**：执行 `python reproduce_all_five_models.py`。它从冻结 Training 输入再次拟合四个基础模型；由**此次新拟合的四份预测**计算 Equal4，并分别对 1,026/218 个样本比较。新加入的三个独立模型重拟合入口已实际测试，且 Stage 1 Publication-grouped **五个模型在 CV/Test 共 10/10 组新拟合比较通过**（最大差异 `1.33e-15`）。**本次打包尚未把全套 80 组通过新的总启动脚本再跑一遍**；旧版中已有的 80/80 历史验证不应被误写成该新启动脚本的完整重跑证明。

在不同计算机、Python/NumPy/Scikit-learn 软件发行包或操作系统上，即便参数和随机种子相同，也不能无条件保证机器精度逐位一致。尤其是 RF 曾出现环境敏感的重新拟合偏差；这里提供已实测可靠的**隔离进程 + 单线程底层数值库环境 + 逐行核查**，出现不匹配时请检查日志与依赖，不要为迎合结果重新调参。

### 3. 本包的目录结构

| 路径 | 用途 |
| --- | --- |
| `input/run.py` | 可移植的冻结算法/预处理/折分/权重/超参数实现（从历史代码仅修改路径） |
| `input/train22.json`, `test22.json`, `train55.json`, `test55.json` | 冻结源数据，原训练代码直接读入 |
| `input/SI_Conventional_RandomSplit_Line3_Stage1-4.docx` | S4 既定 RFE 特征子集及 SI |
| `original_xlsx/` | 四份用户上传的原始 Training/Test Excel（22/55 features） |
| `original_source/` | 历史原始代码和运行/评价材料，保留出处（其中旧绝对路径**不能直接用于新电脑**） |
| `reference/` | Figure 2c PNG、历史 CV/Test 逐行冻结预测 CSV、80 组审计、历史汇总、Excel |
| `verified_base_predictions/` | 三个非 RF 基础模型的原始独立核查后缓存，用于已验证的 RF/Equal4 修复工作流 |
| `reproduce_rf_equal4.py` | 真正重新拟合 RF，并由 RF 与已验证三个基础模型缓存重算 Equal4 |
| `refit_non_rf.py` | 从冻结数据重新拟合 XGBoost / LightGBM / Preprocessed CatBoost |
| `reproduce_all_five_models.py` | 完整独立进程调度器，启动四种模型全部 Stage 与 CV/Test |
| `verify_full_refit.py` | 使用**全部重新拟合**的四个基础模型预测计算 Equal4 并审计 80 组 |
| `verify_all.py` | 验证存档 80 组指标及逐行结果 |
| `verify_package.py` / `PACKAGE_SHA256.json` | 保持原始源数据及程序的 SHA-256 完整性校验 |
| `outputs/` | 已成功的 RF/Equal4 模型预测 `.npz/.npy`、完整 80 组审计 CSV 和 49,760 条逐行对照 |
| `refitted_components/` | 新增全模型复现入口生成的预测；已带 Stage 1 Grouped 的端到端烟雾测试结果 |
| `logs/` | 成功受控重跑日志、重复运行日志、历史问题诊断日志及本次新入口测试日志 |
| `requirements_frozen.txt` | 本次环境精确依赖版本（Python 3.13.5 需单独安装） |
| `*_WINDOWS.bat` | 安装、存档验证、完整重新拟合、RF 修复的 Windows 一键脚本 |

### 4. 冻结方法与边界

- S1：22 个 predictors；S2–S3：55 个 predictors；S4：冻结 RFE 子集大小 RF 14、XGBoost 14、LightGBM 14、Preprocessed CatBoost 28。**不重新执行 RFE 搜索、模型筛选或超参数优化**。
- Original external Test：218 行；所有模型只从原来的 1,026 行 Training 进行训练。评价指标为 `ln(1 + DBP concentration [µg/L])` 上的 R²、RMSE、MAE。
- Publication-grouped CV 与 Conventional random CV 使用锁定 fold。Equal4 是四个基础模型的 prediction mean，权重均为 0.25。
- `reference/Line3_CV_Row_Predictions.csv` 和 `reference/Line3_Test_Row_Predictions.csv` 是**冻结比较目标**，而不是被重新训练程序当作训练标签或用来拟合的数值。
- 文件中有**逐行预测和足以从头训练的代码/数据**，但**没有预先序列化保存的每一棵树或所有 4×5×2 次拟合后的模型二进制对象**；无需这些文件也能重拟合并核对预测。

### 5. Figure 2c Stage 4 外部 Test R² 的验收目标

| 模型 | Random-fold External Test | Grouped-fold External Test |
| --- | ---: | ---: |
| Random Forest | 0.0518 | 0.0610 |
| XGBoost | -0.0121 | 0.0132 |
| LightGBM | 0.2533 | 0.2048 |
| Preprocessed CatBoost | -0.1425 | -0.0811 |
| Equal4 | 0.0700 | 0.0702 |

以上四位数取自 Figure 2c。审计时应使用 `reference/` 中保存的高精度逐行预测比较，不应只检查图中四位小数。

### 6. 故障排查及研究记录

- 安装后先运行 `01_VERIFY_ARCHIVE_WINDOWS.bat`。如 SHA 不通过，重新完整解压，切勿手工修改冻结数据或代码。
- 如完整重拟合出现不匹配，查看 `logs/S{stage}_{scheme}_{model}.log` 和 `logs/S{stage}_RF.log`，保存失败日志。不要用既有冻结预测替换失败的新拟合结果后声称成功。
- 之前发生过 RF 数值偏差，先前失败尝试的说明与日志也保存在 `logs/HISTORICAL_*` 下，便于追踪。
- 重拟合产生的输出 `outputs/`、`refitted_components/` 和 `logs/` 可能更新；`PACKAGE_SHA256.json` 只锁定**不可变源文件**，不会阻止正常重算输出。
- 本包可作为后续 Active Few-Shot Calibration 的冻结模型基础，校准不能改变上述预测开发流程。
