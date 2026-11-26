# Active Few-Shot Calibration：Centroid Initialization + Maximin Diversity Selection

**研究项目**：Microplastics–DBPs；Figure 2b（Line 2）和 Figure 2c（Line 3）之后的 Active Few-Shot Calibration。  
**本文件用途**：交给另一个 ChatGPT 对话即可按既定方案运行。`START_HERE.md` 为最简操作指令，`run_active_fewshot.py` 为已通过端到端 smoke test 的完整实现。  
**方法状态**：本文件将此前已讨论并固定的主要协议，与为可执行性作出的明确技术约定相结合。任何未经过先前实验验证的 Active 优势都不能预先宣称。

## 1. 科学问题、原则与冻结内容

### 1.1 检验的假设

相较于完全随机地挑选目标 publication 中的少数测量样本，基于 **feature-space coverage** 的 **Centroid + Maximin Diversity** 是否可以在相同的测量预算下更准确地估计 publication-specific prediction bias，从而以更少测量实现更高的 external Test R²？主要目标为**校准后的真实 Test R²（以及 RMSE/MAE）**；paired improvement 作为辅助信息。

### 1.2 已锁定、严禁修改

- **Line 2：** Training 995 observations / 22 publications；external Test 249 observations / 5 publications（Ref 7、14、16、21、22）。使用 Figure 2b 冻结的8套原始 predictions：Stage 1–4 × Publication-grouped / Conventional random；每套含 RF、XGBoost、LightGBM、**Native CatBoost**、Equal4。
- **Line 3：** Training 1,026 observations / 20 publications；external Test 218 observations / 7 publications（Ref 4、11、12、13、14、20、26）。使用 Figure 2c 的40套冻结 predictions；对应 RF、XGBoost、LightGBM、**Preprocessed CatBoost**、Equal4。
- **模型绝不重新训练、重新划分 Train/Test、重新调参、重新选择特征或重算 Test prediction。** Active 只改变哪些目标文献中的 label 用于 publication-specific offset 校准。
- 每条 Line 的所有模型、Stages、CV regimes 共用**完全相同的一套 Active sample Test row IDs**；不同 Line 因 Train-only preprocessing 不同，可以有各自独立的序列。
- **校准预算**固定 2.5%、5%、10%、20%。随机基准使用此前的 **500 次 Random Few-Shot** 原始 permutation，不另行生成。
- **target** 为 `y = ln(1 + C_DBP)`，其中 C_DBP 单位 µg/L。offset、Test R²、RMSE 和 MAE 与已锁定 Random Few-Shot 的 metric space 一致，均在这个 ln(1+C) 空间评价。
- 从 Test publications 挑选样本时**严禁读取真实 y、预测残差、R² 或其它需真实响应的统计量**。

## 2. 必要文件及原始行顺序（本 ZIP 已经包含）

### 2.1 Line 2（`inputs/line2/`）

- `train55.json`、`test55.json`：与冻结建模数据库对应的 57 列表结构：第 1–22 列是原始实验特征；第 23 列是 DBP 浓度；第 24 列是 Ref ID；其后还有 33 列分子/聚合物描述符。删除 target 和 Ref ID 后恰好 **55 个 predictors**。
- `AUDIT_FINAL_Stage{1..4}_Publication_grouped.npz` 和 `AUDIT_FINAL_Stage{1..4}_Conventional_random.npz`：每个文件包含 `Test_Ref`、`Actual_Test_ln1p` 和 `Test_pred_RF`、`Test_pred_XGBoost`、`Test_pred_LightGBM`、`Test_pred_CatBoost`、`Test_pred_Equal4`；预测数组与 `test55.json` 原始 Test row 顺序逐行对应。
- `Random_Permutations_500.csv`：500次 Random permutation；`Ordered_Test_Rows_1based` 是**全 Test set 内的 1-based 行号**，不是 publication 内局部序号。
- `Per_Publication_Budgets.csv`：先前 Random 准确的 ceil 预算和原始 per-ref count。

### 2.2 Line 3（`inputs/line3/`）

- `train55.json`、`test55.json`：同样的55原始 predictors，但 Training/Test refs 不同。
- `Line3_Test_Row_Predictions.csv`：40组（Stage × CV regime × model）逐条 external Test 预测；以 `Row_1based` 对应 `test55.json` 第几条 Test record，`Validation_Scheme` 区分 `Publication-grouped` 与 `Conventional random`。
- `Random_Permutations_500.csv`：既有500次 random 选样序列；仍为全 Test 集的 1-based row IDs。
- `Per_Publication_Budgets.csv`：已锁定的每篇文献预算。

**强制检查**：Train/Test 行数、Ref ID、Test label `log1p(C)`, 各 predictions 与 Test 行的对应关系均须检查。Line 2 和 Line 3 **共享 Ref 14 的72条 observations**，不要把两条 Line 的实验证据误认为来自完全不相交的测试论文。

## 3. Active Selection 使用的特征空间

### 3.1 55个原始 predictors（前后顺序同源数据，不要自行修改）

特征列按照 JSON `headers` 排列，只排除 `DBP concentration (µg/L)` 和 `ref ID`，共55项。

**8个类别型 predictors**（精确列名）：

1. `polymer type`
2. `additives`
3. `pretreatment type`
4. `sample phase`
5. `buffer`
6. `disinfectant`
7. `quenching agent`
8. `DBP name`

**其它47项**作为数值型（实验条件、DBP分子描述符及聚合物结构标识等）。注意：原始55个特征中包括 `DBP name`，也包括在实际候选样本信息中能够预先确定的 DBP/聚合物 descriptors。

### 3.2 特征预处理必须 Training-only 拟合

对每条 Line **独立拟合一次**，对全体目标 Test candidates 只做 `transform`，严禁从 Test data 估计 median、mode、mean、std 或 one-hot categories。

数值型（47列）：

1. 从 Training 估计每列的 median，填补 Train 和 Test 缺失值。
2. 如果某列在整个 Training 中均缺失，为避免导入 Test 数据产生泄漏，填补固定的0，保留该列。
3. 使用 Training 的 mean 和 standard deviation 做 `StandardScaler`。恒定列会被处理为零变化列，不引入 Test-only 信息。

类别型（8列）：

1. 用该类别在 Training 中的最频繁**非缺失**取值填补 Train 和 Test 缺失值；仅当 Training 整列都缺失时，使用固定类别 `__MISSING__`。
2. `OneHotEncoder(handle_unknown="ignore")` 仅在 Training 上 fit；Test 中训练时未见过的类别对应 OHE block 全零。
3. Numeric scaled columns 与 one-hot columns **横向拼接**，OHE columns **不再另做 StandardScaler**。

**实际编码后维度（本 ZIP 已经过 smoke test 核实）：Line 2 为120列；Line 3 为116列。** 原来的“55 predictors”是建构特征空间的55个 raw inputs，实际计算 centroid / Euclidean distance 使用的是上述完整编码矩阵。

> **对技术细节的说明**：此前确定了 Training-only median/mode、StandardScaler、OneHotEncoder；对 mode 的并列打平、全空列 fallback、未知类别及浮点 tie-breaking，本交接包为保证执行一致性明确了确定性实现。以上属于可复现性的固定实现细节，不是另外优化出来的新模型。

## 4. Centroid + Maximin：明确的数学定义和确定性算法

对于 Test publication `r`，令其中所有尚未知晓目标真实 DBP 浓度的候选 observations 在编码后的特征为 `z_i ∈ R^d`；按冻结的 **Test row order** 排序，集合为 `U_r`。

**第一步：Centroid Initialization**。使用这一篇 publication 的全部候选特征（不使用 target）计算中心：

`mu_r = (1/N_r) * sum_{i∈U_r} z_i`

选择离中心最近的观测为第一个 calibration 样本：

`i_1 = argmin_{i∈U_r} ||z_i - mu_r||_2^2`

**第二步：Maximin Diversity**。假设已经选中 `S_{k-1}={i_1,...,i_{k-1}}`，为每个未选中候选点计算它到当前 selected set 最近成员的距离：

`d(i, S_{k-1}) = min_{j∈S_{k-1}} ||z_i - z_j||_2^2`

逐次选取 `i_k = argmax_{i∈U_r\S_{k-1}} d(i,S_{k-1})`，一直到形成对该 publication 的**完整、有序 acquisition sequence**。预算取这个序列前 k 个元素，故四个预算的 Active selections **严格嵌套**。

**距离公式**：普通欧氏距离（计算可用平方距离，排序完全相同）；**没有**额外 feature weighting、PCA、按 response 做 stratification、prediction-space rerouting 或根据 y 计算的 uncertainty ranking。

**平局规则**：如果两个距离严格相同，选原始 Test row 顺序最前者。实现等价于在未排序的 Test rows 上调用 `np.argmin` / `np.argmax`。首次和后续的边界情况都按这个规则处理。

**重要约束**：选择算法在每篇 Test publication **分别执行**，不能对所有 Test publications 合并后做一个 centroid。

## 5. Calibration Budget：逐出版物向上取整

对每篇 publication，`k_r(f) = min(N_r - 1, max(1, ceil(f * N_r)))`，其中 `f ∈ {0.025,0.05,0.10,0.20}`；每篇至少选1个，同时至少留下1个供评价。

| Line | N_test / Ref IDs | 2.5% | 5% | 10% | 20% |
|---|---|---:|---:|---:|---:|
| Line 2 | 249 / 7, 14, 16, 21, 22 | **9** | **15** | **28** | **53** |
| Line 3 | 218 / 4, 11, 12, 13, 14, 20, 26 | **10** | **14** | **25** | **47** |

具体每篇文献的 `N_r`、`k_r` 见两个 `Per_Publication_Budgets.csv`。**不要**简单地把整个 Test set 的 2.5%、5%、10%、20% 直接取整：这会破坏先前固定的 publication-wise Random Few-Shot baseline。特别是 Line 3 的 Ref 13 只有4条观测，因此低预算的实际测量比例会高于名义比例。

## 6. Offset Calibration：保留此前 Random Few-Shot 的全部定义

对选中的 publication r 的 calibration indices `C_r`，一旦测得（或回顾性模拟中读取）其真实 DBP 浓度，在固定 target 尺度 `y_i = ln(1+C_i)` 中计算：

`delta_r = (1/|C_r|) * sum_{i∈C_r} (y_i - yhat_i_zero)`

对该 publication 其它 Test rows（包括评价集中的 Test rows）进行：

`yhat_i_active = yhat_i_zero + delta_r`  对于 `i∈r`。

Random baseline **使用相同公式**，但 `C_r` 换成锁定的第 replicate 次 Random permutation 的前 k 个 Test row IDs。

没有 regression intercept+slope 拟合、没有 model retraining、没有 ensemble reweighting、没有 Test label-guided model selection。因 publication 自身已有 measured targets 用于校准，**Few-Shot 评价不再代表完全无需目标文献数据的 Zero-Shot 外部预测**。

## 7. 最重要的公平比较：Active vs Random 要使用完全相同的评价样本

Active 是**确定性**的一套选择（每篇每预算固定）；Random 是该预算下**500套已锁定 permutation**。

对同一预算、同一 target publication r 和 Random replicate t，定义：

- `A_r(f)`：Active selected row set；
- `R_{r,t}(f)`：第 t 次 Random selected row set；
- **共同未用于任何一种校准的评价集** `E_{r,t}(f) = U_r \ (A_r(f) ∪ R_{r,t}(f))`。

合并各 publication 的 `E_{r,t}(f)` 形成 pooled common evaluation set。对**同一批 common evaluation rows** 分别计算：

1. `Zero-Shot`：未经校准的锁定 prediction；
2. `Active`：从 A 的已测目标值计算 offset 后得到的 prediction；
3. `Random`：从 R 的已测目标值计算 offset 后得到的 prediction。

计算 pooled 和逐篇 publication 的 `Test R²`、`RMSE`、`MAE`，并在**每次 replicate 内**先计算：

`DeltaR2_Active_vs_Random(t) = R2_active(E_t) - R2_random(E_t)`

`DeltaR2_Active_vs_Zero(t) = R2_active(E_t) - R2_zero(E_t)`

`DeltaR2_Random_vs_Zero(t) = R2_random(E_t) - R2_zero(E_t)`

`DeltaRMSE_Active_vs_Random(t) = RMSE_active(E_t) - RMSE_random(E_t)`（**负值**代表 Active 更好）。

汇总500次独立于选择算法的 Random permutations 时，报告中位数、5%和95%分位数、Active 胜出比例，并可补充25%/75%分位数。由于 repeated draws **共用同一套 Test outcomes**，不能把500个 replicates当成500篇新文献或500个独立外部验证。

**另存两种辅助评价**：

- **Active-own remainder**：仅排除 Active calibration indices，评价 `U\A`；直接反映实际选定 Active measurements 后的 operational performance，是绝对 R² 的补充展示。
- **Original Random-own remainder**：之前500次 Random baseline，评价 `U\R`。这可以按历史实验做描述性对照，但**禁止直接把 `U\A` 的 Active R² 与 `U\R` 的 Random R² 的差当成严格公平的 Active 优势**；两者评价对象不同。

## 8. 推荐的执行顺序与完整覆盖范围

**先验证的子集**：Line 2、Line 3 各自的 Stage 4，Publication-grouped CV，五个模型 × 4个预算。检查选样顺序、per-publication offsets 和 common remainder 指标合理之后，再跑全部配置。

**最终必须覆盖**：2 Lines × 5 models × 4 Stages × 2 training protocols × 4 budgets；每个组合500次 Random comparison。每条 Line 为 **160个 configuration-budget cell / 80,000条 matched replicate 结果**，两条 Line 合计160,000条 paired comparisons。

主文以校准后的 **Test R²（绝对实际值）** 为主；辅助报告 `Active − Random` 的严格配对 ΔR²、RMSE、MAE。所有样本按固定 reference 和 observation set 划分，不更改 Test membership。

## 9. 运行步骤——无需重新上传源论文或重新训练模型

解压后确保 `run_active_fewshot.py` 和 `inputs/` 文件夹**处于同一级目录**。当前参考实现只需要 `numpy` 和 `scikit-learn`，输入由 JSON、CSV 和 NPZ 构成；不需要在新对话中重新解释或从源论文抽取观测。

```bash
python -m pip install numpy scikit-learn
python run_active_fewshot.py --line both --smoke --out smoke_outputs
python run_active_fewshot.py --line both --out outputs
```

`--smoke` 只用原始500次 permutation 中**前3次**运行，每条 Line 产生：160条 Active-own 评价、480条 matched Active/Random pooled 比较、Line2 的2400条或 Line3 的3360条逐 publication matched 比较。完整运行使用全部500次，生成每条 Line 80,000条 pooled、Line2 的400,000条以及 Line3 的560,000条逐 publication matched 结果。

每条 Line 的主要输出文件：

- `Line{N}_Active_Complete_Sequence.csv`：每篇 publication 的所有候选 Test records 的排序；第一条是 centroid-nearest。
- `Line{N}_Active_Budget_Selections.csv`：各预算精确的 1-based Test row ID、参照 Ref ID、acquisition rank。
- `Line{N}_Active_OwnRemainder_160.csv`：160组 Active-only remaining-evaluation 集的绝对 Test R²、RMSE、MAE；与 Zero-Shot 在对应相同 remainder 配对。
- `Line{N}_ActiveVsRandom_Matched_Replicates.csv`：每条 Line 的80,000次严格 paired Active/Random/Zero test metrics。
- `Line{N}_ActiveVsRandom_Matched_Summary.csv`：160组 pooled 的 median、5%/95% quantiles，以及 Active 比 Random 改善的抽样比例。
- `Line{N}_ActiveVsRandom_PerPublication_Replicates.csv`、`Line{N}_ActiveVsRandom_PerPublication_Summary.csv`：检验提升是否只由某几篇 publication 主导。
- `Line{N}_Execution_Manifest.json`：特征维度、budgets、per-ref count、运行规模和 target 定义。

## 10. 完成标准／必须报告的核查（不可省略）

1. **输入哈希**：先比较 `INPUT_SHA256.json` 与当前输入文件哈希；不要混用新一版 Train/Test 或不同训练阶段 predictions。
2. **行对齐**：Line2 frozen NPZ 的 `Test_Ref/Actual_Test_ln1p` 必须和 `test55.json` 一致；Line3 CSV 的 `Row_1based/Ref_ID/Actual_ln1p` 必须一致。
3. **特征一致**：55原始 predictors = 47 numeric + 8 categorical；编码后维度 Line2 120 / Line3 116；fit只用Training。
4. **抽样一致**：每 Ref、每预算 `ceil(f×N_r)`；各预算 Active set 嵌套；模型/Stage/CV 之间 Active IDs 相同。
5. **没有标签泄漏**：选样代码只读取编码 features；仅选择完成后读取 calibration y；所有校准样本和 Random 校准样本均从共同 evaluation set 排除。
6. **重复一致**：使用随包携带的500×Ref random permutations；不重洗牌。按 `Ref_ID`、原始 Test row index 进行稳定 tie-breaking。
7. **metric space**：评价在 ln(1+C_DBP) 空间；在同一 eval set 上逐次比较，先逐 replicate 求差，后对差取中位数；不做 `median(R2_active) − median(R2_random)` 的替代。
8. **结果解释**：报告原始 Zero-Shot 和 calibrated absolute Test R²；量化逐文献失败案例。Line 2 / Line 3 共有 Ref14，补做 excluding Ref14 的 sensitivity analysis。
9. **无性能窥探调优**：Active acquisition rule、distance 和预处理在读 Test outcome 之前全部冻结。如果 Active 不如 Random，也如实报告；不能看 Test 后调权重、换距离、挑预算来宣称胜出。
10. **复现保存**：输出 data、selected IDs、Python script、包中输入校验和、环境版本及运行说明，制作一个可在新对话中重复运行的 final ZIP。

## 11. 已知限制和必须如实说明的推断

- **每个 Test observation 是一个 DBP-specific label**，而非必然代表一个独立物理实验条件。若一个水样报告多个 DBPs，这些 observations 可能高度相关；若要改为按实际实验条件或分析批次采样，需要另行定义 group IDs 和新实验方案，不能在当前冻结协议中偷偷改变。
- 选择最具特征代表性的样本 **不保证** 恰好具有最具代表性的 prediction residual；Active 可能明显输给 Random。此结果是要检验的假设，不应预先夸大。
- Euclidean distance 经数值标准化和 OHE 后，特征相关性、冗余和类别维数会影响采样。当前协议固定为等权的 encoded columns；如果未来测试其它权重、PCA、distance 或 acquisition strategy，须作为明确命名的**新方法**，并在 Training-only validation 下决定，不可覆盖当前基准。
- Line2 与 Line3 使用不同的 Training/Test 划分且共有 Ref14；重复交叉文献结果是 robustness evidence，不是两套完全相互独立的 pristine external test。
- 新方法使用 target publication 的测量标签后，是 few-shot **domain adaptation**，不能再称为完全盲测的 zero-shot extrapolation。

## 12. 可直接复制粘贴到新聊天框的执行任务

> 请执行我上传 ZIP 里面已经冻结的 Active Few-Shot Calibration 方案。先阅读 `START_HERE.md` 和 `METHOD_FULL_CN.md`。使用其中的 `run_active_fewshot.py` 与 `inputs/` 完成 Line 2 和 Line 3 的 Centroid Initialization + Maximin Diversity Selection。只使用 Training 拟合 55 predictors 的处理；在 Test 的每篇 Ref 独立计算 centroid、以最近中心点初始化、其余用 Euclidean maximin 选取；固定 2.5/5/10/20% per-Ref ceil budgets、嵌套序列、相同 Active IDs 跨 5模型×4Stages×2CV。校准只用所选 Test y 在 ln(1+C) 中算 residual mean，不重训、不修改 Test split。与随包携带的原500次 Random permutations，在两个选样集合并集之外的共同评价 observations 上严格比较 Active/Random/Zero 的最终 Test R²、RMSE、MAE。先跑 smoke test，逐项验证 inputs、features、Ref/row alignment、预算及 sample sequence；再跑每条 Line 的全部80,000次 matched comparisons。输出 pooled/per-ref summary 和逐次记录、报告低预算下的失败案例和 Test Ref14 重叠的 caveat，并打包代码、所有结果、sha256 与配置，不得根据 Test 结果反复调整选择算法。

---

**本交接文件给出了唯一主方案，不需要向另一个对话重新解释研究背景。** 如果运行中发现源数据与文中计数/列结构不符，应立即停止并报告 mismatch，而不是猜测或自动“修复”数据。
