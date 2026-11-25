# 基于 AI 的银行数据管理 —— 交易数据分类与异常交易检测

课程小组大作业。基于 IEEE-CIS Fraud Detection 真实交易数据，完成两个任务：

1. **数据分类**：预测交易产品类别 `ProductCD`（C / H / R / S / W 五分类）
2. **异常交易检测**：识别欺诈交易 `isFraud`（二分类）

## 核心结果

| 任务 | Accuracy | F1 | 误报率 FPR | 其他 |
| --- | --- | --- | --- | --- |
| 异常交易检测（告警线 0.17） | 98.84% | 81.99% | 0.31% | ROC-AUC 0.9710、PR-AUC 0.8659 |
| 数据分类 | 99.89% | Macro-F1 99.63% | — | Weighted-F1 99.89% |

模型统一采用 LightGBM，选型依据见 `线A_模型训练/models/model_selection_*.json`。

正式模型为 v2（`models/lgbm_fraud_final.txt`），按 A8/A9 的实验结论重训：`num_leaves` 64→127、轮数上限 2000（早停 1542）、`scale_pos_weight` 27.58→1。相比期中版本 F1 从 75.77% 提升到 81.99%，误报率从 0.45% 降到 0.31%，精度与召回同时改善。

三级阈值由验证集数据自动推导，随模型产物携带：告警线 0.17（F1 最优）、关注线 0.08、严格线 0.11（误报率 ≤0.5%）。

## 目录结构

| 目录或文件 | 用途 |
| --- | --- |
| [项目分工与进度留档](项目分工与进度留档.md) | 三人分工、已完成工作、下一阶段任务与验收标准 |
| [数据集](数据集/) | 原始数据的获取说明；`processed/` 为线 A 的预处理输出 |
| [线 A 模型训练](线A_模型训练/) | 预处理、模型选型、训练、评估脚本，以及模型、指标和图表 |
| [线 B 数据处理与文档](线B_数据处理与文档/) | EDA 图表、数据卡片、字段说明表、清洗报告、演示样例 |
| [线 C 系统展示与汇报](线C_系统展示与汇报/) | Streamlit 演示系统、告警设计、汇报讲稿、运行截图 |
| [《专综》期中汇报 PPT](《专综》期中汇报-基于AI的对银行的数据管理.pptx) | 12 页期中汇报终稿 |
| [期中汇报模板](《专综》期中汇报-模版.pptx) | 课程原始模板 |

## 数据获取（重要）

原始数据单文件 651 MB，超出 GitHub 100 MB 限制，**未纳入版本库**。克隆后需自行下载并放到 `数据集/`：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `train_transaction.csv` | 651.69 MB | 交易流水表，590,540 行 × 394 列 |
| `train_identity.csv` | 25.30 MB | 身份信息表，144,233 行 × 41 列 |

来源：IEEE-CIS Fraud Detection（IEEE 计算智能学会 + Vesta，2019）公开竞赛数据。

数据放好后，按以下顺序执行即可完整复现：

```powershell
python ".\线A_模型训练\preprocess.py"
python ".\线A_模型训练\model_selection.py"
python ".\线A_模型训练\train_fraud_baseline.py"
python ".\线A_模型训练\train_product_classifier.py"
python ".\线A_模型训练\evaluate_report.py"
python ".\线A_模型训练\retrain_final.py"   # 生成正式模型 v2
```

`数据集/processed/` 已包含预处理结果（`train.csv.gz`、`test.csv.gz`、`category_mapping.json`），只跑训练脚本时可跳过 `preprocess.py`。

## 实验分析（线 A）

训练、选型、评估完成后，可继续复现实验章节的全部图表与数据：

```powershell
python ".\线A_模型训练\training_curve.py"       # A7 训练曲线、学习曲线
python ".\线A_模型训练\hyperparam_search.py"     # A8 四组超参数实验
python ".\线A_模型训练\ablation_study.py"        # A9 六组消融实验
python ".\线A_模型训练\robustness_check.py"      # A10 5 种子 / 5 折 / 时间外推
python ".\线A_模型训练\retrain_final.py"         # A15 主线模型重训（v2）+ 三线阈值
python ".\线A_模型训练\test_inference_api.py"    # A11 接口用例（30 项断言）
python ".\线A_模型训练\external_validation.py"    # A14 外部数据集（ULB 信用卡欺诈）复现
python ".\线A_模型训练\multi_dataset_validation.py"  # A16 四数据集复现验证
python ".\线A_模型训练\batch_predict.py" --input 样例.csv --output 结果.csv --mode raw
```

完整方法、实验数据与结论见 [A13 模型与实验报告](线A_模型训练/A13_模型与实验报告.md)。

关键结论：

- **稳定性**：5 种子 F1 **0.7596 ± 0.0018**，5 折 CV **0.7207 ± 0.0076**
- **时间外推**：F1 降至 0.51，但 AUC 保留 0.91 —— 排序能力几乎不衰减，掉分来自固定阈值失效，部署需按月重校准
- **多数据集复现**：在 ULB（28.5 万）、Nooha（132 万）、Sparkov（30 万）、银行交易合成集（5 千）上用同一流程重训，ROC-AUC 全部 ≥ 0.9689、PR-AUC ≥ 0.8659，说明方法可跨数据集复现；但模型文件不可跨库直接迁移
- **反直觉结论**：类别权重与阈值调优应择一不可叠加（IEEE-CIS 上 spw=1 使 F1 +3.97%，ULB 上 v2 配置比 v1 高 8 倍 F1）

## 运行演示系统

```powershell
pip install -r ".\线C_系统展示与汇报\requirements.txt"
cd ".\线C_系统展示与汇报"
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`。也可双击 `启动演示.bat`，或用 `demo_cli.py` 做无界面自测：

```powershell
python ".\线C_系统展示与汇报\demo_cli.py"
```

## 复现线 B 的 EDA

安装 `pandas`、`numpy`、`matplotlib`。Windows 上图表使用微软雅黑，也可使用 Noto Sans CJK SC 或黑体。

```powershell
python ".\线B_数据处理与文档\复现EDA.py"
```

脚本按批读取原始 CSV，根据线 A 保留的类别映射核对删除列，输出 4 张 PNG 和 `EDA统计.json`。

## 关键口径

- 合并后 434 列 → 剔除缺失率 >99% 的 9 列 → **422 列建模特征**
- 训练 / 测试：**472,432 / 118,108 行**，按 `isFraud` 分层 80/20
- 欺诈样本占比 **3.499%**，采用 `scale_pos_weight = 1` + 验证集阈值调优（不叠加类别权重）
- 模型选型使用 12 万行子集（F1 0.6227），最终指标使用全量（F1 0.8199），两者不可混用
