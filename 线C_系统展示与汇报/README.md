# 线C · 系统展示与汇报

对应分工：**C2 演示系统原型**、**C3 告警展示设计**（C1 汇报 PPT、C4 讲稿见项目根目录）。

> **演示地址：http://localhost:8501**
> 先双击 `启动演示.bat` 启动服务；也可双击 `打开演示网页.url` 直接打开浏览器。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `predict.py` | 加载线A两个 LightGBM 模型，输入校验 + 推理封装 |
| `streamlit_app.py` | C2 演示界面：上传 CSV → 产品分类 + 异常交易告警 |
| `demo_cli.py` | 命令行自测脚本（不启动界面即可验证模型接入） |
| `C3_告警展示设计.md` | 风险分级、标红/徽标规范、演示动线与验收清单 |
| `C4_汇报讲稿.md` | 逐页讲稿、演示操作脚本、Q&A 与进度页周次说明 |
| `requirements.txt` | 依赖清单 |
| `截图/` | 运行截图（演示与验收用） |

> C1 汇报 PPT 终稿位于项目根目录：`《专综》期中汇报-基于AI的对银行的数据管理.pptx`（12 页，按 A/B/C 代码任务的真实产出重新设计，含数据探查、预处理、模型训练、选型、评估与系统演示）。

## 运行方式

方式一（推荐，`uv` 自动准备依赖）：

```powershell
uv run --with streamlit --with pandas --with numpy --with lightgbm streamlit run streamlit_app.py
```

方式二（普通 Python 环境）：

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`。无界面自测：

```powershell
python demo_cli.py            # 使用 10 行演示样例
python demo_cli.py 你的.csv   # 指定预处理后的 CSV
```

## 演示数据

- 页面「使用演示样例数据」开关一键加载 10 行样例，覆盖 **5 类产品 × 2 种欺诈标签**。
- 样例与真实标签来自 `线B_数据处理与文档/演示交接/`；样例仅用于功能核对，不代表模型整体性能。

## 输入格式

- 必须是**线A预处理后的 422 列特征 CSV**（表头与模型特征名一致，类别列已编码）。
- 支持 UTF-8 / GBK；缺列、重复列、非数值内容会给出明确报错；额外列自动忽略并提示。
- 原始交易 CSV 不能直接上传，需先运行线A的 `preprocess.py`。
- 若文件带有 `isFraud` / `ProductCD` 标签列，会自动识别并用于页面内的对照展示。

## 模型接口（来自线A）

| 任务 | 模型文件 | 输出 | 判定 |
| --- | --- | --- | --- |
| 异常交易检测 | `线A_模型训练/models/lgbm_fraud_final.txt` | 欺诈概率 | ≥ **0.17**（验证集 F1 最优阈值）触发告警 |
| 数据分类 | `线A_模型训练/models/lgbm_product_classifier.txt` | 5 类概率 | argmax → C / H / R / S / W |

两个模型特征名与顺序一致（422 列），代码按模型特征顺序自动重排输入。

## 验收情况

- [x] 样例逐行输出产品类别与告警状态（自测：产品 100% 命中、欺诈 100% 一致）
- [x] 缺列 / 非数值文件给出明确报错；额外列提示后忽略
- [x] UTF-8 与 GBK 编码读取
- [x] 结果 CSV 一键下载
- [x] 界面截图见 `截图/`（演示用）
- [ ] 课堂演示前按 `C3_告警展示设计.md` 第 5 节走一遍验收清单
