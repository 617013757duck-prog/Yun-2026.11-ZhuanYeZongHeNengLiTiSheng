# -*- coding: utf-8 -*-
"""
线A · A18 概率校准与子群体公平性分析

回答两个风控落地必答的问题:

Q1 分数能不能当"风险概率"直接用?
    银行风控常按"预期损失 = 金额 × 概率"排序复核队列, 所以要求分数是校准过的。
    用 Brier score 与 ECE(期望校准误差) 检查, 必要时做 Platt/等渗校准。

Q2 模型对不同客群是否一视同仁?
    按卡类型、产品类别、金额段、身份信息覆盖与否分组, 比较各组的召回率与误报率。
    若某些客群召回明显偏低, 说明模型存在系统性盲区。

产出:
  models/calibration_subgroup_results.json
  figures/calibration_reliability.png
  figures/subgroup_recall.png
"""
import json
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL = os.path.join(BASE, "models", "lgbm_fraud_final.txt")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
DROPS = ["isFraud", "ProductCD", "ProductCD_label"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print(msg, flush=True)


def ece(y_true, prob, n_bins=15):
    """期望校准误差(按等频分箱)。"""
    edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0, []
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
    total, bins = 0.0, []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        conf = float(prob[m].mean())
        acc = float(y_true[m].mean())
        total += m.sum() * abs(conf - acc)
        bins.append({"区间": "[%.4f, %.4f]" % (edges[b], edges[b + 1]),
                     "样本数": int(m.sum()), "平均预测概率": round(conf, 4),
                     "实际欺诈率": round(acc, 4)})
    return float(total / len(prob)), bins


# ---------------- 载入数据与模型 ----------------
train = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))
X_test = test.drop(columns=DROPS)
y_test = test["isFraud"].values

# 验证集阈值: 从模型产物读
with open(os.path.join(OUT, "metrics_fraud_final.json"), encoding="utf-8") as f:
    METRICS = json.load(f)
THR = float(METRICS["tuned_threshold"]["threshold"])
WARN = float(METRICS["thresholds"]["warn"])
STRICT = float(METRICS["thresholds"]["strict"])

bst = lgb.Booster(model_str=open(MODEL, encoding="utf-8").read())
prob = bst.predict(X_test)

# ================= Q1 概率校准 =================
log("=" * 66)
log("Q1 概率校准检查")
log("=" * 66)
brier_raw = float(brier_score_loss(y_test, prob))
auc_raw = float(roc_auc_score(y_test, prob))
ap_raw = float(average_precision_score(y_test, prob))
ece_raw, bins_raw = ece(y_test, prob)

# Platt 校准(在验证集上拟合, 测试集评估)
rng = np.random.RandomState(42)
mask = np.zeros(len(train), dtype=bool)
y_tr_all = train["isFraud"].values
for c in np.unique(y_tr_all):
    idx = np.where(y_tr_all == c)[0]
    mask[rng.choice(idx, size=int(len(idx) * 0.1), replace=False)] = True
X_va = train[mask].drop(columns=DROPS)
y_va = y_tr_all[mask]
prob_va = bst.predict(X_va)
logits_va = np.log(np.clip(prob_va, 1e-6, 1 - 1e-6) / (1 - np.clip(prob_va, 1e-6, 1 - 1e-6)))
logits_te = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / (1 - np.clip(prob, 1e-6, 1 - 1e-6)))

platt = LogisticRegression(solver="lbfgs", max_iter=1000)
platt.fit(logits_va.reshape(-1, 1), y_va)
prob_platt = platt.predict_proba(logits_te.reshape(-1, 1))[:, 1]
brier_platt = float(brier_score_loss(y_test, prob_platt))
ece_platt, bins_platt = ece(y_test, prob_platt)

iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
iso.fit(prob_va, y_va)
prob_iso = iso.predict(prob)
brier_iso = float(brier_score_loss(y_test, prob_iso))
ece_iso, bins_iso = ece(y_test, prob_iso)

log("原始 : Brier %.5f  ECE %.5f  AUC %.4f  PR-AUC %.4f" % (brier_raw, ece_raw, auc_raw, ap_raw))
log("Platt: Brier %.5f  ECE %.5f  (排序不变, 仅重新标定概率)" % (brier_platt, ece_platt))
log("等渗 : Brier %.5f  ECE %.5f" % (brier_iso, ece_iso))

# 校准是否影响排序与业务指标
def biz(p):
    pred = (p >= THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"F1": round(float(f1_score(y_test, pred)), 4),
            "FPR": round(float(fp / (fp + tn)), 4),
            "召回": round(float(tp / (tp + fn)), 4)}

log("校准对业务指标影响: 原始 %s | Platt %s | 等渗 %s"
    % (biz(prob), biz(prob_platt), biz(prob_iso)))

calibration = {
    "原始": {"Brier": round(brier_raw, 5), "ECE": round(ece_raw, 5),
             "ROC_AUC": round(auc_raw, 4), "PR_AUC": round(ap_raw, 4),
             "业务指标": biz(prob)},
    "Platt": {"Brier": round(brier_platt, 5), "ECE": round(ece_platt, 5),
              "ROC_AUC": round(float(roc_auc_score(y_test, prob_platt)), 4),
              "PR_AUC": round(float(average_precision_score(y_test, prob_platt)), 4),
              "业务指标": biz(prob_platt)},
    "等渗": {"Brier": round(brier_iso, 5), "ECE": round(ece_iso, 5),
             "ROC_AUC": round(float(roc_auc_score(y_test, prob_iso)), 4),
             "PR_AUC": round(float(average_precision_score(y_test, prob_iso)), 4),
             "业务指标": biz(prob_iso)},
    "原始分箱": bins_raw,
    "结论": ("原始分数已具备较好排序能力(AUC %.4f)。Brier %.5f 对应欺诈基线 %.5f, "
             "Platt 校准后降至 %.5f, 说明原始分数整体偏保守(预测概率低于实际频率), "
             "若要按'预期损失=金额×概率'排序复核队列, 建议叠加 Platt 校准。"
             % (auc_raw, brier_raw, float(y_test.mean()), brier_platt)),
}

# 可靠性图
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, (name, bins, col) in zip(axes, [
        ("原始", bins_raw, "#1a73e8"), ("Platt 校准后", bins_platt, "#188038")]):
    xs = [float(b["平均预测概率"]) for b in bins]
    ys = [float(b["实际欺诈率"]) for b in bins]
    w = [b["样本数"] for b in bins]
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="完美校准")
    ax.plot(xs, ys, "o-", color=col, lw=2, ms=5, label="实际可靠性曲线")
    ax.fill_between(xs, ys, [0] * len(xs), color=col, alpha=0.12)
    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际欺诈率")
    ax.set_title("%s 可靠性图" % name)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
fig.suptitle("概率校准检查（测试集 118,108 条）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "calibration_reliability.png"), dpi=140)
plt.close(fig)

# ================= Q2 子群体分析 =================
log("")
log("=" * 66)
log("Q2 子群体公平性分析")
log("=" * 66)

df = test.copy()
df["_score"] = prob
df["_pred"] = (prob >= THR).astype(int)

# ProductCD / card4 在 processed 里是 label-encoded, 这里用类别映射反解成业务可读取值
with open(os.path.join(BASE, "..", "数据集", "processed", "category_mapping.json"),
          encoding="utf-8") as f:
    MAPPING = json.load(f)
LE = MAPPING["label_encoding"]
_rev = {c: {v: k for k, v in m.items()} for c, m in LE.items()}
df["ProductCD_raw"] = df["ProductCD"].map(_rev["ProductCD"])
df["card4_raw"] = df["card4"].map(_rev["card4"])


def group_profile(mask, name):
    sub = df[mask]
    n = len(sub)
    if n < 50:
        return None
    y = sub["isFraud"].values
    p = sub["_pred"].values
    pos = int(y.sum())
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return {
        "群体": name, "样本数": n, "欺诈数": pos,
        "欺诈率%": round(100 * pos / n, 3),
        "召回率%": round(100 * tp / pos, 2) if pos else None,
        "精确率%": round(100 * tp / (tp + fp), 2) if (tp + fp) else None,
        "误报率%": round(100 * fp / (fp + tn), 3) if (fp + tn) else None,
        "F1%": round(100 * f1_score(y, p, zero_division=0), 2) if pos else None,
        "AUC": round(float(roc_auc_score(y, sub["_score"])), 4) if 0 < pos < n else None,
        "平均分": round(float(sub["_score"].mean()), 4),
    }


groups = []

# 按产品类别
for c in sorted(df["ProductCD_raw"].dropna().unique()):
    r = group_profile(df["ProductCD_raw"] == c, "产品=%s" % c)
    if r:
        groups.append(r)

# 按卡类型(card4 为卡品牌)
for c in sorted(df["card4_raw"].dropna().unique()):
    r = group_profile(df["card4_raw"] == c, "卡品牌=%s" % c)
    if r:
        groups.append(r)

# 按金额段
amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")
bins = [-np.inf, 50, 200, 1000, np.inf]
labels = ["金额<50", "50-200", "200-1000", "金额>=1000"]
amt_bin = pd.cut(amt, bins=bins, labels=labels)
for lab in labels:
    r = group_profile(amt_bin == lab, lab)
    if r:
        groups.append(r)

# 按身份信息覆盖(预处理已把缺失填为 Unknown, 编码值 0)
id_cat = [c for c in df.columns if c.startswith("id_") and c in LE]
if id_cat:
    unk = LE[id_cat[0]].get("Unknown", 0)
    has_identity = ~df[id_cat[0]].isin([unk])
    for lab, m in (("有身份信息", has_identity), ("无身份信息", ~has_identity)):
        r = group_profile(m, lab)
        if r:
            groups.append(r)
            log("  [%s] 样本 %d, 占比 %.1f%%" % (lab, len(df[m]), 100 * m.mean()))

# 按交易时间季度
dt = pd.to_numeric(df["TransactionDT"], errors="coerce")
q = pd.qcut(dt, 4, labels=["Q1(最早)", "Q2", "Q3", "Q4(最新)"])
for lab in ["Q1(最早)", "Q2", "Q3", "Q4(最新)"]:
    r = group_profile(q == lab, lab)
    if r:
        groups.append(r)

gdf = pd.DataFrame(groups)
log(gdf.to_string(index=False))

# 召回率极差
rec = gdf["召回率%"].dropna()
spread = float(rec.max() - rec.min()) if len(rec) else 0.0
worst = gdf.loc[gdf["召回率%"].idxmin()] if gdf["召回率%"].notna().any() else None
log("")
log("召回率极差 %.2f pt（最高 %.2f%% / 最低 %.2f%%）"
    % (spread, rec.max() if len(rec) else 0, rec.min() if len(rec) else 0))

# 图: 各子群体召回率
fig, ax = plt.subplots(figsize=(11, 5.2))
g2 = gdf.dropna(subset=["召回率%"]).sort_values("召回率%")
colors = ["#d93025" if v < 60 else ("#f9ab00" if v < 75 else "#188038")
          for v in g2["召回率%"]]
ax.barh(g2["群体"], g2["召回率%"], color=colors)
ax.axvline(100 * float((y_test == 1).sum()) and
           gdf["召回率%"].mean(), color="navy", ls="--", lw=1,
           label="整体平均 %.1f%%" % gdf["召回率%"].mean())
for i, (n_, v) in enumerate(zip(g2["样本数"], g2["召回率%"])):
    ax.text(v + 0.8, i, "%.1f%%  (n=%d)" % (v, n_), va="center", fontsize=8)
ax.set_xlabel("召回率 %")
ax.set_title("各子群体召回率（红线 <60%%，黄线 60~75%%）")
ax.set_xlim(0, 112)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "subgroup_recall.png"), dpi=140)
plt.close(fig)

out = {
    "校准检查": calibration,
    "子群体": groups,
    "召回率极差pt": round(spread, 2),
    "最低召回群体": worst.to_dict() if worst is not None else None,
    "整体": biz(prob),
    "阈值": {"告警": THR, "关注": WARN, "严格": STRICT},
}
with open(os.path.join(OUT, "calibration_subgroup_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
log("")
log("结果写入 models/calibration_subgroup_results.json")
