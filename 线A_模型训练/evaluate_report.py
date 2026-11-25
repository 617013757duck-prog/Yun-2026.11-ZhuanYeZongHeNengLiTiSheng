# -*- coding: utf-8 -*-
"""
线A · A5 指标汇总与可视化
输出: ROC 曲线 / PR 曲线 / 阈值-指标曲线 / 分类混淆矩阵热力图 / 指标汇总 json
"""
import json
import os
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL_DIR = os.path.join(BASE, "models")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_model(name):
    with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as f:
        return lgb.Booster(model_str=f.read())


log("读取测试集 ...")
test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))
DROP = ["isFraud", "ProductCD", "ProductCD_label"]
X_te = test.drop(columns=DROP)
y_fraud = test["isFraud"].values
y_prod = test["ProductCD_label"].values

# ---------------- 异常检测模型曲线 ----------------
log("加载异常检测模型, 计算 ROC/PR 曲线 ...")
bst = load_model("lgbm_fraud_baseline.txt")
prob = bst.predict(X_te)

fpr, tpr, _ = roc_curve(y_fraud, prob)
roc_auc = auc(fpr, tpr)
prec, rec, ths = precision_recall_curve(y_fraud, prob)
pr_auc = auc(rec, prec)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, lw=2, label="ROC (AUC=%.4f)" % roc_auc)
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("误报率 FPR"); axes[0].set_ylabel("召回率 TPR")
axes[0].set_title("异常交易检测 ROC 曲线"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(rec, prec, lw=2, label="PR (AUC=%.4f)" % pr_auc)
axes[1].axhline(y_fraud.mean(), ls="--", color="gray", label="随机基线 %.3f" % y_fraud.mean())
axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
axes[1].set_title("异常交易检测 PR 曲线"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fraud_roc_pr.png"), dpi=150)
plt.close(fig)

# 阈值-指标曲线
from sklearn.metrics import f1_score
ts = np.arange(0.05, 0.95, 0.01)
f1s, fprs = [], []
for t in ts:
    pred = prob >= t
    f1s.append(f1_score(y_fraud, pred))
    tn, fp, fn, tp = confusion_matrix(y_fraud, pred).ravel()
    fprs.append(fp / (fp + tn))
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ts, f1s, lw=2, label="F1-score")
ax.set_xlabel("阈值"); ax.set_ylabel("F1-score")
ax2 = ax.twinx()
ax2.plot(ts, fprs, lw=2, color="tomato", label="误报率 FPR")
ax2.set_ylabel("误报率 FPR")
best_i = int(np.argmax(f1s))
ax.axvline(ts[best_i], ls="--", color="gray")
ax.set_title("阈值调优曲线 (最优阈值=%.2f, F1=%.4f)" % (ts[best_i], f1s[best_i]))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fraud_threshold_tuning.png"), dpi=150)
plt.close(fig)
log("  最优阈值=%.2f" % ts[best_i])

# ---------------- 分类模型混淆矩阵热力图 ----------------
log("加载分类模型, 绘制混淆矩阵 ...")
bst2 = load_model("lgbm_product_classifier.txt")
pred2 = np.argmax(bst2.predict(X_te), axis=1)
cm = confusion_matrix(y_prod, pred2)
cmn = cm / cm.sum(axis=1, keepdims=True)
CLASSES = ["C", "H", "R", "S", "W"]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cmn, cmap="Blues")
ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
ax.set_title("数据分类 (ProductCD) 归一化混淆矩阵")
for i in range(5):
    for j in range(5):
        ax.text(j, i, "%.2f" % cmn[i, j], ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "product_confusion_matrix.png"), dpi=150)
plt.close(fig)

# ---------------- 汇总 ----------------
with open(os.path.join(MODEL_DIR, "metrics_fraud_baseline.json"), encoding="utf-8") as f:
    mf = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics_product_classifier.json"), encoding="utf-8") as f:
    mp = json.load(f)
summary = {
    "异常交易检测(isFraud)": {
        "Accuracy": mf["tuned_threshold"]["accuracy"],
        "F1-score": mf["tuned_threshold"]["f1"],
        "误报率FPR": mf["tuned_threshold"]["fpr"],
        "ROC-AUC": mf["tuned_threshold"]["roc_auc"],
        "PR-AUC": mf["tuned_threshold"]["pr_auc"],
        "最优阈值": mf["tuned_threshold"]["threshold"],
    },
    "数据分类(ProductCD)": {
        "Accuracy": mp["accuracy"],
        "Macro-F1": mp["f1_macro"],
        "Weighted-F1": mp["f1_weighted"],
    },
}
with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n========== 指标汇总 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
log("A5 完成, 图片输出到 %s" % os.path.abspath(FIG_DIR))
