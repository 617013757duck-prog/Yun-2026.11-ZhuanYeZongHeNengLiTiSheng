# -*- coding: utf-8 -*-
"""
线A · 模型选择环节
目的: 在正式训练前对比候选模型, 用验证集数据支撑技术路线决策
任务类型判定:
  - 异常交易检测 -> 二分类检测任务 -> 关注 Recall / F1 / 误报率 FPR
  - 数据分类     -> 多分类任务     -> 关注 Accuracy / Macro-F1
候选模型(覆盖线性/单树/Bagging/GBDT 四类):
  LR / DecisionTree / RandomForest / HistGradientBoosting / LightGBM
说明: 选型阶段用分层抽样(训练 120k, 验证 25k), 控制对比成本并保证公平
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
SEED = 42
N_TRAIN, N_VALID = 120000, 25000


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---------------- 数据 ----------------
log("读取 processed 数据 ...")
train = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
DROP = ["isFraud", "ProductCD", "ProductCD_label"]
X_all = train.drop(columns=DROP)
y_fraud = train["isFraud"].values
y_prod = train["ProductCD_label"].values

# 分层抽样, 保证少数类比例
idx_tr, idx_va = train_test_split(
    np.arange(len(X_all)), train_size=N_TRAIN, test_size=N_VALID,
    random_state=SEED, stratify=y_prod * 10 + y_fraud)
X_tr, X_va = X_all.iloc[idx_tr], X_all.iloc[idx_va]
yf_tr, yf_va = y_fraud[idx_tr], y_fraud[idx_va]
yp_tr, yp_va = y_prod[idx_tr], y_prod[idx_va]
log("选型样本: 训练 %d (欺诈率 %.3f%%), 验证 %d (欺诈率 %.3f%%)" % (
    len(X_tr), 100 * yf_tr.mean(), len(X_va), 100 * yf_va.mean()))

SPW = (yf_tr == 0).sum() / (yf_tr == 1).sum()
log("scale_pos_weight = %.2f" % SPW)


def make_models(task):
    """task: 'fraud' or 'product'"""
    if task == "fraud":
        lam = SPW
        return {
            "逻辑回归 LR": Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, n_jobs=-1,
                                           class_weight="balanced"))]),
            "决策树 DT": DecisionTreeClassifier(max_depth=18, min_samples_leaf=50,
                                            class_weight="balanced", random_state=SEED),
            "随机森林 RF": RandomForestClassifier(n_estimators=100, max_depth=20,
                                             min_samples_leaf=20, max_samples=0.6,
                                             class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=SEED),
            "梯度提升 HGB": HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                class_weight={0: 1.0, 1: lam}, random_state=SEED),
            "LightGBM": __import__("lightgbm").LGBMClassifier(
                objective="binary", n_estimators=300, learning_rate=0.05,
                num_leaves=64, scale_pos_weight=lam, subsample=0.8,
                bagging_freq=5, colsample_bytree=0.8, n_jobs=-1,
                force_col_wise=True, random_state=SEED, verbose=-1),
        }
    return {
        "逻辑回归 LR": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, n_jobs=-1,
                                       class_weight="balanced"))]),
        "决策树 DT": DecisionTreeClassifier(max_depth=18, min_samples_leaf=50,
                                        class_weight="balanced", random_state=SEED),
        "随机森林 RF": RandomForestClassifier(n_estimators=100, max_depth=20,
                                         min_samples_leaf=20, max_samples=0.6,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=SEED),
        "梯度提升 HGB": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            class_weight="balanced", random_state=SEED),
        "LightGBM": __import__("lightgbm").LGBMClassifier(
            objective="multiclass", num_class=5, n_estimators=300,
            learning_rate=0.05, num_leaves=64, subsample=0.8,
            bagging_freq=5, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, force_col_wise=True, random_state=SEED, verbose=-1),
    }


def best_threshold(y_true, prob):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.02, 0.98, 0.01):
        f = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def run(task):
    label = "异常交易检测(isFraud)" if task == "fraud" else "数据分类(ProductCD)"
    ytr, yva = (yf_tr, yf_va) if task == "fraud" else (yp_tr, yp_va)
    print("\n" + "=" * 72)
    print("任务: %s" % label)
    print("=" * 72)
    rows = []
    for name, mdl in make_models(task).items():
        t0 = time.time()
        mdl.fit(X_tr, ytr)
        train_s = time.time() - t0

        t1 = time.time()
        prob = mdl.predict_proba(X_va)
        infer_s = time.time() - t1

        if task == "fraud":
            p1 = prob[:, 1]
            th, _ = best_threshold(yva, p1)
            pred = (p1 >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(yva, pred).ravel()
            r = {
                "模型": name,
                "Accuracy": round(accuracy_score(yva, pred), 4),
                "Precision": round(precision_score(yva, pred, zero_division=0), 4),
                "Recall": round(recall_score(yva, pred, zero_division=0), 4),
                "F1": round(f1_score(yva, pred, zero_division=0), 4),
                "误报率FPR": round(fp / (fp + tn), 4),
                "ROC-AUC": round(roc_auc_score(yva, p1), 4),
                "最优阈值": round(th, 2),
                "训练耗时s": round(train_s, 1),
                "推理耗时s": round(infer_s, 2),
            }
        else:
            pred = np.argmax(prob, axis=1)
            r = {
                "模型": name,
                "Accuracy": round(accuracy_score(yva, pred), 4),
                "Macro-F1": round(f1_score(yva, pred, average="macro"), 4),
                "Weighted-F1": round(f1_score(yva, pred, average="weighted"), 4),
                "训练耗时s": round(train_s, 1),
                "推理耗时s": round(infer_s, 2),
            }
        rows.append(r)
        print("  %-14s Acc=%.4f  %s  训练=%.1fs" % (
            name, r["Accuracy"],
            ("F1=%.4f FPR=%.4f AUC=%.4f" % (r["F1"], r["误报率FPR"], r["ROC-AUC"]))
            if task == "fraud" else
            ("MacroF1=%.4f" % r["Macro-F1"]), train_s), flush=True)

    dfr = pd.DataFrame(rows)
    fname = "model_selection_%s.json" % task
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        json.dump({"task": label, "sample": {"train": N_TRAIN, "valid": N_VALID},
                   "results": rows}, f, ensure_ascii=False, indent=2)
    dfr.to_csv(os.path.join(OUT, "model_selection_%s.csv" % task),
               index=False, encoding="utf-8-sig")
    return dfr


df_f = run("fraud")
df_p = run("product")

# ---------------- 对比图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(df_f)); w = 0.2
for i, (col, c) in enumerate(zip(["Accuracy", "Precision", "Recall", "F1"],
                                 ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])):
    axes[0].bar(x + (i - 1.5) * w, df_f[col], w, label=col, color=c)
axes[0].set_xticks(x, df_f["模型"], rotation=15, fontsize=9)
axes[0].set_title("异常交易检测 — 候选模型指标对比")
axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

axes2 = axes[0].twinx()
axes2.plot(x, df_f["误报率FPR"], "o--", color="gray", lw=1.5, label="误报率FPR")
axes2.set_ylabel("误报率 FPR"); axes2.legend(loc="lower right", fontsize=8)

x2 = np.arange(len(df_p))
axes[1].bar(x2 - 0.2, df_p["Accuracy"], 0.4, label="Accuracy", color="#1f77b4")
axes[1].bar(x2 + 0.2, df_p["Macro-F1"], 0.4, label="Macro-F1", color="#2ca02c")
axes[1].set_xticks(x2, df_p["模型"], rotation=15, fontsize=9)
axes[1].set_title("数据分类 — 候选模型指标对比")
axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "model_selection_comparison.png"), dpi=150)
plt.close(fig)

# ---------------- 选型结论 ----------------
best_f = df_f.sort_values("F1", ascending=False).iloc[0]["模型"]
best_p = df_p.sort_values("Macro-F1", ascending=False).iloc[0]["模型"]
summary = {"异常交易检测最优模型": best_f, "数据分类最优模型": best_p,
           "最终技术路线": "LightGBM 统一建模(两任务)"}
with open(os.path.join(OUT, "model_selection_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 72)
print("选型结论: 异常检测最优=%s | 分类最优=%s" % (best_f, best_p))
print("=" * 72)
log("模型选择完成")
