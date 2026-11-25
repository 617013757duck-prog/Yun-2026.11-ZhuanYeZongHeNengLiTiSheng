# -*- coding: utf-8 -*-
"""
线A · A14 外部数据集复现验证

目的: 回答"模型是否只在 IEEE-CIS 上好用"这一质疑。
做法: 换一份独立的公开欺诈数据集(ULB Credit Card Fraud, OpenML id=1597),
      用完全相同的建模流程(轻量 LightGBM + 阈值调优)重新训练与评估,
      验证"方法可复现", 而不是"模型文件可迁移"(两份数据特征空间不同, 无法直接套用)。

三组实验:
  E1 随机划分 + 项目原配置(spw = 负正比)
  E2 随机划分 + spw = 1(A8/A9 得出的更优配置)
  E3 时间外推(按 Time 排序, 前 80% 训练 / 后 20% 测试), spw = 1

产出: models/external_validation.json
      figures/external_roc_pr.png
      figures/external_threshold.png
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "..", "_data_src", "external")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
    subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    n_jobs=-1, force_col_wise=True, verbose=-1,
)
ROUNDS = 1000


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_creditcard():
    os.makedirs(CACHE, exist_ok=True)
    bunch = fetch_openml("creditcard", version=1, as_frame=True,
                         data_home=CACHE, parser="auto")
    df = bunch.frame.copy()
    y = pd.to_numeric(df.pop("Class"), errors="coerce").astype(int)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df, y


def eval_block(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def tune_threshold(y_true, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 0.99, 0.01):
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, spw):
    params = dict(PARAMS)
    params["scale_pos_weight"] = spw
    t0 = time.time()
    booster = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=ROUNDS,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    elapsed = time.time() - t0
    prob_va = booster.predict(X_va)
    prob_te = booster.predict(X_te)
    thr, val_f1 = tune_threshold(y_va, prob_va)
    result = eval_block(y_te, prob_te, thr)
    result["val_best_f1"] = round(float(val_f1), 4)
    result["train_seconds"] = round(elapsed, 2)
    result["scale_pos_weight"] = round(float(spw), 2)
    return result, prob_te, thr


def plot_curves(y_true, prob, thr, path_roc, path_thr, title_prefix):
    fpr, tpr, _ = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    ap = average_precision_score(y_true, prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fpr, tpr, color="#1a73e8", lw=2, label="ROC (AUC=%.4f)" % auc)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set_xlabel("假正率 FPR")
    axes[0].set_ylabel("真正率 TPR")
    axes[0].set_title("%s ROC 曲线" % title_prefix)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#d93025", lw=2, label="PR (AP=%.4f)" % ap)
    axes[1].set_xlabel("召回率 Recall")
    axes[1].set_ylabel("精确率 Precision")
    axes[1].set_title("%s PR 曲线" % title_prefix)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_roc, dpi=140)
    plt.close(fig)

    ths = np.arange(0.01, 0.99, 0.01)
    f1s, fprs = [], []
    for t in ths:
        pred = (prob >= t).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
        tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(ths, f1s, color="#188038", lw=2, label="F1")
    ax1.set_xlabel("判定阈值")
    ax1.set_ylabel("F1", color="#188038")
    ax1.tick_params(axis="y", labelcolor="#188038")
    ax2 = ax1.twinx()
    ax2.plot(ths, fprs, color="#d93025", lw=2, ls="--", label="FPR")
    ax2.set_ylabel("误报率 FPR", color="#d93025")
    ax2.tick_params(axis="y", labelcolor="#d93025")
    ax1.axvline(thr, color="grey", ls=":", lw=1.2)
    ax1.set_title("阈值-F1-FPR（虚线为验证集最优阈值 %.2f）" % thr)
    fig.tight_layout()
    fig.savefig(path_thr, dpi=140)
    plt.close(fig)


X, y = load_creditcard()
n, pos = len(X), int(y.sum())
log("外部数据集 ULB Credit Card Fraud: %d 行 × %d 特征, 欺诈 %d 笔 (%.4f%%)"
    % (n, X.shape[1], pos, 100.0 * pos / n))

results = {
    "dataset": {
        "name": "ULB Credit Card Fraud Detection",
        "source": "OpenML id=1597 (Kaggle ULB 原始版本)",
        "rows": n,
        "features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "fraud_count": pos,
        "fraud_rate": round(pos / n, 6),
        "note": "OpenML 版本含 V1-V28 + Amount 共 29 个特征(Time 列被移除); "
                "特征为 PCA 脱敏, 与 IEEE-CIS 特征空间不重叠, 因此只能复现方法, "
                "不能直接套用已训练模型文件",
    },
    "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
    "rounds": ROUNDS,
}

# ---------- E1 / E2 随机划分 60/20/20 ----------
X_tmp, X_te, y_tmp, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
log("随机划分: 训练 %d / 验证 %d / 测试 %d" % (len(X_tr), len(X_va), len(X_te)))

spw_ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
log("E1 原配置 spw=%.2f" % spw_ratio)
e1, prob_e1, thr_e1 = train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, spw_ratio)
log("E1 测试 F1=%.4f FPR=%.5f AUC=%.4f" % (e1["f1"], e1["fpr"], e1["roc_auc"]))

log("E2 spw=1")
e2, prob_e2, thr_e2 = train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, 1.0)
log("E2 测试 F1=%.4f FPR=%.5f AUC=%.4f" % (e2["f1"], e2["fpr"], e2["roc_auc"]))

# ---------- E3 时间外推 ----------
# OpenML 版本已移除 Time 列; ULB 原始数据按交易时间顺序采集, 故用行序近似时间序
order = np.arange(len(X))
X_sorted, y_sorted = X.iloc[order], y.iloc[order]
cut = int(len(X_sorted) * 0.8)
X_tr3, X_te3 = X_sorted.iloc[:cut], X_sorted.iloc[cut:]
y_tr3, y_te3 = y_sorted.iloc[:cut], y_sorted.iloc[cut:]
val_cut = int(len(X_tr3) * 0.8)
X_tr3a, X_va3 = X_tr3.iloc[:val_cut], X_tr3.iloc[val_cut:]
y_tr3a, y_va3 = y_tr3.iloc[:val_cut], y_tr3.iloc[val_cut:]
log("时间外推: 训练 %d / 验证 %d / 测试 %d"
    % (len(X_tr3a), len(X_va3), len(X_te3)))
e3, prob_e3, thr_e3 = train_eval(X_tr3a, y_tr3a, X_va3, y_va3, X_te3, y_te3, 1.0)
log("E3 测试 F1=%.4f FPR=%.5f AUC=%.4f" % (e3["f1"], e3["fpr"], e3["roc_auc"]))

results["E1_random_spw_ratio"] = e1
results["E2_random_spw_1"] = e2
results["E3_time_spw_1"] = {
    **e3,
    "train_rows": int(len(X_tr3a)),
    "val_rows": int(len(X_va3)),
    "test_rows": int(len(X_te3)),
}
results["ieee_reference"] = {
    "random_test_f1": 0.7577,
    "random_test_fpr": 0.0045,
    "cv_f1": 0.7207,
    "time_split_f1": 0.5469,
}

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "external_validation.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

plot_curves(y_te, prob_e2, thr_e2,
            os.path.join(FIG, "external_roc_pr.png"),
            os.path.join(FIG, "external_threshold.png"),
            "ULB 信用卡欺诈")

log("完成, 结果写入 models/external_validation.json")
print(json.dumps({k: results[k] for k in
                  ("E1_random_spw_ratio", "E2_random_spw_1", "E3_time_spw_1")},
                 ensure_ascii=False, indent=2))
