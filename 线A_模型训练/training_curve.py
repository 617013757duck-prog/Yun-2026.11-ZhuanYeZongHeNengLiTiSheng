# -*- coding: utf-8 -*-
"""
线A · A7 训练过程诊断
回答课程问题: 模型训练是否正常? 模型是否真的学会?

产出:
  figures/train_curve_loss.png   Train/Valid LogLoss 随迭代变化
  figures/train_curve_f1.png     Train/Valid F1 随迭代变化(阈值 0.50 与 0.79)
  figures/learning_curve.png     训练样本量 vs 验证集表现
  models/training_log.json       逐轮指标与学习曲线原始数据
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 42
DROPS = ["isFraud", "ProductCD", "ProductCD_label"]
TAG = ["is_unbalance", "auto"]  # LightGBM 参数占位, 实际用 scale_pos_weight

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def f1_at(y, prob, th):
    return f1_score(y, (prob >= th).astype(int), zero_division=0)


def feval_multi(preds, dataset):
    """每轮额外记录两个阈值下的 F1(preds 已是概率, 不可再 sigmoid)"""
    y = dataset.get_label()
    p = preds
    return [
        ("f1_050", f1_at(y, p, 0.50), False),
        ("f1_079", f1_at(y, p, 0.79), False),
    ]


# ---------------- 数据 ----------------
log("读取训练数据 ...")
df = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
X = df.drop(columns=DROPS)
y = df["isFraud"].values
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.10, random_state=SEED, stratify=y)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
log("训练 %d / 验证 %d, 验证欺诈 %d 笔 (%.3f%%), spw=%.2f" % (
    len(X_tr), len(X_va), int(y_va.sum()), 100 * y_va.mean(), spw))

PARAMS = dict(
    objective="binary", metric=["binary_logloss", "auc"], learning_rate=0.05,
    num_leaves=64, subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    scale_pos_weight=spw, n_jobs=-1, force_col_wise=True,
    random_state=SEED, verbose=-1,
)

# ---------------- 主实验: 1000 轮完整训练曲线 ----------------
log("主实验: 训练 1000 轮(不早停), 记录逐轮指标 ...")
dtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
dvalid = lgb.Dataset(X_va, y_va, reference=dtrain, free_raw_data=False)
evals = {}
t0 = time.time()
booster = lgb.train(
    PARAMS, dtrain, num_boost_round=1000,
    valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
    feval=feval_multi,
    callbacks=[lgb.record_evaluation(evals), log_evaluation(period=100)],
)
train_sec = time.time() - t0
log("主实验完成, 耗时 %.1f 秒" % train_sec)

ev = evals
iters = np.arange(1, len(ev["train"]["binary_logloss"]) + 1)
curve = {
    "iteration": iters.tolist(),
    "train_logloss": ev["train"]["binary_logloss"],
    "valid_logloss": ev["valid"]["binary_logloss"],
    "train_auc": ev["train"]["auc"],
    "valid_auc": ev["valid"]["auc"],
    "train_f1_050": ev["train"]["f1_050"],
    "valid_f1_050": ev["valid"]["f1_050"],
    "train_f1_079": ev["train"]["f1_079"],
    "valid_f1_079": ev["valid"]["f1_079"],
}

# ---- 图1: LogLoss 曲线 ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, curve["train_logloss"], label="训练集 LogLoss", lw=1.6)
ax.plot(iters, curve["valid_logloss"], label="验证集 LogLoss", lw=1.6)
ax.set_xlabel("迭代轮次")
ax.set_ylabel("Binary LogLoss")
ax.set_title("训练过程诊断 · LogLoss 随迭代变化")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "train_curve_loss.png"), dpi=150)
plt.close(fig)

# ---- 图2: F1 曲线 ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, curve["train_f1_050"], label="训练集 F1 (阈值0.50)", lw=1.2, alpha=0.75)
ax.plot(iters, curve["valid_f1_050"], label="验证集 F1 (阈值0.50)", lw=1.8)
ax.plot(iters, curve["train_f1_079"], label="训练集 F1 (阈值0.79)", lw=1.2, ls="--", alpha=0.75)
ax.plot(iters, curve["valid_f1_079"], label="验证集 F1 (阈值0.79)", lw=1.8, ls="--")
best_i = int(np.argmax(curve["valid_f1_079"])) + 1
best_f1 = max(curve["valid_f1_079"])
ax.axvline(best_i, color="gray", ls=":", lw=1)
ax.scatter([best_i], [best_f1], color="red", zorder=5)
ax.annotate("验证集最优 F1=%.4f\n第 %d 轮" % (best_f1, best_i),
            xy=(best_i, best_f1), xytext=(best_i * 0.45, best_f1 * 0.75),
            arrowprops=dict(arrowstyle="->", color="red"))
ax.set_xlabel("迭代轮次")
ax.set_ylabel("F1-score")
ax.set_title("训练过程诊断 · F1 随迭代变化(观察过拟合拐点)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "train_curve_f1.png"), dpi=150)
plt.close(fig)
log("已输出 train_curve_loss.png / train_curve_f1.png")

# ---------------- 学习曲线: 样本量 vs 表现 ----------------
log("学习曲线: 25%% / 50%% / 75%% / 100%% 训练样本 ...")
lr_rows = []
for frac in [0.25, 0.50, 0.75, 1.00]:
    if frac < 1.0:
        Xs, _, ys, _ = train_test_split(
            X_tr, y_tr, train_size=frac, random_state=SEED, stratify=y_tr)
    else:
        Xs, ys = X_tr, y_tr
    t1 = time.time()
    m = lgb.train({**PARAMS, "metric": "auc"}, lgb.Dataset(Xs, ys),
                  num_boost_round=400)
    p = m.predict(X_va)
    best_f1 = max(f1_at(y_va, p, t) for t in np.arange(0.30, 0.95, 0.01))
    row = {"训练样本数": int(len(Xs)), "样本占比": frac,
           "验证AUC": round(roc_auc_score(y_va, p), 4),
           "验证最优F1": round(float(best_f1), 4),
           "训练耗时s": round(time.time() - t1, 1)}
    lr_rows.append(row)
    log("  %5.0f%%  n=%6d  AUC=%.4f  F1=%.4f" % (
        100 * frac, len(Xs), row["验证AUC"], row["验证最优F1"]))

fig, ax1 = plt.subplots(figsize=(8, 5))
n = [r["样本占比"] * 100 for r in lr_rows]
ax1.plot(n, [r["验证最优F1"] for r in lr_rows], "o-", color="#d62728", lw=2, label="验证集最优 F1")
ax1.set_xlabel("训练样本占比 (%)")
ax1.set_ylabel("验证集最优 F1", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(n, [r["验证AUC"] for r in lr_rows], "s--", color="#1f77b4", lw=2, label="验证集 AUC")
ax2.set_ylabel("验证集 AUC", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_title("学习曲线 · 增加训练数据是否仍有收益")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "learning_curve.png"), dpi=150)
plt.close(fig)
log("已输出 learning_curve.png")

# ---------------- 落盘 ----------------
best_auc_i = int(np.argmax(curve["valid_auc"])) + 1
summary = {
    "数据集划分": {"训练": int(len(X_tr)), "验证": int(len(X_va)),
                   "验证欺诈数": int(y_va.sum()), "scale_pos_weight": round(spw, 2)},
    "主实验": {
        "迭代轮数": int(iters[-1]), "训练耗时s": round(train_sec, 1),
        "最终训练LogLoss": round(curve["train_logloss"][-1], 5),
        "最终验证LogLoss": round(curve["valid_logloss"][-1], 5),
        "验证LogLoss最低": round(min(curve["valid_logloss"]), 5),
        "验证LogLoss最低轮次": int(np.argmin(curve["valid_logloss"])) + 1,
        "验证AUC最高": round(max(curve["valid_auc"]), 4),
        "验证AUC最高轮次": best_auc_i,
        "验证F1(0.79)最高": round(float(max(curve["valid_f1_079"])), 4),
        "验证F1(0.79)最高轮次": best_i,
        "训练F1(0.79)最终": round(curve["train_f1_079"][-1], 4),
        "训练与验证F1差距": round(curve["train_f1_079"][-1] - curve["valid_f1_079"][-1], 4),
    },
    "学习曲线": lr_rows,
    "逐轮曲线": curve,
}
with open(os.path.join(OUT, "training_log.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 64)
print("A7 训练过程诊断结论")
print("=" * 64)
print("  验证 LogLoss 最低: %.5f (第 %d 轮)" % (
    summary["主实验"]["验证LogLoss最低"], summary["主实验"]["验证LogLoss最低轮次"]))
print("  验证 AUC 最高:    %.4f (第 %d 轮)" % (
    summary["主实验"]["验证AUC最高"], best_auc_i))
print("  验证 F1 最高:     %.4f (第 %d 轮, 阈值0.79)" % (
    summary["主实验"]["验证F1(0.79)最高"], best_i))
print("  训练/验证 F1 差距: %.4f" % summary["主实验"]["训练与验证F1差距"])
print("  学习曲线 F1: " + " -> ".join(
    "%.2f%%:%.4f" % (100 * r["样本占比"], r["验证最优F1"]) for r in lr_rows))
print("=" * 64)
log("A7 完成")
