# -*- coding: utf-8 -*-
"""
线A · A8 超参数实验
回答课程问题: 同样的代码, 为什么别人的模型效果更好?

方法: 单变量控制实验。除被考察参数外, 其余参数固定为基线值。
基线: learning_rate=0.05, num_leaves=64, n_estimators=400, scale_pos_weight=27.58
产出: models/hyperparam_results.json + figures/hp_overview.png
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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
SEED = 42
DROPS = ["isFraud", "ProductCD", "ProductCD_label"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


df = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
X = df.drop(columns=DROPS)
y = df["isFraud"].values
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.10, random_state=SEED, stratify=y)
SPW = (y_tr == 0).sum() / (y_tr == 1).sum()
dtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
log("训练 %d / 验证 %d, spw=%.2f" % (len(X_tr), len(X_va), SPW))

BASE_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
    subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    scale_pos_weight=SPW, n_jobs=-1, force_col_wise=True,
    random_state=SEED, verbose=-1,
)


def evaluate(params, rounds):
    t0 = time.time()
    bst = lgb.train({**BASE_PARAMS, **params}, dtrain, num_boost_round=rounds)
    cost = time.time() - t0
    p = bst.predict(X_va)
    ths = np.arange(0.30, 0.95, 0.01)
    f1s = [f1_score(y_va, (p >= t).astype(int), zero_division=0) for t in ths]
    bi = int(np.argmax(f1s))
    tn, fp, fn, tp = confusion_matrix(y_va, (p >= 0.79).astype(int)).ravel()
    return {
        "轮数": rounds,
        "AUC": round(roc_auc_score(y_va, p), 4),
        "最优F1": round(float(f1s[bi]), 4),
        "最优阈值": round(float(ths[bi]), 2),
        "FPR@0.79": round(fp / (fp + tn), 4),
        "训练耗时s": round(cost, 1),
    }


EXPERIMENTS = [
    ("learning_rate", [0.01, 0.05, 0.10, 0.20], {}, 400),
    ("num_leaves", [15, 31, 64, 127], {}, 400),
    ("n_estimators", [200, 400, 800, 1200], {}, None),
    ("scale_pos_weight", [1.0, 10.0, round(SPW, 2), 50.0], {}, 400),
]

results = {}
for pname, values, fixed, rounds in EXPERIMENTS:
    log("实验变量: %s -> %s" % (pname, values))
    rows = []
    for v in values:
        r = evaluate({pname: v, **fixed}, rounds if rounds else v)
        r[pname] = v
        rows.append(r)
        log("  %-16s AUC=%.4f  F1=%.4f  FPR@0.79=%.4f  %.1fs" % (
            "%s=%s" % (pname, v), r["AUC"], r["最优F1"], r["FPR@0.79"], r["训练耗时s"]))
    results[pname] = rows

with open(os.path.join(OUT, "hyperparam_results.json"), "w", encoding="utf-8") as f:
    json.dump({"baseline": BASE_PARAMS, "results": results}, f,
              ensure_ascii=False, indent=2)

# ---------------- 四联图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
titles = {
    "learning_rate": "学习率 learning_rate",
    "num_leaves": "叶子数 num_leaves",
    "n_estimators": "迭代轮数 n_estimators",
    "scale_pos_weight": "正类权重 scale_pos_weight",
}
for ax, (pname, rows) in zip(axes.ravel(), results.items()):
    xs = [str(r[pname]) for r in rows]
    ax.plot(xs, [r["最优F1"] for r in rows], "o-", lw=2, color="#d62728", label="验证最优 F1")
    ax.plot(xs, [r["AUC"] for r in rows], "s--", lw=1.8, color="#1f77b4", label="验证 AUC")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["FPR@0.79"] for r in rows], "^:", lw=1.5, color="#7f7f7f", label="FPR@0.79")
    ax2.set_ylabel("FPR@0.79", color="#7f7f7f", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#7f7f7f", labelsize=8)
    ax.set_title(titles[pname], fontsize=11)
    ax.set_xlabel(pname)
    ax.set_ylabel("F1 / AUC")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
fig.suptitle("超参数单变量控制实验(其余参数固定为基线)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "hp_overview.png"), dpi=150)
plt.close(fig)

# ---------------- 结论 ----------------
print("\n" + "=" * 62)
print("A8 超参数实验结论(按验证最优 F1 排序)")
print("=" * 62)
for pname, rows in results.items():
    best = max(rows, key=lambda r: r["最优F1"])
    print("  %-18s 最优取值 = %-6s (F1=%.4f, AUC=%.4f)" % (
        pname, best[pname], best["最优F1"], best["AUC"]))
print("=" * 62)
log("A8 完成")
