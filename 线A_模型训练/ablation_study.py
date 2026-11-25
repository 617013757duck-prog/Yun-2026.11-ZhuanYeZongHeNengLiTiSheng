# -*- coding: utf-8 -*-
"""
线A · A9 消融实验
回答课程问题: 我做的改进真的有效吗?

六组对照, 每组只改变一个因素, 其余固定:
  G0 完整基线
  G1 去除身份表特征(id_* / DeviceType / DeviceInfo)
  G2 去除 V 匿名特征(V1-V339)
  G3 去除金额与时间特征(TransactionAmt / TransactionDT / D1-D15)
  G4 不使用类别权重(scale_pos_weight=1)
  G5 不做阈值调优(固定阈值 0.50)
产出: models/ablation_results.json + figures/ablation_comparison.png
"""
import json
import os
import re
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
ROUNDS = 400
DROPS = ["isFraud", "ProductCD", "ProductCD_label"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


df = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
X_all = df.drop(columns=DROPS)
y = df["isFraud"].values
X_tr_full, X_va_full, y_tr, y_va = train_test_split(
    X_all, y, test_size=0.10, random_state=SEED, stratify=y)
SPW = (y_tr == 0).sum() / (y_tr == 1).sum()

# 特征分组
ID_COLS = [c for c in X_all.columns if c.startswith("id_")] + \
          [c for c in X_all.columns if c in ("DeviceType", "DeviceInfo")]
V_COLS = [c for c in X_all.columns if re.fullmatch(r"V\d+", c)]
TIME_AMT_COLS = [c for c in X_all.columns
                 if c in ("TransactionAmt", "TransactionDT") or re.fullmatch(r"D\d+", c)]

log("特征分组: 身份 %d 列, V 匿名 %d 列, 金额时间 %d 列, 总计 %d 列" % (
    len(ID_COLS), len(V_COLS), len(TIME_AMT_COLS), X_all.shape[1]))

BASE_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
    subsample=0.8, bagging_freq=5, colsample_bytree=0.8, n_jobs=-1,
    force_col_wise=True, random_state=SEED, verbose=-1,
)


def run(name, drop_cols, spw, tune_threshold=True):
    cols = [c for c in X_all.columns if c not in drop_cols]
    Xtr, Xva = X_tr_full[cols], X_va_full[cols]
    params = {**BASE_PARAMS, "scale_pos_weight": spw}
    t0 = time.time()
    bst = lgb.train(params, lgb.Dataset(Xtr, y_tr), num_boost_round=ROUNDS)
    cost = time.time() - t0
    p = bst.predict(Xva)

    if tune_threshold:
        ths = np.arange(0.30, 0.95, 0.01)
        f1s = [f1_score(y_va, (p >= t).astype(int), zero_division=0) for t in ths]
        bi = int(np.argmax(f1s))
        best_th, best_f1 = float(ths[bi]), float(f1s[bi])
    else:
        best_th = 0.50
        best_f1 = f1_score(y_va, (p >= 0.50).astype(int), zero_division=0)

    tn79, fp79, _, _ = confusion_matrix(y_va, (p >= 0.79).astype(int)).ravel()
    tnb, fpb, _, _ = confusion_matrix(y_va, (p >= best_th).astype(int)).ravel()
    row = {
        "组别": name, "特征数": len(cols), "scale_pos_weight": round(spw, 2),
        "最优阈值": round(best_th, 2), "最优F1": round(best_f1, 4),
        "F1@0.50": round(f1_score(y_va, (p >= 0.50).astype(int), zero_division=0), 4),
        "AUC": round(roc_auc_score(y_va, p), 4),
        "FPR@最优阈值": round(fpb / (fpb + tnb), 4),
        "FPR@0.79": round(fp79 / (fp79 + tn79), 4),
        "训练耗时s": round(cost, 1),
    }
    log("  %-18s 特征%3d  F1=%.4f  AUC=%.4f  FPR=%.4f" % (
        name, len(cols), row["最优F1"], row["AUC"], row["FPR@最优阈值"]))
    return row


log("开始六组消融实验(每组 %d 轮) ..." % ROUNDS)
rows = [
    run("G0 完整基线", [], SPW, True),
    run("G1 去身份表", ID_COLS, SPW, True),
    run("G2 去V匿名特征", V_COLS, SPW, True),
    run("G3 去金额与时间", TIME_AMT_COLS, SPW, True),
    run("G4 无类别权重", [], 1.0, True),
    run("G5 不调阈值", [], SPW, False),
]

base_f1 = rows[0]["最优F1"]
for r in rows:
    r["F1相对基线变化"] = round(r["最优F1"] - base_f1, 4)
    r["F1相对变化%"] = round(100 * (r["最优F1"] - base_f1) / base_f1, 2)

with open(os.path.join(OUT, "ablation_results.json"), "w", encoding="utf-8") as f:
    json.dump({"rounds": ROUNDS, "baseline_scale_pos_weight": round(SPW, 2),
               "results": rows}, f, ensure_ascii=False, indent=2)

# ---------------- 图 ----------------
labels = [r["组别"] for r in rows]
f1v = [r["最优F1"] for r in rows]
fprv = [r["FPR@最优阈值"] for r in rows]
colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(rows))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ypos = np.arange(len(rows))[::-1]

axes[0].barh(ypos, f1v, color=colors)
axes[0].set_yticks(ypos, labels, fontsize=9)
axes[0].set_xlim(0, max(f1v) * 1.18)
axes[0].set_xlabel("验证集最优 F1")
axes[0].set_title("消融实验 · F1 对比(绿色为完整基线)")
for yv, v, r in zip(ypos, f1v, rows):
    txt = "%.4f" % v if r["组别"].startswith("G0") else "%.4f (%+.2f%%)" % (v, r["F1相对变化%"])
    axes[0].text(v + 0.004, yv, txt, va="center", fontsize=8)
axes[0].grid(axis="x", alpha=0.3)

axes[1].barh(ypos, fprv, color=colors)
axes[1].set_yticks(ypos, labels, fontsize=9)
axes[1].set_xlabel("误报率 FPR(最优阈值下)")
axes[1].set_title("消融实验 · 误报率对比(越低越好)")
for yv, v in zip(ypos, fprv):
    axes[1].text(v + 0.0002, yv, "%.4f" % v, va="center", fontsize=8)
axes[1].grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "ablation_comparison.png"), dpi=150)
plt.close(fig)

print("\n" + "=" * 68)
print("A9 消融实验结论(相对完整基线的 F1 变化)")
print("=" * 68)
for r in rows[1:]:
    print("  %-18s F1=%s  (%+.2f%%)   FPR=%s" % (
        r["组别"], r["最优F1"], r["F1相对变化%"], r["FPR@最优阈值"]))
print("=" * 68)
log("A9 完成")
