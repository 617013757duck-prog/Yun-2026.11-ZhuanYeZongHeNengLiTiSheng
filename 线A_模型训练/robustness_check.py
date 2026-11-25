# -*- coding: utf-8 -*-
"""
线A · A10 结果可信性验证
回答课程问题: 实验结果真的可信吗?

三项验证:
  1) 5 个随机种子重复训练 -> 指标均值 ± 标准差
  2) 5 折分层交叉验证   -> 每折 F1 与箱线图
  3) 时间外推验证       -> 按 TransactionDT 排序, 前 80% 训练 / 后 20% 测试
  4) 配对 t 检验        -> LightGBM 与梯度提升 HGB 在 5 折上的差异是否显著
产出: models/robustness_results.json
      figures/cv_boxplot.png
      figures/time_split_compare.png
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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
SPW = (y == 0).sum() / (y == 1).sum()
log("全量 %d 行, 欺诈 %d 笔, spw=%.2f" % (len(X), int(y.sum()), SPW))

BASE_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
    subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    scale_pos_weight=SPW, n_jobs=-1, force_col_wise=True, verbose=-1,
)


def metrics(y_true, prob):
    ths = np.arange(0.30, 0.95, 0.01)
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in ths]
    bi = int(np.argmax(f1s))
    th = float(ths[bi])
    tn, fp, _, _ = confusion_matrix(y_true, (prob >= th).astype(int)).ravel()
    return {
        "AUC": round(roc_auc_score(y_true, prob), 4),
        "最优F1": round(float(f1s[bi]), 4),
        "最优阈值": round(th, 2),
        "FPR": round(fp / (fp + tn), 4),
    }


# ---------------- 1) 多种子 ----------------
log("1/4 多种子重复(5 个种子, 各 1000 轮) ...")
seed_rows = []
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
for s in [42, 7, 2024, 123, 999]:
    t0 = time.time()
    bst = lgb.train({**BASE_PARAMS, "random_state": s}, lgb.Dataset(X_tr, y_tr),
                    num_boost_round=1000)
    m = metrics(y_va, bst.predict(X_va))
    m["seed"] = s
    m["耗时s"] = round(time.time() - t0, 1)
    seed_rows.append(m)
    log("  seed=%-5d F1=%.4f  AUC=%.4f  FPR=%.4f" % (s, m["最优F1"], m["AUC"], m["FPR"]))

f1_arr = np.array([r["最优F1"] for r in seed_rows])
auc_arr = np.array([r["AUC"] for r in seed_rows])
seed_summary = {
    "F1均值": round(float(f1_arr.mean()), 4), "F1标准差": round(float(f1_arr.std(ddof=1)), 4),
    "F1最小": round(float(f1_arr.min()), 4), "F1最大": round(float(f1_arr.max()), 4),
    "AUC均值": round(float(auc_arr.mean()), 4), "AUC标准差": round(float(auc_arr.std(ddof=1)), 4),
}

# ---------------- 2) 5 折交叉验证 + 与 HGB 对比 ----------------
log("2/4 5 折分层交叉验证(LightGBM 600 轮 vs HGB 200 轮) ...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_lgb, cv_hgb = [], []
for k, (tr, va) in enumerate(skf.split(X, y), 1):
    bst = lgb.train(BASE_PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=600)
    ml = metrics(y[va], bst.predict(X.iloc[va]))
    ml["fold"] = k
    cv_lgb.append(ml)

    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        class_weight={0: 1.0, 1: SPW}, random_state=SEED)
    hgb.fit(X.iloc[tr], y[tr])
    mh = metrics(y[va], hgb.predict_proba(X.iloc[va])[:, 1])
    mh["fold"] = k
    cv_hgb.append(mh)
    log("  第%d折  LightGBM F1=%.4f  |  HGB F1=%.4f" % (k, ml["最优F1"], mh["最优F1"]))

lgb_f1 = np.array([r["最优F1"] for r in cv_lgb])
hgb_f1 = np.array([r["最优F1"] for r in cv_hgb])
t_stat, p_val = stats.ttest_rel(lgb_f1, hgb_f1)
cv_summary = {
    "LightGBM_5折F1": [r["最优F1"] for r in cv_lgb],
    "LightGBM_F1均值": round(float(lgb_f1.mean()), 4),
    "LightGBM_F1标准差": round(float(lgb_f1.std(ddof=1)), 4),
    "HGB_5折F1": [r["最优F1"] for r in cv_hgb],
    "HGB_F1均值": round(float(hgb_f1.mean()), 4),
    "配对t检验_t值": round(float(t_stat), 4),
    "配对t检验_p值": round(float(p_val), 4),
    "差异是否显著(p<0.05)": bool(p_val < 0.05),
    "提升幅度": round(float(lgb_f1.mean() - hgb_f1.mean()), 4),
}

# ---------------- 3) 时间外推 ----------------
log("3/4 时间外推验证(按 TransactionDT 排序, 前 80%% / 后 20%%) ...")
order = df["TransactionDT"].argsort().values
X_time, y_time = X.iloc[order], y[order]
cut = int(len(X_time) * 0.80)
Xtr_t, ytr_t = X_time.iloc[:cut], y_time[:cut]
Xva_t, yva_t = X_time.iloc[cut:], y_time[cut:]
log("  训练 %d (欺诈率 %.3f%%) / 测试 %d (欺诈率 %.3f%%)" % (
    len(Xtr_t), 100 * ytr_t.mean(), len(Xva_t), 100 * yva_t.mean()))
bst_t = lgb.train(BASE_PARAMS, lgb.Dataset(Xtr_t, ytr_t), num_boost_round=1000)
time_res = metrics(yva_t, bst_t.predict(Xva_t))
time_res["训练行数"] = int(len(Xtr_t))
time_res["测试行数"] = int(len(Xva_t))
time_res["测试欺诈率"] = round(float(yva_t.mean()), 5)
log("  时间外推 F1=%.4f  AUC=%.4f  FPR=%.4f" % (
    time_res["最优F1"], time_res["AUC"], time_res["FPR"]))

# ---------------- 先落盘, 避免画图异常导致实验数据丢失 ----------------
out = {
    "多种子重复": {"明细": seed_rows, "汇总": seed_summary},
    "5折交叉验证": {"明细LightGBM": cv_lgb, "明细HGB": cv_hgb, "汇总": cv_summary},
    "时间外推": time_res,
}
with open(os.path.join(OUT, "robustness_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log("指标已落盘 robustness_results.json")

# ---------------- 图1: 交叉验证箱线图 ----------------
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([lgb_f1, hgb_f1], tick_labels=["LightGBM", "梯度提升 HGB"],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)
for i, arr in enumerate([lgb_f1, hgb_f1], start=1):
    ax.scatter(np.full(len(arr), i) + np.random.uniform(-0.06, 0.06, len(arr)),
               arr, color="black", s=22, zorder=5)
ax.set_ylabel("5 折验证集最优 F1")
ax.set_title("5 折交叉验证 · LightGBM vs HGB\n配对 t 检验 p=%.4f (%s)" % (
    p_val, "显著" if p_val < 0.05 else "不显著"))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cv_boxplot.png"), dpi=150)
plt.close(fig)

# ---------------- 图2: 随机划分 vs 时间外推 ----------------
fig, ax = plt.subplots(figsize=(8, 5))
names = ["随机划分\n(5 种子均值)", "时间外推\n(前80%→后20%)"]
vals_f1 = [seed_summary["F1均值"], time_res["最优F1"]]
vals_fpr = [float(np.mean([r["FPR"] for r in seed_rows])), time_res["FPR"]]
x = np.arange(2)
w = 0.35
b1 = ax.bar(x - w / 2, vals_f1, w, label="F1", color="#1f77b4")
b2 = ax.bar(x + w / 2, vals_fpr, w, label="误报率 FPR", color="#d62728")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                "%.4f" % b.get_height(), ha="center", fontsize=9)
ax.set_xticks(x, names)
ax.set_ylim(0, max(vals_f1) * 1.25)
ax.set_ylabel("指标值")
ax.set_title("随机划分 vs 时间外推 · 模型在'未来数据'上的真实表现")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "time_split_compare.png"), dpi=150)
plt.close(fig)

# ---------------- 落盘(二次确认) ----------------
with open(os.path.join(OUT, "robustness_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 68)
print("A10 可信性验证结论")
print("=" * 68)
print("  5 种子 F1: %.4f ± %.4f  (范围 %.4f ~ %.4f)" % (
    seed_summary["F1均值"], seed_summary["F1标准差"],
    seed_summary["F1最小"], seed_summary["F1最大"]))
print("  5 折  F1: %.4f ± %.4f" % (
    cv_summary["LightGBM_F1均值"], cv_summary["LightGBM_F1标准差"]))
print("  vs HGB : LGBM %.4f / HGB %.4f, 配对 t 检验 p=%.4f (%s)" % (
    cv_summary["LightGBM_F1均值"], cv_summary["HGB_F1均值"], p_val,
    "差异显著" if p_val < 0.05 else "差异不显著"))
print("  时间外推: F1=%.4f, AUC=%.4f, FPR=%.4f" % (
    time_res["最优F1"], time_res["AUC"], time_res["FPR"]))
print("=" * 68)
log("A10 完成")
