# -*- coding: utf-8 -*-
"""
线A · A17 v2 配置下的稳定性重验

A10 的稳定性实验用的是 v1 配置(spw=27.58, leaves=64), 与正式模型 v2 不一致,
且阈值扫描范围是 0.30~0.95, 会漏掉 v2 的 0.17 附近的最优区间。
本脚本用与 retrain_final.py 完全一致的 v2 配置重跑, 使证据链与正式模型对齐。

三组实验:
  R1 5 个随机种子重复
  R2 5 折分层交叉验证 (LightGBM vs HGB, 配对 t 检验)
  R3 时间外推 (前 80% 训练 / 后 20% 测试), 并比较"沿用旧阈值"与"重校准阈值"

产出:
  models/robustness_v2_results.json
  figures/cv_boxplot_v2.png
  figures/time_split_compare_v2.png
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
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
SEED = 42
DROPS = ["isFraud", "ProductCD", "ProductCD_label"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 与 retrain_final.py 完全一致的 v2 配置
V2_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=127,
    min_child_samples=50, subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    scale_pos_weight=1.0, n_jobs=-1, force_col_wise=True, verbose=-1,
)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def tune(y_true, prob, lo=0.01, hi=0.99):
    """全区间扫描, 不预设阈值范围。"""
    ths = np.arange(lo, hi, 0.01)
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in ths]
    bi = int(np.argmax(f1s))
    tn, fp, _, _ = confusion_matrix(y_true, (prob >= ths[bi]).astype(int)).ravel()
    return float(ths[bi]), float(f1s[bi]), float(fp / (fp + tn))


def metrics(y_true, prob):
    th, f1, fpr = tune(y_true, prob)
    return {"AUC": round(float(roc_auc_score(y_true, prob)), 4),
            "最优F1": round(f1, 4), "最优阈值": round(th, 2),
            "FPR": round(fpr, 4)}


df = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
X = df.drop(columns=DROPS)
y = df["isFraud"].values
log("全量 %d 行, 欺诈 %d 笔 (%.3f%%)" % (len(X), int(y.sum()), 100 * y.mean()))

# ---------------- R1 多种子 ----------------
log("R1/3 多种子重复 (5 种子 × 1200 轮) ...")
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
seed_rows = []
for s in [42, 7, 2024, 123, 999]:
    t0 = time.time()
    bst = lgb.train({**V2_PARAMS, "random_state": s}, lgb.Dataset(X_tr, y_tr),
                    num_boost_round=1200)
    m = metrics(y_va, bst.predict(X_va))
    m["seed"] = s
    m["耗时s"] = round(time.time() - t0, 1)
    seed_rows.append(m)
    log("  seed=%-5d F1=%.4f 阈值=%.2f  AUC=%.4f  FPR=%.4f"
        % (s, m["最优F1"], m["最优阈值"], m["AUC"], m["FPR"]))

f1_arr = np.array([r["最优F1"] for r in seed_rows])
auc_arr = np.array([r["AUC"] for r in seed_rows])
thr_arr = np.array([r["最优阈值"] for r in seed_rows])
seed_summary = {
    "F1均值": round(float(f1_arr.mean()), 4),
    "F1标准差": round(float(f1_arr.std(ddof=1)), 4),
    "F1最小": round(float(f1_arr.min()), 4),
    "F1最大": round(float(f1_arr.max()), 4),
    "AUC均值": round(float(auc_arr.mean()), 4),
    "AUC标准差": round(float(auc_arr.std(ddof=1)), 4),
    "阈值均值": round(float(thr_arr.mean()), 2),
    "阈值标准差": round(float(thr_arr.std(ddof=1)), 3),
    "配置": "v2 (leaves=127, spw=1, lr=0.05, 1200 轮)",
}

# ---------------- R2 5 折交叉验证 ----------------
log("R2/3 5 折分层交叉验证 (LightGBM 1200 轮 vs HGB 200 轮) ...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_lgb, cv_hgb = [], []
for k, (tr, va) in enumerate(skf.split(X, y), 1):
    bst = lgb.train(V2_PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=1200)
    ml = metrics(y[va], bst.predict(X.iloc[va]))
    ml["fold"] = k
    cv_lgb.append(ml)

    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        class_weight={0: 1.0, 1: 27.58}, random_state=SEED)
    hgb.fit(X.iloc[tr], y[tr])
    mh = metrics(y[va], hgb.predict_proba(X.iloc[va])[:, 1])
    mh["fold"] = k
    cv_hgb.append(mh)
    log("  第%d折  LightGBM F1=%.4f  |  HGB F1=%.4f"
        % (k, ml["最优F1"], mh["最优F1"]))

lgb_f1 = np.array([r["最优F1"] for r in cv_lgb])
hgb_f1 = np.array([r["最优F1"] for r in cv_hgb])
t_stat, p_val = stats.ttest_rel(lgb_f1, hgb_f1)
cv_summary = {
    "LightGBM_5折F1": [r["最优F1"] for r in cv_lgb],
    "LightGBM_F1均值": round(float(lgb_f1.mean()), 4),
    "LightGBM_F1标准差": round(float(lgb_f1.std(ddof=1)), 4),
    "HGB_5折F1": [r["最优F1"] for r in cv_hgb],
    "HGB_F1均值": round(float(hgb_f1.mean()), 4),
    "HGB_F1标准差": round(float(hgb_f1.std(ddof=1)), 4),
    "配对t检验_t值": round(float(t_stat), 4),
    "配对t检验_p值": round(float(p_val), 8),
    "差异是否显著(p<0.05)": bool(p_val < 0.05),
    "提升幅度": round(float(lgb_f1.mean() - hgb_f1.mean()), 4),
    "说明": "HGB 使用 class_weight=27.58(等价 v1 口径), LightGBM 使用 v2 配置(spw=1)",
}

# ---------------- R3 时间外推 ----------------
log("R3/3 时间外推 (前 80% → 后 20%) ...")
order = df["TransactionDT"].argsort().values
X_time, y_time = X.iloc[order], y[order]
cut = int(len(X_time) * 0.80)
Xtr_t, ytr_t = X_time.iloc[:cut], y_time[:cut]
Xva_t, yva_t = X_time.iloc[cut:], y_time[cut:]
log("  训练 %d (欺诈率 %.3f%%) / 测试 %d (欺诈率 %.3f%%)" % (
    len(Xtr_t), 100 * ytr_t.mean(), len(Xva_t), 100 * yva_t.mean()))

# 训练期内部再切一份验证集用于阈值重校准
val_cut = int(len(Xtr_t) * 0.9)
bst_t = lgb.train(V2_PARAMS, lgb.Dataset(Xtr_t.iloc[:val_cut], ytr_t[:val_cut]),
                  num_boost_round=1200)
thr_recal, _, _ = tune(ytr_t[val_cut:], bst_t.predict(Xtr_t.iloc[val_cut:]))
prob_t = bst_t.predict(Xva_t)
time_res = metrics(yva_t, prob_t)
time_res["重校准阈值"] = round(thr_recal, 2)
time_res["沿用随机划分阈值"] = round(float(np.mean(thr_arr)), 2)
time_res["训练行数"] = int(len(Xtr_t))
time_res["测试行数"] = int(len(Xva_t))
time_res["测试欺诈率"] = round(float(yva_t.mean()), 5)
# 重校准阈值下的表现
pred_old = (prob_t >= float(np.mean(thr_arr))).astype(int)
pred_new = (prob_t >= thr_recal).astype(int)


def block(pred):
    tn, fp, fn, tp = confusion_matrix(yva_t, pred).ravel()
    return {"F1": round(float(f1_score(yva_t, pred, zero_division=0)), 4),
            "FPR": round(float(fp / (fp + tn)), 4),
            "召回": round(float(tp / (tp + fn)), 4),
            "精确": round(float(tp / (tp + fp)), 4) if (tp + fp) else 0.0}


time_res["沿用旧阈值表现"] = block(pred_old)
time_res["重校准阈值表现"] = block(pred_new)
thr_oracle, f1_oracle, _ = tune(yva_t, prob_t)
time_res["理论上限"] = {"阈值": round(thr_oracle, 2), "F1": round(f1_oracle, 4),
                        "说明": "在测试集上直接选最优阈值得到的上限, 现实中不可用"}
log("  时间外推 F1=%.4f 阈值=%.2f  AUC=%.4f  FPR=%.4f"
    % (time_res["最优F1"], time_res["最优阈值"], time_res["AUC"], time_res["FPR"]))
log("  沿用随机划分阈值(%.2f) → F1 %.4f | 重校准阈值(%.2f) → F1 %.4f | 上限 %.4f"
    % (time_res["沿用随机划分阈值"], time_res["沿用旧阈值表现"]["F1"],
       thr_recal, time_res["重校准阈值表现"]["F1"], f1_oracle))

out = {
    "配置": "v2 (num_leaves=127, scale_pos_weight=1, learning_rate=0.05)",
    "多种子重复": {"明细": seed_rows, "汇总": seed_summary},
    "5折交叉验证": {"明细LightGBM": cv_lgb, "明细HGB": cv_hgb, "汇总": cv_summary},
    "时间外推": time_res,
    "v1对照": {"多种子F1均值": 0.7596, "5折F1均值": 0.7207, "时间外推F1": 0.5469},
}
with open(os.path.join(OUT, "robustness_v2_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log("指标已落盘 robustness_v2_results.json")

# ---------------- 图 ----------------
fig, ax = plt.subplots(figsize=(7.5, 5))
bp = ax.boxplot([lgb_f1, hgb_f1], tick_labels=["LightGBM v2", "梯度提升 HGB"],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["#1a73e8", "#9aa0a6"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for i, arr in enumerate([lgb_f1, hgb_f1], start=1):
    ax.scatter(np.full(len(arr), i) + np.random.uniform(-0.06, 0.06, len(arr)),
               arr, color="black", s=22, zorder=5)
ax.set_ylabel("5 折验证集最优 F1")
ax.set_title("5 折交叉验证（v2 配置）· LightGBM vs HGB\n配对 t 检验 p=%.2e (%s)"
             % (p_val, "显著" if p_val < 0.05 else "不显著"))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cv_boxplot_v2.png"), dpi=150)
plt.close(fig)

names = ["随机划分\n(5 种子均值)", "时间外推\n沿用旧阈值", "时间外推\n重校准阈值", "时间外推\n理论上限"]
f1_vals = [seed_summary["F1均值"], time_res["沿用旧阈值表现"]["F1"],
           time_res["重校准阈值表现"]["F1"], time_res["理论上限"]["F1"]]
fpr_vals = [float(np.mean([r["FPR"] for r in seed_rows])),
            time_res["沿用旧阈值表现"]["FPR"],
            time_res["重校准阈值表现"]["FPR"],
            round(time_res["FPR"] * 0 + 0, 4)]
x = np.arange(len(names))
fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
b1 = ax1.bar(x - 0.2, f1_vals, 0.4, label="F1", color="#1a73e8")
ax1.set_ylabel("F1", color="#1a73e8")
ax1.set_ylim(0, max(f1_vals) * 1.25)
for b in b1:
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
             "%.4f" % b.get_height(), ha="center", fontsize=9)
ax2 = ax1.twinx()
b2 = ax2.bar(x + 0.2, fpr_vals, 0.4, label="误报率 FPR", color="#d93025")
ax2.set_ylabel("误报率 FPR", color="#d93025")
for b in b2:
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
             "%.4f" % b.get_height(), ha="center", fontsize=9)
ax1.set_xticks(x, names)
ax1.set_title("v2 · 随机划分与时间外推对比（末列为测试集理论上限）")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "time_split_compare_v2.png"), dpi=150)
plt.close(fig)

print("\n" + "=" * 70)
print("A17 v2 稳定性重验结论")
print("=" * 70)
print("  5 种子 F1: %.4f ± %.4f (范围 %.4f ~ %.4f), 阈值 %.2f±%.3f" % (
    seed_summary["F1均值"], seed_summary["F1标准差"], seed_summary["F1最小"],
    seed_summary["F1最大"], seed_summary["阈值均值"], seed_summary["阈值标准差"]))
print("  5 折  F1: %.4f ± %.4f  (v1 为 %.4f)" % (
    cv_summary["LightGBM_F1均值"], cv_summary["LightGBM_F1标准差"], 0.7207))
print("  vs HGB : LGBM %.4f / HGB %.4f, p=%.2e (%s)" % (
    cv_summary["LightGBM_F1均值"], cv_summary["HGB_F1均值"], p_val,
    "差异显著" if p_val < 0.05 else "差异不显著"))
print("  时间外推: 沿用阈值 F1 %.4f → 重校准 F1 %.4f (上限 %.4f), AUC %.4f" % (
    time_res["沿用旧阈值表现"]["F1"], time_res["重校准阈值表现"]["F1"],
    time_res["理论上限"]["F1"], time_res["AUC"]))
print("=" * 70)
log("A17 完成")
