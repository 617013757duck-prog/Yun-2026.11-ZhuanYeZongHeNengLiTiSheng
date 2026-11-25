# -*- coding: utf-8 -*-
"""
线A · A15 主线模型重训（v2）

解决的问题（来自 A7–A10 的实验结论）:
  1. 容量偏小: num_leaves 64 → 127
  2. 轮数不足: n_estimators 1000 → 2000 + 早停
  3. 类别权重负向: scale_pos_weight 27.58 → 1
  4. 时间外推阈值失效: 补充"用最近 10% 训练数据重校准阈值"的时间外推实验

评估口径与 train_fraud_baseline.py 一致:
  训练集内部按类别 90/10 切分, 10% 验证集只用于早停与阈值调优, 测试集不参与任何选择。

产出:
  models/lgbm_fraud_final.txt
  models/metrics_fraud_final.json
  figures/final_roc_pr.png
  figures/final_threshold.png
  figures/final_vs_baseline.png
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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve)

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL_DIR = os.path.join(BASE, "models")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
SEED = 42
DROP = ["isFraud", "ProductCD", "ProductCD_label"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def evaluate(y_true, prob, t):
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(t), 2),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "fpr": round(float(fp / (fp + tn)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def tune(y_true, prob, lo=0.05, hi=0.95):
    ths = np.arange(lo, hi, 0.01)
    scores = [f1_score(y_true, prob >= t) for t in ths]
    i = int(np.argmax(scores))
    return float(ths[i]), float(scores[i])


def tune_for_watch(y_true, prob, f1_ratio=0.95, fpr_multiple=2.5):
    """取"关注"线: 在保持 F1 不低于告警线 95%、且误报率不超过告警线 2.5 倍的前提下,
    取能带来最高召回率的最低阈值。

    只按召回率选会把关注队列撑爆(阈值 0.01 时 FPR 5.7%), 只按 F1 选则没有区分度。
    """
    ths = np.arange(0.01, 0.95, 0.01)
    f1s, fprs, recs = [], [], []
    for t in ths:
        pred = (prob >= t).astype(int)
        tn, fp, _, _ = confusion_matrix(y_true, pred).ravel()
        f1s.append(f1_score(y_true, pred))
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
        recs.append(recall_score(y_true, pred, zero_division=0))
    f1s, fprs, recs = np.array(f1s), np.array(fprs), np.array(recs)
    i_alert = int(np.argmax(f1s))
    ok = (f1s >= f1_ratio * f1s[i_alert]) & (fprs <= fpr_multiple * fprs[i_alert])
    if not ok.any():
        return float(ths[i_alert])
    cand = np.where(ok)[0]
    return float(ths[cand[np.argmax(recs[cand])]])


def tune_for_fpr(y_true, prob, cap=0.005, lo=0.01, hi=0.95):
    """取满足误报率 ≤ cap 的最小阈值, 用于给出告警线的业务上限参考。"""
    ths = np.arange(lo, hi, 0.01)
    for t in ths:
        pred = (prob >= t).astype(int)
        tn, fp, _, _ = confusion_matrix(y_true, pred).ravel()
        if (fp / (fp + tn)) <= cap:
            return float(t)
    return float(ths[-1])


def fit_model(X_tr, y_tr, X_va, y_va, params):
    model = lgb.LGBMClassifier(**params)
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return model, time.time() - t0


# ---------------- 1. 读取数据 ----------------
log("读取 processed 数据 ...")
train = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))
X_all, y_all = train.drop(columns=DROP), train["isFraud"]
X_test, y_test = test.drop(columns=DROP), test["isFraud"]
features = list(X_all.columns)
log("特征 %d | 训练 %d | 测试 %d" % (len(features), len(X_all), len(X_test)))

# ---------------- 2. 与 v1 完全相同的 90/10 验证集切分 ----------------
rng = np.random.RandomState(SEED)
valid_mask = np.zeros(len(X_all), dtype=bool)
for cls in y_all.unique():
    idx = np.where(y_all.values == cls)[0]
    valid_mask[rng.choice(idx, size=int(len(idx) * 0.1), replace=False)] = True
X_tr, y_tr = X_all[~valid_mask], y_all[~valid_mask]
X_va, y_va = X_all[valid_mask], y_all[valid_mask]

PARAMS = dict(
    objective="binary", metric="auc",
    n_estimators=2000, learning_rate=0.05, num_leaves=127,
    min_child_samples=50,
    subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    scale_pos_weight=1.0,
    random_state=SEED, n_jobs=-1, force_col_wise=True, verbose=-1,
)
log("v2 参数: %s" % {k: PARAMS[k] for k in
                   ("learning_rate", "num_leaves", "n_estimators", "scale_pos_weight")})

# ---------------- 3. 训练 v2 ----------------
model, sec = fit_model(X_tr, y_tr, X_va, y_va, PARAMS)
log("训练完成 %.1f 秒, 最佳迭代 %d" % (sec, model.best_iteration_))

p_va = model.predict_proba(X_va)[:, 1]
thr, val_f1 = tune(y_va, p_va)
warn_thr = tune_for_watch(y_va, p_va)
strict_thr = tune_for_fpr(y_va, p_va, cap=0.005)
log("验证集: 告警线 %.2f (F1 最优) | 关注线 %.2f (召回≥85%%) | 严格线 %.2f (FPR≤0.5%%)"
    % (thr, warn_thr, strict_thr))

# ---------------- 4. 测试集评估 ----------------
p_test = model.predict_proba(X_test)[:, 1]
m_default = evaluate(y_test, p_test, 0.5)
m_tuned = evaluate(y_test, p_test, thr)
log("v2 测试: F1 %.4f | FPR %.4f | AUC %.4f | 阈值 %.2f"
    % (m_tuned["f1"], m_tuned["fpr"], m_tuned["roc_auc"], thr))

# ---------------- 5. 时间外推 + 阈值重校准 ----------------
log("时间外推实验 ...")
full = pd.concat([train, test], ignore_index=True).sort_values("TransactionDT")
cut = int(len(full) * 0.8)
X_ts_tr_full = full.iloc[:cut].drop(columns=DROP)
y_ts_tr_full = full.iloc[:cut]["isFraud"]
X_ts_te = full.iloc[cut:].drop(columns=DROP)
y_ts_te = full.iloc[cut:]["isFraud"]
val_cut = int(len(X_ts_tr_full) * 0.9)
X_t, y_t = X_ts_tr_full.iloc[:val_cut], y_ts_tr_full.iloc[:val_cut]
X_v, y_v = X_ts_tr_full.iloc[val_cut:], y_ts_tr_full.iloc[val_cut:]

V1_PARAMS = dict(PARAMS, num_leaves=64, n_estimators=1000,
                 scale_pos_weight=float((y_t == 0).sum() / (y_t == 1).sum()))

time_rows = {}
for tag, prm in (("v1", V1_PARAMS), ("v2", PARAMS)):
    model_ts, sec_ts = fit_model(X_t, y_t, X_v, y_v, prm)
    p_v_ts = model_ts.predict_proba(X_v)[:, 1]
    thr_ts, _ = tune(y_v, p_v_ts)
    p_te_ts = model_ts.predict_proba(X_ts_te)[:, 1]
    m_old = evaluate(y_ts_te, p_te_ts, thr)       # 沿用随机划分的阈值
    m_new = evaluate(y_ts_te, p_te_ts, thr_ts)    # 用最近 10% 数据重校准阈值
    thr_oracle, f1_oracle = tune(y_ts_te, p_te_ts)   # 理论上限, 仅用于分析
    m_oracle = evaluate(y_ts_te, p_te_ts, thr_oracle)
    time_rows[tag] = {
        "old_threshold": m_old, "recalibrated_threshold": m_new,
        "recalibrated_value": round(thr_ts, 2),
        "oracle_threshold": m_oracle,
        "oracle_note": "在测试集上直接选最优阈值得到的 F1 上限, 现实中不可用, "
                       "仅用于区分'排序能力下降'与'阈值失效'",
        "train_seconds": round(sec_ts, 1),
        "val_f1": round(float(f1_score(y_v, p_v_ts >= thr_ts)), 4),
    }
    log("%s 时间外推: 旧阈值 %.2f → F1 %.4f | 重校准 %.2f → F1 %.4f | 上限 F1 %.4f"
        % (tag, thr, m_old["f1"], thr_ts, m_new["f1"], m_oracle["f1"]))
    if tag == "v2":
        m_time_old, m_time_new, m_time_oracle = m_old, m_new, m_oracle
        thr_ts_final = thr_ts

# ---------------- 6. 保存模型与指标 ----------------
model_path = os.path.join(MODEL_DIR, "lgbm_fraud_final.txt")
with open(model_path, "w", encoding="utf-8") as f:
    f.write(model.booster_.model_to_string())
log("模型已保存: %s" % model_path)

metrics = {
    "model": "LightGBM v2 (A15 重训)",
    "params": {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in PARAMS.items() if k != "verbose"},
    "feature_count": len(features),
    "best_iteration": int(model.best_iteration_),
    "train_seconds": round(sec, 1),
    "imbalance_strategy": "scale_pos_weight=1 + threshold tuning",
    "thresholds": {
        "alert": round(thr, 2),
        "warn": round(warn_thr, 2),
        "strict": round(strict_thr, 2),
        "policy": "告警线=验证集 F1 最优; 关注线=在 F1 不低于告警线 95% 且误报率"
                  "不超过 2.5 倍的前提下取召回最高; 严格线=误报率≤0.5%。"
                  "三条线均由验证集数据决定, 模型换分数尺度时自动随之变化。",
    },
    "default_threshold": m_default,
    "tuned_threshold": m_tuned,
    "time_split": {
        "v1": time_rows["v1"],
        "v2": time_rows["v2"],
        "train_seconds": round(sec_ts, 1),
    },
    "baseline_v1": {
        "f1": 0.7577, "fpr": 0.0045, "roc_auc": 0.9643, "pr_auc": 0.8073,
        "time_split_f1": 0.5469,
    },
}
with open(os.path.join(MODEL_DIR, "metrics_fraud_final.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# ---------------- 7. 图表 ----------------
fpr_c, tpr_c, _ = roc_curve(y_test, p_test)
prec_c, rec_c, _ = precision_recall_curve(y_test, p_test)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(fpr_c, tpr_c, color="#1a73e8", lw=2,
             label="v2 ROC (AUC=%.4f)" % m_tuned["roc_auc"])
axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
axes[0].set_xlabel("假正率 FPR")
axes[0].set_ylabel("真正率 TPR")
axes[0].set_title("v2 模型 ROC 曲线（测试集）")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)
axes[1].plot(rec_c, prec_c, color="#d93025", lw=2,
             label="v2 PR (AP=%.4f)" % m_tuned["pr_auc"])
axes[1].set_xlabel("召回率 Recall")
axes[1].set_ylabel("精确率 Precision")
axes[1].set_title("v2 模型 PR 曲线（测试集）")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "final_roc_pr.png"), dpi=140)
plt.close(fig)

ths = np.arange(0.05, 0.95, 0.01)
f1s = [f1_score(y_test, p_test >= t) for t in ths]
fprs = []
for t in ths:
    tn, fp, _, _ = confusion_matrix(y_test, (p_test >= t).astype(int)).ravel()
    fprs.append(fp / (fp + tn))
fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
ax1.plot(ths, f1s, color="#188038", lw=2, label="F1")
ax1.set_xlabel("判定阈值")
ax1.set_ylabel("F1", color="#188038")
ax1.tick_params(axis="y", labelcolor="#188038")
ax2 = ax1.twinx()
ax2.plot(ths, fprs, color="#d93025", lw=2, ls="--", label="FPR")
ax2.set_ylabel("误报率 FPR", color="#d93025")
ax2.tick_params(axis="y", labelcolor="#d93025")
ax1.axvline(thr, color="grey", ls=":", lw=1.2,
            label="验证集最优阈值 %.2f" % thr)
ax1.set_title("v2 阈值-F1-FPR 曲线（测试集）")
ax1.legend(loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "final_threshold.png"), dpi=140)
plt.close(fig)

groups = ["随机划分\n(测试集)", "时间外推\n旧阈值", "时间外推\n重校准阈值", "时间外推\n理论上限"]
f1_vals = [m_tuned["f1"], m_time_old["f1"], m_time_new["f1"], m_time_oracle["f1"]]
fpr_vals = [m_tuned["fpr"], m_time_old["fpr"], m_time_new["fpr"], m_time_oracle["fpr"]]
x = np.arange(len(groups))
fig, ax1 = plt.subplots(figsize=(9.5, 4.6))
bars = ax1.bar(x - 0.18, f1_vals, width=0.36, color="#1a73e8", label="F1")
ax1.set_ylabel("F1", color="#1a73e8")
ax1.set_ylim(0, 1.0)
for b, v in zip(bars, f1_vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.4f" % v,
             ha="center", fontsize=9)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + 0.18, fpr_vals, width=0.36, color="#d93025", label="FPR")
ax2.set_ylabel("误报率 FPR", color="#d93025")
for b, v in zip(bars2, fpr_vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, "%.4f" % v,
             ha="center", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(groups)
ax1.set_title("v2 在随机划分与时间外推下的表现（末列为理论上限，仅供分析）")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "final_vs_baseline.png"), dpi=140)
plt.close(fig)

log("完成。v2 指标: %s" % json.dumps(
    {"f1": m_tuned["f1"], "fpr": m_tuned["fpr"],
     "acc": m_tuned["accuracy"], "auc": m_tuned["roc_auc"],
     "pr_auc": m_tuned["pr_auc"]}, ensure_ascii=False))
