# -*- coding: utf-8 -*-
"""
线A · A2 类别不平衡处理 + A3 异常检测基线模型
策略:
  A2 - 不平衡处理: scale_pos_weight(类别权重) + 验证集阈值调优(以F1最优)
  A3 - 基线模型:   LightGBM 二分类 (isFraud)
评估: Accuracy / Precision / Recall / F1 / FPR(误报率) / ROC-AUC / PR-AUC
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---------------- 1. 读取 processed 数据 ----------------
log("读取 train.csv.gz ...")
train = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
log("读取 test.csv.gz ...")
test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))

DROP = ["isFraud", "ProductCD", "ProductCD_label"]
X_train_full = train.drop(columns=DROP)
y_train_full = train["isFraud"]
X_test = test.drop(columns=DROP)
y_test = test["isFraud"]
log("特征列: %d, 训练集: %d, 测试集: %d" % (X_train_full.shape[1], len(train), len(test)))

# ---------------- 2. 从训练集切验证集(阈值调优用, 避免测试集泄漏) ----------------
rng = np.random.RandomState(SEED)
valid_mask = np.zeros(len(X_train_full), dtype=bool)
for cls in y_train_full.unique():
    idx = np.where(y_train_full.values == cls)[0]
    valid_mask[rng.choice(idx, size=int(len(idx) * 0.1), replace=False)] = True
X_tr, y_tr = X_train_full[~valid_mask], y_train_full[~valid_mask]
X_va, y_va = X_train_full[valid_mask], y_train_full[valid_mask]
log("子训练集: %d | 验证集: %d (验证集欺诈率 %.3f%%)" % (
    len(X_tr), len(X_va), 100 * y_va.mean()))

# ---------------- 3. A2 不平衡处理: scale_pos_weight ----------------
neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
spw = neg / pos
log("A2 不平衡处理: scale_pos_weight = %.2f (负例 %d / 正例 %d)" % (spw, neg, pos))

# ---------------- 4. A3 LightGBM 基线 ----------------
params = dict(
    objective="binary",
    metric="auc",            # 早停跟踪 auc; spw 会使 logloss 失真
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    bagging_freq=5,          # subsample 必须搭配 bagging_freq 才生效
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    random_state=SEED,
    n_jobs=-1,
    force_col_wise=True,
)
log("训练 LightGBM 基线 (最多 1000 轮, early stopping) ...")
model = lgb.LGBMClassifier(**params)
model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
log("最佳迭代轮数: %d" % model.best_iteration_)

# ---------------- 5. A2 阈值调优(在验证集上选 F1 最优阈值) ----------------
p_va = model.predict_proba(X_va)[:, 1]
ths = np.arange(0.05, 0.95, 0.01)
scores = [f1_score(y_va, p_va >= t) for t in ths]
best_t = float(ths[int(np.argmax(scores))])
log("验证集最优阈值(F1): %.2f" % best_t)


# ---------------- 6. 测试集评估(0.5 默认 vs 调优阈值) ----------------
def evaluate(y_true, prob, t, name):
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    m = {
        "threshold": round(t, 2),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "fpr": round(fp / (fp + tn), 4),  # 误报率
        "roc_auc": round(roc_auc_score(y_true, prob), 4),
        "pr_auc": round(average_precision_score(y_true, prob), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }
    print("\n[%s] 阈值=%.2f" % (name, t))
    print("  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f" % (
        m["accuracy"], m["precision"], m["recall"], m["f1"]))
    print("  FPR(误报率)=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f" % (
        m["fpr"], m["roc_auc"], m["pr_auc"]))
    print("  混淆矩阵: TP=%d FP=%d TN=%d FN=%d" % (tp, fp, tn, fn))
    return m


p_test = model.predict_proba(X_test)[:, 1]
m_default = evaluate(y_test, p_test, 0.5, "默认阈值")
m_tuned = evaluate(y_test, p_test, best_t, "调优阈值")

# ---------------- 7. 保存模型与指标 ----------------
# LightGBM C API 不支持中文路径, 用 model_to_string + Python 写文件绕过
model_path = os.path.join(MODEL_DIR, "lgbm_fraud_baseline.txt")
with open(model_path, "w", encoding="utf-8") as f:
    f.write(model.booster_.model_to_string())
log("模型已保存: %s" % model_path)
with open(os.path.join(MODEL_DIR, "metrics_fraud_baseline.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model": "LightGBM", "params": {k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in params.items()},
        "best_iteration": int(model.best_iteration_),
        "imbalance_strategy": "scale_pos_weight=%.2f + threshold tuning" % spw,
        "default_threshold": m_default,
        "tuned_threshold": m_tuned,
    }, f, ensure_ascii=False, indent=2)

# 特征重要性 top15
imp = pd.Series(model.feature_importances_, index=X_train_full.columns)
imp = imp.sort_values(ascending=False).head(15)
print("\n特征重要性 Top15:")
for k, v in imp.items():
    print("  %-18s %d" % (k, v))

log("A2+A3 完成, 模型与指标已保存到 %s" % os.path.abspath(MODEL_DIR))
