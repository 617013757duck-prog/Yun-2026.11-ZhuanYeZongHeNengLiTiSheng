# -*- coding: utf-8 -*-
"""
线A · A20 数据分类模型重训（v2）

修复原 train_product_classifier.py 的两个方法论问题:
  1) **数据泄漏**: 原脚本用测试集作为 early stopping 的验证集
     (model.fit(X_tr, y_tr, eval_X=X_te, ...)), 测试集参与了模型选择。
     本脚本从训练集内部切 10% 验证集, 测试集只在最后评估一次。
  2) **超参数未调**: 原配置为 600 轮 / 64 叶, 未做任何实验。
     本脚本做 learning_rate × num_leaves × 轮数 的小规模实验后确定最终配置。

同时检验 A9 的结论在多分类任务上是否成立:
  类别权重与阈值调优的取舍在多分类任务里没有阈值可调, 只能保留类别权重,
  这里对比"有/无类别权重"两种配置。

产出:
  models/lgbm_product_classifier_final.txt
  models/metrics_product_classifier_final.json
  figures/product_confusion_matrix_final.png
  figures/product_hyperparam.png
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL_DIR = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
SEED = 42
CLASSES = ["C", "H", "R", "S", "W"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def evaluate(y_true, pred):
    return {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
            "f1_macro": round(float(f1_score(y_true, pred, average="macro")), 4),
            "f1_weighted": round(float(f1_score(y_true, pred, average="weighted")), 4),
            "f1_per_class": {CLASSES[i]: round(float(f1_score(
                y_true == i, pred == i, zero_division=0)), 4)
                for i in range(len(CLASSES))}}


train = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))
DROP = ["isFraud", "ProductCD", "ProductCD_label"]
X_all, y_all = train.drop(columns=DROP), train["ProductCD_label"]
X_te, y_te = test.drop(columns=DROP), test["ProductCD_label"]
log("特征 %d | 训练 %d | 测试 %d" % (X_all.shape[1], len(X_all), len(X_te)))

# 内部验证集(分层), 测试集不参与任何选择
rng = np.random.RandomState(SEED)
vmask = np.zeros(len(X_all), dtype=bool)
y_arr = y_all.values
for c in np.unique(y_arr):
    idx = np.where(y_arr == c)[0]
    vmask[rng.choice(idx, size=int(len(idx) * 0.1), replace=False)] = True
X_tr, y_tr = X_all[~vmask], y_all[~vmask]
X_va, y_va = X_all[vmask], y_all[vmask]
log("子训练 %d | 验证 %d | 测试 %d" % (len(X_tr), len(X_va), len(X_te)))

cnt = y_tr.value_counts().sort_index()
cw = {int(k): float(len(y_tr) / (len(cnt) * v)) for k, v in cnt.items()}
log("类别权重: %s" % {CLASSES[k]: round(v, 2) for k, v in cw.items()})

BASE_PARAMS = dict(
    objective="multiclass", num_class=5, metric="multi_logloss",
    subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, force_col_wise=True, verbose=-1,
)

# ---------------- 1) 超参数小实验 ----------------
log("1/3 超参数实验 (lr × leaves × 轮数, 验证集评估) ...")
hp_rows = []
for lr in [0.05, 0.1]:
    for leaves in [64, 127]:
        for rounds in [600, 1200]:
            p = {**BASE_PARAMS, "learning_rate": lr, "num_leaves": leaves,
                 "n_estimators": rounds, "class_weight": cw}
            t0 = time.time()
            m = lgb.LGBMClassifier(**p)
            m.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            pred = m.predict(X_va)
            row = {"learning_rate": lr, "num_leaves": leaves, "轮数": rounds,
                   "实际轮数": int(m.best_iteration_),
                   "验证Accuracy": round(float(accuracy_score(y_va, pred)), 4),
                   "验证MacroF1": round(float(f1_score(y_va, pred, average="macro")), 4),
                   "耗时s": round(time.time() - t0, 1)}
            hp_rows.append(row)
            log("  lr=%.2f leaves=%-4d 轮数=%-5d → Acc %.4f  MacroF1 %.4f (实际 %d 轮, %.0fs)"
                % (lr, leaves, rounds, row["验证Accuracy"], row["验证MacroF1"],
                   row["实际轮数"], row["耗时s"]))

hp_rows.sort(key=lambda r: -r["验证MacroF1"])
best_hp = hp_rows[0]
log("最优超参: %s" % {k: best_hp[k] for k in ("learning_rate", "num_leaves", "轮数")})

# ---------------- 2) 类别权重消融 ----------------
log("2/3 类别权重消融 (多分类任务没有阈值可调, 只能保留或去掉权重) ...")
abl = []
for tag, use_cw in (("有类别权重", True), ("无类别权重", False)):
    p = {**BASE_PARAMS, "learning_rate": best_hp["learning_rate"],
         "num_leaves": best_hp["num_leaves"],
         "n_estimators": best_hp["轮数"]}
    if use_cw:
        p["class_weight"] = cw
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = m.predict(X_va)
    row = {"配置": tag, **evaluate(y_va, pred),
           "实际轮数": int(m.best_iteration_)}
    abl.append(row)
    log("  %s → 验证 Acc %.4f  MacroF1 %.4f" % (tag, row["accuracy"], row["f1_macro"]))
abl.sort(key=lambda r: -r["f1_macro"])
USE_CW = abl[0]["配置"] == "有类别权重"
log("选用: %s" % abl[0]["配置"])

# ---------------- 3) 最终训练与测试集评估 ----------------
log("3/3 用训练集全量重训, 测试集只评估一次 ...")
final_params = {**BASE_PARAMS, "learning_rate": best_hp["learning_rate"],
                "num_leaves": best_hp["num_leaves"],
                "n_estimators": best_hp["轮数"]}
if USE_CW:
    final_params["class_weight"] = cw
model = lgb.LGBMClassifier(**final_params)
model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
log("最佳迭代 %d" % model.best_iteration_)

pred = model.predict(X_te)
m_test = evaluate(y_te, pred)
cm = confusion_matrix(y_te, pred)
print("\n[测试集] Accuracy=%.4f  Macro-F1=%.4f  Weighted-F1=%.4f"
      % (m_test["accuracy"], m_test["f1_macro"], m_test["f1_weighted"]))
print(classification_report(y_te, pred, target_names=CLASSES, digits=4))

# 与 v1 对比
v1 = {}
vp = os.path.join(MODEL_DIR, "metrics_product_classifier.json")
if os.path.exists(vp):
    with open(vp, encoding="utf-8") as f:
        v1 = json.load(f)
log("v1 对照: Acc %.4f MacroF1 %.4f | v2: Acc %.4f MacroF1 %.4f"
    % (v1.get("accuracy", 0), v1.get("f1_macro", 0),
       m_test["accuracy"], m_test["f1_macro"]))

# 保存
mpath = os.path.join(MODEL_DIR, "lgbm_product_classifier_final.txt")
with open(mpath, "w", encoding="utf-8") as f:
    f.write(model.booster_.model_to_string())
with open(os.path.join(MODEL_DIR, "metrics_product_classifier_final.json"), "w",
          encoding="utf-8") as f:
    json.dump({
        "model": "LightGBM multiclass v2",
        "params": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in final_params.items() if k != "verbose"},
        "feature_count": int(X_all.shape[1]),
        "best_iteration": int(model.best_iteration_),
        "测试集": m_test,
        "混淆矩阵": cm.tolist(),
        "classes": CLASSES,
        "超参数实验": hp_rows,
        "类别权重消融": abl,
        "v1对照": {"accuracy": v1.get("accuracy"), "f1_macro": v1.get("f1_macro"),
                   "best_iteration": v1.get("best_iteration")},
        "泄漏修复说明": "v1 用测试集做 early stopping 验证集, 测试集参与了模型选择; "
                        "v2 改为训练集内部切 10% 验证集, 测试集仅最后评估一次",
    }, f, ensure_ascii=False, indent=2)
log("模型与指标已保存")

# ---------------- 图1: 混淆矩阵 ----------------
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, "%.3f\n(n=%d)" % (cmn[i, j], cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xticks(range(len(CLASSES)), CLASSES)
ax.set_yticks(range(len(CLASSES)), CLASSES)
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
ax.set_title("ProductCD 五分类归一化混淆矩阵（v2，测试集）")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "product_confusion_matrix_final.png"), dpi=140)
plt.close(fig)

# ---------------- 图2: 超参数实验 ----------------
labels = ["lr%.2f\nL%d\n%d轮" % (r["learning_rate"], r["num_leaves"], r["轮数"])
          for r in hp_rows]
fig, ax1 = plt.subplots(figsize=(11, 4.4))
xs = np.arange(len(hp_rows))
b1 = ax1.bar(xs, [r["验证MacroF1"] for r in hp_rows], 0.55,
             color=["#188038" if i == 0 else "#1a73e8" for i in range(len(hp_rows))])
ax1.set_ylim(0.98, 1.0)
ax1.set_ylabel("验证集 Macro-F1", color="#1a73e8")
ax1.set_xticks(xs)
ax1.set_xticklabels(labels, fontsize=7)
for b, r in zip(b1, hp_rows):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0003,
             "%.4f" % r["验证MacroF1"], ha="center", fontsize=8)
ax2 = ax1.twinx()
ax2.plot(xs, [r["实际轮数"] for r in hp_rows], "s--", color="#d93025", ms=5)
ax2.set_ylabel("早停实际轮数", color="#d93025")
ax1.set_title("分类模型超参数实验（绿色为最优）")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "product_hyperparam.png"), dpi=140)
plt.close(fig)

log("A20 完成")
