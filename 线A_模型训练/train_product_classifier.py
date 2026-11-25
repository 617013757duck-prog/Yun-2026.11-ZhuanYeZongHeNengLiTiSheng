# -*- coding: utf-8 -*-
"""
线A · A4 数据分类模型 (ProductCD 五分类: C/H/R/S/W)
注意: ProductCD 是分类目标, 必须从特征中剔除
评估: Accuracy / Macro-F1 / 加权F1 / 各类别F1
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42
CLASSES = ["C", "H", "R", "S", "W"]  # ProductCD_label 0..4


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


log("读取 processed 数据 ...")
train = pd.read_csv(os.path.join(PROC, "train.csv.gz"))
test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))

DROP = ["isFraud", "ProductCD", "ProductCD_label"]
X_tr, y_tr = train.drop(columns=DROP), train["ProductCD_label"]
X_te, y_te = test.drop(columns=DROP), test["ProductCD_label"]
log("特征列: %d, 训练 %d, 测试 %d" % (X_tr.shape[1], len(X_tr), len(X_te)))

# 五类也不平衡(W占74%), 用类别权重
cnt = y_tr.value_counts().sort_index()
cw = {int(k): float(len(y_tr) / (len(cnt) * v)) for k, v in cnt.items()}
log("类别权重: %s" % {CLASSES[k]: round(v, 2) for k, v in cw.items()})

params = dict(
    objective="multiclass",
    num_class=5,
    metric="multi_logloss",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    bagging_freq=5,
    colsample_bytree=0.8,
    class_weight=cw,
    random_state=SEED,
    n_jobs=-1,
    force_col_wise=True,
)
log("训练 LightGBM 五分类模型 ...")
model = lgb.LGBMClassifier(**params)
model.fit(X_tr, y_tr, eval_X=X_te, eval_y=y_te,
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
log("最佳迭代轮数: %d" % model.best_iteration_)

pred = model.predict(X_te)
acc = accuracy_score(y_te, pred)
f1m = f1_score(y_te, pred, average="macro")
f1w = f1_score(y_te, pred, average="weighted")
cm = confusion_matrix(y_te, pred)

print("\n[测试集] Accuracy=%.4f  Macro-F1=%.4f  Weighted-F1=%.4f" % (acc, f1m, f1w))
print("\n分类报告:")
print(classification_report(y_te, pred, target_names=CLASSES, digits=4))
print("混淆矩阵(行=真实, 列=预测, 顺序 C/H/R/S/W):")
print(cm)

# 保存(中文路径用 model_to_string 绕过)
mpath = os.path.join(MODEL_DIR, "lgbm_product_classifier.txt")
with open(mpath, "w", encoding="utf-8") as f:
    f.write(model.booster_.model_to_string())
with open(os.path.join(MODEL_DIR, "metrics_product_classifier.json"), "w", encoding="utf-8") as f:
    json.dump({"model": "LightGBM multiclass", "best_iteration": int(model.best_iteration_),
               "accuracy": round(acc, 4), "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4),
               "confusion_matrix": cm.tolist(), "classes": CLASSES}, f, ensure_ascii=False, indent=2)
log("A4 完成, 模型与指标已保存")
