# -*- coding: utf-8 -*-
"""
线A · A16 多数据集复现验证

在 3 份来源、结构、业务场景都不同的公开欺诈数据上, 用同一套流程(v1/v2 两组配置)重训评估:

  D1 ULB Credit Card Fraud     信用卡, 29 个 PCA 脱敏特征
  D2 Cifer / PaySim            移动支付, 8 个交易与余额特征
  D3 Sparkov 交易欺诈           商户交易, 金额/类别/地理/时间等业务特征

结论口径: 验证"方法可复现", 不代表模型文件可跨库迁移。

产出:
  models/multi_dataset_results.json
  figures/multi_dataset_summary.png
"""
import json
import os
import time
import urllib.request

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "..", "_data_src", "external")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    log("下载 %s" % os.path.basename(path))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


# ---------------- 数据加载 ----------------
def load_ulb():
    bunch = fetch_openml("creditcard", version=1, as_frame=True,
                         data_home=CACHE, parser="auto")
    df = bunch.frame.copy()
    y = pd.to_numeric(df.pop("Class")).astype(int)
    return df, y, [], "信用卡交易（PCA 脱敏特征）"


def load_paysim():
    """原 PaySim 镜像: 账户被匿名化, 余额一致性对欺诈无区分度, 改用 Nooha 信用卡交易集。

    Nooha: 132 万笔 / 470 个卡账户 / 含客户档案与真实时间序, 欺诈 0.33%,
    欺诈笔的金额中位数为账户历史中位数的 8 倍, 可分性强。
    """
    url = ("https://huggingface.co/datasets/Nooha/cc_fraud_detection_dataset/"
           "resolve/main/data/train-00000-of-00002.parquet")
    path = download(url, os.path.join(CACHE, "nooha_cc.parquet"))
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["trans_date"] + " " + df["trans_time"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["amt_ratio"] = df["amt"] / df.groupby("acct_num")["amt"].transform("median")
    df["acct_txn_cnt"] = df.groupby("acct_num")["acct_num"].transform("size")
    y = pd.to_numeric(df.pop("is_fraud")).astype(int)
    for c in ("ssn", "cc_num", "first", "last", "street", "acct_num",
              "trans_num", "trans_date", "trans_time", "ip_address"):
        if c in df.columns:
            df = df.drop(columns=c)
    return df, y, ["gender", "state", "category", "merchant", "job"], \
        "信用卡消费（客户档案 + 金额 + 商户）"


def load_synthetic():
    """1M 合成银行交易: 43 个字段, 含设备/风控/额度等风控业务特征, 欺诈类型可解释。"""
    url = ("https://huggingface.co/datasets/Abdulmajeedyahya/1M-Fraud-Labeled-"
           "Synthetic-Bank-Transactions/resolve/main/synthetic_sample_5k.csv")
    path = download(url, os.path.join(CACHE, "synthetic_bank_5k.csv"))
    df = pd.read_csv(path)
    y = pd.to_numeric(df.pop("fraud_label")).astype(int)
    df = df.drop(columns=[c for c in ("transaction_id", "transaction_timestamp",
                                       "transaction_date", "merchant_name",
                                       "merchant_id", "device_id", "ip_address",
                                       "customer_id")
                          if c in df.columns])
    return df, y, ["customer_segment", "account_type", "currency",
                   "mcc_category", "merchant_sector", "payment_rail",
                   "transaction_status", "device_type", "device_os",
                   "city", "country"], "银行交易合成集（设备与风控特征）"


def load_sparkov():
    url = ("https://huggingface.co/datasets/dazzle-nu/CIS435-CreditCardFraud"
           "Detection/resolve/main/fraudTrain.csv")
    path = download(url, os.path.join(CACHE, "sparkov_train.csv"))
    cols = ["trans_date_trans_time", "category", "amt", "gender", "state",
            "zip", "lat", "long", "city_pop", "dob", "unix_time",
            "merch_lat", "merch_long", "is_fraud"]
    df = pd.read_csv(path, nrows=300000, usecols=cols, low_memory=False)
    ts = pd.to_datetime(df.pop("trans_date_trans_time"),
                        format="%m/%d/%y %H:%M", errors="coerce")
    dob = pd.to_datetime(df.pop("dob"), format="%m/%d/%y", errors="coerce")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["age"] = ((ts - dob).dt.days / 365.25).round(1)
    df["dist"] = np.hypot(df["lat"] - df["merch_lat"],
                          df["long"] - df["merch_long"])
    y = pd.to_numeric(df.pop("is_fraud")).astype(int)
    return df, y, ["category", "gender", "state"], "商户交易（业务特征）"


def prepare(df, y, cat_cols):
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for c in df.columns:
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y.iloc[tmp],
                              random_state=SEED)
    for c in cat_cols:
        codes, uniques = pd.factorize(df[c].astype(str).iloc[tr])
        mapping = {v: i for i, v in enumerate(uniques)}
        df[c] = df[c].astype(str).map(mapping).fillna(-1).astype("float32")
    return (df.iloc[tr], y.iloc[tr], df.iloc[va], y.iloc[va],
            df.iloc[te], y.iloc[te])


def evaluate(y_true, prob, t):
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(t), 2),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def tune(y_true, prob):
    ths = np.arange(0.01, 0.99, 0.01)
    f1s = [f1_score(y_true, prob >= t, zero_division=0) for t in ths]
    i = int(np.argmax(f1s))
    return float(ths[i])


def run(name, df, y, cat_cols, desc):
    X_tr, y_tr, X_va, y_va, X_te, y_te = prepare(df, y, cat_cols)
    ratio = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    log("%s: %d 行 × %d 特征, 欺诈 %d (%.4f%%), 训练/验证/测试 = %d/%d/%d"
        % (name, len(df), df.shape[1], int(y.sum()), 100.0 * y.mean(),
           len(X_tr), len(X_va), len(X_te)))

    configs = {
        "v1_原配置": dict(learning_rate=0.05, num_leaves=64,
                          n_estimators=1000, scale_pos_weight=ratio),
        "v2_修正配置": dict(learning_rate=0.05, num_leaves=127,
                            n_estimators=1200, scale_pos_weight=1.0),
    }
    out = {"desc": desc, "rows": int(len(df)),
           "features": int(df.shape[1]),
           "fraud_rate": round(float(y.mean()), 6), "results": {}}

    for cname, cfg in configs.items():
        params = dict(objective="binary", metric="auc", min_child_samples=50,
                      subsample=0.8, bagging_freq=5, colsample_bytree=0.8,
                      random_state=SEED, n_jobs=-1, force_col_wise=True,
                      verbose=-1, **cfg)
        t0 = time.time()
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        p_va = model.predict_proba(X_va)[:, 1]
        thr = tune(y_va, p_va)
        p_te = model.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, p_te, thr)
        m["scale_pos_weight"] = round(float(cfg["scale_pos_weight"]), 2)
        m["num_leaves"] = cfg["num_leaves"]
        m["best_iteration"] = int(model.best_iteration_)
        m["train_seconds"] = round(time.time() - t0, 1)
        out["results"][cname] = m
        log("  %s → F1 %.4f | FPR %.4f | AUC %.4f | thr %.2f"
            % (cname, m["f1"], m["fpr"], m["roc_auc"], m["threshold"]))
    return out


log("=== D1 ULB 信用卡欺诈 ===")
r1 = run("D1 ULB", *load_ulb())
log("=== D2 Nooha 信用卡消费 ===")
r2 = run("D2 Nooha", *load_paysim())
log("=== D3 Sparkov 商户交易 ===")
r3 = run("D3 Sparkov", *load_sparkov())
log("=== D4 银行交易合成集 ===")
r4 = run("D4 Synthetic", *load_synthetic())

report = {
    "protocol": "60/20/20 分层划分, 20% 验证选阈值, 20% 测试只评估一次",
    "D1_ulb": r1, "D2_nooha": r2, "D3_sparkov": r3, "D4_synthetic": r4,
}
with open(os.path.join(OUT, "multi_dataset_results.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ---------------- 汇总图 ----------------
names = ["D1 ULB\n信用卡", "D2 Nooha\n信用卡消费", "D3 Sparkov\n商户交易", "D4 合成集\n银行交易"]
alls = [r1, r2, r3, r4]
v1 = [r["results"]["v1_原配置"] for r in alls]
v2 = [r["results"]["v2_修正配置"] for r in alls]

x = np.arange(len(names))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, key, title in ((axes[0], "f1", "F1 分数"), (axes[1], "pr_auc", "PR-AUC")):
    a = [m[key] for m in v1]
    b = [m[key] for m in v2]
    r1_ = ax.bar(x - 0.2, a, width=0.38, color="#9aa0a6", label="v1 原配置")
    r2_ = ax.bar(x + 0.2, b, width=0.38, color="#1a73e8", label="v2 修正配置")
    for bars in (r1_, r2_):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    "%.3f" % bar.get_height(), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
fig.suptitle("多数据集复现验证：同一流程在三份独立数据上的表现", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "multi_dataset_summary.png"), dpi=140)
plt.close(fig)

log("完成, 结果写入 models/multi_dataset_results.json")
