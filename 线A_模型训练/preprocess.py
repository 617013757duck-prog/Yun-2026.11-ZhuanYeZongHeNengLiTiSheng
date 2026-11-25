# -*- coding: utf-8 -*-
"""
线A · A1 数据预处理流水线
数据集: IEEE-CIS Fraud Detection (train_transaction + train_identity)
输出: 分层划分后的 train/test (csv.gz) + 类别编码映射 + 处理报告
"""
import json
import os
import time

import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "数据集")
OUT = os.path.join(DATA, "processed")
os.makedirs(OUT, exist_ok=True)

TRANS_CSV = os.path.join(DATA, "train_transaction.csv")
IDENT_CSV = os.path.join(DATA, "train_identity.csv")

TEST_SIZE = 0.2
SEED = 42
MISSING_DROP_THRESHOLD = 0.99  # 缺失率>99% 的列剔除


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---------------- 1. 读取 ----------------
log("读取交易流水表 ...")
df_t = pd.read_csv(TRANS_CSV)
log("  transaction: %d 行 x %d 列" % df_t.shape)

log("读取客户/身份信息表 ...")
df_i = pd.read_csv(IDENT_CSV)
log("  identity:    %d 行 x %d 列" % df_i.shape)

# ---------------- 2. 合并 ----------------
log("按 TransactionID 左连接合并 ...")
df = df_t.merge(df_i, on="TransactionID", how="left")
del df_t, df_i
log("  合并后: %d 行 x %d 列" % df.shape)

# ---------------- 3. 剔除缺失率>99% 的列 ----------------
missing = df.isna().mean()
drop_cols = missing[missing > MISSING_DROP_THRESHOLD].index.tolist()
df = df.drop(columns=drop_cols)
log("缺失率>99%% 剔除 %d 列: %s" % (len(drop_cols), drop_cols))

# ---------------- 4. 拆分标签与特征 ----------------
y_fraud = df["isFraud"].astype(int)          # 异常检测标签
y_product = df["ProductCD"].copy()           # 数据分类标签(5类)
feat = df.drop(columns=["isFraud", "TransactionID"])
log("特征列数: %d (含 ProductCD)" % feat.shape[1])

# ---------------- 5. 缺失值填充 ----------------
cat_cols = feat.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = [c for c in feat.columns if c not in cat_cols]
log("数值列 %d 个, 类别列 %d 个" % (len(num_cols), len(cat_cols)))

log("数值列缺失值 -> 中位数填充 ...")
med = feat[num_cols].median()
feat[num_cols] = feat[num_cols].fillna(med)

log("类别列缺失值 -> 'Unknown' 填充 ...")
feat[cat_cols] = feat[cat_cols].fillna("Unknown")

# ---------------- 6. 类别编码 (label encoding, 保存映射) ----------------
log("类别列 label encoding ...")
mapping = {}
for c in cat_cols:
    codes, uniques = pd.factorize(feat[c])
    feat[c] = codes.astype(np.int32)
    mapping[c] = {str(v): int(i) for i, v in enumerate(uniques)}
log("  编码完成, 映射表 %d 个字段" % len(mapping))

# ProductCD 单独保留编码映射(作为分类标签)
prod_map = {v: int(i) for i, v in enumerate(sorted(y_product.unique()))}
y_product_enc = y_product.map(prod_map).astype(int)
log("ProductCD 分类标签映射: %s" % prod_map)

# ---------------- 7. 分层划分 80/20 ----------------
log("按 isFraud 分层划分 train/test (80/20, seed=%d) ..." % SEED)
rng = np.random.RandomState(SEED)
test_mask = np.zeros(len(feat), dtype=bool)
for cls in y_fraud.unique():
    idx = np.where(y_fraud.values == cls)[0]
    k = int(round(len(idx) * TEST_SIZE))
    test_mask[rng.choice(idx, size=k, replace=False)] = True

X_train, X_test = feat[~test_mask], feat[test_mask]
out_train = X_train.copy()
out_train["isFraud"] = y_fraud[~test_mask].values
out_train["ProductCD_label"] = y_product_enc[~test_mask].values
out_test = X_test.copy()
out_test["isFraud"] = y_fraud[test_mask].values
out_test["ProductCD_label"] = y_product_enc[test_mask].values

# ---------------- 8. 保存 ----------------
log("写出 processed 文件 ...")
out_train.to_csv(os.path.join(OUT, "train.csv.gz"), index=False, compression="gzip")
out_test.to_csv(os.path.join(OUT, "test.csv.gz"), index=False, compression="gzip")
with open(os.path.join(OUT, "category_mapping.json"), "w", encoding="utf-8") as f:
    json.dump({"label_encoding": mapping, "ProductCD_label": prod_map,
               "dropped_columns_missing_gt_99pct": drop_cols}, f,
              ensure_ascii=False, indent=2)

# ---------------- 9. 报告 ----------------
def dist(s, name):
    vc = s.value_counts().sort_index()
    total = len(s)
    return "  %s: %s" % (name, "  ".join(
        "%s=%d(%.2f%%)" % (k, v, 100 * v / total) for k, v in vc.items()))

print("\n" + "=" * 60)
print("处理报告")
print("=" * 60)
print("合并后样本: %d, 剔除列: %d, 最终特征列: %d" % (len(df), len(drop_cols), feat.shape[1]))
print("训练集: %d 行 | 测试集: %d 行" % (len(out_train), len(out_test)))
print("\nisFraud 标签分布:")
print(dist(out_train["isFraud"], "train"))
print(dist(out_test["isFraud"], "test"))
print("\nProductCD 分类标签分布:")
print(dist(out_train["ProductCD_label"], "train"))
print(dist(out_test["ProductCD_label"], "test"))
print("\n输出目录: %s" % os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print("  %-28s %.2f MB" % (f, os.path.getsize(p) / 1024 / 1024))
log("A1 完成")
