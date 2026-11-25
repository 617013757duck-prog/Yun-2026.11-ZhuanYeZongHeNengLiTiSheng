# -*- coding: utf-8 -*-
"""
线C · 模型推理封装

加载线A训练好的两个 LightGBM 模型, 提供统一的输入校验与预测接口, 供
Streamlit 演示页和命令行自测脚本复用。

- 欺诈模型: isFraud 二分类概率; 告警阈值读取自 metrics_fraud_final.json
  的 tuned_threshold.threshold, 关注线读取 thresholds.warn
- 产品模型: ProductCD 五分类 (C/H/R/S/W)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "线A_模型训练" / "models"
PROC_DIR = ROOT / "数据集" / "processed"

FRAUD_MODEL = MODEL_DIR / "lgbm_fraud_final.txt"
PRODUCT_MODEL = MODEL_DIR / "lgbm_product_classifier_final.txt"
FRAUD_METRICS = MODEL_DIR / "metrics_fraud_final.json"
PRODUCT_METRICS = MODEL_DIR / "metrics_product_classifier_final.json"

WARN_THRESHOLD = 0.08  # “关注”预警线兜底值, 正常由线A metrics 产物覆盖

# 可选标签列: 上传文件中若带有真实标签, 仅用于对照, 不参与预测
LABEL_COLUMNS = ["isFraud", "ProductCD", "ProductCD_label"]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ModelBundle:
    """两个模型 + 阈值 + 特征名的只读集合。"""

    def __init__(self):
        # LightGBM C API 对中文路径支持不好, 先读成字符串再加载
        self.fraud = lgb.Booster(
            model_str=FRAUD_MODEL.read_text(encoding="utf-8"))
        self.product = lgb.Booster(
            model_str=PRODUCT_MODEL.read_text(encoding="utf-8"))

        fm = _read_json(FRAUD_METRICS)
        pm = _read_json(PRODUCT_METRICS)
        self.alert_threshold = float(fm["tuned_threshold"]["threshold"])
        th = fm.get("thresholds") or {}
        self.warn_threshold = float(th.get("warn", WARN_THRESHOLD))
        self.strict_threshold = float(th.get("strict", self.alert_threshold))
        self.threshold_policy = th.get("policy", "")
        self.classes = list(pm.get("classes") or ["C", "H", "R", "S", "W"])
        self.fraud_metrics = fm.get("tuned_threshold", {})
        self.features = list(self.fraud.feature_name())
        self.product_features = list(self.product.feature_name())

        if self.features != self.product_features:
            raise RuntimeError("两个模型的特征名/顺序不一致, 请检查线A产物")


def read_csv_any(path_or_buffer):
    """读取 CSV, 兼容 UTF-8(BOM) 与 GBK(Excel 另存)。"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别 CSV 编码，请使用 UTF-8 或 GBK 保存")


def validate_and_align(df, features):
    """校验输入并返回按模型特征顺序排列的数值矩阵。

    返回: (X, info)
    info: {missing_columns, extra_columns, bad_columns, nan_cells, rows}
    缺列抛 ValueError; 无法转为数值的列抛 ValueError。
    """
    if df is None or len(df) == 0:
        raise ValueError("输入文件没有数据行")

    cols = list(df.columns)
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        raise ValueError("输入文件存在重复列名: %s" % "、".join(dup[:10]))

    extra = [c for c in cols if c not in features]
    missing = [c for c in features if c not in cols]

    info = {
        "rows": len(df),
        "missing_columns": missing,
        "extra_columns": extra,
        "bad_columns": [],
        "nan_cells": 0,
    }
    if missing:
        head = "、".join(missing[:20]) + (" 等" if len(missing) > 20 else "")
        raise ValueError("缺少 %d 个模型所需字段（应为预处理后的 %d 列）：%s"
                         % (len(missing), len(features), head))

    converted = {}
    for c in features:
        s = df[c]
        if s.dtype.kind in "iufb":
            converted[c] = s.astype(float)
            continue
        conv = pd.to_numeric(s, errors="coerce")
        invalid = int(conv.isna().sum() - s.isna().sum())
        if invalid > 0:
            info["bad_columns"].append(c)
        converted[c] = conv.astype(float)

    if info["bad_columns"]:
        head = "、".join(info["bad_columns"][:10])
        raise ValueError("以下字段包含无法解析为数值的内容: %s" % head)

    X = pd.DataFrame(converted, index=df.index)  # 一次性建表, 避免逐列插入产生碎片
    info["nan_cells"] = int(X.isna().sum().sum())
    return X, info


def predict(bundle, df):
    """对已预处理的数据做推理, 返回逐行结果表与补充信息。

    返回: (result_df, info, product_proba)
    - result_df: 行号 / 产品类别 / 产品置信度 / 欺诈分数 / 风险等级 / 处置建议 / 是否告警
    - product_proba: ndarray(n, 5), 各类别概率
    """
    X, info = validate_and_align(df, bundle.features)

    fraud_score = np.asarray(bundle.fraud.predict(X), dtype=float).ravel()
    proba = np.asarray(bundle.product.predict(X), dtype=float)
    if proba.ndim == 1:  # 理论上不会发生: 产品模型是多分类
        proba = np.column_stack([1 - proba, proba])
    pred_idx = proba.argmax(axis=1)

    result = pd.DataFrame({
        "行号": np.arange(1, len(X) + 1),
        "产品类别": [bundle.classes[i] for i in pred_idx],
        "产品置信度": proba.max(axis=1).round(4),
        "欺诈分数": fraud_score.round(4),
    })

    alert = result["欺诈分数"] >= bundle.alert_threshold
    warn = (result["欺诈分数"] >= bundle.warn_threshold) & ~alert
    result["风险等级"] = np.select([alert, warn], ["高风险", "关注"], default="正常")
    result["是否告警"] = alert
    result["处置建议"] = np.select(
        [alert, warn],
        ["立即拦截 / 人工复核", "加入观察名单"],
        default="正常放行")

    info["alert_count"] = int(alert.sum())
    info["warn_count"] = int(warn.sum())
    info["alert_rate"] = float(alert.mean()) if len(result) else 0.0
    info["avg_score"] = float(np.mean(fraud_score)) if len(result) else 0.0
    return result, info, proba


def class_probability_frame(bundle, proba):
    """把 n×5 概率矩阵转成带类别名的 DataFrame。"""
    return pd.DataFrame(proba, columns=bundle.classes)
