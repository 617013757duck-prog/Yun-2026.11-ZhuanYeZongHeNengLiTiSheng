# -*- coding: utf-8 -*-
"""
线A · A11 统一推理接口

设计目标: 训练与推理共用同一套预处理口径, 避免"两套编码漂移"。
只暴露三个函数, 线C 前端与批处理脚本均通过本模块调用:

    predict_batch(df, threshold=None, mode="raw")  -> DataFrame
    predict_one(record, threshold=None)            -> dict
    quality_report(df)                             -> dict

输入模式:
    mode="raw"        原始交易字段(含或不含身份字段), 内部自动对齐与编码
    mode="processed"  已是 422 列建模特征, 直接推理

依赖产物(均由 preprocess.py 生成):
    数据集/processed/category_mapping.json
    线A_模型训练/models/lgbm_fraud_final.txt
    线A_模型训练/models/lgbm_product_classifier.txt
    线A_模型训练/models/metrics_fraud_final.json
"""
from __future__ import annotations

import json
import os
import io
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROC = BASE.parent / "数据集" / "processed"
MODEL_DIR = BASE / "models"

FRAUD_MODEL = MODEL_DIR / "lgbm_fraud_final.txt"
PRODUCT_MODEL = MODEL_DIR / "lgbm_product_classifier.txt"
FRAUD_METRICS = MODEL_DIR / "metrics_fraud_final.json"
PRODUCT_METRICS = MODEL_DIR / "metrics_product_classifier.json"
MAPPING = PROC / "category_mapping.json"
SPEC_CACHE = MODEL_DIR / "feature_spec.json"

LABEL_COLUMNS = ["isFraud", "ProductCD", "ProductCD_label"]
REQUIRED_RAW = ["TransactionAmt", "TransactionDT", "card1", "card4", "card6"]

WARN_THRESHOLD = 0.08  # 关注线兜底值, 正常由 metrics_fraud_final.json 的 thresholds.warn 覆盖


# ============ 异常类型(与前端错误码 E01-E07 对应) ============
class InferenceError(Exception):
    code = "E00"


class MissingColumnsError(InferenceError):
    code = "E01"

    def __init__(self, columns):
        self.columns = list(columns)
        super().__init__("缺少必填列: %s" % ", ".join(self.columns))


class InvalidTypeError(InferenceError):
    code = "E02"

    def __init__(self, row, column, value):
        self.row, self.column, self.value = row, column, value
        super().__init__("第 %s 行 %s 的值 %r 无法解析为数值" % (row, column, value))


class OutOfRangeError(InferenceError):
    code = "E03"

    def __init__(self, row, column, value, allowed=None):
        msg = "第 %s 行 %s=%r 超出允许范围" % (row, column, value)
        if allowed:
            msg += ", 允许值: %s" % "/".join(map(str, allowed))
        super().__init__(msg)


class EmptyDataError(InferenceError):
    code = "E04"


class TooManyRowsError(InferenceError):
    code = "E05"


class EncodingError(InferenceError):
    code = "E06"


class ModelMissingError(InferenceError):
    code = "E07"


# ============ 模型与特征规格 ============
class _Bundle:
    def __init__(self):
        for p in (FRAUD_MODEL, PRODUCT_MODEL, FRAUD_METRICS, MAPPING):
            if not p.exists():
                raise ModelMissingError("模型或映射文件缺失: %s" % p)

        # LightGBM C API 不支持中文路径, 先读字符串
        self.fraud = lgb.Booster(model_str=FRAUD_MODEL.read_text(encoding="utf-8"))
        self.product = lgb.Booster(model_str=PRODUCT_MODEL.read_text(encoding="utf-8"))
        self.features = list(self.fraud.feature_name())

        with open(FRAUD_METRICS, encoding="utf-8") as f:
            fm = json.load(f)
        self.threshold = float(fm["tuned_threshold"]["threshold"])
        th = fm.get("thresholds") or {}
        self.warn_threshold = float(th.get("warn", WARN_THRESHOLD))
        self.strict_threshold = float(th.get("strict", self.threshold))
        self.model_version = str(fm.get("model", "LightGBM"))
        with open(PRODUCT_METRICS, encoding="utf-8") as f:
            pm = json.load(f)
        self.product_reverse = {v: k for k, v in pm["classes"].items()} \
            if isinstance(pm.get("classes"), dict) else None

        with open(MAPPING, encoding="utf-8") as f:
            mp = json.load(f)
        self.label_encoding = mp["label_encoding"]
        self.product_label = mp["ProductCD_label"]
        self.dropped = mp["dropped_columns_missing_gt_99pct"]
        self.categorical = list(self.label_encoding.keys())
        self.spec = self._load_spec()

    def _load_spec(self):
        if SPEC_CACHE.exists():
            with open(SPEC_CACHE, encoding="utf-8") as f:
                return json.load(f)
        # 首次调用: 从 processed 采样 2 万行估计填充值
        sample = pd.read_csv(PROC / "train.csv.gz", nrows=20000)
        fill = {}
        for c in self.features:
            if c not in sample.columns:
                fill[c] = 0.0
            elif c in self.categorical:
                mode = sample[c].mode(dropna=True)
                fill[c] = int(mode.iloc[0]) if len(mode) else 0
            else:
                fill[c] = float(sample[c].median())
        spec = {"features": self.features, "fill_values": fill,
                "categorical": self.categorical, "dropped": self.dropped}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(SPEC_CACHE, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        return spec


_BUNDLE = None


def load():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = _Bundle()
    return _BUNDLE


# ============ 输入对齐 ============
def _align(df: pd.DataFrame, mode: str):
    b = load()
    if mode == "raw":
        missing = [c for c in REQUIRED_RAW if c not in df.columns]
        if missing:
            raise MissingColumnsError(missing)

    X = pd.DataFrame(index=df.index)
    cols = {}
    for c in b.features:
        if c in df.columns:
            if c in b.categorical:
                mapping = b.label_encoding[c]
                s = df[c].astype(str).map(mapping)
                unknown = int(mapping.get("Unknown", 0))
                cols[c] = s.fillna(unknown).astype("int32")
            else:
                raw_s = df[c]
                v = pd.to_numeric(raw_s, errors="coerce")
                # 原值非空但转不成数值 = 脏数据, 报错(E02); 原值为空则按缺失填充
                bad = v.isna() & raw_s.notna()
                if bad.any():
                    idx = raw_s.index[bad][0]
                    raise InvalidTypeError(int(idx) + 2, c, raw_s.at[idx])
                cols[c] = v.fillna(b.spec["fill_values"][c]).astype("float64")
        else:
            fill = b.spec["fill_values"][c]
            cols[c] = pd.Series(
                np.int32(fill) if c in b.categorical else np.float64(fill),
                index=df.index)
    X = pd.DataFrame(cols, index=df.index)[b.features]
    return X


def _risk(score: float, threshold: float, warn: float = None):
    if score >= threshold:
        return "高风险", "拦截"
    if warn is None:
        warn = WARN_THRESHOLD
    if score >= warn:
        return "关注", "复核"
    return "正常", "放行"


def _product_names(idx):
    b = load()
    rev = {v: k for k, v in b.product_label.items()}
    return [rev.get(int(i), str(i)) for i in np.asarray(idx)]


# ============ 对外接口 ============
def predict_batch(df: pd.DataFrame, threshold: float | None = None,
                  mode: str = "raw", max_rows: int = 200000) -> pd.DataFrame:
    """批量预测。返回原始列 + 结果列。"""
    if not isinstance(df, pd.DataFrame):
        raise InvalidTypeError("-", "input", type(df).__name__)
    if df.empty:
        raise EmptyDataError("上传文件为空, 至少需要 1 行数据")
    if len(df) > max_rows:
        raise TooManyRowsError("行数 %d 超过上限 %d, 请拆分后上传" % (len(df), max_rows))

    b = load()
    th = float(threshold if threshold is not None else b.threshold)
    X = _align(df, mode)

    score = b.fraud.predict(X)
    pidx = np.argmax(b.product.predict(X), axis=1)
    pconf = b.product.predict(X).max(axis=1)

    out = df.copy()
    out["ProductCD_pred"] = _product_names(pidx)
    out["ProductCD_confidence"] = np.round(pconf, 4)
    out["fraud_score"] = np.round(score, 6)
    out["risk_level"] = [ _risk(s, th, b.warn_threshold)[0] for s in score ]
    out["decision"] = [ _risk(s, th, b.warn_threshold)[1] for s in score ]
    return out


def predict_one(record: dict, threshold: float | None = None) -> dict:
    """单笔预测, 额外返回 Top3 贡献特征。"""
    df = pd.DataFrame([record])
    out = predict_batch(df, threshold=threshold, mode="raw", max_rows=1)
    row = out.iloc[0]

    b = load()
    X = _align(df, "raw")
    contrib = b.fraud.predict(X, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(contrib))[:3]
    reasons = [{"feature": b.features[i],
                "value": float(X.iloc[0, i]),
                "contribution": round(float(contrib[i]), 4)} for i in order]

    return {
        "TransactionID": row.get("TransactionID"),
        "product": row["ProductCD_pred"],
        "product_confidence": float(row["ProductCD_confidence"]),
        "fraud_score": float(row["fraud_score"]),
        "risk_level": row["risk_level"],
        "decision": row["decision"],
        "top_reasons": reasons,
    }


def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    """从原始字节读取 CSV, 兼容 UTF-8(BOM) 与 GBK(Excel 另存)。"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise EncodingError("无法识别 CSV 编码, 请使用 UTF-8 或 GBK 保存")


def quality_report(df: pd.DataFrame) -> dict:
    """上传数据的即时质量报告(供前端渲染)。"""
    if df.empty:
        raise EmptyDataError("文件为空")
    missing = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
    invalid = []
    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")
        bad = df.index[(amt.isna()) | (amt <= 0)]
        for i in bad[:50]:
            invalid.append({"row": int(i) + 2, "column": "TransactionAmt",
                            "error": "金额必须大于 0",
                            "value": None if pd.isna(df.at[i, "TransactionAmt"])
                            else str(df.at[i, "TransactionAmt"])})
    if "ProductCD" in df.columns:
        allowed = set(load().product_label)
        bad = df.index[~df["ProductCD"].astype(str).isin(allowed)]
        for i in bad[:50]:
            invalid.append({"row": int(i) + 2, "column": "ProductCD",
                            "error": "不在允许类别内",
                            "value": str(df.at[i, "ProductCD"])})
    dist = {}
    if "ProductCD" in df.columns:
        dist = df["ProductCD"].astype(str).value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_top": [{"column": c, "missing_pct": float(v)}
                        for c, v in missing.head(10).items() if v > 0],
        "invalid_rows": invalid,
        "distribution": dist,
    }


if __name__ == "__main__":
    # 自测: 用线B交付的 10 行样例跑一次完整流程
    sample_path = BASE.parent / "线B_数据处理与文档" / "演示交接" / "模型输入样例.csv"
    if not sample_path.exists():
        print("未找到样例文件:", sample_path)
        raise SystemExit(1)
    d = pd.read_csv(sample_path)
    b = load()
    print("特征数 %d, 告警阈值 %.2f" % (len(b.features), b.threshold))
    res = predict_batch(d, mode="processed")
    cols = ["TransactionID", "ProductCD_pred", "ProductCD_confidence",
            "fraud_score", "risk_level", "decision"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))
    print("\n质量报告:", json.dumps(quality_report(d), ensure_ascii=False)[:300])
    one = predict_one(d.iloc[0].to_dict())
    print("\n单笔:", json.dumps(one, ensure_ascii=False))
