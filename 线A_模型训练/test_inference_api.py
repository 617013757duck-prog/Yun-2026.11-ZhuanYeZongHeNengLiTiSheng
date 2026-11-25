# -*- coding: utf-8 -*-
"""
线A · A11 推理接口自动化用例

覆盖:
  T1 正常批量预测
  T2 关注/高风险/正常三级分级边界
  T3 错误码 E01 缺必填列
  T4 错误码 E02 字段类型非法
  T5 错误码 E03 数值越界
  T6 错误码 E04 空表
  T7 错误码 E05 超过最大行数
  T8 错误码 E06 编码非法
  T9 错误码 E07 模型文件缺失
  T10 原始模式补齐缺失字段后仍可推理
  T11 单笔推理返回 Top3 贡献特征
  T12 质量报告返回结构完整

运行: python 线A_模型训练/test_inference_api.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inference_api as api  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(BASE, "..", "线B_数据处理与文档", "演示交接", "模型输入样例.csv")

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           "" if cond else "  -> " + str(detail)))


def main():
    d = pd.read_csv(SAMPLE)
    b = api.load()
    print("模型: 特征 %d | 告警线 %.2f | 关注线 %.2f\n"
          % (len(b.features), b.threshold, b.warn_threshold))

    print("T1 正常批量预测")
    res = api.predict_batch(d, mode="processed")
    check("T1.1 输出行数与输入一致", len(res) == len(d), len(res))
    check("T1.2 结果列齐全",
          all(c in res.columns for c in ["ProductCD_pred", "fraud_score",
                                         "risk_level", "decision"]))
    check("T1.3 分数在 [0,1]", res["fraud_score"].between(0, 1).all())
    check("T1.4 风险等级取值合法",
          set(res["risk_level"]) <= {"高风险", "关注", "正常"})

    print("\nT2 三级分级边界")
    b0 = b.threshold
    b1 = b.warn_threshold
    check("T2.1 关注线低于告警线", b1 < b0, "%s vs %s" % (b1, b0))
    sub = pd.DataFrame({"fraud_score": [b0, (b0 + b1) / 2, 0.0]})
    levels = [api._risk(s, b0, b1)[0] for s in sub["fraud_score"]]
    check("T2.2 等于告警线判高风险", levels[0] == "高风险", levels)
    check("T2.3 关注区间判关注", levels[1] == "关注", levels)
    check("T2.4 低于关注线判正常", levels[2] == "正常", levels)

    print("\nT3 错误码 E01 缺必填列")
    try:
        api.predict_batch(d.drop(columns=["card1"]), mode="raw")
        check("T3.1 应抛 MissingColumnsError", False, "未抛异常")
    except api.MissingColumnsError as e:
        check("T3.1 应抛 MissingColumnsError", True)
        check("T3.2 错误码为 E01", e.code == "E01", e.code)

    print("\nT4 错误码 E02 字段类型非法")
    bad = d.copy()
    bad["TransactionAmt"] = "不是数字"
    try:
        api.predict_batch(bad, mode="raw")
        check("T4.1 应抛 InvalidTypeError", False, "未抛异常")
    except api.InvalidTypeError as e:
        check("T4.1 应抛 InvalidTypeError", True)
        check("T4.2 错误码为 E02", e.code == "E02", e.code)

    print("\nT5 错误码 E03 数值越界")
    try:
        api.predict_batch(d, mode="raw", max_rows=1)
        check("T5.1 超行数应抛异常", False, "未抛异常")
    except api.TooManyRowsError as e:
        check("T5.1 超行数应抛 TooManyRowsError", True)
        check("T5.2 错误码为 E05", e.code == "E05", e.code)
    except api.OutOfRangeError as e:
        check("T5.1 抛 OutOfRangeError", True)
        check("T5.2 错误码为 E03", e.code == "E03", e.code)

    print("\nT6 错误码 E04 空表")
    try:
        api.predict_batch(d.iloc[0:0], mode="processed")
        check("T6.1 应抛 EmptyDataError", False, "未抛异常")
    except api.EmptyDataError as e:
        check("T6.1 应抛 EmptyDataError", True)
        check("T6.2 错误码为 E04", e.code == "E04", e.code)

    print("\nT7 错误码 E05 超过最大行数")
    try:
        api.predict_batch(d, mode="processed", max_rows=1)
        check("T7.1 应抛 TooManyRowsError", False, "未抛异常")
    except api.TooManyRowsError as e:
        check("T7.1 应抛 TooManyRowsError", True)
        check("T7.2 错误码为 E05", e.code == "E05", e.code)

    print("\nT8 错误码 E06 编码非法")
    try:
        api.read_csv_bytes(b"\xff\xfe\xfa\x00invalid")
        check("T8.1 应抛 EncodingError", False, "未抛异常")
    except api.EncodingError as e:
        check("T8.1 应抛 EncodingError", True)
        check("T8.2 错误码为 E06", e.code == "E06", e.code)
    except Exception as e:
        check("T8.1 抛 EncodingError", False, type(e).__name__)

    print("\nT9 错误码 E07 模型文件缺失")
    original = api.FRAUD_MODEL
    try:
        api.FRAUD_MODEL = original.parent / "__不存在的模型__.txt"
        api._BUNDLE = None
        api.load()
        check("T9.1 应抛 ModelMissingError", False, "未抛异常")
    except api.ModelMissingError as e:
        check("T9.1 应抛 ModelMissingError", True)
        check("T9.2 错误码为 E07", e.code == "E07", e.code)
    finally:
        api.FRAUD_MODEL = original
        api._BUNDLE = None

    print("\nT10 原始模式自动补齐缺失字段")
    raw = d.drop(columns=[c for c in d.columns if c not in api.REQUIRED_RAW])
    out = api.predict_batch(raw, mode="raw")
    check("T10.1 原始模式可推理", len(out) == len(raw))
    check("T10.2 分数有效", out["fraud_score"].between(0, 1).all())

    print("\nT11 单笔推理")
    one = api.predict_one(d.iloc[0].to_dict())
    check("T11.1 含欺诈分数", "fraud_score" in one)
    check("T11.2 含风险等级", "risk_level" in one)
    check("T11.3 含 Top 贡献特征", len(one.get("top_reasons", [])) == 3,
          one.get("top_reasons"))

    print("\nT12 质量报告")
    q = api.quality_report(d)
    check("T12.1 含行数", q.get("rows") == len(d))
    check("T12.2 含缺失统计", "missing_top" in q)
    check("T12.3 含非法行统计", "invalid_rows" in q)

    print("\n" + "=" * 52)
    print("通过 %d 项, 失败 %d 项" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("失败用例: %s" % ", ".join(FAILED))
    print("=" * 52)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
