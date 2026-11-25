# -*- coding: utf-8 -*-
"""
线A · A12 批处理预测与分数漂移监控

用法:
    python batch_predict.py --input 交易.csv --output 结果.csv
    python batch_predict.py --input 交易.csv --mode processed --threshold 0.79

输出:
    1) 结果 CSV(原始列 + ProductCD_pred / fraud_score / risk_level / decision)
    2) 同名 .summary.json(日报口径 + 分数分布 + Top10 高风险)
    3) 追加一行到 models/monitor_history.jsonl(漂移监控基线对比)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_api as api

BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(BASE, "models", "monitor_history.jsonl")


def read_any(path):
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise api.EncodingError("无法识别文件编码, 请使用 UTF-8 或 GBK")


def summarize(df, elapsed):
    scores = df["fraud_score"].values
    n = len(df)
    lv = df["risk_level"].value_counts().to_dict()
    top = df.nlargest(10, "fraud_score")
    keep = [c for c in ["TransactionID", "TransactionAmt", "fraud_score",
                        "risk_level", "decision"] if c in top.columns]
    return {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总笔数": int(n),
        "高风险": int(lv.get("高风险", 0)),
        "关注": int(lv.get("关注", 0)),
        "正常": int(lv.get("正常", 0)),
        "告警率%": round(100 * (lv.get("高风险", 0) + lv.get("关注", 0)) / n, 2),
        "拦截率%": round(100 * lv.get("高风险", 0) / n, 2),
        "平均分数": round(float(scores.mean()), 4),
        "分数P50": round(float(np.percentile(scores, 50)), 4),
        "分数P95": round(float(np.percentile(scores, 95)), 4),
        "分数P99": round(float(np.percentile(scores, 99)), 4),
        "耗时秒": round(elapsed, 2),
        "处理速度行每秒": int(n / elapsed) if elapsed > 0 else 0,
        "Top10高风险": top[keep].round(4).to_dict("records"),
    }


def check_drift(summary):
    """与历史平均分数对比, 偏移超过 50% 记为漂移提醒。"""
    if not os.path.exists(HISTORY):
        return None
    hist = []
    with open(HISTORY, encoding="utf-8") as f:
        for line in f:
            try:
                hist.append(json.loads(line)["平均分数"])
            except Exception:
                continue
    if len(hist) < 3:
        return None
    base = float(np.mean(hist))
    cur = summary["平均分数"]
    ratio = (cur - base) / base if base else 0
    return {"历史基线均值": round(base, 4), "本次均值": cur,
            "相对偏移%": round(100 * ratio, 2),
            "是否漂移": bool(abs(ratio) > 0.5)}


def main():
    ap = argparse.ArgumentParser(description="批量交易风险预测")
    ap.add_argument("--input", required=True, help="输入 CSV 路径")
    ap.add_argument("--output", default=None, help="输出 CSV 路径")
    ap.add_argument("--mode", default="raw", choices=["raw", "processed"])
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    t0 = time.time()
    df = read_any(args.input)
    print("读入 %d 行 × %d 列, 模式=%s" % (len(df), df.shape[1], args.mode))

    res = api.predict_batch(df, threshold=args.threshold, mode=args.mode)
    elapsed = time.time() - t0

    out = args.output or os.path.splitext(args.input)[0] + "_预测结果.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")

    summary = summarize(res, elapsed)
    sp = os.path.splitext(out)[0] + ".summary.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    drift = check_drift(summary)
    if drift:
        summary["漂移检查"] = drift

    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"时间": summary["生成时间"],
                            "平均分数": summary["平均分数"],
                            "高风险": summary["高风险"],
                            "总笔数": summary["总笔数"]}, ensure_ascii=False) + "\n")

    print("\n" + "=" * 56)
    print("批处理完成")
    print("=" * 56)
    for k in ["总笔数", "高风险", "关注", "正常", "告警率%",
              "平均分数", "分数P95", "耗时秒", "处理速度行每秒"]:
        print("  %-12s %s" % (k, summary[k]))
    if drift:
        print("  漂移检查     %s (偏移 %+.2f%%)" % (
            "检测到漂移" if drift["是否漂移"] else "正常", drift["相对偏移%"]))
    print("  结果文件     %s" % out)
    print("  汇总文件     %s" % sp)
    print("=" * 56)


if __name__ == "__main__":
    main()
