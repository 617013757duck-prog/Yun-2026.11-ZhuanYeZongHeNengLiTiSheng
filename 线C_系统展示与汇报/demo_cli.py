# -*- coding: utf-8 -*-
"""
线C · 命令行自测脚本

不启动界面即可验证模型接入是否正常：
  python demo_cli.py                 # 用 10 行演示样例
  python demo_cli.py 某文件.csv      # 指定预处理后的 CSV
"""
import argparse
from pathlib import Path

import pandas as pd

import predict as core

SAMPLE = core.ROOT / "线B_数据处理与文档" / "演示交接" / "模型输入样例.csv"
LABELS = core.ROOT / "线B_数据处理与文档" / "演示交接" / "样例真实标签.csv"


def main():
    parser = argparse.ArgumentParser(description="线C 演示系统命令行自测")
    parser.add_argument("csv", nargs="?", default=str(SAMPLE),
                        help="预处理后的交易 CSV（默认使用演示样例）")
    parser.add_argument("--show", type=int, default=10, help="打印前 N 行")
    args = parser.parse_args()

    bundle = core.ModelBundle()
    print("=" * 66)
    print("模型: LightGBM 欺诈二分类 + ProductCD 五分类")
    print("特征: %d 列 | 告警阈值: %.2f | 产品类别: %s"
          % (len(bundle.features), bundle.alert_threshold,
             "/".join(bundle.classes)))
    print("=" * 66)

    df = core.read_csv_any(args.csv)
    try:
        result, info, _ = core.predict(bundle, df)
    except ValueError as exc:
        print("[输入校验失败] %s" % exc)
        raise SystemExit(2)

    print("输入: %s" % args.csv)
    print("行数 %d | 高风险告警 %d | 关注 %d | 告警率 %.1f%% | 平均分数 %.4f"
          % (info["rows"], info["alert_count"], info["warn_count"],
             100 * info["alert_rate"], info["avg_score"]))
    if info["extra_columns"]:
        print("已忽略额外列: %s" % "、".join(info["extra_columns"][:10]))

    print("-" * 66)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(result.head(args.show).to_string(index=False))
    print("-" * 66)

    if (Path(args.csv).resolve() == SAMPLE.resolve()) and LABELS.exists():
        labels = core.read_csv_any(str(LABELS))
        if len(labels) == len(result):
            same_product = (result["产品类别"].values
                            == labels["ProductCD"].values).mean()
            same_fraud = (result["是否告警"].astype(int).values
                          == labels["isFraud"].values).mean()
            print("样例真实标签对照: 产品命中 %.0f%% | 欺诈判定一致 %.0f%%"
                  % (100 * same_product, 100 * same_fraud))

    out = Path("预测结果_%s.csv" % Path(args.csv).stem)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print("完整结果已保存: %s" % out.resolve())
    print("界面演示: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
