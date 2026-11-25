"""从项目内原始 CSV 分块统计并生成线 B 图表与核查数据。"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "数据集"
FIGURES = HERE / "B1_EDA图表"
SOURCE = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"


def describe(col):
    named = {
        "TransactionID": ("主键", "交易标识，两张表的关联键"),
        "isFraud": ("标签", "竞赛目标，0 为非欺诈，1 为欺诈"),
        "TransactionDT": ("交易属性", "相对参考时点的时间差"),
        "TransactionAmt": ("交易属性", "交易金额"),
        "ProductCD": ("标签", "产品类别代码，项目五分类目标"),
        "P_emaildomain": ("邮箱", "付款方邮箱域名"),
        "R_emaildomain": ("邮箱", "收款方邮箱域名"),
        "DeviceType": ("设备", "设备类型"),
        "DeviceInfo": ("设备", "设备信息"),
    }
    if col in named:
        return named[col]
    groups = [
        (r"card\d+", "卡片", "匿名卡片属性，按类别理解"),
        (r"addr\d+", "地址", "匿名地址代码，按类别理解"),
        (r"dist\d+", "距离", "距离类匿名字段，具体口径未公开"),
        (r"C\d+", "计数 C", "匿名计数类特征，具体口径未公开"),
        (r"D\d+", "时差 D", "匿名时间差类特征，具体口径未公开"),
        (r"M\d+", "匹配 M", "匿名匹配类字段，具体口径未公开"),
        (r"V\d+", "匿名 V", "Vesta 工程特征，单列定义未公开"),
        (r"id_\d+", "身份 id", "匿名身份或设备相关字段，单列定义未公开"),
    ]
    for pattern, group, note in groups:
        if re.fullmatch(pattern, col):
            return group, note
    return "其他", "参见竞赛数据说明"


def read_statistics():
    identity = pd.read_csv(DATA / "train_identity.csv").set_index("TransactionID")
    if not identity.index.is_unique or identity.index.hasnans:
        raise ValueError("身份表交易键必须非空且唯一")
    transaction_file = DATA / "train_transaction.csv"
    missing = pd.Series(dtype="int64")
    ids, labels = [], []
    text_columns = set(identity.select_dtypes(exclude="number"))
    for chunk in pd.read_csv(transaction_file, chunksize=25_000, low_memory=False):
        ids.append(chunk["TransactionID"])
        labels.append(chunk[["isFraud", "ProductCD"]])
        text_columns.update(chunk.select_dtypes(exclude="number"))
        missing = missing.add(chunk.isna().sum(), fill_value=0)
    ids = pd.Index(pd.concat(ids, ignore_index=True), name="TransactionID")
    if not ids.is_unique or ids.hasnans:
        raise ValueError("交易表交易键必须非空且唯一")
    label_frame = pd.concat(labels, ignore_index=True)
    if label_frame.isna().any().any():
        raise ValueError("发现缺失标签")
    joined_identity = identity.reindex(ids)
    transaction_columns = pd.read_csv(transaction_file, nrows=0).columns
    mapping = json.loads((DATA / "processed/category_mapping.json").read_text(encoding="utf-8"))
    dropped = mapping["dropped_columns_missing_gt_99pct"]
    rows = []
    for filename, columns, raw, joined, total in [
        ("train_transaction.csv", transaction_columns, missing, missing, len(ids)),
        ("train_identity.csv", ["TransactionID", *identity.columns], identity.isna().sum(),
         joined_identity.isna().sum(), len(identity)),
    ]:
        for col in columns:
            raw_count = int(raw.get(col, 0))
            merged_count = int(joined.get(col, 0))
            group, note = describe(col)
            if col == "TransactionID":
                treatment = "关联键，建模时移除"
            elif col == "isFraud":
                treatment = "二分类标签"
            elif col == "ProductCD":
                treatment = "生成五分类标签，两个模型均排除原字段"
            elif col in dropped:
                treatment = "剔除（合并后缺失率 >99%）"
            elif col in mapping["label_encoding"]:
                treatment = "Unknown 填充后编码"
            else:
                treatment = "数值中位数填充"
            rows.append([col, filename, group, "文本" if col in text_columns else "数值",
                         note, raw_count, raw_count / total, merged_count, merged_count / len(ids), treatment])
    actual_drop = [r[0] for r in rows if r[8] > .99]
    assert set(actual_drop) == set(dropped), (actual_drop, dropped)
    return {
        "transaction_rows": len(ids), "transaction_cols": len(transaction_columns),
        "identity_rows": len(identity), "identity_cols": len(identity.columns) + 1,
        "matched": int(ids.isin(identity.index).sum()),
        "fraud": {str(k): int(v) for k, v in label_frame.isFraud.value_counts().sort_index().items()},
        "product": {str(k): int(v) for k, v in label_frame.ProductCD.value_counts().items()},
        "dropped": dropped, "fields": rows, "source": SOURCE,
    }


def save_charts(stats):
    for font in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if font in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams.update({
        "font.size": 13, "text.color": "#162B43", "axes.labelcolor": "#344C62",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.bottom": False, "xtick.color": "#4B6075", "ytick.color": "#344C62",
    })
    FIGURES.mkdir(exist_ok=True)
    n = stats["transaction_rows"]

    def bars(filename, title, subtitle, labels, values, colors, axis_label, show_pct=False):
        fig, ax = plt.subplots(figsize=(10, 5.1), layout="constrained")
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors, height=.60, zorder=3)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.36)
        ax.set_xlabel(axis_label)
        from matplotlib.ticker import StrMethodFormatter
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="x", alpha=.20, zorder=0)
        ax.tick_params(length=0, pad=7)
        for iy, v in zip(y, values):
            label = f"{v:,}  ({v / n:.2%})" if show_pct else f"{v:,}"
            ax.text(v + max(values) * .025, iy, label, va="center", weight="bold")
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=17, weight="bold", pad=17)
        fig.savefig(FIGURES / filename, dpi=200, facecolor="white")
        plt.close(fig)

    for metric, filename, title, names, colors in [
        ("product", "ProductCD五类分布.png", "产品类别分布", {}, ["#216D81"] * 5),
        ("fraud", "isFraud分布.png", "欺诈标签分布", {"0": "非欺诈（0）", "1": "欺诈（1）"},
         ["#216D81", "#C3683A"]),
    ]:
        counts = stats[metric]
        bars(filename, title, f"原始交易表 · {n:,} 笔", [names.get(k, k) for k in counts],
             list(counts.values()), colors, "交易笔数", show_pct=True)
    groups = pd.Series([row[2] for row in stats["fields"]]).value_counts()
    bars("字段分组分布.png", "原始字段分组", "两张原始表 · 共同主键在两表各计一次",
         list(groups.index), groups.tolist(), ["#216D81"] * len(groups), "字段数")

    fields = [row for row in stats["fields"] if row[0] != "TransactionID"]
    values = np.full((14, 31), np.nan)
    values.flat[:len(fields)] = [row[8] * 100 for row in fields]
    fig, ax = plt.subplots(figsize=(11, 6.1), layout="constrained")
    cmap = matplotlib.colormaps["YlOrRd"].with_extremes(bad="#EBEEF2")
    img = ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(0, 31, 5), np.arange(1, 32, 5))
    ax.set_yticks(np.arange(14), [str(i * 31 + 1) for i in range(14)])
    ax.set_xlabel("行内位置（每行 31 列）")
    ax.set_ylabel("本行首字段序号")
    ax.set_title("左连接后的逐字段缺失率\n按原表列顺序排列，先交易字段，再身份字段",
                 loc="left", fontsize=17, weight="bold", pad=14)
    ax.axvline(20.5, ymin=1 / 14, ymax=2 / 14, color="#162B43", linewidth=1.5)
    fig.colorbar(img, ax=ax, fraction=.032, pad=.03, label="缺失率（%）")
    fig.savefig(FIGURES / "合并后缺失率热力图.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    result = read_statistics()
    save_charts(result)
    (HERE / "EDA统计.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
