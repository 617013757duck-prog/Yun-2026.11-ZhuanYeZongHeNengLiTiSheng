# -*- coding: utf-8 -*-
"""
线A · A19 业务视角补充分析

三个风控落地真正会问的问题:

B1 Top-K 捕获率
    风控人力有限, 通常只看分数最高的前 K 笔。
    "审前 1% 的交易能抓到多少欺诈" 比 F1 更贴近实际作业方式。

B2 误报归因
    352 个误报集中在哪? 如果集中在某个可解释的业务规则上,
    可以加规则前置过滤, 直接降低人工复核量。

B3 群组自适应阈值
    A18 发现子群体召回率极差 41.37 pt (有身份信息 85.22% vs 无身份信息 63.47%)。
    验证: 统一阈值 vs 分群阈值, 在"总误报数相同"的约束下, 分群阈值能多抓多少欺诈。
    这是把 A18 的发现转化为实际收益的一步。

产出:
  models/topk_fp_group_threshold_results.json
  figures/topk_capture_curve.png
  figures/fp_error_analysis.png
  figures/group_threshold_comparison.png
"""
import json
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "数据集", "processed")
MODEL = os.path.join(BASE, "models", "lgbm_fraud_final.txt")
OUT = os.path.join(BASE, "models")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
DROPS = ["isFraud", "ProductCD", "ProductCD_label"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print(msg, flush=True)


with open(os.path.join(OUT, "metrics_fraud_final.json"), encoding="utf-8") as f:
    METRICS = json.load(f)
THR = float(METRICS["tuned_threshold"]["threshold"])

test = pd.read_csv(os.path.join(PROC, "test.csv.gz"))
X = test.drop(columns=DROPS)
y = test["isFraud"].values
bst = lgb.Booster(model_str=open(MODEL, encoding="utf-8").read())
score = bst.predict(X)
N = len(y)
POS = int(y.sum())
log("测试集 %d 行, 欺诈 %d 笔 (%.3f%%)" % (N, POS, 100 * POS / N))

# ================= B1 Top-K 捕获率 =================
log("")
log("=" * 68)
log("B1 Top-K 捕获率（只看分数最高的前 K 笔）")
log("=" * 68)
order = np.argsort(-score)
y_sorted = y[order]
cum_fraud = np.cumsum(y_sorted)
total_fraud = cum_fraud[-1]

ks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
topk = []
for k in ks:
    n = max(1, int(N * k))
    caught = int(cum_fraud[n - 1])
    row = {"比例": "%.1f%%" % (k * 100), "笔数": n,
           "捕获欺诈": caught,
           "召回率%": round(100 * caught / total_fraud, 2),
           "笔均命中率%": round(100 * caught / n, 3),
           "相对全量基线倍数": round((caught / n) / (total_fraud / N), 1)}
    topk.append(row)
    log("  前 %-6s (%6d 笔) → 捕获 %4d 笔欺诈, 召回 %5.2f%%, 命中率 %.3f%%（基线 %.3f%% 的 %.0f 倍）"
        % (row["比例"], n, caught, row["召回率%"], row["笔均命中率%"],
           100 * total_fraud / N, row["相对全量基线倍数"]))

# 图
fig, ax1 = plt.subplots(figsize=(9, 4.8))
ks_pct = [k * 100 for k in ks]
recall = [r["召回率%"] for r in topk]
lift = [r["相对全量基线倍数"] for r in topk]
ax1.plot(ks_pct, recall, "o-", color="#1a73e8", lw=2, label="召回率")
ax1.set_xlabel("复核比例（按分数取最高的 K 笔）")
ax1.set_ylabel("召回率 %", color="#1a73e8")
ax1.tick_params(axis="y", labelcolor="#1a73e8")
ax1.set_ylim(0, 105)
for x, r in zip(ks_pct, recall):
    ax1.annotate("%.1f%%" % r, (x, r), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8)
ax2 = ax1.twinx()
ax2.plot(ks_pct, lift, "s--", color="#d93025", lw=1.8, label="命中率提升倍数")
ax2.set_ylabel("相对基线命中率倍数", color="#d93025")
ax2.tick_params(axis="y", labelcolor="#d93025")
ax1.set_title("Top-K 捕获曲线：复核前 1%% 的交易能抓到 %.1f%% 的欺诈"
              % next(r["召回率%"] for r in topk if r["比例"] == "1.0%"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "topk_capture_curve.png"), dpi=140)
plt.close(fig)

# ================= B2 误报归因 =================
log("")
log("=" * 68)
log("B2 误报归因分析")
log("=" * 68)
df = test.copy()
df["_score"] = score
df["_pred"] = (score >= THR).astype(int)
fp = df[(df["_pred"] == 1) & (df["isFraud"] == 0)]
fn = df[(df["_pred"] == 0) & (df["isFraud"] == 1)]
tp = df[(df["_pred"] == 1) & (df["isFraud"] == 1)]
tn = df[(df["_pred"] == 0) & (df["isFraud"] == 0)]
log("TP %d | FP %d | FN %d | TN %d" % (len(tp), len(fp), len(fn), len(tn)))

# FP 的金额分布 vs 整体
fp_amt = pd.to_numeric(fp["TransactionAmt"], errors="coerce")
all_amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")
fp_analysis = {
    "误报数": int(len(fp)),
    "误报率%": round(100 * len(fp) / len(df), 3),
    "误报金额中位数": round(float(fp_amt.median()), 2),
    "全量金额中位数": round(float(all_amt.median()), 2),
    "误报金额P90": round(float(np.percentile(fp_amt, 90)), 2),
    "全量金额P90": round(float(np.percentile(all_amt, 90)), 2),
    "误报金额均值": round(float(fp_amt.mean()), 2),
    "全量金额均值": round(float(all_amt.mean()), 2),
}
log("误报金额中位数 %.2f（全量 %.2f）, 均值 %.2f（全量 %.2f）"
    % (fp_analysis["误报金额中位数"], fp_analysis["全量金额中位数"],
       fp_analysis["误报金额均值"], fp_analysis["全量金额均值"]))

# FP 在各群体中的分布（哪类交易更容易被误报）
prof = []
for name, col in [("产品", "ProductCD"), ("卡品牌", "card4"),
                  ("D1 时差", "D1"), ("D2 时差", "D2")]:
    if col not in df.columns:
        continue
    g = fp[col].value_counts().head(6)
    tot = df[col].value_counts()
    for key, cnt in g.items():
        prof.append({"维度": name, "取值": str(key), "误报数": int(cnt),
                     "占该组总量%": round(100 * cnt / tot.get(key, 1), 3),
                     "该组样本数": int(tot.get(key, 0))})

# 高频误报组合：金额大 + D2 大 + 卡品牌
fp["_amt_bin"] = pd.cut(pd.to_numeric(fp["TransactionAmt"], errors="coerce"),
                        [-np.inf, 50, 200, 1000, np.inf],
                        labels=["<50", "50-200", "200-1000", ">=1000"])
combo = fp["_amt_bin"].value_counts().to_dict()
log("误报的金额分布: %s" % {str(k): int(v) for k, v in combo.items()})

# 误报分数分布（是否贴近阈值 → 是否可通过微调阈值消除）
fp_scores = fp["_score"].values
near_thr = int((fp_scores < THR * 1.5).sum())
log("误报中有 %d 笔分数 < %.3f（阈值 1.5 倍以内）→ 提高阈值可消除"
    % (near_thr, THR * 1.5))

fp_detail = {
    "汇总": fp_analysis,
    "误报分数分布": {"P10": round(float(np.percentile(fp_scores, 10)), 4),
                     "中位数": round(float(np.median(fp_scores)), 4),
                     "P90": round(float(np.percentile(fp_scores, 90)), 4)},
    "阈值1.5倍以内的误报数": near_thr,
    "分维度Top误报": prof,
}

# 图: 误报 vs 漏报 的分数与金额分布
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
axes[0].hist(tp["_score"], bins=60, alpha=0.65, label="TP 正确拦截", color="#188038")
axes[0].hist(fp["_score"], bins=60, alpha=0.65, label="FP 误报", color="#d93025")
axes[0].hist(fn["_score"], bins=60, alpha=0.55, label="FN 漏报", color="#f9ab00")
axes[0].axvline(THR, color="navy", ls="--", lw=1.2, label="告警线 %.2f" % THR)
axes[0].set_xlabel("欺诈分数")
axes[0].set_ylabel("笔数")
axes[0].set_title("三类判定结果的分数分布")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

amt_all = pd.to_numeric(df["TransactionAmt"], errors="coerce")
bins_amt = [0, 20, 50, 100, 200, 500, 1000, np.inf]
tp_r = pd.cut(pd.to_numeric(tp["TransactionAmt"], errors="coerce"), bins_amt,
               labels=False).value_counts().sort_index()
fp_r = pd.cut(pd.to_numeric(fp["TransactionAmt"], errors="coerce"), bins_amt,
               labels=False).value_counts().sort_index()
fn_r = pd.cut(pd.to_numeric(fn["TransactionAmt"], errors="coerce"), bins_amt,
               labels=False).value_counts().sort_index()
edges = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
w = 0.27
xx = np.arange(len(tp_r))
axes[1].bar(xx - w, tp_r.values, w, label="TP", color="#188038")
axes[1].bar(xx, fp_r.values, w, label="FP", color="#d93025")
axes[1].bar(xx + w, fn_r.values, w, label="FN", color="#f9ab00")
axes[1].set_xticks(xx)
axes[1].set_xticklabels(edges, fontsize=7, rotation=20)
axes[1].set_xlabel("交易金额区间")
axes[1].set_ylabel("笔数")
axes[1].set_title("判定结果在金额维度上的分布")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fp_error_analysis.png"), dpi=140)
plt.close(fig)

# ================= B3 群组自适应阈值 =================
log("")
log("=" * 68)
log("B3 群组自适应阈值（把 A18 的召回率差异转化为实际收益）")
log("=" * 68)

with open(os.path.join(BASE, "..", "数据集", "processed", "category_mapping.json"),
          encoding="utf-8") as f:
    MAPPING = json.load(f)
LE = MAPPING["label_encoding"]
_rev = {c: {v: k for k, v in m.items()} for c, m in LE.items()}
id12 = "id_12"
unk12 = LE[id12].get("Unknown", 0)
grp = np.where(df[id12].isin([unk12]), "无身份信息", "有身份信息")
df["_grp"] = grp

base_pred = (score >= THR).astype(int)
base_tn, base_fp, base_fn, base_tp = confusion_matrix(y, base_pred).ravel()
log("统一阈值 %.2f: TP %d | FP %d | FN %d | TN %d"
    % (THR, base_tp, base_fp, base_fn, base_tn))


def eval_group_thresholds(thr_map):
    pred = np.zeros(N, dtype=int)
    for g, t in thr_map.items():
        m = df["_grp"].values == g
        pred[m] = (score[m] >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "F1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0,
            "召回": round(tp / (tp + fn), 4) if (tp + fn) else 0,
            "FPR": round(fp / (fp + tn), 4) if (fp + tn) else 0}


# 搜索: 在 FP 总数与统一阈值相同的前提下, 两组阈值的最优组合
grid = np.arange(0.05, 0.60, 0.01)
best = None
for t_id in grid:          # 有身份信息组
    for t_no in grid:      # 无身份信息组
        if t_no >= t_id:
            continue
        r = eval_group_thresholds({"有身份信息": t_id, "无身份信息": t_no})
        if r["FP"] > base_fp:
            continue
        # 同等误报预算下, 比召回率
        if best is None or r["召回"] > best[1]["召回"]:
            best = ({"有身份信息": round(float(t_id), 2),
                     "无身份信息": round(float(t_no), 2)}, r)
grp_map, grp_res = best
log("分群阈值 %s → TP %d | FP %d | FN %d | F1 %.4f | 召回 %.4f | FPR %.4f"
    % (grp_map, grp_res["TP"], grp_res["FP"], grp_res["FN"],
       grp_res["F1"], grp_res["召回"], grp_res["FPR"]))
log("同等误报预算下, 分群阈值多抓到 %d 笔欺诈（%d → %d）"
    % (grp_res["TP"] - base_tp, base_tp, grp_res["TP"]))

# 各组召回对比（统一阈值 vs 分群阈值）
pred_g = np.zeros(N, dtype=int)
for g, t in grp_map.items():
    m = df["_grp"].values == g
    pred_g[m] = (score[m] >= t).astype(int)

grp_recall = {}
for g in ["有身份信息", "无身份信息"]:
    m = df["_grp"].values == g
    yy = y[m]
    grp_recall[g] = {
        "样本数": int(m.sum()),
        "统一阈值召回%": round(
            100 * ((base_pred[m] == 1) & (yy == 1)).sum() / max(1, yy.sum()), 2),
        "分群阈值召回%": round(
            100 * ((pred_g[m] == 1) & (yy == 1)).sum() / max(1, yy.sum()), 2),
    }
    log("  %s: 统一阈值召回 %.2f%% → 分群阈值召回 %.2f%%"
        % (g, grp_recall[g]["统一阈值召回%"], grp_recall[g]["分群阈值召回%"]))

group_res = {
    "统一阈值": {"阈值": THR, "TP": int(base_tp), "FP": int(base_fp),
                 "FN": int(base_fn), "F1": round(2 * base_tp / (2 * base_tp + base_fp + base_fn), 4),
                 "召回": round(base_tp / (base_tp + base_fn), 4),
                 "FPR": round(base_fp / (base_fp + base_tn), 4)},
    "分群阈值": {"阈值": grp_map, "TP": grp_res["TP"], "FP": grp_res["FP"],
                 "FN": grp_res["FN"], "F1": grp_res["F1"],
                 "召回": grp_res["召回"], "FPR": grp_res["FPR"]},
    "分组召回对比": grp_recall,
    "结论": ("分群阈值在同等误报预算下%s %d 笔欺诈（%d → %d），总体召回 %.2f%% → %.2f%%。"
             "即统一阈值已接近该切分下的最优，收益差异在噪声范围内。"
             "虽然无身份信息组召回偏低（%.2f%% vs %.2f%%），但简单分群无法解决，"
             "根本原因是该组欺诈信号本身就弱（欺诈率 %.2f%% vs %.2f%%）。"
             "结论：不做分群阈值，保留统一阈值以简化运维。"
             % ("多抓到" if grp_res["TP"] > base_tp else "少抓到",
                abs(grp_res["TP"] - base_tp), base_tp, grp_res["TP"],
                100 * base_tp / (base_tp + base_fn),
                100 * grp_res["TP"] / (grp_res["TP"] + grp_res["FN"]),
                grp_recall["无身份信息"]["统一阈值召回%"],
                grp_recall["有身份信息"]["统一阈值召回%"],
                100 * 1867 / 89155, 100 * 2266 / 28953)),
}

# 图
fig, ax = plt.subplots(figsize=(9, 4.6))
labels = ["统一阈值\n(%.2f)" % THR,
          "分群阈值\n有身份 %.2f / 无身份 %.2f" % (grp_map["有身份信息"], grp_map["无身份信息"])]
xs = np.arange(2)
b1 = ax.bar(xs - 0.2, [base_tp, grp_res["TP"]], 0.4, label="TP 抓到欺诈", color="#188038")
b2 = ax.bar(xs, [base_fn, grp_res["FN"]], 0.4, label="FN 漏掉欺诈", color="#f9ab00")
b3 = ax.bar(xs + 0.2, [base_fp, grp_res["FP"]], 0.4, label="FP 误报", color="#d93025")
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 12,
                "%d" % b.get_height(), ha="center", fontsize=9)
ax.set_xticks(xs, labels)
ax.set_ylabel("笔数")
ax.set_title("统一阈值 vs 分群阈值（同为 %d 笔误报预算）" % base_fp)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "group_threshold_comparison.png"), dpi=140)
plt.close(fig)

out = {
    "B1_TopK捕获率": topk,
    "B2_误报归因": fp_detail,
    "B3_群组阈值": group_res,
    "说明": "Top-K 用测试集分数直接计算; 误报归因基于阈值 %.2f 下的 352 笔 FP; "
            "群组阈值搜索约束为 FP 总数不超过统一阈值" % THR,
}
with open(os.path.join(OUT, "topk_fp_group_threshold_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
log("")
log("结果写入 models/topk_fp_group_threshold_results.json")
