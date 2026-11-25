# -*- coding: utf-8 -*-
"""
线C · C2 演示系统原型（Streamlit · 风控工作台版）

设计参考：
- Microsoft Streamlit UI Template（自定义 CSS 的嵌入方式）
- awesome-streamlit-themes 的 Financial 主题（冷静蓝、克制配色、统一令牌）
- ACI 欺诈分析仪表盘复盘（风险色贯穿图表/徽标/表格，概览到告警下钻）

功能保持不变：上传/样例 → 校验 → 产品分类 + 欺诈告警 → 明细 → 导出。
"""
import html
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import predict as core

SAMPLE_CSV = core.ROOT / "线B_数据处理与文档" / "演示交接" / "模型输入样例.csv"
SAMPLE_LABELS = core.ROOT / "线B_数据处理与文档" / "演示交接" / "样例真实标签.csv"

# ---------- 设计令牌 ----------
INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
CANVAS = "#F4F6FA"
PRIMARY = "#1D4ED8"
RED = "#B91C1C"
AMBER = "#B45309"
GREEN = "#15803D"
PALETTE = ["#1D4ED8", "#0E7490", "#B45309", "#15803D", "#7C3AED"]

st.set_page_config(page_title="异常交易监控 | 银行数据管理", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
:root {
  --ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --canvas:#F4F6FA;
  --primary:#1D4ED8; --red:#B91C1C; --amber:#B45309; --green:#15803D;
}
html, body, .stApp, [class*="st-"] {
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
}
.stApp { background: var(--canvas); }
#MainMenu, footer, [data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1220px; padding-top: 0.6rem; padding-bottom: 2rem; }

/* ---------- 顶栏 ---------- */
.topbar {
  display:flex; justify-content:space-between; align-items:center;
  padding: 12px 4px 14px 4px; border-bottom: 1px solid var(--line);
  margin-bottom: 18px;
}
.tb-left { display:flex; align-items:center; gap:11px; }
.mark {
  width:38px; height:38px; border-radius:11px;
  background: linear-gradient(135deg,#1D4ED8,#0E7490);
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 4px 12px rgba(29,78,216,.25);
}
.tb-name { font-size:14.5px; font-weight:700; color:var(--ink); line-height:1.25; }
.tb-sub { font-size:11px; color:var(--muted); margin-top:1px; }
.tb-right { display:flex; align-items:center; gap:8px; }
.top-chip {
  font-size:11.5px; color:#334155; background:#fff; border:1px solid var(--line);
  padding:4px 10px; border-radius:7px; white-space:nowrap;
}
.top-chip.live { color:#0E7490; border-color:#BAE6FD; background:#F0F9FF; font-weight:600; }
.top-chip.live::before {
  content:""; display:inline-block; width:6px; height:6px; border-radius:50%;
  background:#0E7490; margin-right:6px; vertical-align:middle;
}

/* ---------- 页头 ---------- */
.page-head { display:flex; justify-content:space-between; align-items:flex-end; margin: 6px 0 18px 0; }
.eyebrow { font-size:10.5px; letter-spacing:.16em; color:#94A3B8; font-weight:700; }
.page-head h1 { margin:6px 0 4px 0; font-size:26px; font-weight:700; color:var(--ink); letter-spacing:-.01em; }
.page-head p { margin:0; color:var(--muted); font-size:12.5px; }
.head-meta { color:#94A3B8; font-size:11.5px; }

/* ---------- 卡片容器 ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:#fff; border:1px solid var(--line); border-radius:14px;
  box-shadow: 0 1px 2px rgba(15,23,42,.03), 0 10px 24px rgba(15,23,42,.04);
}
.card-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.card-title { font-size:13px; font-weight:700; color:var(--ink); }
.card-hint { font-size:11px; color:#94A3B8; }

/* ---------- KPI ---------- */
.kpi {
  background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px 15px 13px 15px;
  box-shadow: 0 1px 2px rgba(15,23,42,.03);
}
.kpi-top { display:flex; align-items:center; gap:8px; }
.kpi-icon {
  width:28px; height:28px; border-radius:8px; display:flex; align-items:center;
  justify-content:center; flex:0 0 28px;
}
.kpi-label { font-size:11.5px; color:var(--muted); }
.kpi-value { font-size:25px; font-weight:700; color:var(--ink); margin:8px 0 1px 0;
  font-variant-numeric: tabular-nums; letter-spacing:-.02em; }
.kpi-sub { font-size:10.5px; color:#94A3B8; }

/* ---------- 表格 ---------- */
.table-wrap { overflow-x:auto; }
table.grid { width:100%; border-collapse:separate; border-spacing:0; font-size:12px; }
table.grid th {
  background:#F8FAFC; color:#475569; font-weight:600; font-size:11px;
  letter-spacing:.02em; padding:9px 12px; text-align:left;
  border-bottom:1px solid var(--line); white-space:nowrap;
}
table.grid th.num, table.grid td.num { text-align:right; font-variant-numeric: tabular-nums; }
table.grid td { padding:8px 12px; border-bottom:1px solid #F1F5F9; color:#1E293B; }
table.grid tbody tr:hover td { background:#F8FAFC; }
table.grid tr.high td { background:#FEF7F7; }
table.grid tr.warn td { background:#FFFAF2; }
table.grid tr.high:hover td { background:#FDEFEF; }
table.grid tr.warn:hover td { background:#FFF6E8; }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; }
.badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600; }
.b-high { background:#FEE2E2; color:#B91C1C; }
.b-warn { background:#FEF3C7; color:#B45309; }
.b-ok   { background:#DCFCE7; color:#15803D; }
.b-alert{ background:#B91C1C; color:#fff; }
.pchip {
  display:inline-block; min-width:24px; text-align:center; padding:1px 7px;
  border-radius:5px; background:#EEF2FF; color:#1D4ED8; font-weight:700; font-size:11.5px;
}
.scorebar { display:inline-block; width:44px; height:4px; background:#EEF2F6;
  border-radius:2px; margin-left:8px; vertical-align:middle; overflow:hidden; }
.scorebar span { display:block; height:100%; border-radius:2px; }

/* ---------- 仪表盘 ---------- */
.gauge-flex { display:flex; align-items:center; justify-content:center; padding: 4px 0 0 0; }
.gauge-meta { display:flex; justify-content:space-between; font-size:10.5px; color:#94A3B8; margin-top:-6px; }
.detail-foot { margin-top:10px; font-size:12px; color:#334155; }

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }
.side-label { font-size:10.5px; font-weight:700; letter-spacing:.12em; color:#94A3B8; margin:16px 0 8px 0; }
.side-card { border:1px solid var(--line); border-radius:12px; padding:12px 13px; margin-bottom:10px; }
.model-row { display:flex; justify-content:space-between; align-items:center;
  font-size:12px; color:#334155; padding:3px 0; }
.model-row b { color:var(--ink); font-variant-numeric: tabular-nums; }
.tag { background:#EFF6FF; color:#1D4ED8; border-radius:5px; padding:1px 7px; font-size:10px; font-weight:700; }
.side-div { height:1px; background:#F1F5F9; margin:7px 0; }
.side-note { font-size:11px; color:var(--muted); line-height:1.7; }

div.stButton > button, div.stDownloadButton > button {
  border-radius:9px; border:1px solid #D8E0EA; background:#fff; color:#0F172A;
  font-weight:600; font-size:12.5px;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  border-color:#1D4ED8; color:#1D4ED8;
}
div[data-testid="stAlert"] { border-radius:10px; }
[data-testid="stFileUploaderDropzone"] {
  background:#FAFBFC; border:1px dashed #CBD5E1; border-radius:10px;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display:none; }
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChip"]) button[aria-label="Add files"] {
  display:none;
}
[data-testid="stFileChip"] { background:#F1F5F9; border-radius:8px; }
details { border-radius:12px !important; border:1px solid var(--line) !important; background:#fff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return core.ModelBundle()


def query_flag(name):
    try:
        return str(st.query_params.get(name, "")).lower() in ("1", "true", "yes")
    except Exception:
        return False


ICONS = {
    "activity": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "eye": '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z"/><circle cx="12" cy="12" r="3"/>',
    "pie": '<path d="M21.2 15.9A10 10 0 1 1 8 2.8"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
}


def icon(name, color, size=15):
    return ('<svg width="%d" height="%d" viewBox="0 0 24 24" fill="none" '
            'stroke="%s" stroke-width="1.9" stroke-linecap="round" '
            'stroke-linejoin="round">%s</svg>' % (size, size, color, ICONS[name]))


def kpi(label, value, sub, icon_name, fg, bg):
    return ('<div class="kpi"><div class="kpi-top">'
            '<div class="kpi-icon" style="background:%s">%s</div>'
            '<div class="kpi-label">%s</div></div>'
            '<div class="kpi-value">%s</div><div class="kpi-sub">%s</div></div>'
            % (bg, icon(icon_name, fg), label, value, sub))


def section_head(s):
    return ('<div class="page-head" style="margin:26px 0 12px 0">'
            '<div><h1 style="font-size:16px;margin:0">%s</h1></div></div>' % s)


def score_cell(score, threshold):
    color = RED if score >= threshold else (AMBER if score >= 0.5 else GREEN)
    return ('<span class="num">%.4f</span>'
            '<span class="scorebar"><span style="width:%.0f%%;background:%s"></span></span>'
            % (score, min(score, 1.0) * 100, color))


def gauge_svg(score, threshold, color):
    r, cx, cy = 104, 130, 128
    length = math.pi * r
    dash = max(0.0, min(score, 1.0)) * length
    theta = math.pi * threshold
    tx1 = cx - (r - 13) * math.cos(theta)
    ty1 = cy - (r - 13) * math.sin(theta)
    tx2 = cx - (r + 13) * math.cos(theta)
    ty2 = cy - (r + 13) * math.sin(theta)
    return (
        '<svg width="270" height="160" viewBox="0 0 260 160">'
        '<path d="M26 128 A104 104 0 0 1 234 128" fill="none" stroke="#EEF2F6" '
        'stroke-width="16" stroke-linecap="round"/>'
        '<path d="M26 128 A104 104 0 0 1 234 128" fill="none" stroke="%s" '
        'stroke-width="16" stroke-linecap="round" stroke-dasharray="%.1f 999"/>'
        '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0F172A" '
        'stroke-width="2" stroke-linecap="round"/>'
        '<text x="130" y="106" text-anchor="middle" font-size="34" font-weight="700" '
        'fill="%s" font-family="Segoe UI, Microsoft YaHei">%.4f</text>'
        '<text x="130" y="126" text-anchor="middle" font-size="10.5" fill="#94A3B8" '
        'font-family="Segoe UI, Microsoft YaHei">欺诈分数</text>'
        '<text x="22" y="150" font-size="10" fill="#94A3B8">0</text>'
        '<text x="238" y="150" text-anchor="end" font-size="10" fill="#94A3B8">1.0</text>'
        '<text x="130" y="150" text-anchor="middle" font-size="10" fill="#64748B">'
        '阈值 %.2f</text></svg>' % (color, dash, tx1, ty1, tx2, ty2, color, score, threshold))


def product_donut(counts, total):
    df = pd.DataFrame({"类别": list(counts.index), "笔数": [int(v) for v in counts.values]})
    return alt.Chart(df).mark_arc(innerRadius=62, outerRadius=88, cornerRadius=3,
                                  padAngle=0.012).encode(
        theta=alt.Theta("笔数:Q"),
        color=alt.Color("类别:N", scale=alt.Scale(domain=list(counts.index),
                                                  range=PALETTE),
                        legend=alt.Legend(orient="right", title=None,
                                          labelFontSize=11, symbolSize=90)),
        tooltip=[alt.Tooltip("类别:N"), alt.Tooltip("笔数:Q")],
    ).properties(height=240).configure_view(strokeWidth=0)


def score_hist(scores, threshold):
    df = pd.DataFrame({"score": np.asarray(scores, dtype=float)})
    xenc = alt.X("score:Q", bin=alt.Bin(step=0.05, extent=[0, 1]),
                 axis=alt.Axis(title="欺诈分数", labelFontSize=10, titleFontSize=10,
                               grid=False, tickCount=11))
    yenc = alt.Y("count():Q", axis=alt.Axis(title="交易笔数", labelFontSize=10,
                                            titleFontSize=10, grid=True,
                                            gridColor="#F1F5F9"))
    base = alt.Chart(df)
    normal = (base.transform_filter(alt.datum.score < threshold)
              .mark_bar(color=PRIMARY, cornerRadius=2)
              .encode(x=xenc, y=yenc,
                      tooltip=[alt.Tooltip("count():Q", title="笔数")]))
    alert = (base.transform_filter(alt.datum.score >= threshold)
             .mark_bar(color=RED, cornerRadius=2)
             .encode(x=xenc, y=yenc,
                     tooltip=[alt.Tooltip("count():Q", title="笔数")]))
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(
        color=INK, strokeDash=[4, 4], size=1.2).encode(x="x:Q")
    label = alt.Chart(pd.DataFrame({"x": [threshold], "t": ["阈值 %.2f" % threshold]})).mark_text(
        align="left", dx=6, dy=-10, fontSize=10, color=INK).encode(x="x:Q", text="t:N")
    return (normal + alert + rule + label).properties(height=240).configure_view(strokeWidth=0)


def prob_bars(prob_row):
    df = pd.DataFrame({"类别": list(prob_row.index),
                       "概率": [round(float(v) * 100, 2) for v in prob_row.values]})
    base = alt.Chart(df).encode(
        x=alt.X("概率:Q", scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title=None, grid=False, labels=False, ticks=False)),
        y=alt.Y("类别:N", sort=list(prob_row.index),
                axis=alt.Axis(title=None, labelFontSize=12, ticks=False)),
    )
    bars = base.mark_bar(cornerRadius=4, color=PRIMARY, height=18)
    text = base.mark_text(align="left", dx=4, fontSize=11, color="#475569").encode(
        x="概率:Q", text=alt.Text("概率:Q", format=".2f"))
    return (bars + text).properties(height=200).configure_view(strokeWidth=0)


bundle = load_bundle()
threshold = bundle.alert_threshold

# ---------- 顶栏 ----------
st.markdown(
    '<div class="topbar"><div class="tb-left">'
    '<div class="mark"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
    'stroke="#fff" stroke-width="1.8" stroke-linejoin="round">'
    '<path d="M12 2 4 5v6c0 5 3.5 9.5 8 11 4.5-1.5 8-6 8-11V5l-8-3z"/>'
    '<path d="m9 12 2 2 4-4"/></svg></div>'
    '<div><div class="tb-name">银行数据管理平台</div>'
    '<div class="tb-sub">风控分析工作台 · 期中演示</div></div></div>'
    '<div class="tb-right"><span class="top-chip">LightGBM ×2</span>'
    '<span class="top-chip">告警阈值 %.2f</span>'
    '<span class="top-chip live">演示环境</span></div></div>' % threshold,
    unsafe_allow_html=True)

# ---------- 页头 ----------
st.markdown(
    '<div class="page-head"><div>'
    '<div class="eyebrow">RISK MONITORING</div>'
    '<h1>异常交易监控</h1>'
    '<p>上传预处理后的交易数据，自动输出产品分类标签与异常交易告警</p>'
    '</div><div class="head-meta">%d 维特征 · 五分类 + 二分类</div></div>'
    % len(bundle.features), unsafe_allow_html=True)

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown('<div class="side-label">模型</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-card">'
        '<div class="model-row"><span>异常交易检测</span><span class="tag">LightGBM</span></div>'
        '<div class="model-row"><span>交易产品分类</span><span class="tag">LightGBM</span></div>'
        '<div class="side-div"></div>'
        '<div class="model-row"><span>告警阈值</span><b>%.2f</b></div>'
        '<div class="model-row"><span>关注线</span><b>%.2f</b></div>'
        '<div class="model-row"><span>输入特征</span><b>%d 列</b></div>'
        '</div>' % (threshold, bundle.warn_threshold, len(bundle.features)),
        unsafe_allow_html=True)
    st.markdown('<div class="side-label">输入数据</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-card"><div class="side-note">'
        '线A预处理后的 CSV：<br>· 含全部 %d 个特征列<br>· 类别列已编码<br>'
        '· 可附带 isFraud / ProductCD 标签列用于对照<br>· 缺列 / 非数值会明确报错'
        '</div></div>' % len(bundle.features), unsafe_allow_html=True)
    if SAMPLE_CSV.exists():
        st.download_button("下载演示样例 CSV（10 行）", data=SAMPLE_CSV.read_bytes(),
                           file_name="模型输入样例.csv", mime="text/csv",
                           use_container_width=True)
    st.caption("模型与预处理来自线A，样例来自线B。仅用于课程演示。")

# ---------- 数据入口 ----------
with st.container(border=True):
    c1, c2 = st.columns([1.05, 2.1], gap="large")
    with c1:
        use_sample = st.toggle("使用演示样例数据（10 行）",
                               value=query_flag("sample"), key="use_sample")
    with c2:
        if use_sample:
            st.caption("已加载演示样例（5 类产品 × 2 种欺诈标签）；关闭开关可上传文件。")
        else:
            st.caption("支持 UTF-8 / GBK；表头需与模型特征名一致（共 %d 列）。"
                       % len(bundle.features))
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("上传预处理后的交易 CSV（UTF-8 / GBK）", type=["csv"])

df_raw = None
labels = None
source_name = ""
if use_sample:
    if SAMPLE_CSV.exists():
        df_raw = core.read_csv_any(str(SAMPLE_CSV))
        source_name = "演示样例（10 行）"
        if SAMPLE_LABELS.exists():
            labels = core.read_csv_any(str(SAMPLE_LABELS))
    else:
        st.error("未找到演示样例：%s" % SAMPLE_CSV)
elif uploaded is not None:
    try:
        df_raw = core.read_csv_any(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error("读取文件失败：%s" % exc)

if df_raw is None:
    with st.container(border=True):
        st.markdown(
            '<div class="card-title">开始使用</div>'
            '<div class="card-hint" style="margin-top:6px;line-height:1.9">'
            '① 打开上方「使用演示样例数据」开关，或<br>'
            '② 上传一份线A预处理后的交易 CSV（%d 列特征）</div>' % len(bundle.features),
            unsafe_allow_html=True)
    st.stop()

try:
    result, info, proba = core.predict(bundle, df_raw)
except ValueError as exc:
    st.error("输入校验未通过：%s" % exc)
    st.stop()
except Exception as exc:
    st.exception(exc)
    st.stop()

if info["extra_columns"]:
    shown = "、".join(info["extra_columns"][:8]) + (
        " 等" if len(info["extra_columns"]) > 8 else "")
    st.warning("已忽略 %d 个额外列：%s" % (len(info["extra_columns"]), shown))

# ---------- KPI ----------
cols = st.columns(5, gap="small")
cols[0].markdown(kpi("交易总数", "%d" % info["rows"], source_name,
                     "activity", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
cols[1].markdown(kpi("高风险告警", "%d" % info["alert_count"],
                     "分数 ≥ %.2f" % threshold, "alert", RED, "#FEF2F2"),
                 unsafe_allow_html=True)
cols[2].markdown(kpi("关注", "%d" % info["warn_count"],
                     "%.2f ≤ 分数 < %.2f" % (bundle.warn_threshold, threshold),
                     "eye", AMBER, "#FFFBEB"),
                 unsafe_allow_html=True)
cols[3].markdown(kpi("告警率", "%.1f%%" % (100 * info["alert_rate"]),
                     "占全部交易比例", "pie", PRIMARY, "#EFF6FF"),
                 unsafe_allow_html=True)
cols[4].markdown(kpi("平均欺诈分数", "%.4f" % info["avg_score"],
                     "模型输出均值", "target", "#0E7490", "#ECFEFF"),
                 unsafe_allow_html=True)

# ---------- 图表 ----------
st.markdown(section_head("分布概览"), unsafe_allow_html=True)
chart_l, chart_r = st.columns([1.6, 1], gap="large")
with chart_l:
    with st.container(border=True):
        st.markdown('<div class="card-head"><div class="card-title">欺诈分数分布</div>'
                    '<div class="card-hint">红色为告警区间（≥ %.2f）</div></div>' % threshold,
                    unsafe_allow_html=True)
        st.altair_chart(score_hist(result["欺诈分数"].values, threshold),
                        use_container_width=True)
with chart_r:
    with st.container(border=True):
        st.markdown('<div class="card-head"><div class="card-title">产品分类构成</div>'
                    '<div class="card-hint">共 %d 笔</div></div>' % len(result),
                    unsafe_allow_html=True)
        counts = (result["产品类别"].value_counts()
                  .reindex(bundle.classes).fillna(0).astype(int))
        st.altair_chart(product_donut(counts, len(result)), use_container_width=True)

# ---------- 告警列表 ----------
st.markdown(section_head("逐笔告警"), unsafe_allow_html=True)

tool_l, tool_r = st.columns([2.4, 1], gap="small")
with tool_l:
    st.markdown(
        '<span class="badge b-high">高风险 %d</span>&nbsp;&nbsp;'
        '<span class="badge b-warn">关注 %d</span>&nbsp;&nbsp;'
        '<span class="badge b-ok">正常 %d</span>'
        % (info["alert_count"], info["warn_count"],
           info["rows"] - info["alert_count"] - info["warn_count"]),
        unsafe_allow_html=True)
with tool_r:
    st.download_button("导出预测结果 CSV",
                       data=result.to_csv(index=False).encode("utf-8-sig"),
                       file_name="预测结果.csv", mime="text/csv",
                       use_container_width=True)

max_rows = 200
rows_html = []
for _, r in result.head(max_rows).iterrows():
    risk = r["风险等级"]
    dot_color = {"高风险": RED, "关注": AMBER, "正常": GREEN}[risk]
    alert = ('<span class="badge b-alert">告警</span>'
             if r["是否告警"] else '<span style="color:#CBD5E1">—</span>')
    rows_html.append(
        '<tr class="%s"><td class="num">%d</td>'
        '<td><span class="dot" style="background:%s"></span>'
        '<span class="pchip">%s</span></td>'
        '<td class="num">%.4f</td><td class="num">%s</td>'
        '<td><span class="badge %s">%s</span></td>'
        '<td>%s</td><td>%s</td></tr>' % (
            {"高风险": "high", "关注": "warn", "正常": ""}[risk],
            int(r["行号"]), dot_color, html.escape(str(r["产品类别"])),
            r["产品置信度"],
            score_cell(float(r["欺诈分数"]), threshold),
            {"高风险": "b-high", "关注": "b-warn", "正常": "b-ok"}[risk], risk,
            alert, html.escape(str(r["处置建议"]))))

st.markdown(
    '<div class="table-wrap"><table class="grid"><thead><tr>'
    '<th class="num">行号</th><th>产品类别</th><th class="num">置信度</th>'
    '<th class="num">欺诈分数</th><th>风险等级</th><th>告警</th><th>处置建议</th>'
    '</tr></thead><tbody>%s</tbody></table></div>' % "".join(rows_html),
    unsafe_allow_html=True)
if len(result) > max_rows:
    st.caption("共 %d 行，仅展示前 %d 行；完整结果请导出 CSV。" % (len(result), max_rows))

# ---------- 单笔明细 ----------
st.markdown(section_head("单笔分析"), unsafe_allow_html=True)
with st.container(border=True):
    default_idx = int(np.flatnonzero(result["是否告警"].values)[0]) if info["alert_count"] else 0
    csel, _ = st.columns([1, 2.4], gap="large")
    with csel:
        row_id = st.selectbox("选择行号", result["行号"].tolist(), index=default_idx)
    idx = int(row_id) - 1
    score = float(result.loc[idx, "欺诈分数"])
    risk = result.loc[idx, "风险等级"]
    advice = str(result.loc[idx, "处置建议"])
    color = RED if score >= threshold else (AMBER if score >= bundle.warn_threshold else GREEN)
    prob_row = pd.Series(proba[idx], index=bundle.classes)

    d1, d2 = st.columns([1.35, 1], gap="large")
    with d1:
        st.markdown('<div class="card-title">产品类别概率（%）</div>', unsafe_allow_html=True)
        st.altair_chart(prob_bars(prob_row), use_container_width=True)
    with d2:
        st.markdown('<div class="gauge-flex">%s</div>' % gauge_svg(score, threshold, color),
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="detail-foot">风险等级 <span class="badge %s">%s</span>'
            '　处置建议：%s</div>' % (
                {"高风险": "b-high", "关注": "b-warn", "正常": "b-ok"}[risk],
                risk, html.escape(advice)),
            unsafe_allow_html=True)

# ---------- 样例对照 ----------
if labels is not None and len(labels) == len(result):
    st.markdown(section_head("样例核对"), unsafe_allow_html=True)
    cmp = result.copy()
    cmp["真实产品"] = labels["ProductCD"].values
    cmp["真实欺诈"] = labels["isFraud"].values
    ok_prod = int((cmp["产品类别"] == cmp["真实产品"]).sum())
    ok_fraud = int((cmp["是否告警"].astype(int) == cmp["真实欺诈"]).sum())
    with st.container(border=True):
        st.markdown(
            '<span class="badge b-ok">产品命中 %d / %d</span>&nbsp;&nbsp;'
            '<span class="badge b-ok">欺诈判定一致 %d / %d</span>'
            '<span style="color:#94A3B8;font-size:11px;margin-left:10px;">'
            '仅演示核对，不代表整体性能</span>'
            % (ok_prod, len(cmp), ok_fraud, len(cmp)), unsafe_allow_html=True)
        rows = []
        for _, r in cmp.iterrows():
            pm_ok = r["产品类别"] == r["真实产品"]
            fm_ok = int(r["是否告警"]) == int(r["真实欺诈"])
            rows.append(
                '<tr><td class="num">%d</td><td>%s</td><td>%s</td>'
                '<td>%s</td><td class="num">%.4f</td><td>%s</td><td>%s</td></tr>' % (
                    int(r["行号"]), html.escape(str(r["产品类别"])),
                    html.escape(str(r["真实产品"])),
                    '<span class="badge b-ok">✓</span>' if pm_ok
                    else '<span class="badge b-high">✗</span>',
                    r["欺诈分数"], "是" if int(r["真实欺诈"]) else "否",
                    '<span class="badge b-ok">✓</span>' if fm_ok
                    else '<span class="badge b-high">✗</span>'))
        st.markdown(
            '<div class="table-wrap"><table class="grid"><thead><tr>'
            '<th class="num">行号</th><th>预测产品</th><th>真实产品</th><th>产品</th>'
            '<th class="num">欺诈分数</th><th>真实欺诈</th><th>判定</th>'
            '</tr></thead><tbody>%s</tbody></table></div>' % "".join(rows),
            unsafe_allow_html=True)

st.markdown(
    '<div style="margin-top:28px;padding-top:14px;border-top:1px solid #E2E8F0;'
    'display:flex;justify-content:space-between;color:#94A3B8;font-size:10.5px;">'
    '<span>银行数据管理平台 · 风控分析工作台</span>'
    '<span>模型与预处理来自线A · 样例来自线B · 阈值 %.2f</span></div>' % threshold,
    unsafe_allow_html=True)
