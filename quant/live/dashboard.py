#!/usr/bin/env python3
"""
Paper Trading 看板：持仓 + 净值 + 交易记录
启动：streamlit run live/dashboard.py
"""
import os, sys, json, glob, time, subprocess
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 路径
QUANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB_DIR = os.path.dirname(QUANT_DIR)
NEW_DIR = os.path.join(QLIB_DIR, "new_quant")
LEDGER_DIR = os.path.join(QUANT_DIR, "live", "ledger")
SIG_DIR = os.path.join(QUANT_DIR, "live", "signals")
ETF_DIR = os.path.join(NEW_DIR, "data", "etf")
LOF_DIR = os.path.join(NEW_DIR, "data", "lof")
PT_SIG_DIR = os.path.join(NEW_DIR, "outputs", "pt_signals")
TENCENT_DIR = os.path.join(QLIB_DIR, "data_cache", "tencent")
# 单元清单: 旧三腿 (paper_trade.py 维护) + 新三腿 (experiments/live_pt.py 维护) + 三组合 (combo_track.py 加权)
OLD_LEGS = ["ICW_SW", "VG", "VGH"]
NEW_LEGS = ["M4", "LOF", "TREND"]
COMBOS = ["COMBO_A", "COMBO_B", "COMBO_C"]
LEGS6 = OLD_LEGS + NEW_LEGS
STRATS = LEGS6 + COMBOS
IS_COMBO = set(COMBOS)
STRAT_LABELS = {
    "ICW_SW": "ICW双周+熊市切VGH（消融版）",
    "VG": "VG Top10（纯规则价值+盈利）",
    "VGH": "VGH Top10（纯规则结构化剥离）",
    "M4": "M4排雷质量动量(~30只等权)",
    "LOF": "LOF折价Top10等权",
    "TREND": "跨资产趋势ETF(逆波动率)",
    "COMBO_A": "【组合】A进攻-数字王(ICW60/M440)",
    "COMBO_B": "【组合】B进攻+LOF(ICW54/M436/LOF10)",
    "COMBO_C": "【组合】C风平x1.33(六腿风险平价+融资垫)",
}

# ─────────────── 股票名称映射 ───────────────
@st.cache_data(ttl=3600)
def load_name_map():
    """从最近一次信号的 orders.csv 加载 代码→名称 映射"""
    name_map = {}
    # 遍历所有信号日，找最新的 orders.csv
    for sig_dir in sorted(glob.glob(os.path.join(SIG_DIR, "*")), reverse=True):
        for strat in STRATS:
            orders_p = os.path.join(sig_dir, strat, "orders.csv")
            if os.path.exists(orders_p):
                try:
                    df = pd.read_csv(orders_p)
                    # orders.csv 没有 name 列，从 rebalance.csv 或 meta.json 获取
                    meta_p = os.path.join(sig_dir, strat, "meta.json")
                    if os.path.exists(meta_p):
                        # 尝试从信号输出日志或持仓文件获取名称
                        pass
                except:
                    pass
    # 常见股票名称硬编码（从 2026-09-09 信号提取）
    COMMON_NAMES = {
        "SH601601": "中国太保", "SH601336": "新华保险", "SH600507": "方大特钢",
        "SH601665": "齐鲁银行", "SZ000719": "中原传媒", "SH600839": "四川长虹",
        "SH600262": "北方股份", "SH603816": "顾家家居", "SZ000028": "国药一致",
        "SZ002039": "黔源电力", "SZ002294": "信立泰", "SZ002763": "汇洁股份",
        "SZ002415": "海康威视",
    }
    name_map.update(COMMON_NAMES)
    # 港股名称: 港股不在 fcf_result.csv 中, 从港股清单 (tools/fetch_hk_data.py) 构建
    try:
        from tools.fetch_hk_data import HK_STOCKS
        for c, n in HK_STOCKS:
            name_map[f"hk{c.split(':')[0].zfill(5)}"] = n
    except Exception:
        pass
    return name_map

NAME_MAP = load_name_map()

def get_stock_name(code):
    """代码转名称，未知代码返回原代码"""
    return NAME_MAP.get(code, code)


# ─────────────── 实时价格 + 雪球链接 ───────────────

def xueqiu_url(inst):
    """雪球个股页: A股带市场前缀大写(SH600519), 港股/B股去掉前两位(00700)"""
    u = inst.upper()
    if u.startswith("HK") or (u[:2] in ("SH", "SZ") and len(u) == 8 and u[2] in "92"):
        return f"https://xueqiu.com/S/{u[2:]}"
    return f"https://xueqiu.com/S/{u}"


def _src_last_close(unit, inst):
    """行情源最新真实收盘价: TREND→ETF parquet, LOF→场内CSV, 其他→tencent parquet"""
    try:
        if unit == "TREND":
            df = pd.read_parquet(os.path.join(ETF_DIR, f"{inst.lower()}.parquet"),
                                 columns=["date", "close", "factor"])
        elif unit == "LOF":
            df = pd.read_csv(os.path.join(LOF_DIR, f"{inst[2:]}_px.csv"),
                             usecols=["date", "close"]).assign(factor=1.0)
        else:
            df = pd.read_parquet(os.path.join(TENCENT_DIR, f"{inst.lower()}.parquet"),
                                 columns=["date", "close", "factor"])
        if len(df) == 0:
            return None, None
        row = df.iloc[-1]
        fac = row["factor"]
        if pd.isna(fac) or fac <= 0:
            return None, None
        return float(row["close"] / fac), str(pd.Timestamp(row["date"]).date())
    except Exception:
        return None, None


@st.cache_data(ttl=30)
def load_live_prices(unit, insts):
    """当天最新价: 优先腾讯实时报价(盘中), 失败/无数据回退行情源最新收盘。

    返回 {inst: (价格, 来源标签)}。
    """
    out = {}
    a_insts = [i for i in insts if not i.startswith("hk")]
    hk_insts = [i for i in insts if i.startswith("hk")]
    # 1) 腾讯实时报价 (A股+港股, 一次批量请求)
    got = set()
    if a_insts or hk_insts:
        try:
            import requests
            codes = ",".join([i.lower() for i in a_insts + hk_insts])
            r = requests.get(f"https://qt.gtimg.cn/q={codes}", timeout=6,
                             headers={"Referer": "https://gu.qq.com",
                                      "User-Agent": "Mozilla/5.0"})
            r.encoding = "gbk"
            now = pd.Timestamp.now()
            today = now.strftime("%Y%m%d")
            today_slash = now.strftime("%Y/%m/%d")
            for seg in r.text.split(";"):
                seg = seg.strip()
                if not seg.startswith("v_") or "=" not in seg:
                    continue
                sym = seg[2:seg.index("=")].upper()
                try:
                    payload = seg[seg.index('"') + 1:seg.rindex('"')]
                    f = payload.split("~")
                    px = float(f[3]) if len(f) > 4 and f[3] else 0.0
                except Exception:
                    continue
                if px > 0:
                    tag = "实时"
                    t30 = f[30] if len(f) > 30 else ""
                    if t30.startswith(today):
                        tag = f"盘中 {t30[8:10]}:{t30[10:12]}"
                    elif t30.startswith(today_slash) and " " in t30:
                        # 港股时间格式: 2026/09/17 16:08:13
                        tag = f"盘中 {t30.split(' ')[-1][:5]}"
                    out[sym] = (px, tag)
                    got.add(sym)
        except Exception:
            pass
    # 2) 回退: 行情源最新收盘 (实时失败的兜底; 港股无源数据则缺价)
    for inst in insts:
        if inst in got:
            continue
        px, dt = _src_last_close(unit, inst)
        if px is not None:
            out[inst] = (px, f"{dt}收盘" if dt else "收盘")
    return out


# ─────────────── 页面加载自动同步 (刷新页面 = 数据全链路刷新) ───────────────

def ensure_fresh_data():
    """与 daily_mark.sh 同链路: 补持仓行情(股票/ETF/LOF) → 六腿补历史漏记+今日盯市
    → 组合加权。返回 (状态文本, 是否有新净值写入)。"""
    env = {**os.environ, "PYTHONPATH": QUANT_DIR}
    env_nq = {**os.environ, "PYTHONPATH": NEW_DIR}
    notes, new_data = [], False
    try:
        # 1) 汇总持仓代码与各腿最新净值日期
        syms, etf_syms, lof_syms, last_dates = set(), set(), set(), {}
        for strat in STRATS:
            p = os.path.join(LEDGER_DIR, strat, "state.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                st_ = json.load(f)
            for s in st_.get("positions", {}):
                if s.lower().startswith("hk"):
                    continue
                if strat == "TREND":
                    etf_syms.add(s)
                elif strat == "LOF":
                    lof_syms.add(s)
                else:
                    syms.add(s)
            hist = st_.get("nav_history", [])
            if strat in LEGS6:
                last_dates[strat] = hist[-1]["date"] if hist else None
        # 2) 行情刷新: 股票(baostock增量) + ETF(腾讯增量) + LOF(新浪覆盖)
        if syms:
            sym_file = os.path.join("/tmp", "dash_held_syms.txt")
            with open(sym_file, "w") as fp:
                fp.write("\n".join(sorted(s.lower() for s in syms)))
            r = subprocess.run([sys.executable, "tools/update_ohlcv_bs.py",
                                "--syms", f"@{sym_file}"],
                               cwd=QUANT_DIR, env=env,
                               capture_output=True, text=True, timeout=300)
            notes.append("股票行情已更新" if r.returncode == 0 else "股票行情更新失败")
        if etf_syms:
            r = subprocess.run([sys.executable, "tools/fetch_etf_ohlcv.py", "--update"],
                               cwd=NEW_DIR, env=env_nq,
                               capture_output=True, text=True, timeout=300)
            notes.append("ETF行情已更新" if r.returncode == 0 else "ETF行情更新失败")
        if lof_syms:
            r = subprocess.run([sys.executable, "tools/fetch_lof.py", "--update"]
                               + sorted(lof_syms),
                               cwd=NEW_DIR, env=env_nq,
                               capture_output=True, text=True, timeout=600)
            notes.append("LOF行情已更新" if r.returncode == 0 else "LOF行情更新失败")
        # 3) 交易日历 (取任一股票 parquet 的 date 列)
        cal = []
        if syms:
            p0 = os.path.join(TENCENT_DIR, f"{sorted(syms)[0].lower()}.parquet")
            if os.path.exists(p0):
                cal = [d.strftime("%Y-%m-%d")
                       for d in pd.read_parquet(p0, columns=["date"])["date"]]
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        # 4) 旧三腿盯市: 补历史漏记 + 今日 (今日无收盘数据时自动回退实时价)
        for strat in OLD_LEGS:
            if strat not in last_dates:
                continue
            last = last_dates[strat]
            for d in [x for x in cal if (last is None or x > last) and x < today]:
                subprocess.run([sys.executable, "live/paper_trade.py",
                                "--strategy", strat, "--mark", d],
                               cwd=QUANT_DIR, env=env,
                               capture_output=True, text=True, timeout=120)
            if (last or "1970-01-01") < today:
                subprocess.run([sys.executable, "live/paper_trade.py",
                                "--strategy", strat, "--mark"],
                               cwd=QUANT_DIR, env=env,
                               capture_output=True, text=True, timeout=120)
        # 5) 新三腿盯市: 同链路 (experiments.live_pt)
        for leg in NEW_LEGS:
            if leg not in last_dates:
                continue
            last = last_dates[leg]
            for d in [x for x in cal if (last is None or x > last) and x < today]:
                subprocess.run([sys.executable, "-m", "experiments.live_pt",
                                "--leg", leg, "--mark", d],
                               cwd=NEW_DIR, env=env_nq,
                               capture_output=True, text=True, timeout=120)
            if (last or "1970-01-01") < today:
                subprocess.run([sys.executable, "-m", "experiments.live_pt",
                                "--leg", leg, "--mark"],
                               cwd=NEW_DIR, env=env_nq,
                               capture_output=True, text=True, timeout=120)
        # 6) 组合加权 (必须在六腿盯市之后)
        r = subprocess.run([sys.executable, "live/combo_track.py", "--mark"],
                           cwd=QUANT_DIR, env=env,
                           capture_output=True, text=True, timeout=120)
        notes.append("组合已更新" if r.returncode == 0 else "组合更新失败")
        # 7) 是否有新净值写入
        for strat, last in last_dates.items():
            p = os.path.join(LEDGER_DIR, strat, "state.json")
            with open(p) as f:
                hist = json.load(f).get("nav_history", [])
            if hist and hist[-1]["date"] != last:
                new_data = True
                notes.append(f"{strat} 净值已更新")
    except Exception as e:
        notes.append(f"同步异常: {e}")
    ts = pd.Timestamp.now().strftime("%H:%M:%S")
    return f"{' | '.join(notes) if notes else '无变更'} · {ts}", new_data


# 进程级 memo: 3 分钟内重复加载/交互不再重复同步 (streamlit 脚本重跑共享进程全局变量)
_SYNC_MEMO = {"t": 0.0, "msg": "", "new": False}


def auto_sync():
    """页面加载时调用: 返回 (状态文本, 本次是否实际执行了同步)"""
    now = time.time()
    if now - _SYNC_MEMO["t"] > 180:
        msg, new = ensure_fresh_data()
        _SYNC_MEMO.update(t=now, msg=msg, new=new)
        return msg, True
    return _SYNC_MEMO["msg"], False


st.set_page_config(page_title="V18 Paper Trading 看板", layout="wide")
st.title("📈 V18 量化策略 Paper Trading 看板")

# ─────────────── 侧边栏 ───────────────
with st.sidebar:
    st.header("策略选择")
    sel = st.selectbox("策略", STRATS, format_func=lambda x: STRAT_LABELS[x])
    # 刷新页面即自动同步: 补行情 + 补齐漏掉的交易日 + 今日盯市 (3分钟内不重复)
    with st.spinner("🔄 正在同步最新行情与盯市..."):
        sync_msg, sync_ran = auto_sync()
    if sync_ran and _SYNC_MEMO["new"]:
        st.cache_data.clear()  # 台账有新净值, 清掉旧缓存重读
    st.caption(f"🔄 自动同步: {sync_msg}")
    st.divider()
    st.caption(f"台账目录: `{LEDGER_DIR}/{sel}/`")
    st.caption(f"信号目录: `{SIG_DIR}/`")

# ─────────────── 加载数据 ───────────────
@st.cache_data(ttl=60)
def load_state(strat):
    p = os.path.join(LEDGER_DIR, strat, "state.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_trades(strat):
    p = os.path.join(LEDGER_DIR, strat, "trades.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(ttl=60)
def load_signals(strat):
    """加载所有信号日的 holdings_log"""
    out = []
    for sig_dir in sorted(glob.glob(os.path.join(SIG_DIR, "*"))):
        meta_p = os.path.join(sig_dir, strat, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p) as f:
                meta = json.load(f)
            out.append(meta)
    return out

@st.cache_data(ttl=60)
def load_pt_signal(leg):
    """新三腿最新目标持仓信号 (new_quant/outputs/pt_signals/{leg}_latest.json)"""
    p = os.path.join(PT_SIG_DIR, f"{leg}_latest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

state = load_state(sel)
trades = load_trades(sel)
signals = load_signals(sel)

if state is None:
    st.error("台账为空：腿请先建仓（旧三腿 `paper_trade.py --fill` / 新三腿 "
             "`experiments/live_pt.py --fill`），组合请 `combo_track.py --init`")
    st.stop()

# ─────────────── 顶部指标卡 ───────────────
col1, col2, col3, col4 = st.columns(4)
nav_init = state.get("nav_init", 0)
nav_hist = state.get("nav_history", [])
last_nav = nav_hist[-1]["nav"] if nav_hist else nav_init
last_ret = (last_nav / nav_init - 1) * 100 if nav_init else 0

col1.metric("初始资金", f"{nav_init:,.0f} 元")
last_intraday = nav_hist[-1].get("intraday") if nav_hist else False
last_date = nav_hist[-1]["date"] if nav_hist else ""
col2.metric("当前净值", f"{last_nav:,.0f} 元", f"{last_ret:+.2f}%")
if sel in IS_COMBO:
    wsum = sum(state.get("weights", {}).values())
    col3.metric("权重和", f"{wsum:.2f}" + (" (含融资垫)" if wsum > 1.05 else ""))
    col4.metric("组成腿数", f"{len(state.get('weights', {}))} 条")
else:
    col3.metric("现金余额", f"{state.get('cash', 0):,.0f} 元")
    col4.metric("持仓数", f"{len(state.get('positions', {}))} 只")
if last_date:
    tag = " (盘中实时, 晚间自动更新为收盘价)" if last_intraday else " (收盘价)"
    st.caption(f"净值最新记录: {last_date}{tag}")

# ─────────────── 净值曲线 ───────────────
st.subheader("净值走势")
if nav_hist:
    df_nav = pd.DataFrame(nav_hist)
    df_nav["date"] = pd.to_datetime(df_nav["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_nav["date"], y=df_nav["nav"],
        mode="lines+markers", name="净值",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6)))
    fig.add_hline(y=nav_init, line_dash="dash", line_color="gray",
                  annotation_text="初始资金")
    fig.update_layout(
        xaxis_title="日期", yaxis_title="净值 (元)",
        hovermode="x unified", height=350, margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("暂无净值历史，等待下次调仓")

# ─────────────── 当前持仓 / 组合构成 ───────────────
positions = state.get("positions", {})
if sel in IS_COMBO:
    st.subheader("组合构成（腿权重 × 腿日收益加权）")
    leg_rows = []
    for leg, w in state.get("weights", {}).items():
        leg_state = load_state(leg)
        if leg_state is None or not leg_state.get("nav_history"):
            leg_rows.append({"腿": leg, "权重": f"{w:.2%}", "腿最新净值": "-",
                             "腿累计": "-", "腿最新日期": "-"})
            continue
        lh = leg_state["nav_history"][-1]
        leg_rows.append({"腿": leg, "权重": f"{w:.2%}", "腿最新净值": f"{lh['nav']:,.0f}",
                         "腿累计": f"{lh['ret_pct']:+.2f}%", "腿最新日期": lh["date"]})
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)
    st.caption("组合净值 = Σ 腿权重 × 腿日收益 (日频加权, 与 six_leg_combo 回测同口径); "
               "基点仅记账锚点, 组合层不做独立交易, 明细见各腿页")
    st.stop()  # 组合层无持仓/交易/信号明细
st.subheader("当前持仓")
if positions:
    live_map = load_live_prices(sel, list(positions.keys()))
    pos_rows = []
    total_mv = 0.0
    for inst, pos in sorted(positions.items(), key=lambda x: -x[1]["shares"] * x[1]["avg_cost"]):
        shares = pos["shares"]
        cost = pos["avg_cost"]
        px, tag = live_map.get(inst, (state.get("last_px", {}).get(inst, cost), "台账"))
        mv = shares * px
        total_mv += mv
        pnl = (px - cost) * shares
        name = get_stock_name(inst)
        pos_rows.append({
            "名称": name, "链接": xueqiu_url(inst), "代码": inst, "持仓": shares,
            "成本价": f"{cost:.2f}", "现价": f"{px:.2f}", "价格时间": tag,
            "市值": f"{mv:,.0f}", "浮动盈亏": f"{pnl:+,.0f}",
            "收益率": f"{(px / cost - 1) * 100:+.1f}%",
            "占比": f"{mv / last_nav * 100:.1f}%"})

    # HTML 表格渲染: 名称为雪球链接 (LinkColumn 的 display_text 引用隐藏列有 bug,
    # 显示为字面量 {名称}, 故改用 HTML <a> 标签, 100% 可控)
    TH = ("padding:6px 10px;text-align:right;font-weight:600;"
          "border-bottom:2px solid rgba(128,128,128,.35);white-space:nowrap;")
    TD = ("padding:6px 10px;text-align:right;"
          "border-bottom:1px solid rgba(128,128,128,.12);white-space:nowrap;")
    TDL = TD + "text-align:left;"

    def _pnl_html(v):
        try:
            neg = float(str(v).replace(",", "").replace("+", "")) < 0
        except Exception:
            return ""
        return f"color:{'#ef5350' if neg else '#26a69a'};"

    hdr = (f"<tr><th style='{TH}text-align:left;'>名称</th><th style='{TH}text-align:left;'>代码</th>"
           f"<th style='{TH}'>持仓</th><th style='{TH}'>成本价</th><th style='{TH}'>现价</th>"
           f"<th style='{TH}'>价格时间</th><th style='{TH}'>市值</th><th style='{TH}'>浮动盈亏</th>"
           f"<th style='{TH}'>收益率</th><th style='{TH}'>占比</th></tr>")
    rows = []
    for r in pos_rows:
        link = (f"<a href='{r['链接']}' target='_blank' style='text-decoration:none;'>"
                f"{r['名称']}</a>")
        rows.append(
            f"<tr><td style='{TDL}'>{link}</td><td style='{TDL}'>{r['代码']}</td>"
            f"<td style='{TD}'>{r['持仓']}</td><td style='{TD}'>{r['成本价']}</td>"
            f"<td style='{TD}'>{r['现价']}</td><td style='{TD}'>{r['价格时间']}</td>"
            f"<td style='{TD}'>{r['市值']}</td>"
            f"<td style='{TD}{_pnl_html(r['浮动盈亏'])}'>{r['浮动盈亏']}</td>"
            f"<td style='{TD}{_pnl_html(r['收益率'])}'>{r['收益率']}</td>"
            f"<td style='{TD}'>{r['占比']}</td></tr>")
    st.markdown(
        "<table style='border-collapse:collapse;font-size:14px;'><thead>"
        + hdr + "</thead><tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True)
    live_nav = total_mv + state.get("cash", 0)
    st.caption(f"按当天最新股价估算: 持仓市值 {total_mv:,.0f} 元 + 现金 "
               f"{state.get('cash', 0):,.0f} 元 = **{live_nav:,.0f} 元** "
               f"({(live_nav / nav_init - 1) * 100:+.2f}%) | 官方盯市净值见上方曲线", unsafe_allow_html=True)

    # 持仓饼图
    fig_pie = go.Figure(data=[go.Pie(
        labels=[r["名称"] for r in pos_rows],
        values=[float(r["市值"].replace(",", "")) for r in pos_rows],
        hole=0.3, textinfo="label+percent")])
    fig_pie.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("空仓")

# ─────────────── 交易记录 ───────────────
st.subheader("交易记录")
if not trades.empty:
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.sort_values("date", ascending=False)
    # 添加名称列
    trades_display = trades.copy()
    trades_display["名称"] = trades_display["instrument"].apply(get_stock_name)
    # 调整列顺序
    cols = ["date", "signal_date", "名称", "instrument", "action", "shares", "price", "gross", "fee", "net", "pnl"]
    cols = [c for c in cols if c in trades_display.columns]
    trades_display = trades_display[cols]
    st.dataframe(trades_display, use_container_width=True, hide_index=True)

    # 交易统计
    n_buys = len(trades[trades["action"] == "buy"])
    n_sells = len(trades[trades["action"] == "sell"])
    total_fee = trades["fee"].sum()
    realized = trades[trades["action"] == "sell"]["pnl"].sum() if "pnl" in trades.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("买入笔数", n_buys)
    c2.metric("卖出笔数", n_sells)
    c3.metric("总手续费", f"{total_fee:,.0f} 元")
    c4.metric("已实现P&L", f"{realized:+,.0f} 元")
else:
    st.info("暂无交易记录")

# ─────────────── 信号历史 ───────────────
st.subheader("信号历史")
if sel in NEW_LEGS:
    sig = load_pt_signal(sel)
    if sig:
        w = sig.get("weights", {})
        st.dataframe(pd.DataFrame([{
            "信号期": sig.get("signal_asof"), "标的数": len(w),
            "权重和": f"{sum(w.values()):.4f}",
            "费用率(往返)": f"{sig.get('fee_rt', 0):.3%}"}]),
            use_container_width=True, hide_index=True)
        st.caption(f"信号文件: `{os.path.join(PT_SIG_DIR, sel + '_latest.json')}`")
    else:
        st.info("暂无信号文件")
elif signals:
    sig_rows = []
    for meta in signals:
        sig_dt = meta.get("signal_date", "")
        regime = meta.get("regime", "")
        n_pos = meta.get("summary", {}).get("n_positions", 0)
        invested = meta.get("summary", {}).get("invested", 0)
        sig_rows.append({
            "信号日": sig_dt, "市场环境": regime,
            "持仓数": n_pos, "投入金额": f"{invested:,.0f}"})
    df_sig = pd.DataFrame(sig_rows)
    st.dataframe(df_sig, use_container_width=True, hide_index=True)
else:
    st.info("暂无信号历史")

# ─────────────── 回测对照 (如有 NAV 文件) ───────────────
nav_csv = os.path.join(QUANT_DIR, "quick_backtest_nav.csv")
if os.path.exists(nav_csv):
    st.subheader("回测 NAV 对照")
    df_bt = pd.read_csv(nav_csv, index_col=0, parse_dates=True)
    if sel in df_bt.columns:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=df_bt.index, y=df_bt[sel],
            mode="lines", name=f"{sel} 回测NAV",
            line=dict(color="#d62728", width=2)))
        fig_bt.update_layout(
            xaxis_title="日期", yaxis_title="净值",
            hovermode="x unified", height=300,
            margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.info(f"回测 NAV 文件中无 {sel} 列")
