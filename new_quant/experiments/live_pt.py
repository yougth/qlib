#!/usr/bin/env python3
"""experiments.live_pt —— 新腿(M4/LOF/TREND) Paper Trading 台账(信号/成交/盯市)
================================================================================
与 qlib/quant/live/paper_trade.py 共用台账根 (qlib/quant/live/ledger/{leg}/),
state.json / trades.csv 结构兼容; 各腿口径与回测一致:
  · M4    : 双正池 score_mv 打分, buffered(12,25,44) + 剔14 → ~30只等权; 往返0.4%
  · LOF   : 月末折价最深 Top10 等权; 往返0.1%
  · TREND : 11只ETF 12-1动量>0 逆波动率配权; 往返0.1%

成交价: A股/ETF 用真实价 close/factor (与 paper_trade 同口径), LOF 用场内收盘价。
口径提示: 分红/利息未入账 (与现有三策略台账一致), 净值略低于全收益口径。

用法:
  python3 -m experiments.live_pt --leg TREND --signal            # 生成最新一期目标持仓
  python3 -m experiments.live_pt --leg TREND --fill 2026-09-18   # 按该日收盘价建仓/换仓
  python3 -m experiments.live_pt --leg TREND --mark 2026-09-18   # 按该日收盘价盯市
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402

LEDGER_DIR = os.path.join(config.QLIB_DIR, "quant", "live", "ledger")
SIG_DIR = os.path.join(config.NEW_DIR, "outputs", "pt_signals")
ETF_END = "2026-12-31"          # ETF 面板取数上界 (覆盖最新数据)

LEGS = {
    "M4":    {"label": "M4排雷质量动量(~30只等权)", "fee_rt": config.FEE_RT_STOCK,
              "capital": 500000},
    "LOF":   {"label": "LOF折价Top10等权", "fee_rt": 0.001, "capital": 50000},
    "TREND": {"label": "跨资产趋势ETF(逆波动率)", "fee_rt": config.FEE_RT_ETF,
              "capital": 200000},   # 20万: 国债ETF一手1.4万, 5万装不下81%权重
}


# ─────────────────── 信号生成 (最新一期目标权重) ───────────────────

def sig_m4():
    """最新一期 M4 目标持仓 {SH600007: w} (复刻 g_ml_m4_dump 链路)"""
    from core import data
    from strategies import stock_strategies as ss
    from experiments.lowatt_opt import buffered
    from experiments.lowatt_mv import load_mv_panel, score_mv
    TOPK, ENTRY, EXIT, KDROP = 30, 12, 25, 14
    cal = data.load_calendar(config.BT_START, config.BT_END)
    panels = data.load_panels(config.BT_START, config.BT_END)
    fin = data.load_financials()
    mvp = load_mv_panel()
    signals = data.month_end_signals(cal, config.BT_START, config.BT_END)
    sigs = []
    for t, e in signals:
        cand = ss.candidates_at(panels, fin, t)
        s = score_mv(mvp, fin, t, cand).dropna()
        sigs.append((t, e, s.sort_values(ascending=False)))   # buffered按传入序排名,必须降序
    with open(os.path.join(config.OUT_DIR, "g_ml_preds.pkl"), "rb") as f:
        preds, _, _ = pickle.load(f)
    e2t = {e: t for t, e, _ in sigs}

    def apply_drop(rebals):
        out = []
        for e, w in rebals:
            t = e2t.get(e)
            if t in preds and len(w) > TOPK:
                p = preds[t].reindex(list(w)).dropna()
                nd = min(len(w) - TOPK, KDROP)
                bad = set(p.nsmallest(nd).index)
                keep = {s: 1.0 / (len(w) - nd) for s in w if s not in bad}
                out.append((e, keep))
            else:
                out.append((e, w))
        return out

    M4 = apply_drop(buffered(sigs, ENTRY, EXIT, TOPK + KDROP))
    e, w = M4[-1]
    return e, w


def sig_lof():
    """最新一期 LOF 目标持仓 {161725: w} (月末折价最深 Top10)"""
    from experiments.lof_research import load_panels as lof_panels
    from experiments.lof_strategy import build as lof_build
    P, A, N = lof_panels()
    Nf = N.reindex(P.index).ffill()
    D = P / Nf.shift(1) - 1
    me = P.resample("ME").last()
    D_me = D.resample("ME").last()
    days = A.notna().resample("ME").sum()
    amt20 = A.rolling(20).mean().resample("ME").last()
    tgt = lof_build(D_me, me, amt20, days, "topN", n=10)
    t = sorted(tgt)[-1]
    return t, tgt[t]


def sig_trend():
    """最新一期 TREND 目标持仓 {sh513100: w} (逆波动率, 含现金腿=权重和<1)"""
    from strategies.trend_etf import load_etf_panel, trend_rebalances
    px = load_etf_panel(config.BT_START, ETF_END)
    reb = trend_rebalances(px)
    e, w = reb[-1]
    return e, w


def gen_sig(leg):
    if leg == "M4":
        return sig_m4()
    if leg == "LOF":
        return sig_lof()
    return sig_trend()


# ─────────────────── 代码格式: 信号源 ↔ 台账 ───────────────────

def to_ledger(leg, code):
    """信号代码 → 台账代码 (统一 SH/SZ 前缀大写)"""
    if leg == "TREND":                 # sh513100 → SH513100
        return code.upper()
    if leg == "LOF":                   # 161725 → SZ161725
        return ("SH" if code[0] == "5" else "SZ") + code
    return code                        # M4 已是 SH600007


# ─────────────────── 价格获取 (真实可成交价, <= dt 最后一行) ───────────────────

def _last_row(df, dt):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])          # ETF parquet date 为字符串
    df = df[df["date"] <= pd.Timestamp(dt)]
    return df.iloc[-1] if len(df) else None


def get_px(leg, syms, dt):
    """{台账代码: 价格}; dt=None 取最新一行"""
    dt = pd.Timestamp(dt) if dt is not None else pd.Timestamp("2099-12-31")
    out = {}
    for sym in syms:
        if leg == "M4":
            p = os.path.join(config.TENCENT_DIR, f"{sym.lower()}.parquet")
            if not os.path.exists(p):
                continue
            row = _last_row(pd.read_parquet(p, columns=["date", "close", "factor"]), dt)
            if row is not None and not pd.isna(row["factor"]) and row["factor"] > 0:
                out[sym] = float(row["close"] / row["factor"])
        elif leg == "LOF":
            code = sym[2:]
            p = os.path.join(config.NEW_DIR, "data", "lof", f"{code}_px.csv")
            if not os.path.exists(p):
                continue
            row = _last_row(pd.read_csv(p, parse_dates=["date"]), dt)
            if row is not None and np.isfinite(row["close"]) and row["close"] > 0:
                out[sym] = float(row["close"])
        else:                          # TREND: SH513100 → sh513100.parquet
            p = os.path.join(config.ETF_DIR, f"{sym.lower()}.parquet")
            if not os.path.exists(p):
                continue
            row = _last_row(pd.read_parquet(p, columns=["date", "close", "factor"]), dt)
            if row is not None and not pd.isna(row["factor"]) and row["factor"] > 0:
                out[sym] = float(row["close"] / row["factor"])
    return out


def fetch_realtime_close(syms):
    """腾讯实时报价 (A股/ETF/LOF 场内), 返回 (价格dict, 是否当日数据)。

    与 quant/live/paper_trade.py 同口径: 仅用于盘中盯市 —— 当日净值先按实时价
    记一条 (标 intraday), 晚间收盘 mark 同日期覆盖回官方收盘价。
    """
    try:
        import requests
        codes = ",".join(s.lower() for s in syms)
        r = requests.get(f"https://qt.gtimg.cn/q={codes}", timeout=6,
                         headers={"Referer": "https://gu.qq.com",
                                  "User-Agent": "Mozilla/5.0"})
        r.encoding = "gbk"
        today = pd.Timestamp.now().strftime("%Y%m%d")
        out, is_today = {}, False
        for seg in r.text.split(";"):
            seg = seg.strip()
            if not seg.startswith("v_") or "=" not in seg:
                continue
            sym = seg[2:seg.index("=")].upper()
            try:
                f = seg[seg.index('"') + 1:seg.rindex('"')].split("~")
                px = float(f[3]) if len(f) > 4 and f[3] else 0.0
            except Exception:
                continue
            if px > 0:
                out[sym] = px
                if len(f) > 30 and f[30] and f[30].startswith(today):
                    is_today = True
        return out, is_today
    except Exception:
        return {}, False


def _snap_dates(leg, syms):
    """各标的行情数据的最新日期 {sym: 'YYYY-MM-DD'}"""
    dts = {}
    for sym in syms:
        if leg == "M4":
            p = os.path.join(config.TENCENT_DIR, f"{sym.lower()}.parquet")
        elif leg == "LOF":
            p = os.path.join(config.NEW_DIR, "data", "lof", f"{sym[2:]}_px.csv")
        else:
            p = os.path.join(config.ETF_DIR, f"{sym.lower()}.parquet")
        if not os.path.exists(p):
            continue
        col = pd.read_parquet(p, columns=["date"]) if p.endswith(".parquet") \
            else pd.read_csv(p, usecols=["date"])
        dts[sym] = pd.Timestamp(col["date"].iloc[-1]).strftime("%Y-%m-%d")
    return dts


def px_snap_date(leg, syms):
    """各标的行情数据的最新日期 (核对用)"""
    dts = _snap_dates(leg, syms)
    return max(dts.values()) if dts else None


# ─────────────────── 台账读写 ───────────────────

def leg_dir(leg):
    d = os.path.join(LEDGER_DIR, leg)
    os.makedirs(d, exist_ok=True)
    return d


def load_state(leg):
    p = os.path.join(leg_dir(leg), "state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"cash": 0.0, "positions": {}, "nav_init": None, "nav_history": []}


def save_state(leg, state):
    with open(os.path.join(leg_dir(leg), "state.json"), "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def append_trades(leg, trades):
    if not trades:
        return
    p = os.path.join(leg_dir(leg), "trades.csv")
    df = pd.DataFrame(trades)
    if os.path.exists(p):
        df.to_csv(p, index=False, mode="a", header=False)
    else:
        df.to_csv(p, index=False)


def calc_nav(state, px):
    nav = state["cash"]
    for sym, pos in state["positions"].items():
        nav += pos["shares"] * px.get(sym, pos.get("avg_cost", 0))
    return nav


# ─────────────────── --signal: 生成并落盘目标持仓 ───────────────────

def cmd_signal(args):
    leg = args.leg
    asof, w_raw = gen_sig(leg)
    w = {to_ledger(leg, c): float(x) for c, x in w_raw.items()}
    print(f"\n[{leg}] {LEGS[leg]['label']} | 信号期 {pd.Timestamp(asof).date()} | "
          f"标的 {len(w)} 只 | 权重和 {sum(w.values()):.3f}", flush=True)
    for c, x in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<10} {x:>7.3f}", flush=True)
    os.makedirs(SIG_DIR, exist_ok=True)
    fp = os.path.join(SIG_DIR, f"{leg}_latest.json")
    with open(fp, "w") as f:
        json.dump({"leg": leg, "signal_asof": str(pd.Timestamp(asof).date()),
                   "weights": w, "fee_rt": LEGS[leg]["fee_rt"]}, f, indent=2)
    print(f"[+] 信号已存 {fp}", flush=True)
    return w


# ─────────────────── --fill: 按指定日期收盘价建仓/换仓 ───────────────────

def cmd_fill(args):
    leg = args.leg
    dt = pd.Timestamp(args.fill)
    spec = LEGS[leg]
    half_fee = spec["fee_rt"] / 2.0

    fp = os.path.join(SIG_DIR, f"{leg}_latest.json")
    if not os.path.exists(fp):
        sys.exit(f"[ERROR] 无信号 {fp}, 先跑 --signal")
    with open(fp) as f:
        sig = json.load(f)
    w = sig["weights"]

    state = load_state(leg)
    if state["nav_init"] is None:
        cap = args.capital or spec["capital"]
        state["nav_init"] = cap
        state["cash"] = cap
        state["leg"] = leg
        state["label"] = spec["label"]
        print(f"[fill] 首次建仓 | 初始资金 {cap:,.0f} 元 | 信号期 {sig['signal_asof']}", flush=True)

    px = get_px(leg, set(list(w)) | set(state["positions"]), dt)
    snap = px_snap_date(leg, list(state["positions"]) or list(w))
    print(f"[fill] {leg} 成交日 {dt.date()} (行情最新 {snap}) | 涉及 "
          f"{len(set(list(w)) | set(state['positions']))} 只, 有价 {len(px)} 只", flush=True)

    # 按当前总资产计算目标手数
    nav = calc_nav(state, px)
    tgt_shares = {}
    for sym, wi in w.items():
        p = px.get(sym)
        if not p or p <= 0:
            print(f"  [SKIP] {sym}: 无价格", flush=True)
            continue
        lot = int(wi * nav / p // 100) * 100
        if lot > 0:
            tgt_shares[sym] = lot

    # 先卖后买
    trades = []
    sell_list = []
    for sym, pos in state["positions"].items():
        tgt = tgt_shares.get(sym, 0)
        if tgt < pos["shares"]:
            sell_list.append((sym, pos["shares"] - tgt))
    for sym, n in sell_list:
        p = px.get(sym)
        if not p or n <= 0:
            continue
        gross = n * p
        fee = gross * half_fee
        pos = state["positions"][sym]
        pnl = (p - pos["avg_cost"]) * n - fee
        state["cash"] += gross - fee
        left = pos["shares"] - n
        if left <= 0:
            del state["positions"][sym]
        else:
            pos["shares"] = left
        trades.append({"date": dt.strftime("%Y-%m-%d"), "instrument": sym,
                       "action": "sell", "shares": n, "price": round(p, 4),
                       "gross": round(gross, 2), "fee": round(fee, 2),
                       "net": round(gross - fee, 2), "pnl": round(pnl, 2)})
        print(f"  [SELL] {sym}: {n} @ {p:.3f} → 净 {gross - fee:,.0f} (P&L {pnl:+,.0f})",
              flush=True)

    for sym, tgt in sorted(tgt_shares.items(), key=lambda kv: -kv[1]):
        cur = state["positions"].get(sym, {}).get("shares", 0)
        n = tgt - cur
        if n <= 0:
            continue
        p = px.get(sym)
        if not p or p <= 0:
            continue
        cost = n * p * (1 + half_fee)
        if cost > state["cash"]:
            n = int(state["cash"] / (p * (1 + half_fee)) // 100) * 100
            if n <= 0:
                print(f"  [SKIP] {sym}: 资金不足", flush=True)
                continue
            cost = n * p * (1 + half_fee)
        fee = n * p * half_fee
        state["cash"] -= cost
        pos = state["positions"].get(sym, {"shares": 0, "avg_cost": 0})
        old_val = pos["shares"] * pos["avg_cost"]
        pos["shares"] = cur + n
        pos["avg_cost"] = (old_val + n * p) / pos["shares"]
        state["positions"][sym] = pos
        trades.append({"date": dt.strftime("%Y-%m-%d"), "instrument": sym,
                       "action": "buy", "shares": n, "price": round(p, 4),
                       "gross": round(n * p, 2), "fee": round(fee, 2),
                       "net": round(-cost, 2), "pnl": 0})
        print(f"  [BUY]  {sym}: {n} @ {p:.3f} → 花费 {cost:,.0f}", flush=True)

    nav = calc_nav(state, px)
    entry = {"date": dt.strftime("%Y-%m-%d"), "nav": round(nav, 2),
             "ret_pct": round((nav / state["nav_init"] - 1) * 100, 2)}
    hist = state["nav_history"]
    if hist and hist[-1]["date"] == entry["date"]:
        hist[-1] = entry
    else:
        hist.append(entry)
    state["last_px"] = {k: round(v, 4) for k, v in px.items()}
    state["last_px_date"] = snap
    state["signal_asof"] = sig["signal_asof"]
    save_state(leg, state)
    append_trades(leg, trades)

    inv = nav - state["cash"]
    print(f"\n[fill] 成交 {len(trades)} 笔 | 净值 {nav:,.0f} (初始 {state['nav_init']:,.0f}) | "
          f"仓位 {inv / nav:.1%} | 现金 {state['cash']:,.0f}", flush=True)
    print(f"[archive] 台账 → {leg_dir(leg)}/", flush=True)


# ─────────────────── --mark: 盯市快照 ───────────────────

def cmd_mark(args):
    leg = args.leg
    state = load_state(leg)
    if state["nav_init"] is None:
        sys.exit(f"[ERROR] 台账为空, 先跑 --fill")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    dt = None if args.mark in (None, "today") else pd.Timestamp(args.mark)
    syms = list(state["positions"])
    px = get_px(leg, syms, dt)
    snap = px_snap_date(leg, syms) if dt is None else dt.strftime("%Y-%m-%d")
    # 盘中模式: 任一持仓行情还停在旧交易日 → 全部改用腾讯实时价, 保证同一快照时点
    # (标 intraday, 晚间收盘 mark 同日期覆盖回官方收盘价)。指定日期的补记不回退实时价。
    intraday = False
    snap_dts = _snap_dates(leg, syms)
    if dt is None and any(d < today for d in snap_dts.values()):
        rt, is_today = fetch_realtime_close([s for s in syms if not s.startswith("hk")])
        if is_today and rt:
            px.update(rt)
            snap = today
            intraday = True
        else:
            print(f"[mark] 提示: 无当日实时/收盘数据, 沿用 {snap} 收盘价", flush=True)
    if not px:
        sys.exit(f"[ERROR] 无价格数据")
    nav = calc_nav(state, px)
    entry = {"date": snap, "nav": round(nav, 2),
             "ret_pct": round((nav / state["nav_init"] - 1) * 100, 2)}
    if intraday:
        entry["intraday"] = True
    hist = state["nav_history"]
    if hist and hist[-1]["date"] == snap:
        hist[-1] = entry
        action = "覆盖"
    else:
        hist.append(entry)
        action = "新增"
    state["last_px"] = {k: round(v, 4) for k, v in px.items()}
    state["last_px_date"] = snap
    save_state(leg, state)
    tag = "盘中实时" if intraday else "收盘"
    print(f"[mark] {leg} {snap} {action} [{tag}] | 净值 {nav:,.0f} "
          f"(累计 {entry['ret_pct']:+.2f}%) | 价格 {len(px)}/{len(syms)} 只", flush=True)


# ─────────────────── main ───────────────────

def main():
    ap = argparse.ArgumentParser(description="新腿 Paper Trading 台账")
    ap.add_argument("--leg", required=True, choices=list(LEGS.keys()))
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--signal", action="store_true", help="生成并落盘最新一期目标持仓")
    sub.add_argument("--fill", metavar="YYYY-MM-DD", help="按该日收盘价建仓/换仓")
    sub.add_argument("--mark", metavar="YYYY-MM-DD", nargs="?", const="today",
                     help="盯市快照 (不交易)")
    sub.add_argument("--reset", action="store_true", help="清空台账重新开始")
    ap.add_argument("--capital", type=float, default=0, help="初始资金 (仅首次 fill)")
    args = ap.parse_args()

    if args.reset:
        d = leg_dir(args.leg)
        for fn in ["state.json", "trades.csv"]:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                os.remove(p)
        print(f"[reset] {args.leg} 台账已清空")
        return

    if args.signal:
        cmd_signal(args)
    elif args.fill:
        cmd_fill(args)
    elif args.mark:
        cmd_mark(args)


if __name__ == "__main__":
    main()
