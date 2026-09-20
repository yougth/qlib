#!/usr/bin/env python3
"""
live/paper_trade.py —— Paper Trading 台账: T+1 模拟成交 + 净值追踪
================================================================================
monthly_signal.py 产出"应该买什么"的指令; 本脚本负责"执行后赚了多少"的记账。

工作流:
  1. monthly_signal.py --strategy VG --capital 50000       → 生成信号 + rebalance.csv
  2. paper_trade.py --strategy VG --fill 2026-07-23        → T+1 收盘价模拟成交, 更新台账
  3. paper_trade.py --strategy VG --report                 → 拉最新价, 算市值/收益率

台账结构 (live/ledger/{STRATEGY}/):
  state.json    : {cash, positions: {inst: {shares, avg_cost}}, nav_init, nav_history: [...]}
  trades.csv    : 全部成交记录 (日期/代码/方向/股数/价格/金额/费用)

口径:
  · 成交价 = T+1 (信号日次一交易日) 收盘价, 与回测一致
  · 交易费用 = 单边 FEE_ROUNDTRIP/2 (买入卖出各 half)
  · 净值 = cash + sum(持仓 × 最新收盘价)
"""
import os, sys, json, argparse, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from qlib.data import D

from core import config, data as datalayer

QUANT_DIR = config.QUANT_DIR
SIG_DIR = os.path.join(QUANT_DIR, "live", "signals")
LEDGER_DIR = os.path.join(QUANT_DIR, "live", "ledger")
HALF_FEE = config.FEE_ROUNDTRIP / 2  # 单边费率


# ─────────────────── 价格获取 ───────────────────

def fetch_t1_close(insts, sig_dt, cal):
    """信号日 T+1 的收盘价 (次一交易日)。若 T+1 不在日历 (如末日), 退到最近可用日。"""
    after = [d for d in cal if d > sig_dt]
    if not after:
        # 末日信号: 用信号日本身 (无 T+1 数据)
        t1 = sig_dt
    else:
        t1 = after[0]

    a_insts = [i for i in insts if not i.startswith("hk")]
    hk_insts = [i for i in insts if i.startswith("hk")]
    out = {}

    if a_insts:
        px = D.features(a_insts, ["$close", "$factor"],
                        start_time=t1 - pd.Timedelta(days=5), end_time=t1)
        if px is not None and len(px) > 0:
            real = px["$close"] / px["$factor"]
            s = real.unstack(level=0).sort_index().ffill().iloc[-1]
            out.update({k: float(v) for k, v in s.items() if np.isfinite(v) and v > 0})

    if hk_insts:
        out.update(_fetch_hk_close(hk_insts, t1))
    return out, t1


def fetch_latest_close(insts):
    """最新收盘价 (A 股 bin + 港股腾讯接口)。"""
    a_insts = [i for i in insts if not i.startswith("hk")]
    hk_insts = [i for i in insts if i.startswith("hk")]
    out = {}

    if a_insts:
        px = D.features(a_insts, ["$close", "$factor"],
                        start_time=pd.Timestamp.today() - pd.Timedelta(days=10),
                        end_time=pd.Timestamp.today())
        if px is not None and len(px) > 0:
            real = px["$close"] / px["$factor"]
            s = real.unstack(level=0).sort_index().ffill().iloc[-1]
            out.update({k: float(v) for k, v in s.items() if np.isfinite(v) and v > 0})

    if hk_insts:
        out.update(_fetch_hk_close(hk_insts, pd.Timestamp.today()))
    return out


def _fetch_hk_close(hk_insts, dt):
    """腾讯接口抓港股不复权收盘价。"""
    import time, requests
    out = {}
    for inst in hk_insts:
        code5 = inst[2:]
        url = ("https://web.ifzq.gtimg.cn/appstock/app/kline/kline?"
               f"param=hk{code5},day,,{dt.strftime('%Y-%m-%d')},5,")
        try:
            r = requests.get(url, timeout=10,
                             headers={"Referer": "https://gu.qq.com",
                                      "User-Agent": "Mozilla/5.0"}).json()
            rows = (r.get("data") or {}).get(f"hk{code5}", {}).get("day") or []
            for row in reversed(rows):
                if pd.Timestamp(row[0]) <= dt:
                    p = float(row[2])
                    if p > 0:
                        out[inst] = p
                    break
        except Exception:
            pass
        time.sleep(0.15)
    return out


def fetch_close_on(insts, dt=None):
    """轻量盯市价: 直接读 parquet 源 (真实价 = close/factor), 不依赖 qlib bin。

    dt=None   → 各票取 parquet 最后一行 (最新收盘价)
    dt=某日期 → 取 <= dt 的最后一行 (停牌沿用此前收盘价)
    返回 (价格dict, 实际行情日期)。
    """
    dt = pd.Timestamp(dt) if dt is not None else None
    tencent_dir = os.path.join(os.path.dirname(QUANT_DIR), "data_cache", "tencent")
    out, dates = {}, []
    hk_insts = []
    for inst in insts:
        if inst.startswith("hk"):
            hk_insts.append(inst)
            continue
        p = os.path.join(tencent_dir, f"{inst.lower()}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["date", "close", "factor"])
        if dt is not None:
            df = df[df["date"] <= dt]
        if len(df) == 0:
            continue
        row = df.iloc[-1]
        fac = row["factor"]
        if pd.isna(fac) or fac <= 0:
            continue
        px = float(row["close"] / fac)
        if np.isfinite(px) and px > 0:
            out[inst] = px
            dates.append(pd.Timestamp(row["date"]))
    if hk_insts:
        ref = dt if dt is not None else pd.Timestamp.today()
        out.update(_fetch_hk_close(hk_insts, ref))
    snap = max(dates).strftime("%Y-%m-%d") if dates else None
    return out, snap


def fetch_realtime_close(a_insts):
    """腾讯实时报价 (盘中). 返回 (价格dict, 是否当日数据)。

    仅用于盘中盯市: 当日净值先按实时价记一条 (标 intraday),
    晚间收盘 mark 同日期覆盖回官方收盘价。
    """
    try:
        import requests
        codes = ",".join(i.lower() for i in a_insts)
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


# ─────────────────── 台账读写 ───────────────────

def ledger_path(strategy):
    d = os.path.join(LEDGER_DIR, strategy)
    os.makedirs(d, exist_ok=True)
    return d


def load_state(strategy):
    p = os.path.join(ledger_path(strategy), "state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"cash": 0.0, "positions": {}, "nav_init": None, "nav_history": []}


def save_state(strategy, state):
    p = os.path.join(ledger_path(strategy), "state.json")
    with open(p, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def append_trades(strategy, trades):
    """trades: list[dict] → 追加到 trades.csv"""
    if not trades:
        return
    p = os.path.join(ledger_path(strategy), "trades.csv")
    df = pd.DataFrame(trades)
    if os.path.exists(p):
        df.to_csv(p, index=False, mode="a", header=False)
    else:
        df.to_csv(p, index=False)


# ─────────────────── --fill: T+1 模拟成交 ───────────────────

def cmd_fill(args):
    strategy = args.strategy
    sig_dt = pd.Timestamp(args.fill)

    # 读取信号
    sig_dir = os.path.join(SIG_DIR, sig_dt.strftime("%Y-%m-%d"), strategy)
    reb_path = os.path.join(sig_dir, "rebalance.csv")
    orders_path = os.path.join(sig_dir, "orders.csv")
    if not os.path.exists(reb_path):
        sys.exit(f"[ERROR] 找不到信号 {sig_dir}/rebalance.csv, 先跑 monthly_signal.py")
    reb = pd.read_csv(reb_path)

    datalayer.init_qlib()
    cal = datalayer.get_calendar()

    # 全部涉及标的
    insts = list(set(reb["instrument"].dropna()))
    px_map, t1_date = fetch_t1_close(insts, sig_dt, cal)
    print(f"[fill] 信号日 {sig_dt.date()} → T+1 成交日 {t1_date.date()}", flush=True)
    print(f"       涉及 {len(insts)} 只标的, 获取价格 {len(px_map)} 只", flush=True)

    state = load_state(strategy)
    if state["nav_init"] is None:
        state["nav_init"] = args.capital
        state["cash"] = args.capital
        print(f"       首次建仓, 初始资金 {args.capital:,.0f} 元", flush=True)

    trades = []
    # 先卖后买 (资金优先回笼)
    sells = reb[reb["delta"] < 0].copy()
    buys = reb[reb["delta"] > 0].copy()
    sells = sells.sort_values("delta")  # 卖最多先执行
    buys = buys.sort_values("delta", ascending=False)

    for _, row in pd.concat([sells, buys]).iterrows():
        inst = row["instrument"]
        delta = int(row["delta"])  # 正=买, 负=卖
        px = px_map.get(inst)
        if px is None or px <= 0:
            print(f"  [SKIP] {inst}: 无 T+1 价格, 跳过", flush=True)
            continue

        if delta < 0:
            # 卖出: shares 取绝对值
            sell_shares = abs(delta)
            pos = state["positions"].get(inst, {})
            held = pos.get("shares", 0)
            sell_shares = min(sell_shares, held)  # 不超持
            if sell_shares <= 0:
                continue
            gross = sell_shares * px
            fee = gross * HALF_FEE
            net = gross - fee
            avg_cost = pos.get("avg_cost", 0)
            pnl = (px - avg_cost) * sell_shares - fee
            state["cash"] += net
            new_held = held - sell_shares
            if new_held <= 0:
                del state["positions"][inst]
            else:
                state["positions"][inst]["shares"] = new_held
            trades.append({
                "date": t1_date.strftime("%Y-%m-%d"), "signal_date": sig_dt.strftime("%Y-%m-%d"),
                "instrument": inst, "action": "sell", "shares": sell_shares,
                "price": px, "gross": round(gross, 2), "fee": round(fee, 2),
                "net": round(net, 2), "pnl": round(pnl, 2),
            })
            print(f"  [SELL] {inst}: {sell_shares} 股 @ {px:.2f} → 净 {net:,.0f} (P&L {pnl:+,.0f})",
                  flush=True)

        elif delta > 0:
            # 买入
            buy_shares = delta
            gross = buy_shares * px
            fee = gross * HALF_FEE
            total_cost = gross + fee
            if total_cost > state["cash"]:
                # 资金不足, 按可用现金量买
                affordable = int(state["cash"] / (px * (1 + HALF_FEE)) // 100) * 100
                if affordable < 100:
                    print(f"  [SKIP] {inst}: 资金不足 (需 {total_cost:,.0f}, 有 {state['cash']:,.0f})",
                          flush=True)
                    continue
                buy_shares = affordable
                gross = buy_shares * px
                fee = gross * HALF_FEE
                total_cost = gross + fee

            state["cash"] -= total_cost
            pos = state["positions"].get(inst, {"shares": 0, "avg_cost": 0})
            old_total = pos["shares"] * pos["avg_cost"]
            new_total = old_total + gross
            new_shares = pos["shares"] + buy_shares
            pos["shares"] = new_shares
            pos["avg_cost"] = new_total / new_shares if new_shares > 0 else 0
            state["positions"][inst] = pos
            trades.append({
                "date": t1_date.strftime("%Y-%m-%d"), "signal_date": sig_dt.strftime("%Y-%m-%d"),
                "instrument": inst, "action": "buy", "shares": buy_shares,
                "price": px, "gross": round(gross, 2), "fee": round(fee, 2),
                "net": round(-total_cost, 2), "pnl": 0,
            })
            print(f"  [BUY]  {inst}: {buy_shares} 股 @ {px:.2f} → 花费 {total_cost:,.0f} (费 {fee:.0f})",
                  flush=True)

    # 记录净值
    nav = _calc_nav(state, px_map)
    nav_entry = {"date": t1_date.strftime("%Y-%m-%d"), "nav": round(nav, 2),
                 "ret_pct": round((nav / state["nav_init"] - 1) * 100, 2) if state["nav_init"] else 0}
    state["nav_history"].append(nav_entry)
    save_state(strategy, state)
    append_trades(strategy, trades)

    print(f"\n[fill] 成交 {len(trades)} 笔 | 净值 {nav:,.0f} "
          f"(初始 {state['nav_init']:,.0f}, 收益 {nav/state['nav_init']-1:+.1%})",
          flush=True)
    print(f"[archive] 台账 → {ledger_path(strategy)}/", flush=True)


def _calc_nav(state, px_map):
    """净值 = 现金 + 持仓 × 最新价"""
    nav = state["cash"]
    for inst, pos in state["positions"].items():
        px = px_map.get(inst, pos.get("avg_cost", 0))
        nav += pos["shares"] * px
    return nav


# ─────────────────── --report: 持仓与收益报表 ───────────────────

def cmd_report(args):
    strategy = args.strategy
    state = load_state(strategy)
    if not state["positions"] and state["nav_init"] is None:
        sys.exit(f"[ERROR] 台账为空, 先跑 --fill 执行首次建仓")

    datalayer.init_qlib()
    insts = list(state["positions"].keys())
    px_map = fetch_latest_close(insts) if insts else {}

    print(f"\n{'='*64}")
    print(f"Paper Trading 报表 | {strategy}")
    print(f"{'='*64}")
    print(f"初始资金: {state['nav_init']:,.0f} 元")
    print(f"现金余额: {state['cash']:,.0f} 元")

    if state["positions"]:
        print(f"\n{'代码':<12} {'持仓':>8} {'成本':>10} {'现价':>10} {'市值':>12} {'P&L':>10} {'收益率':>8}")
        print("-" * 80)
        total_mv = 0
        total_pnl = 0
        for inst, pos in sorted(state["positions"].items(), key=lambda x: -x[1]["shares"]):
            shares = pos["shares"]
            cost = pos["avg_cost"]
            px = px_map.get(inst, cost)
            mv = shares * px
            pnl = (px - cost) * shares
            ret = (px / cost - 1) * 100 if cost > 0 else 0
            total_mv += mv
            total_pnl += pnl
            print(f"{inst:<12} {shares:>8} {cost:>10.2f} {px:>10.2f} {mv:>12,.0f} {pnl:>+10,.0f} {ret:>+7.1f}%")

        nav = state["cash"] + total_mv
        total_ret = (nav / state["nav_init"] - 1) * 100 if state["nav_init"] else 0
        print("-" * 80)
        print(f"{'合计':<12} {'':>8} {'':>10} {'':>10} {total_mv:>12,.0f} {total_pnl:>+10,.0f}")
        print(f"\n净值: {nav:,.0f} 元 | 总收益 {nav - state['nav_init']:+,.0f} 元 ({total_ret:+.1f}%)")
    else:
        nav = state["cash"]
        total_ret = (nav / state["nav_init"] - 1) * 100 if state["nav_init"] else 0
        print(f"\n空仓 | 净值 {nav:,.0f} ({total_ret:+.1f}%)")

    if state["nav_history"]:
        print(f"\n--- 净值历史 ---")
        for h in state["nav_history"][-10:]:
            print(f"  {h['date']}  净值 {h['nav']:>12,.0f}  累计 {h['ret_pct']:+.1f}%")

    # 交易统计
    trades_path = os.path.join(ledger_path(strategy), "trades.csv")
    if os.path.exists(trades_path):
        td = pd.read_csv(trades_path)
        n_buys = len(td[td["action"] == "buy"])
        n_sells = len(td[td["action"] == "sell"])
        total_fee = td["fee"].sum()
        realized_pnl = td[td["action"] == "sell"]["pnl"].sum()
        print(f"\n--- 交易统计 ---")
        print(f"  买入 {n_buys} 笔 | 卖出 {n_sells} 笔 | 总费用 {total_fee:,.0f} 元 | "
              f"已实现 P&L {realized_pnl:+,.0f} 元")

    print(f"\n台账目录: {ledger_path(strategy)}/")


# ─────────────────── --mark: 每日盯市快照 ───────────────────

def cmd_mark(args):
    """按某日收盘价记录净值, 不交易。用于每日盯市或事后补记。

    --mark        → 用持仓最新可得收盘价记一条快照
    --mark DATE   → 补记某日净值 (回溯)
    同一日期重复 mark 会覆盖旧记录 (幂等, 可放心重跑)。
    """
    strategy = args.strategy
    state = load_state(strategy)
    if state["nav_init"] is None:
        sys.exit(f"[ERROR] 台账为空, 先跑 --fill 执行首次建仓")

    insts = list(state["positions"].keys())

    mark_dt = None if args.mark in (None, "today") else pd.Timestamp(args.mark)
    # 轻量盯市: 直读 parquet 源, 不 init qlib / 不依赖 bin, 秒级完成
    px_map, snap_date = fetch_close_on(insts, mark_dt)
    if not px_map:
        sys.exit(f"[ERROR] 无价格数据: 请先更新行情 (tools/update_ohlcv_bs.py) 或检查日期")
    if snap_date is None:
        snap_date = mark_dt.strftime("%Y-%m-%d") if mark_dt is not None \
            else pd.Timestamp.today().strftime("%Y-%m-%d")

    # 盘中模式: parquet 最新收盘仍是旧交易日 → 用腾讯实时价记当日净值 (标 intraday,
    # 晚间收盘 mark 同日期覆盖回官方收盘价)。指定日期的补记不回退实时价。
    intraday = False
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    if mark_dt is None and snap_date < today_str:
        rt_map, is_today = fetch_realtime_close([i for i in insts if not i.startswith("hk")])
        if is_today and rt_map:
            for inst, px in rt_map.items():
                px_map[inst] = px
            snap_date = today_str
            intraday = True
        else:
            print(f"[mark] 提示: 无当日实时/收盘数据, 沿用 {snap_date} 收盘价", flush=True)

    nav = _calc_nav(state, px_map)
    entry = {"date": snap_date, "nav": round(nav, 2),
             "ret_pct": round((nav / state["nav_init"] - 1) * 100, 2)}
    if intraday:
        entry["intraday"] = True

    hist = state["nav_history"]
    if hist and hist[-1]["date"] == snap_date:
        hist[-1] = entry
        action = "覆盖"
    else:
        hist.append(entry)
        action = "新增"
    # 快照价写入 state, 供看板显示浮动盈亏
    state["last_px"] = px_map
    state["last_px_date"] = snap_date
    save_state(strategy, state)

    n_px = len(px_map)
    tag = "盘中实时" if intraday else "收盘"
    print(f"[mark] {snap_date} {action} [{tag}] | 净值 {nav:,.0f} "
          f"(累计 {entry['ret_pct']:+.2f}%) | 价格 {n_px}/{len(insts)} 只", flush=True)


# ─────────────────── main ───────────────────

def main():
    ap = argparse.ArgumentParser(description="Paper Trading 台账")
    ap.add_argument("--strategy", required=True,
                    choices=["ICW_SW", "VG", "VGH"], help="策略名")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--fill", metavar="YYYY-MM-DD",
                     help="执行 T+1 模拟成交 (信号日)")
    sub.add_argument("--report", action="store_true",
                     help="输出持仓与收益报表")
    sub.add_argument("--mark", metavar="YYYY-MM-DD", nargs="?", const="today",
                     help="每日盯市快照 (不交易); 带日期则补记某日净值")
    sub.add_argument("--reset", action="store_true",
                     help="清空台账重新开始")
    ap.add_argument("--capital", type=float, default=50000,
                    help="初始资金 (仅首次 fill 时使用)")
    args = ap.parse_args()

    if args.reset:
        d = ledger_path(args.strategy)
        for fn in ["state.json", "trades.csv"]:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                os.remove(p)
        print(f"[reset] {args.strategy} 台账已清空")
        return

    if args.fill:
        cmd_fill(args)
    elif args.report:
        cmd_report(args)
    elif args.mark:
        cmd_mark(args)


if __name__ == "__main__":
    main()
