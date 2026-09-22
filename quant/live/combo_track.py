#!/usr/bin/env python3
"""
live/combo_track.py —— 组合层加权净值跟踪 (COMBO_A / COMBO_B / COMBO_C)
================================================================================
三组合 = 六腿台账净值的固定权重加权 (日频再平衡口径, 与 six_leg_combo.py 回测一致):

  COMBO_A 进攻-数字王 : ICW_SW 60% / M4 40%                        (回测 23.86%/1.25)
  COMBO_B 进攻+LOF    : ICW_SW 54% / M4 36% / LOF 10%              (回测 23.61%/1.29)
  COMBO_C 风平x1.33   : 六腿逆波动率归一化 x1.33 (权重和 1.33 = 33% 融资垫)
                        ICW/VG/VGH/M4/TREND/LOF=14.5/19.7/23.0/11.4/50.4/14.0
                        (回测 19.49% / Sharpe 1.57 / 回撤 -15.6%)

腿净值来源: live/ledger/{leg}/state.json 的 nav_history
  · 旧三腿 ICW_SW/VG/VGH: paper_trade.py --mark 维护
  · 新三腿 M4/LOF/TREND : new_quant experiments/live_pt.py --mark 维护
基点: 每组合 100,000 元 (仅记账锚点; 收益率与基点无关, 腿台账名义资金亦与基点无关)。
组合口径: r_c(t) = Σ w_i · r_i(t); 腿间再平衡成本未计 (与回测口径一致);
          COMBO_C 的融资利息未计 (与回测一致, 实盘需另扣 ~3-4%/年 x 33%).

用法:
  python3 live/combo_track.py --init 2026-09-18          # 初始化组合起点
  python3 live/combo_track.py --mark                     # 重算到最新 (幂等)
  python3 live/combo_track.py --report                   # 三组合报表
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402

QUANT_DIR = config.QUANT_DIR
LEDGER_DIR = os.path.join(QUANT_DIR, "live", "ledger")

COMBOS = {
    "COMBO_A": {"label": "A进攻-数字王(ICW60/M440)",
                "weights": {"ICW_SW": 0.60, "M4": 0.40}, "nav_init": 100000},
    "COMBO_B": {"label": "B进攻+LOF(ICW54/M436/LOF10)",
                "weights": {"ICW_SW": 0.54, "M4": 0.36, "LOF": 0.10}, "nav_init": 100000},
    "COMBO_C": {"label": "C风平x1.33(六腿风险平价+融资垫)",
                "weights": {"ICW_SW": 0.1454, "VG": 0.1971, "VGH": 0.2300,
                            "M4": 0.1141, "TREND": 0.5040, "LOF": 0.1395},
                "nav_init": 100000},
}

# 回测基线 (six_leg_combo.py, 2020-02-03~2026-07-23 公共窗口; C 为台账4位小数权重口径)
BACKTEST = {
    "COMBO_A": {"cagr": 0.2386, "sharpe": 1.25, "mdd": -0.200},
    "COMBO_B": {"cagr": 0.2361, "sharpe": 1.29, "mdd": -0.205},
    "COMBO_C": {"cagr": 0.1950, "sharpe": 1.57, "mdd": -0.156},
}


def load_leg_nav(leg):
    """腿净值序列 (date → nav), 来自 ledger/{leg}/state.json"""
    p = os.path.join(LEDGER_DIR, leg, "state.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        st = json.load(f)
    hist = st.get("nav_history", [])
    if not hist:
        return None
    s = pd.Series({pd.Timestamp(h["date"]): float(h["nav"]) for h in hist})
    return s.sort_index()


def combo_path(combo):
    d = os.path.join(LEDGER_DIR, combo)
    os.makedirs(d, exist_ok=True)
    return d


def load_combo(combo):
    p = os.path.join(combo_path(combo), "state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_combo(combo, st):
    with open(os.path.join(combo_path(combo), "state.json"), "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False, default=str)


def compute(combo, start_date):
    """从 start_date 起重算组合净值序列 (全量, 幂等)"""
    spec = COMBOS[combo]
    legs = list(spec["weights"])
    navs = {}
    for leg in legs:
        s = load_leg_nav(leg)
        if s is None:
            sys.exit(f"[ERROR] {combo}: 腿 {leg} 无净值 (ledger/{leg}/state.json)")
        if s.index.max() < pd.Timestamp(start_date):
            sys.exit(f"[ERROR] {combo}: 腿 {leg} 净值止于 {s.index.max().date()}, "
                     f"晚于起点 {start_date}")
        navs[leg] = s

    idx = sorted(set().union(*[set(s.index) for s in navs.values()]))
    idx = [d for d in idx if d >= pd.Timestamp(start_date)]
    M = pd.DataFrame({leg: navs[leg].reindex(idx).ffill() for leg in legs})
    M = M.dropna()
    r = M.pct_change().fillna(0.0)
    rc = sum(spec["weights"][leg] * r[leg] for leg in legs)
    nav_c = spec["nav_init"] * (1.0 + rc).cumprod()
    ret_pct = (nav_c / spec["nav_init"] - 1.0) * 100
    return M, nav_c, ret_pct


def cmd_init(args):
    start = pd.Timestamp(args.init)
    for combo in COMBOS:
        M, nav_c, ret_pct = compute(combo, start)
        # 腿在起点日的净值 (记录用)
        leg_start = {leg: float(M[leg].iloc[0]) for leg in M.columns}
        st = {"combo": combo, "label": COMBOS[combo]["label"],
              "weights": COMBOS[combo]["weights"],
              "nav_init": COMBOS[combo]["nav_init"],
              "start_date": start.strftime("%Y-%m-%d"),
              "leg_nav_at_start": leg_start,
              "nav_history": []}
        for d, nav, rp in zip(nav_c.index, nav_c.values, ret_pct.values):
            st["nav_history"].append({"date": d.strftime("%Y-%m-%d"),
                                      "nav": round(float(nav), 2),
                                      "ret_pct": round(float(rp), 2)})
        save_combo(combo, st)
        print(f"[init] {combo} 起点 {start.date()} | 权重 "
              f"{COMBOS[combo]['weights']} | 基点净值 {nav_c.iloc[0]:,.0f}", flush=True)
    print(f"[+] 组合台账 → {LEDGER_DIR}/COMBO_*/", flush=True)


def cmd_mark(args):
    for combo in COMBOS:
        st = load_combo(combo)
        if st is None:
            sys.exit(f"[ERROR] {combo} 未初始化, 先跑 --init")
        start = st["start_date"]
        M, nav_c, ret_pct = compute(combo, start)
        hist = [{"date": d.strftime("%Y-%m-%d"), "nav": round(float(nav), 2),
                 "ret_pct": round(float(rp), 2)}
                for d, nav, rp in zip(nav_c.index, nav_c.values, ret_pct.values)]
        # 盘中标记传播: 最新日任一腿快照为盘中实时 (intraday), 组合同日同步标记
        if hist:
            last_date = hist[-1]["date"]
            for leg in COMBOS[combo]["weights"]:
                p = os.path.join(LEDGER_DIR, leg, "state.json")
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    lh = json.load(f).get("nav_history", [])
                if lh and lh[-1].get("date") == last_date and lh[-1].get("intraday"):
                    hist[-1]["intraday"] = True
                    break
        st["nav_history"] = hist
        st["weights"] = COMBOS[combo]["weights"]
        st["nav_init"] = COMBOS[combo]["nav_init"]
        st["leg_nav_at_start"] = {leg: float(M[leg].iloc[0]) for leg in M.columns}
        save_combo(combo, st)
        last = hist[-1]
        print(f"[mark] {combo} 至 {last['date']} | 净值 {last['nav']:,.0f} "
              f"({last['ret_pct']:+.2f}%) | {len(hist)} 个快照", flush=True)


def cmd_report(args):
    print(f"\n{'='*100}\n  组合层 Paper Trading 报表 (口径: 日频加权, 与回测一致)\n{'='*100}",
          flush=True)
    hdr = f"{'组合':<10}{'起点':<12}{'最新':<12}{'净值':>12}{'累计':>9}{'天数':>6}{'回测基线':>12}{'回测Sharpe':>10}"
    print(hdr, flush=True)
    for combo in COMBOS:
        st = load_combo(combo)
        if st is None or not st["nav_history"]:
            print(f"{combo:<10} 未初始化", flush=True)
            continue
        hist = st["nav_history"]
        last = hist[-1]
        bt = BACKTEST[combo]
        print(f"{combo:<10}{st['start_date']:<12}{last['date']:<12}"
              f"{last['nav']:>12,.0f}{last['ret_pct']:>8.2f}%{len(hist):>6}"
              f"{bt['cagr']:>11.2%}{bt['sharpe']:>10.2f}", flush=True)

    # 腿明细
    print(f"\n--- 腿净值明细 ---", flush=True)
    for leg in ["ICW_SW", "VG", "VGH", "M4", "TREND", "LOF"]:
        s = load_leg_nav(leg)
        if s is None:
            print(f"  {leg:<8} 无台账", flush=True)
            continue
        n0, n1 = s.iloc[0], s.iloc[-1]
        print(f"  {leg:<8} {s.index[0].date()} ~ {s.index[-1].date()}  "
              f"最新 {n1:>12,.0f}  累计 {n1/n0-1:>+7.2%}  ({len(s)}快照)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="组合层加权净值跟踪")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--init", metavar="YYYY-MM-DD", help="初始化组合起点")
    sub.add_argument("--mark", action="store_true", help="重算到最新 (幂等)")
    sub.add_argument("--report", action="store_true", help="三组合报表")
    args = ap.parse_args()

    if args.init:
        cmd_init(args)
    elif args.mark:
        cmd_mark(args)
    elif args.report:
        cmd_report(args)


if __name__ == "__main__":
    main()
