#!/usr/bin/env python3
"""experiments.g_ml_port —— M4超排(剔14) 的组合层与安全带实测
================================================================================
1) M4(净网44剔14留30) 日收益 → 与 TREND 组合 (w=0.3/0.5/0.7)
2) QVIX安全带: 仅在 QVIX 252日分位>0.9 的月份降仓(k=0.7 / 0.5)
   (此前的月度恐慌半仓已证伪, 这里只测"极端状态")
PIT: QVIX T日晚披露 → 分位shift(1); 组合为日频混合近似(两腿再平衡成本未计)。
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402

TOPK, ENTRY, EXIT = 30, 12, 25
KDROP = 14  # 剔除数(上限扫描峰值)


def _apply(rebals, e2t, preds, n_drop, min_hold):
    out = []
    for e, w in rebals:
        t = e2t.get(e)
        if t in preds and len(w) > min_hold and n_drop > 0:
            p = preds[t].reindex(list(w)).dropna()
            nd = min(len(w) - min_hold, n_drop)
            bad = set(p.nsmallest(nd).index)
            keep = {s: 1.0 / len([x for x in w if x not in bad])
                    for s in w if s not in bad}
            out.append((e, keep))
        else:
            out.append((e, w))
    return out


def main():
    cal = data.load_calendar(config.BT_START, config.BT_END)
    panels = data.load_panels(config.BT_START, config.BT_END)
    fin = data.load_financials()
    mvp = load_mv_panel()
    signals = data.month_end_signals(cal, config.BT_START, config.BT_END)
    sigs = []
    for t, e in signals:
        cand = ss.candidates_at(panels, fin, t)
        s = score_mv(mvp, fin, t, cand).dropna()
        sigs.append((t, e, s.sort_values(ascending=False)))
    with open(os.path.join(config.OUT_DIR, "g_ml_preds.pkl"), "rb") as f:
        preds, ws, ic_log = pickle.load(f)
    e2t = {e: t for t, e, _ in sigs}
    e0_exec = min(e for t, e, _ in sigs if t in preds)

    M4 = _apply(buffered(sigs, ENTRY, EXIT, TOPK + KDROP), e2t, preds,
                KDROP, TOPK)
    M4 = [(e, w) for e, w in M4 if e >= e0_exec]
    r_m4, to_m4, _, _ = run_portfolio(panels["adj_close"], M4,
                                      config.FEE_RT_STOCK, panels["last_date"])
    m = calc_metrics(r_m4)
    print(f"[CHECK] M4 剔{KDROP} 单腿: 年化 {m['年化收益']:.2%} | Sharpe {m['Sharpe']:.2f} "
          f"| 回撤 {m['最大回撤']:.1%} | 换手 {to_m4:.1%} | 起点 {r_m4.index[0].date()}", flush=True)

    # ---- QVIX 分位 (PIT: shift(1)) ----
    qv = pd.read_csv(os.path.join(config.NEW_DIR, "data",
                                  "index_option_50etf_qvix.csv"))
    qv["date"] = pd.to_datetime(qv["date"])
    q = qv.set_index("date")["close"].sort_index()
    q_pct = q.rolling(252, min_periods=126).apply(
        lambda w: float((w.iloc[-1] >= w).mean()))
    q_pct = q_pct.shift(1)

    # ---- 1) QVIX 安全带 ----
    print(f"\n{'='*100}\n  1) QVIX 安全带 (仅极端恐慌月降仓; 对照: M4 恒仓)\n{'='*100}", flush=True)
    print(f"{'方案':<22}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}", flush=True)
    variants = {"M4 恒仓(对照)": M4}
    for k, tag in [(0.7, "k=0.7"), (0.5, "k=0.5")]:
        seat = []
        for e, w in M4:
            p = q_pct.asof(e)
            kk = k if (not np.isnan(p) and p > 0.9) else 1.0
            seat.append((e, {s: x * kk for s, x in w.items()}))
        variants[f"QVIX>90分位→{tag}"] = seat
    for name, rb in variants.items():
        r, to, _, _ = run_portfolio(panels["adj_close"], rb,
                                    config.FEE_RT_STOCK, panels["last_date"])
        m = calc_metrics(r)
        print(f"{name:<22}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}", flush=True)

    # ---- 2) M4 + TREND 组合 ----
    combo = pd.read_csv(os.path.join(config.OUT_DIR, "lowatt_combo_returns.csv"),
                        index_col=0, parse_dates=True)
    tr = combo["TREND"].reindex(r_m4.index).fillna(0.0)
    print(f"\n{'='*100}\n  2) M4 超排 + TREND 组合 (日频混合近似, 两腿再平衡成本未计)\n{'='*100}", flush=True)
    print(f"{'组合':<22}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}", flush=True)
    for w in [0.3, 0.5, 0.7]:
        r = w * r_m4 + (1 - w) * tr
        m = calc_metrics(r)
        print(f"M4 {w:.0%} + TREND {1-w:.0%}   {m['年化收益']:>9.2%}"
              f"{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}"
              f"{m['Calmar']:>8.2f}", flush=True)
    # 月胜率
    mr = pd.concat({"M4": (1 + r_m4).resample("ME").prod() - 1,
                    "TREND": (1 + tr).resample("ME").prod() - 1}, axis=1).dropna()
    mc = 0.5 * mr["M4"] + 0.5 * mr["TREND"]
    print(f"  50/50 月胜率 {(mc > 0).mean():.0%} | 最差月 {mc.min():.1%} | 最好月 {mc.max():.1%}", flush=True)


if __name__ == "__main__":
    main()
