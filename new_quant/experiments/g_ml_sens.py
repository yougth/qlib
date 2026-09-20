#!/usr/bin/env python3
"""experiments.g_ml_sens —— M4排雷版参数敏感性 (剔除数量 4/6/10/14只)
================================================================================
读 preds 缓存, 只重跑回测。若剔除数量敏感=运气; 若平台状=真效应。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics, yearly_returns  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402
import pickle  # noqa: E402

TOPK, ENTRY, EXIT = 30, 12, 25


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
    print(f"[CHECK] 回测区间 {e0_exec.date()} ~ {config.BT_END}", flush=True)

    variants = {}
    for n_drop, extra in [(0, 0), (4, 4), (6, 6), (10, 10), (14, 14)]:
        G0h = buffered(sigs, ENTRY, EXIT, TOPK + extra)
        out = []
        for e, w in G0h:
            t = e2t.get(e)
            if t in preds and len(w) > TOPK and n_drop > 0:
                p = preds[t].reindex(list(w)).dropna()
                nd = min(len(w) - TOPK, n_drop)
                bad = set(p.nsmallest(nd).index)
                keep = {s_: 1.0 / len([x for x in w if x not in bad])
                        for s_ in w if s_ not in bad}
                out.append((e, keep))
            else:
                out.append((e, w))
        out = [(e, w) for e, w in out if e >= e0_exec]
        variants[f"M4 剔除{n_drop}" if n_drop else "G0 对照(Top30)"] = out

    print(f"\n{'='*96}\n  M4 排雷 敏感性: 剔除数量 (G规则Top30+缓冲 → 剔除模型最差N只)\n{'='*96}",
          flush=True)
    print(f"{'方案':<18}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}"
          f"{'Calmar':>8}{'换手':>8}{'成本x2':>9}", flush=True)
    for name, rebals in variants.items():
        r, to, buys, fz = run_portfolio(panels["adj_close"], rebals,
                                        config.FEE_RT_STOCK, panels["last_date"])
        r2, *_ = run_portfolio(panels["adj_close"], rebals,
                               config.FEE_RT_STOCK * config.COST_STRESS,
                               panels["last_date"])
        m, m2 = calc_metrics(r), calc_metrics(r2)
        print(f"{name:<18}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}"
              f"{m['Calmar']:>8.2f}{to:>8.1%}{m2['年化收益']:>9.2%}", flush=True)


if __name__ == "__main__":
    main()
