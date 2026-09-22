#!/usr/bin/env python3
"""experiments.g_ml_m4_dump —— 导出 M4(净网44剔14) 日收益到 CSV, 供组合实验复用"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
from core import config, data  # noqa: E402
from core.backtest import run_portfolio  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402

TOPK, ENTRY, EXIT, KDROP = 30, 12, 25, 14


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
    M4 = [(e, w) for e, w in M4 if e >= e0_exec]
    r_m4, _, _, _ = run_portfolio(panels["adj_close"], M4,
                                  config.FEE_RT_STOCK, panels["last_date"])
    out = os.path.join(config.NEW_DIR, "data", "m4_daily.csv")
    r_m4.to_csv(out, header=["ret"])
    print(f"[+] M4 日收益 {len(r_m4)} 行 | {r_m4.index[0].date()} ~ "
          f"{r_m4.index[-1].date()} → {out}", flush=True)


if __name__ == "__main__":
    main()
