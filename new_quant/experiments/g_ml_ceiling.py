#!/usr/bin/env python3
"""experiments.g_ml_ceiling —— 排雷剔除上限扫描 + 成本归因
================================================================================
Q1: 剔除越多越好, 上限在哪?
  A扫描: 净网扩大(池=30+k), 固定持仓30, k=0/4/6/10/14/18/22/26/30
  B扫描: 净网固定40, 持仓收缩, nd=0/4/6/10/14/18/22/26/30 (floor=8)
Q2: M1纯模型为什么失败 → C: G0/M1/M4 在 fee=0/0.4%/0.8% 下 gross vs net
读 outputs/g_ml_preds.pkl 缓存, 只重跑回测。
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402

TOPK, ENTRY, EXIT = 30, 12, 25
FLOOR = 8


def _apply(rebals, e2t, preds, n_drop, min_hold):
    """模型剔除: 每个调仓日剔掉预测最差 n_drop 只 (至少保留 min_hold)"""
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
    print(f"[CHECK] 回测区间 {e0_exec.date()} ~ {config.BT_END} | preds {len(preds)}个月", flush=True)

    def run(rebals):
        r, to, _, _ = run_portfolio(panels["adj_close"], rebals,
                                    config.FEE_RT_STOCK, panels["last_date"])
        r2, *_ = run_portfolio(panels["adj_close"], rebals,
                               config.FEE_RT_STOCK * config.COST_STRESS,
                               panels["last_date"])
        m, m2 = calc_metrics(r), calc_metrics(r2)
        nh = np.mean([len(w) for _, w in rebals]) if rebals else np.nan
        return m, m2, to, nh

    hdr = f"{'方案':<10}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}{'换手':>8}{'成本x2':>9}{'持仓':>7}"

    # ---- A: 净网扩大, 固定持仓30 ----
    print(f"\n{'='*104}\n  A 净网扩大: 池=Top(30+k) → 剔除最差k只 → 固定持仓30\n{'='*104}", flush=True)
    print(hdr, flush=True)
    for k in [0, 4, 6, 10, 14, 18, 22, 26, 30]:
        rb = _apply(buffered(sigs, ENTRY, EXIT, TOPK + k), e2t, preds, k, TOPK)
        rb = [(e, w) for e, w in rb if e >= e0_exec]
        m, m2, to, nh = run(rb)
        print(f"k={k:<8}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}"
              f"{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}{to:>8.1%}"
              f"{m2['年化收益']:>9.2%}{nh:>7.1f}", flush=True)

    # ---- B: 净网固定40, 持仓收缩 ----
    print(f"\n{'='*104}\n  B 净网固定Top40 → 剔除最差nd只 (收缩持仓, floor={FLOOR})\n{'='*104}", flush=True)
    print(hdr, flush=True)
    for nd in [0, 4, 6, 10, 14, 18, 22, 26, 30]:
        rb = _apply(buffered(sigs, ENTRY, EXIT, TOPK + 10), e2t, preds, nd, FLOOR)
        rb = [(e, w) for e, w in rb if e >= e0_exec]
        m, m2, to, nh = run(rb)
        print(f"nd={nd:<7}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}"
              f"{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}{to:>8.1%}"
              f"{m2['年化收益']:>9.2%}{nh:>7.1f}", flush=True)

    # ---- C: 成本归因 G0/M1/M4 ----
    print(f"\n{'='*104}\n  C 成本归因: fee=0 / 0.4% / 0.8% (解释M1为什么失败)\n{'='*104}", flush=True)
    ml_sigs = [(t, e, preds[t].sort_values(ascending=False))
               for t, e, _ in sigs if t in preds]
    M1 = [(e, w) for e, w in buffered(ml_sigs, ENTRY, EXIT, TOPK) if e >= e0_exec]
    G0 = [(e, w) for e, w in buffered(sigs, ENTRY, EXIT, TOPK) if e >= e0_exec]
    M4 = [(e, w) for e, w in _apply(buffered(sigs, ENTRY, EXIT, TOPK + 10),
                                    e2t, preds, 10, TOPK) if e >= e0_exec]
    print(f"{'策略':<8}{'gross(0)':>10}{'net(0.4%)':>11}{'x2(0.8%)':>11}{'费用拖累':>10}{'换手':>8}", flush=True)
    for name, rb in [("G0", G0), ("M1", M1), ("M4", M4)]:
        r0, to, _, _ = run_portfolio(panels["adj_close"], rb, 0.0, panels["last_date"])
        r1, *_ = run_portfolio(panels["adj_close"], rb, 0.004, panels["last_date"])
        r2, *_ = run_portfolio(panels["adj_close"], rb, 0.008, panels["last_date"])
        m0, m1v, m2 = calc_metrics(r0), calc_metrics(r1), calc_metrics(r2)
        drag = m0["年化收益"] - m1v["年化收益"]
        print(f"{name:<8}{m0['年化收益']:>10.2%}{m1v['年化收益']:>11.2%}"
              f"{m2['年化收益']:>11.2%}{drag:>10.2%}{to:>8.1%}", flush=True)


if __name__ == "__main__":
    main()
