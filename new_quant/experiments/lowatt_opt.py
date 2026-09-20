#!/usr/bin/env python3
"""experiments.lowatt_opt —— 低关注质量(Top15) 的提升收益优化实验
================================================================================
实验组 (全部同一打分, 只改执行层):
  A. 原版 Top15                 (对照, 应与 run_all 一致)
  B. 缓冲带 entry=10/exit=20    新股须进Top10才买, 旧股跌出Top20才卖
  C. 缓冲带 entry=12/exit=25    (更宽缓冲)
  D. Top30                      (分散敏感性, 检验收益是否靠集中度)
每组: 标准成本0.4% + 成本×2 压力; 另报逐年候选池规模。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics, yearly_returns  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402


def plain(sigs, topk):
    out = []
    for t, e, s in sigs:
        pick = list(s.index)[:topk]
        out.append((e, {x: 1.0 / len(pick) for x in pick} if pick else {}))
    return out


def buffered(sigs, entry, exit_rank, topk):
    """缓冲带: 旧持仓保留到跌出 exit_rank, 新面孔须进 entry 才买"""
    holds, out = [], []
    for t, e, s in sigs:
        ranked = list(s.index)
        rpos = {x: i for i, x in enumerate(ranked)}
        keep = [h for h in holds if h in rpos and rpos[h] < exit_rank]
        for x in list(ranked[:entry]) + ranked:
            if len(keep) >= topk:
                break
            if x not in keep:
                keep.append(x)
        if len(keep) > topk:
            keep.sort(key=lambda x: rpos[x])
            keep = keep[:topk]
        holds = keep
        out.append((e, {x: 1.0 / len(keep) for x in keep} if keep else {}))
    return out


def main():
    cal = data.load_calendar(config.BT_START, config.BT_END)
    panels = data.load_panels(config.BT_START, config.BT_END)
    fin = data.load_financials()
    signals = data.month_end_signals(cal, config.BT_START, config.BT_END)

    sigs = []
    for t, e in signals:
        cand = ss.candidates_at(panels, fin, t)
        s = ss.score_lowatt(panels, fin, t, cand).dropna()
        sigs.append((t, e, s.sort_values(ascending=False)))
        print(f"  [SIG] {t.date()} 池={len(cand)} 打分={len(s)}", flush=True)

    df = pd.DataFrame([dict(year=t.year, n=len(c))
                       for t, e, c in [(a, b, c) for a, b, c in
                                       [(t, e, s) for t, e, s in sigs]]])
    print("\n== 逐年候选池规模 (质量池×可交易过滤后, Top15 从中挑选) ==",
          flush=True)
    print(df.groupby("year")["n"].agg(["mean", "min", "max"]).round(0).to_string(),
          flush=True)

    variants = {
        "A 原版Top15": plain(sigs, 15),
        "B 缓冲10-20": buffered(sigs, 10, 20, 15),
        "C 缓冲12-25": buffered(sigs, 12, 25, 15),
        "D 原版Top30": plain(sigs, 30),
    }
    print(f"\n{'='*96}\n  LOWATT 优化实验 (2019-02~2026-09, 成本后)\n{'='*96}",
          flush=True)
    hdr = (f"{'方案':<12}{'年化':>8}{'波动':>8}{'Sharpe':>8}{'回撤':>8}"
           f"{'换手':>8}{'买入':>6}{'成本x2年化':>10}")
    print(hdr, flush=True)
    for name, rebals in variants.items():
        r, to, buys, fz = run_portfolio(
            panels["adj_close"], rebals, config.FEE_RT_STOCK, panels["last_date"])
        m = calc_metrics(r)
        r2, *_ = run_portfolio(
            panels["adj_close"], rebals,
            config.FEE_RT_STOCK * config.COST_STRESS, panels["last_date"])
        m2 = calc_metrics(r2)
        print(f"{name:<12}{m['年化收益']:>8.2%}{m['年化波动']:>8.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>8.1%}{to:>8.1%}{buys:>6}"
              f"{m2['年化收益']:>10.2%}", flush=True)
        yr = yearly_returns(r)
        print("  分年: " + "  ".join(f"{y}:{v:.0%}" for y, v in yr.items()),
              flush=True)


if __name__ == "__main__":
    main()
