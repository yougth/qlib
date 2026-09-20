#!/usr/bin/env python3
"""experiments.lowatt_mv —— LOWATT 升级实验: 成交额代理 → 真市值因子
================================================================================
背景: 原版"低关注"用20日均成交额代理 (无市值数据)。现用百度个股历史总市值
(new_quant/data/marketcap_baidu.csv, 5天采样, PIT最长滞后5天) 做"真小市值"检验。
上一轮实验结论: 缓冲带12-25省换手不损收益; Top30≥Top15 (收益不靠集中度)。

实验组 (质量门槛不变, 只换打分与执行):
  A  成交额代理 Top15            (对照, = run_all 原版)
  E  真市值   Top15              (纯替换: 0.6×小市值 + 0.4×现金转换)
  F  真市值   Top15 + 缓冲12-25
  G  真市值   Top30 + 缓冲12-25  (全套优化组合)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics, yearly_returns, rank_ic  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import plain, buffered  # noqa: E402

MV_FILE = os.path.join(config.NEW_DIR if hasattr(config, "NEW_DIR") else
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "marketcap_baidu.csv")


def load_mv_panel():
    mv = pd.read_csv(MV_FILE, dtype={"code": str})
    mv["date"] = pd.to_datetime(mv["date"])
    mv["sym"] = np.where(mv["code"].str[:2].isin(["60", "68"]),
                         "SH" + mv["code"], "SZ" + mv["code"])
    p = mv.pivot(index="date", columns="sym", values="mv").sort_index()
    p = p.ffill()   # 各股采样相位不同, 必须ffill: 任一行=各股最近一次已知值(PIT)
    print(f"[CHECK] 市值面板: {p.shape[1]} 只 | {p.index[0].date()} ~ "
          f"{p.index[-1].date()} | 采样点/只中位 {int(p.notna().sum().median())}",
          flush=True)
    return p


def score_mv(mvp, fin, t, cand):
    """0.6×小市值分位 + 0.4×现金转换分位 (真市值替换成交额)"""
    if not cand:
        return pd.Series(dtype=float)
    row = mvp.loc[:t].iloc[-1] if len(mvp.loc[:t]) else pd.Series(dtype=float)
    mv = row.reindex(cand)
    cc = data.cash_conversion(fin, t).reindex(cand)
    return 0.6 * (-mv).rank(pct=True) + 0.4 * cc.rank(pct=True)


def main():
    cal = data.load_calendar(config.BT_START, config.BT_END)
    panels = data.load_panels(config.BT_START, config.BT_END)
    fin = data.load_financials()
    mvp = load_mv_panel()
    signals = data.month_end_signals(cal, config.BT_START, config.BT_END)

    sigs_amt, sigs_mv = [], []
    for t, e in signals:
        cand = ss.candidates_at(panels, fin, t)
        s_amt = ss.score_lowatt(panels, fin, t, cand).dropna()
        s_mv = score_mv(mvp, fin, t, cand).dropna()
        sigs_amt.append((t, e, s_amt.sort_values(ascending=False)))
        sigs_mv.append((t, e, s_mv.sort_values(ascending=False)))
    cov = np.mean([len(s) for _, _, s in sigs_mv]) / max(
        1, np.mean([len(s) for _, _, s in sigs_amt]))
    print(f"[CHECK] 真市值打分覆盖 {cov:.1%} (相对成交额代理)", flush=True)

    variants = {
        "A 代理Top15": plain(sigs_amt, 15),
        "E 市值Top15": plain(sigs_mv, 15),
        "F 市值15+缓冲": buffered(sigs_mv, 12, 25, 15),
        "G 市值30+缓冲": buffered(sigs_mv, 12, 25, 30),
    }
    print(f"\n{'='*96}\n  LOWATT 真市值升级实验 (2019-02~2026-09, 成本后)\n{'='*96}",
          flush=True)
    print(f"{'方案':<14}{'年化':>8}{'波动':>8}{'Sharpe':>8}{'回撤':>8}"
          f"{'换手':>8}{'买入':>6}{'成本x2':>9}{'RankIC':>8}", flush=True)
    for name, rebals in variants.items():
        r, to, buys, fz = run_portfolio(
            panels["adj_close"], rebals, config.FEE_RT_STOCK, panels["last_date"])
        m = calc_metrics(r)
        r2, *_ = run_portfolio(
            panels["adj_close"], rebals,
            config.FEE_RT_STOCK * config.COST_STRESS, panels["last_date"])
        m2 = calc_metrics(r2)
        src = sigs_mv if name[0] in "EFG" else sigs_amt
        _, ric, _ = rank_ic(panels["adj_close"], {t: s for t, e, s in src})
        print(f"{name:<14}{m['年化收益']:>8.2%}{m['年化波动']:>8.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>8.1%}{to:>8.1%}{buys:>6}"
              f"{m2['年化收益']:>9.2%}{ric:>8.4f}", flush=True)
        yr = yearly_returns(r)
        print("  分年: " + "  ".join(f"{y}:{v:.0%}" for y, v in yr.items()),
              flush=True)


if __name__ == "__main__":
    main()
