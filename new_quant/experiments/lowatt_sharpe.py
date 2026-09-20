#!/usr/bin/env python3
"""experiments.lowatt_sharpe —— G(真市值Top30+缓冲) 收益归因 + Sharpe优化实验
================================================================================
归因 (按博文三层结构 Beta/风险溢价/纯Alpha):
  · G日收益 对 CSI300 / CSI500 回归 → beta, 年化alpha, R², 月度胜率
实验组 (Sharpe优化, 不换选股, 只加"控风险"层):
  G0       G 原版 (对照)
  G1       市场趋势过滤-半仓: CSI500 12-1动量<0 → 仓位×0.5
  G2       市场趋势过滤-清仓: CSI500 12-1动量<0 → 持现金
  G3       Top50分散: 选50只, 缓冲20-40
  G4       Top50 + 趋势过滤半仓
映射: 趋势过滤 = 用户 value_comp 体系的"宏观仓位阀门"同款思路。
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


def load_index(stem):
    f = os.path.join(config.TENCENT_DIR, stem + ".parquet")
    d = pd.read_parquet(f)[["date", "close"]]
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float).sort_index()


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

    c500 = load_index("sh000905")
    # 市场趋势信号: 每个信号日的 CSI500 12-1 动量 (PIT)
    mt = {}
    for t, e, _ in sigs:
        h = c500.loc[:t]
        mt[t] = bool(len(h) >= 250 and h.iloc[-21] / h.iloc[-250] - 1 > 0)

    def apply_mt(rebals, mode):
        out = []
        for t, e, _ in sigs:
            w = dict(rebals[len(out)][1])
            if not mt[t]:
                w = {s: x * 0.5 for s, x in w.items()} if mode == "half" else {}
            out.append((e, w))
        return out

    G0 = buffered(sigs, 12, 25, 30)
    G1 = apply_mt(G0, "half")
    G2 = apply_mt(G0, "full")
    G3 = buffered(sigs, 20, 40, 50)
    G4 = apply_mt(G3, "half")
    off = sum(1 for t in mt if not mt[t])
    print(f"\n[CHECK] 市场动量<0 的信号月: {off}/{len(mt)} "
          f"({off/len(mt):.0%}), 这些月减仓/空仓", flush=True)

    variants = {"G0 G原版": G0, "G1 趋势半仓": G1, "G2 趋势清仓": G2,
                "G3 Top50": G3, "G4 Top50+半仓": G4}
    print(f"\n{'='*100}")
    print("  Sharpe 优化实验 (2019-02~2026-09, 成本后, 选股完全相同)")
    print(f"{'='*100}")
    print(f"{'方案':<14}{'年化':>8}{'波动':>8}{'Sharpe':>8}{'回撤':>9}"
          f"{'Calmar':>8}{'换手':>8}{'成本x2':>9}", flush=True)
    rets = {}
    for name, rebals in variants.items():
        r, to, buys, fz = run_portfolio(
            panels["adj_close"], rebals, config.FEE_RT_STOCK, panels["last_date"])
        rets[name] = r
        m = calc_metrics(r)
        r2, *_ = run_portfolio(
            panels["adj_close"], rebals,
            config.FEE_RT_STOCK * config.COST_STRESS, panels["last_date"])
        m2 = calc_metrics(r2)
        print(f"{name:<14}{m['年化收益']:>8.2%}{m['年化波动']:>8.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}"
              f"{to:>8.1%}{m2['年化收益']:>9.2%}", flush=True)
        yr = yearly_returns(r)
        print("  分年: " + "  ".join(f"{y}:{v:.0%}" for y, v in yr.items()),
              flush=True)

    # ---- 归因: G0 vs 指数 ----
    print(f"\n{'='*100}\n  收益归因 (G0 日收益 vs 指数回归, 全区间)\n{'='*100}",
          flush=True)
    c300 = load_index("sh000300")
    for tag, idx in [("沪深300", c300), ("中证500", c500)]:
        ir = idx.pct_change()
        df = pd.concat({"g": rets["G0 G原版"], "m": ir}, axis=1).dropna()
        X = np.column_stack([np.ones(len(df)), df["m"]])
        coef, *_ = np.linalg.lstsq(X, df["g"], rcond=None)
        a_d, b = coef
        resid = df["g"] - (a_d + b * df["m"])
        r2 = 1 - resid.var() / df["g"].var()
        a_ann = (1 + a_d) ** 244 - 1
        print(f"  vs {tag}: beta {b:.2f} | 年化alpha {a_ann:.2%} | R² {r2:.2f} | "
              f"日相关 {df['g'].corr(df['m']):.2f}", flush=True)
    # 逆势月统计
    m_g = pd.concat({n: (1 + r).resample("M").prod() - 1 for n, r in rets.items()},
                    axis=1).dropna()
    m_idx = (1 + c500.pct_change()).resample("M").prod() - 1
    m_idx = m_idx.reindex(m_g.index)
    down = m_idx < 0
    up_when_down = {n: (m_g[n][down] > 0).mean() for n in m_g.columns}
    print(f"\n  中证500下跌月 ({int(down.sum())}个) 中各方案正收益占比:", flush=True)
    for n, v in up_when_down.items():
        print(f"    {n}: {v:.0%}", flush=True)


if __name__ == "__main__":
    main()
