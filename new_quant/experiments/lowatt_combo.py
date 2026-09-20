#!/usr/bin/env python3
"""experiments.lowatt_combo —— G(真市值Top30) 的 Sharpe 优化: 仓位层 + 组合层
================================================================================
对应博文《量化共识全景复盘》的指导:
  · 第五条第5点 "仓位规则是收益的一半" (Kelly/波动管理)
  · 第二条 "择时用来控风险可以, 用来放大收益就是数据挖掘"
  · √N 数学: 两个低相关策略 50/50 混合, 组合 Sharpe 高于单腿

变体:
  G0  G原版 (对照)
  G1  G0 + CSI500 12-1 动量<0 半仓       (趋势过滤, 同 lowatt_sharpe)
  G5  G0 + 自身波动率目标 20%             (vol-managed, 每调仓日按60d实测波动缩放仓位)
  G4  Top50 + 趋势半仓                    (分散+趋势)
  TREND  跨资产趋势ETF (修复后引擎: 无信号=真清仓, 降杠杆部分=真现金)

输出: 全网格 w×G + (1-w)×TREND 的组合指标 + 相关矩阵; 日收益落盘供复用。
注意: 组合为日频混合近似 (未计两腿间再平衡成本), 权重为演示, 非最终配置。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics, yearly_returns  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from strategies import trend_etf as te  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402
from experiments.lowatt_sharpe import load_index  # noqa: E402


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

    # ---- 市场趋势信号 (CSI500 12-1 动量, PIT) ----
    c500 = load_index("sh000905")
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
    G3 = buffered(sigs, 20, 40, 50)
    G4 = apply_mt(G3, "half")

    r0, to0, *_ = run_portfolio(panels["adj_close"], G0, config.FEE_RT_STOCK,
                                panels["last_date"])

    # ---- G5: 波动率目标 (vol-managed) ----
    # 每调仓日按 G0 自身 60 日已实现波动率缩放下月仓位, 权重<1 部分=现金
    TGT = 0.20
    nav0 = (1 + r0).cumprod()
    G5 = []
    ks = []
    for idx, (t, e, _) in enumerate(sigs):
        w = dict(G0[idx][1])
        h = nav0.loc[:t]
        sd = (h.pct_change().iloc[-60:].std() * np.sqrt(244)
              if len(h) >= 60 else np.nan)
        k = min(1.0, TGT / sd) if sd and sd > 0 else 1.0
        ks.append(k)
        G5.append((e, {s: x * k for s, x in w.items()}))
    print(f"[CHECK] G5 波动率目标 20%: 平均仓位系数 {np.mean(ks):.2f} "
          f"(min {np.min(ks):.2f} / max {np.max(ks):.2f})", flush=True)

    rets = {}
    for name, rebals, fee in [("G0", G0, config.FEE_RT_STOCK),
                              ("G1", G1, config.FEE_RT_STOCK),
                              ("G5", G5, config.FEE_RT_STOCK),
                              ("G4", G4, config.FEE_RT_STOCK)]:
        r, to, buys, fz = run_portfolio(panels["adj_close"], rebals, fee,
                                        panels["last_date"])
        rets[name] = r
        m = calc_metrics(r)
        print(f"[{name}] 年化 {m['年化收益']:.2%} | 波动 {m['年化波动']:.2%} | "
              f"Sharpe {m['Sharpe']:.2f} | 回撤 {m['最大回撤']:.1%} | "
              f"换手 {to:.1%}", flush=True)

    epx = te.load_etf_panel("2017-01-01", config.BT_END)
    rt, tot, buyst, _ = run_portfolio(epx, te.trend_rebalances(epx),
                                      config.FEE_RT_ETF)
    rets["TREND"] = rt
    mt_ = calc_metrics(rt)
    print(f"[TREND] 年化 {mt_['年化收益']:.2%} | 波动 {mt_['年化波动']:.2%} | "
          f"Sharpe {mt_['Sharpe']:.2f} | 回撤 {mt_['最大回撤']:.1%} | "
          f"换手 {tot:.1%} | 买入 {buyst}", flush=True)

    ids = ["G0", "G1", "G5", "G4", "TREND"]
    df = pd.concat({n: rets[n] for n in ids}, axis=1).dropna()
    print(f"\n{'='*100}\n  日收益相关矩阵 (区间 {df.index[0].date()} ~ "
          f"{df.index[-1].date()}, {len(df)}天)\n{'='*100}")
    print(df.corr().round(3).to_string(), flush=True)

    out_csv = os.path.join(config.OUT_DIR, "lowatt_combo_returns.csv")
    df.to_csv(out_csv)
    print(f"[+] 日收益落盘 {out_csv}", flush=True)

    # ---- G0 画像: 回撤区间 + 月度分布 ----
    nav0 = (1 + rets["G0"]).cumprod()
    dd = nav0 / nav0.cummax() - 1
    tr = dd.idxmin()
    pk = nav0.loc[:tr].idxmax()
    mo = (1 + rets["G0"]).resample("ME").prod() - 1
    print(f"\n  G0 画像: 最大回撤 {dd.min():.1%} ({pk.date()} → {tr.date()}) | "
          f"月度胜率 {(mo > 0).mean():.0%} | 最好月 {mo.max():.1%} | "
          f"最差月 {mo.min():.1%}", flush=True)

    # ---- 组合网格 ----
    print(f"\n{'='*100}\n  组合网格: w×G + (1-w)×TREND (日频混合近似, 成本后)\n{'='*100}",
          flush=True)
    print(f"{'组合':<22}{'年化':>8}{'波动':>8}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}",
          flush=True)
    rows = []
    for gname, ws in [("G0", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                      ("G5", [0.5, 0.6]),
                      ("G4", [0.5, 0.6])]:
        for w in ws:
            r = w * df[gname] + (1 - w) * df["TREND"]
            m = calc_metrics(r)
            tag = f"{gname} {w:.0%} + TREND {1-w:.0%}"
            rows.append((tag, r, m))
            print(f"{tag:<22}{m['年化收益']:>8.2%}{m['年化波动']:>8.2%}"
                  f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}"
                  f"{m['Calmar']:>8.2f}", flush=True)

    best = max(rows, key=lambda x: x[2]["Sharpe"])
    print(f"\n  Sharpe 最高: {best[0]} → Sharpe {best[2]['Sharpe']:.2f}", flush=True)
    by_tag = {t: r for t, r, _ in rows}
    for tag in [best[0], "G0 50% + TREND 50%", "G0 60% + TREND 40%"]:
        yr = yearly_returns(by_tag[tag])
        print(f"  分年 {tag}: " + "  ".join(f"{y}:{v:+.0%}"
                                           for y, v in yr.items()), flush=True)


if __name__ == "__main__":
    main()
