#!/usr/bin/env python3
"""experiments.lof_strategy —— 方向2 LOF折价 正式策略回测(含成本)
================================================================================
规则:
  · 月末T: 折价率 d = close(T)/nav(T-1) - 1  (净值T晚披露, PIT用T-1)
  · 过滤: 月内≥10交易日有成交 且 20日均成交额≥50万
  · 买入: 折价最深的N只等权 (N=5/10/20) ; 变体: d<-1%全买(上限20只)
  · 持有: T月末收盘 → 下月末收盘; 日频盯市; ETF往返成本0.1%按换手计
产出: 年化/波动/Sharpe/回撤/Calmar/月胜率/换手/分年
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402
from core.backtest import calc_metrics  # noqa: E402
from experiments.lof_research import load_panels  # noqa: E402

FEE = 0.001          # ETF往返0.1%
MIN_DAYS = 10
MIN_AMT = 50e4       # 20日均成交额≥50万
NS = [5, 10, 20]


def build(D_me, me, amt20, days, mode, n=10, thr=-0.01, cap=20):
    """生成每月目标持仓 {code: weight}; mode: 'topN' 或 'deep'"""
    targets = {}
    ts = list(D_me.index)
    for i, t in enumerate(ts[:-1]):
        v_d = days.loc[t] if t in days.index else None
        v_a = amt20.loc[t] if t in amt20.index else None
        x = D_me.loc[t]
        ok = x.notna()
        if v_d is not None:
            ok &= (v_d.reindex(x.index) >= MIN_DAYS).fillna(False)
        if v_a is not None:
            ok &= (v_a.reindex(x.index) >= MIN_AMT).fillna(False)
        x = x[ok]
        min_need = n if mode == "topN" else 5
        if len(x) < min_need:
            targets[t] = {}
            continue
        if mode == "topN":
            pick = x.nsmallest(n).index
        else:
            pick = x[x < thr].index[:cap]
        targets[t] = {c: 1.0 / len(pick) for c in pick} if len(pick) else {}
    return targets


def backtest(targets, P):
    """日频盯市: 目标权重在再平衡日生效(下一交易日起), 日频drift, 成本按换手"""
    idx = P.index
    dates = sorted(targets)
    nav = 1.0
    w = {}           # 当前权重(日频drift)
    navs, dts, inv = [], [], []
    next_i = 0
    daily = P.pct_change()
    for t in idx:
        r_gross = sum(wi * daily.loc[t, c] for c, wi in w.items()
                      if not np.isnan(daily.loc[t, c]))
        nav *= (1.0 + r_gross)
        # drift
        if w and (1.0 + r_gross) != 0:
            w = {c: wi * (1 + (daily.loc[t, c] if not np.isnan(daily.loc[t, c]) else 0.0))
                 / (1.0 + r_gross) for c, wi in w.items()}
        # 再平衡(月末信号 → 下一交易日生效)
        if next_i < len(dates) and t >= dates[next_i]:
            tgt = targets[dates[next_i]]
            turn = sum(abs(tgt.get(c, 0.0) - w.get(c, 0.0)) for c in set(tgt) | set(w))
            nav *= (1.0 - FEE * turn)
            w = dict(tgt)
            next_i += 1
        navs.append(nav)
        dts.append(t)
        inv.append(bool(w))
    r = pd.Series(navs, index=pd.DatetimeIndex(dts)).pct_change().fillna(0.0)
    m_inv = pd.Series(inv, index=pd.DatetimeIndex(dts)).resample("ME").max()
    return r, m_inv


def main():
    print("[1] 加载 LOF 面板...", flush=True)
    P, A, N = load_panels()
    Nf = N.reindex(P.index).ffill()
    D = P / Nf.shift(1) - 1
    me = P.resample("ME").last()
    D_me = D.resample("ME").last()
    days = A.notna().resample("ME").sum()
    amt20 = A.rolling(20).mean().resample("ME").last()
    print(f"    面板 {P.shape} | {P.index.min().date()} ~ {P.index.max().date()}", flush=True)

    variants = {}
    for n in NS:
        variants[f"Top{n}深折价等权"] = build(D_me, me, amt20, days, "topN", n=n)
    variants["折价>1%全买(≤20)"] = build(D_me, me, amt20, days, "deep")

    print(f"\n{'='*100}\n  LOF折价策略 (成本后, 日频盯市)\n{'='*100}", flush=True)
    hdr = f"{'策略':<18}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}{'月胜率':>8}{'年换手':>8}{'持仓':>7}"
    print(hdr, flush=True)
    for name, tgt in variants.items():
        r, m_inv = backtest(tgt, P)
        m = calc_metrics(r)
        mr = (1 + r).resample("ME").prod() - 1
        mr = mr[m_inv > 0]          # 仅统计持仓月
        # 年换手: 用目标权重变化估算
        ts = sorted(tgt)
        turns = []
        for i in range(1, len(ts)):
            a, b = tgt[ts[i - 1]], tgt[ts[i]]
            turns.append(sum(abs(b.get(c, 0.0) - a.get(c, 0.0)) for c in set(a) | set(b)))
        yr_to = np.mean(turns) * 12 if turns else 0.0
        nh = np.mean([len(v) for v in tgt.values()]) if tgt else 0
        print(f"{name:<18}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}"
              f"{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}{(mr > 0).mean():>8.0%}"
              f"{yr_to:>8.0%}{nh:>7.1f}", flush=True)
        yr = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        print("    分年: " + "  ".join(f"{k.year}:{v:+.0%}" for k, v in yr.items()), flush=True)

    # ---- 与 M4 股票腿的相关性 + 组合价值 ----
    m4fp = os.path.join(config.NEW_DIR, "data", "m4_daily.csv")
    if os.path.exists(m4fp):
        r_m4 = pd.read_csv(m4fp, index_col=0, parse_dates=True)["ret"]
        print(f"\n{'='*100}\n  LOF腿 × M4股票腿 (2020-02起, 日频混合近似, 腿间再平衡成本未计)\n{'='*100}", flush=True)
        print(f"{'组合':<22}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}", flush=True)
        r_lof, _ = backtest(variants["Top10深折价等权"], P)
        mr_m4 = (1 + r_m4).resample("ME").prod() - 1
        mr_lof = (1 + r_lof).resample("ME").prod() - 1
        cor = pd.concat({"M4": mr_m4, "LOF": mr_lof}, axis=1).dropna().corr()
        print(f"  月度收益相关: {cor.iloc[0, 1]:.3f}", flush=True)
        common = r_m4.index.intersection(r_lof.index)
        a = r_m4.reindex(common)
        b = r_lof.reindex(common)
        m = calc_metrics(a)
        print(f"{'M4 单腿':<22}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}"
              f"{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}", flush=True)
        for wgt in [0.9, 0.8]:
            r = wgt * a + (1 - wgt) * b
            m = calc_metrics(r)
            print(f"{'M4 %d%% + LOF %d%%' % (wgt*100, (1-wgt)*100):<22}{m['年化收益']:>9.2%}"
                  f"{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}"
                  f"{m['Calmar']:>8.2f}", flush=True)

    out = os.path.join(config.OUT_DIR, "lof_strategy_report.txt")
    print(f"\n[+] 完成 → {out}", flush=True)


if __name__ == "__main__":
    main()
