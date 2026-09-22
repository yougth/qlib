#!/usr/bin/env python3
"""experiments.trend_attrib —— TREND收益归因 + 危机窗口保护 + 动量窗口体检 + 三腿组合
================================================================================
归因口径: 日频盯市, 调仓日权重生效(T+1), 权重日频drift, 每只ETF的
  每日贡献 = w_i(t)×r_i(t), 加总≈组合毛收益; 成本单列。
分组: 美股(513100/513500) 黄金(518880) 国债(511010) A股宽基(300/500/创业板)
      A股红利(510880) 港股(159920) 商品(501018原油/159985豆粕)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402
from core.backtest import calc_metrics  # noqa: E402
from strategies.trend_etf import load_etf_panel, trend_rebalances  # noqa: E402
from experiments.lof_strategy import build as lof_build, backtest as lof_backtest  # noqa: E402
from experiments.lof_research import load_panels as lof_panels  # noqa: E402

FEE = 0.001
GROUPS = {
    "美股(纳指+标普)": ["sh513100", "sh513500"],
    "黄金": ["sh518880"],
    "国债": ["sh511010"],
    "A股宽基": ["sh510300", "sh510500", "sz159915"],
    "A股红利": ["sh510880"],
    "港股": ["sz159920"],
    "商品": ["sh501018", "sz159985"],
}


def backtest_attrib(px, rebals):
    """日频盯市+drift+成本; 返回 (日收益, 每只ETF每日贡献DataFrame, 成本合计)"""
    dret = px.pct_change().fillna(0.0)
    reb_map = {e: dict(wt) for e, wt in rebals}
    exec_dates = sorted(reb_map)
    cur = {}
    nav, cost_total = 1.0, 0.0
    cols = list(px.columns)
    contrib = {c: np.zeros(len(px)) for c in cols}
    W = np.zeros((len(px), len(cols)))
    navs = np.zeros(len(px))
    j = 0
    for k, t in enumerate(px.index):
        r_g = sum(wi * dret.iloc[k][c] for c, wi in cur.items())
        nav *= (1.0 + r_g)
        if cur and (1.0 + r_g) != 0:
            for c in list(cur):
                contrib[c][k] = cur[c] * dret.iloc[k][c]
                cur[c] *= (1.0 + dret.iloc[k][c]) / (1.0 + r_g)
        while j < len(exec_dates) and t >= exec_dates[j]:
            tgt = reb_map[exec_dates[j]]
            turn = sum(abs(tgt.get(c, 0.0) - cur.get(c, 0.0))
                       for c in set(tgt) | set(cur))
            cst = FEE * turn
            cost_total += cst
            nav *= (1.0 - cst)
            cur = dict(tgt)
            j += 1
        for ci, c in enumerate(cols):
            W[k, ci] = cur.get(c, 0.0)
        navs[k] = nav
    r = pd.Series(navs, index=px.index).pct_change().fillna(0.0)
    w_daily = pd.DataFrame(W, index=px.index, columns=cols)
    return r, pd.DataFrame(contrib, index=px.index), cost_total, w_daily


def mom_variants(px):
    """动量窗口体检: 12-1(基线) / 6-1 / 含近月 / 双确认"""
    idx = px.index
    out = {}
    specs = {"12-1(基线)": (21, 250), "6-1": (21, 126),
             "12-0含近月": (1, 250), "双确认6-1&12-1": None}
    from strategies.trend_etf import _month_end_positions
    for name, spec in specs.items():
        reb, holds = [], []
        for i in _month_end_positions(idx):
            if i < 250:
                continue
            if spec is None:
                ok_mask = ((px.iloc[i - 21] / px.iloc[i - 250] - 1) > 0) & \
                          ((px.iloc[i - 21] / px.iloc[i - 126] - 1) > 0)
            else:
                skip, win = spec
                ok_mask = (px.iloc[i - skip] / px.iloc[i - win] - 1) > 0
            vol = px.iloc[i - 60:i].pct_change().std() * np.sqrt(244)
            ok = (ok_mask & (vol > 0) & vol.notna())
            ok = ok[ok].index
            if not len(ok):
                reb.append((idx[i + 1], {}))
                continue
            iv = 1.0 / vol[ok]
            w = iv / iv.sum()
            reb.append((idx[i + 1], {s: float(x) for s, x in w.items()}))
        r, _, _, _ = backtest_attrib(px, reb)
        m = calc_metrics(r)
        out[name] = (m, r)
        print(f"{name:<18}年化 {m['年化收益']:7.2%} 波动 {m['年化波动']:6.2%} "
              f"Sharpe {m['Sharpe']:5.2f} 回撤 {m['最大回撤']:7.1%} "
              f"Calmar {m['Calmar']:5.2f}", flush=True)
    return out


def main():
    px = load_etf_panel(config.BT_START, config.BT_END)
    print(f"[CHECK] ETF面板 {px.shape} | {px.index.min().date()} ~ "
          f"{px.index.max().date()}", flush=True)

    rebals = trend_rebalances(px)
    r_tr, contrib, cost, w_daily = backtest_attrib(px, rebals)
    m = calc_metrics(r_tr)
    print(f"\n[CHECK] TREND复算: 年化 {m['年化收益']:.2%} | Sharpe {m['Sharpe']:.2f} "
          f"| 回撤 {m['最大回撤']:.1%} | 成本合计 {cost:.2%}", flush=True)

    # ---- 1) 分组归因 ----
    print(f"\n{'='*100}\n  1) 收益归因 (毛贡献, 全区间)\n{'='*100}", flush=True)
    gross_total = contrib.sum().sum()
    hdr = (f"{'组别':<20}{'毛贡献':>9}{'贡献占比':>9}{'平均仓位':>9}"
             f"{'全程buy&hold':>13}")
    print(hdr, flush=True)
    for g, members in GROUPS.items():
        cg = contrib[members].sum().sum()
        avg_w = w_daily[members].sum(axis=1).mean()
        bh = float((px[members].iloc[-1] / px[members].apply(
            lambda s: s.dropna().iloc[0]) - 1).mean())
        print(f"{g:<20}{cg:>9.2%}{cg / gross_total:>9.0%}{avg_w:>9.1%}"
              f"{bh:>13.1%}", flush=True)
    print(f"{'合计(毛)':<20}{gross_total:>9.2%}{1.0:>9.0%}", flush=True)
    print(f"{'交易成本':<20}{-cost:>9.2%}", flush=True)

    # ---- 2) 危机窗口: M4大跌期 TREND在干嘛 ----
    print(f"\n{'='*100}\n  2) 危机窗口保护 (M4回撤段 vs TREND同期)\n{'='*100}",
          flush=True)
    m4fp = os.path.join(config.NEW_DIR, "data", "m4_daily.csv")
    r_m4 = pd.read_csv(m4fp, index_col=0, parse_dates=True)["ret"]
    nav4 = (1 + r_m4).cumprod()
    dd = nav4 / nav4.cummax() - 1
    troughs = dd.nsmallest(3).index.sort_values()
    shown = set()
    print(f"{'M4回撤段':<26}{'M4收益':>9}{'TREND同期':>10}", flush=True)
    for tr in troughs:
        peak = nav4.loc[:tr].idxmax()
        key = (peak.year, tr.year)
        if key in shown:
            continue
        shown.add(key)
        seg = slice(peak, tr)
        print(f"{peak.date()} ~ {tr.date()}   "
              f"{nav4.loc[tr] / nav4.loc[peak] - 1:>9.1%}"
              f"{(1 + r_tr.loc[seg]).prod() - 1:>10.1%}", flush=True)

    # ---- 3) TREND最差6个月归因 ----
    print(f"\n{'='*100}\n  3) TREND最差月份归因\n{'='*100}", flush=True)
    mr = (1 + r_tr).resample("ME").prod() - 1
    for t, v in mr.nsmallest(6).items():
        seg = contrib.loc[t.replace(day=1):t]
        cg = seg.sum()
        top = cg.abs().nlargest(2)
        det = " + ".join(f"{c}:{cg[c]:+.1%}" for c in top.index)
        print(f"{t.date()} {v:>7.1%}  主因 {det}", flush=True)

    # ---- 4) 动量窗口体检 ----
    print(f"\n{'='*100}\n  4) 动量窗口体检 (博文下一步#2)\n{'='*100}", flush=True)
    mom_variants(px)

    # ---- 5) 三腿组合: M4 × TREND × LOF ----
    print(f"\n{'='*100}\n  5) 三腿组合 (2020-02起, 腿间再平衡成本未计)\n{'='*100}",
          flush=True)
    P, A, N = lof_panels()
    Nf = N.reindex(P.index).ffill()
    D = P / Nf.shift(1) - 1
    me = P.resample("ME").last()
    D_me = D.resample("ME").last()
    days = A.notna().resample("ME").sum()
    amt20 = A.rolling(20).mean().resample("ME").last()
    r_lof, _ = lof_backtest(lof_build(D_me, me, amt20, days, "topN", n=10), P)
    common = r_m4.index.intersection(r_tr.index).intersection(r_lof.index)
    a = r_m4.reindex(common)
    b = r_tr.reindex(common)
    c = r_lof.reindex(common)
    print(f"{'组合 M4/TREND/LOF':<24}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}"
          f"{'Calmar':>8}{'月胜率':>8}", flush=True)

    def show(tag, wm, wt, wc):
        r = wm * a + wt * b + wc * c
        m = calc_metrics(r)
        mret = (1 + r).resample("ME").prod() - 1
        print(f"{tag:<24}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}"
              f"{(mret > 0).mean():>8.0%}", flush=True)

    show("M4 100/0/0 (单腿)", 1.0, 0.0, 0.0)
    show("50/50/0 (现行)", 0.5, 0.5, 0.0)
    show("60/30/10", 0.6, 0.3, 0.1)
    show("50/35/15", 0.5, 0.35, 0.15)
    show("40/40/20", 0.4, 0.4, 0.2)
    # 等风险版: 权重 ∝ 1/波动
    vs = [x.std() * np.sqrt(244) for x in (a, b, c)]
    iv = [1 / v for v in vs]
    s = sum(iv)
    w = [x / s for x in iv]
    show(f"等风险 {w[0]:.0%}/{w[1]:.0%}/{w[2]:.0%}", *w)
    # 等风险×1.5(满仓杠杆现金垫)
    show(f"等风险x1.33", *[min(x * 1.33, 1.0) for x in w])
    print(f"  腿波动: M4 {vs[0]:.1%} TREND {vs[1]:.1%} LOF {vs[2]:.1%} | "
          f"月度相关: M4-TR {pd.concat({'a': (1+a).resample('ME').prod()-1, 'b': (1+b).resample('ME').prod()-1}, axis=1).dropna().corr().iloc[0,1]:.2f} "
          f"M4-LOF {pd.concat({'a': (1+a).resample('ME').prod()-1, 'c': (1+c).resample('ME').prod()-1}, axis=1).dropna().corr().iloc[0,1]:.2f} "
          f"TR-LOF {pd.concat({'b': (1+b).resample('ME').prod()-1, 'c': (1+c).resample('ME').prod()-1}, axis=1).dropna().corr().iloc[0,1]:.2f}", flush=True)

    print("\n[+] 完成", flush=True)


if __name__ == "__main__":
    main()
