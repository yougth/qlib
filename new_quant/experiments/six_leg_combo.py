#!/usr/bin/env python3
"""experiments.six_leg_combo —— 六策略统一年化口径 + 关联矩阵 + 博文理论组合
================================================================================
六腿:
  旧三 (qlib/quant rolling10y_nav.csv, 2020-01~2026-07): ICW_SW / VG / VGH
  新三 (new_quant): M4排雷 / TREND跨资产趋势 / LOF折价
口径: 公共日历内日频收益混合, 腿间再平衡成本未计(乐观偏差, 月频腿影响<0.3pp)
组合(博文√N+风险平价):
  等权6 / 等权3簇 / 风险平价6(满仓与1.33现金垫) / 簇风险平价
  进攻网格: ICW_SW与M4内部配比 × TREND% × LOF% × VGH/VG补充, 单腿≤60%
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402
from core.backtest import calc_metrics  # noqa: E402
from strategies.trend_etf import load_etf_panel, trend_rebalances  # noqa: E402
from experiments.trend_attrib import backtest_attrib  # noqa: E402
from experiments.lof_strategy import build as lof_build, backtest as lof_backtest  # noqa: E402
from experiments.lof_research import load_panels as lof_panels  # noqa: E402

OLD_NAV = "/Users/11164591/Documents/Qoder目录/qlib/quant/outputs/rolling10y_nav.csv"


def load_legs():
    legs = {}
    old = pd.read_csv(OLD_NAV, sep="\t", index_col=0, parse_dates=True)
    for c in ["ICW_SW", "VG", "VGH"]:
        legs[c] = old[c].pct_change().dropna()

    legs["M4"] = pd.read_csv(os.path.join(config.NEW_DIR, "data", "m4_daily.csv"),
                             index_col=0, parse_dates=True)["ret"]

    px = load_etf_panel(config.BT_START, config.BT_END)
    r_tr, _, _, _ = backtest_attrib(px, trend_rebalances(px))
    legs["TREND"] = r_tr

    P, A, N = lof_panels()
    Nf = N.reindex(P.index).ffill()
    D = P / Nf.shift(1) - 1
    me = P.resample("ME").last()
    D_me = D.resample("ME").last()
    days = A.notna().resample("ME").sum()
    amt20 = A.rolling(20).mean().resample("ME").last()
    r_lof, _ = lof_backtest(lof_build(D_me, me, amt20, days, "topN", n=10), P)
    legs["LOF"] = r_lof
    return legs


def main():
    legs = load_legs()
    print("[CHECK] 各腿原始区间:", flush=True)
    for k, r in legs.items():
        print(f"  {k:<8}{r.index.min().date()} ~ {r.index.max().date()}  "
              f"{len(r)}行", flush=True)

    df = pd.DataFrame(legs)
    common = df.dropna().index                     # 六腿齐备日
    X = df.loc[common]
    print(f"\n[CHECK] 公共窗口: {common.min().date()} ~ {common.max().date()} "
          f"({len(common)}日)", flush=True)

    print(f"\n{'='*100}\n  1) 六腿单跑 (统一公共窗口, 成本已在各腿内部)\n{'='*100}",
          flush=True)
    hdr = f"{'策略':<10}{'年化':>9}{'波动':>9}{'Sharpe':>8}{'回撤':>9}{'Calmar':>8}{'月胜率':>8}"
    print(hdr, flush=True)
    stats = {}
    for c in X.columns:
        m = calc_metrics(X[c])
        mr = (1 + X[c]).resample("ME").prod() - 1
        stats[c] = m
        print(f"{c:<10}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}{m['Sharpe']:>8.2f}"
              f"{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}{(mr > 0).mean():>8.0%}",
              flush=True)

    print(f"\n{'='*100}\n  2) 月度收益相关矩阵 (六腿)\n{'='*100}", flush=True)
    mr = (1 + X).resample("ME").prod() - 1
    print(mr.corr().round(3).to_string(), flush=True)

    # ---- 3) 理论组合 ----
    print(f"\n{'='*100}\n  3) 博文理论组合 (√N低相关 + 风险平价)\n{'='*100}",
          flush=True)
    print(hdr, flush=True)
    vols = X.std() * np.sqrt(244)

    def show(tag, w, normalize=True):
        w = pd.Series(w, dtype=float)
        w = w[w > 0]
        if w.sum() == 0:
            return None
        if normalize:
            w = w / w.sum()
        r = (X[w.index] * w).sum(axis=1)
        m = calc_metrics(r)
        wr = (1 + r).resample("ME").prod() - 1
        wt = "/".join(f"{k}{v:.0%}" for k, v in w.items())
        print(f"{tag:<10}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}"
              f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}"
              f"{(wr > 0).mean():>8.0%}  [{wt}]", flush=True)
        return m

    eq6 = {c: 1 for c in X.columns}
    show("等权6", eq6)
    eq3 = {"ICW_SW": 0.125, "VG": 0.125, "VGH": 0.125, "M4": 0.125,
           "TREND": 0.25, "LOF": 0.25}
    show("等权3簇", eq3)
    rp = (1 / vols)
    show("风险平价6", rp.to_dict())
    rp133 = (1 / vols) * (1.33 / (1 / vols).sum())
    show("RPx1.33", rp133.to_dict(), normalize=False)
    # 簇风险平价: 股簇内部等权后按1/簇波动
    eq_cluster = (0.25 * X[["ICW_SW", "VG", "VGH", "M4"]].sum(axis=1))
    cl = pd.DataFrame({"股簇": eq_cluster, "TREND": X["TREND"],
                       "LOF": X["LOF"]})
    cv = cl.std() * np.sqrt(244)
    cw = (1 / cv)
    r = (cl * (cw / cw.sum())).sum(axis=1)
    m = calc_metrics(r)
    print(f"{'簇RP':<10}{m['年化收益']:>9.2%}{m['年化波动']:>9.2%}"
          f"{m['Sharpe']:>8.2f}{m['最大回撤']:>9.1%}{m['Calmar']:>8.2f}"
          f"  [股簇{(cw/cw.sum())['股簇']:.0%}/TREND{(cw/cw.sum())['TREND']:.0%}/"
          f"LOF{(cw/cw.sum())['LOF']:.0%}]", flush=True)

    # ---- 4) 进攻网格: 找最高年化 (约束: 相关<0.5优先, 单腿≤60%, 六腿可用) ----
    print(f"\n{'='*100}\n  4) 进攻网格 (ICW_SW/M4配比 × TREND × LOF × VG/VGH, 单腿≤60%)\n"
          f"{'='*100}", flush=True)
    rows = []
    for p_icw in range(0, 61, 10):              # ICW_SW 占股池比例
        for w_tr in range(0, 61, 10):           # TREND
            for w_lof in range(0, 41, 10):      # LOF
                rest = 100 - w_tr - w_lof
                if rest <= 0:
                    continue
                for vgh_pct in range(0, 61, 20):  # 防御股(VG+VGH)占股池比
                    off = rest * (100 - vgh_pct) / 100
                    w_icw = off * p_icw / 100
                    w_m4 = off * (100 - p_icw) / 100
                    w_vg = rest * (vgh_pct / 2) / 100
                    w_vgh = rest * (vgh_pct / 2) / 100
                    w = {"ICW_SW": w_icw, "M4": w_m4, "VG": w_vg,
                         "VGH": w_vgh, "TREND": w_tr, "LOF": w_lof}
                    if max(w.values()) > 60:
                        continue
                    r = sum(X[k] * (v / 100) for k, v in w.items())
                    m = calc_metrics(r)
                    rows.append({"w": w, **m})
    R = pd.DataFrame(rows)
    top_cagr = R.sort_values("年化收益", ascending=False).head(6)
    top_sharpe = R.sort_values("Sharpe", ascending=False).head(6)
    print("\n  ★ 年化Top6 (满仓约束, 权重和=100%):", flush=True)
    for _, row in top_cagr.iterrows():
        wt = "/".join(f"{k}{v:.0f}%" for k, v in row["w"].items() if v > 0)
        print(f"  年化{row['年化收益']:>7.2%} Sharpe{row['Sharpe']:>5.2f} "
              f"回撤{row['最大回撤']:>7.1%} Calmar{row['Calmar']:>5.2f}  [{wt}]",
              flush=True)
    print("\n  ★ SharpeTop6:", flush=True)
    for _, row in top_sharpe.iterrows():
        wt = "/".join(f"{k}{v:.0f}%" for k, v in row["w"].items() if v > 0)
        print(f"  Sharpe{row['Sharpe']:>5.2f} 年化{row['年化收益']:>7.2%} "
              f"回撤{row['最大回撤']:>7.1%} Calmar{row['Calmar']:>5.2f}  [{wt}]",
              flush=True)

    print("\n[+] 完成", flush=True)


if __name__ == "__main__":
    main()
