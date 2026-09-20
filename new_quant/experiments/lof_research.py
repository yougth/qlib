#!/usr/bin/env python3
"""experiments.lof_research —— 方向2 LOF折溢价 研究版
================================================================================
数据: data/lof/ 80只(成交额前80) px(新浪场内) + nav(东财净值)
PIT: 折价率 d(T) = close(T)/nav(T-1) - 1  (净值T晚披露 → 用T-1净值, T月末决策, T+1起持有)
研究:
  R1 横截面: 月末按折价率分3组 → 次月场内收益 (折价收敛是否真实赚差)
  R2 时序:   每只LOF折价率自身历史z分位 < -1.5 (异常深折价) → 次月收益
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402

LOF_DIR = os.path.join(config.NEW_DIR, "data", "lof")
MIN_DAYS = 10  # 月内至少10个交易日有成交才纳入(过滤定开LOF)


def load_panels():
    px, nav, amt = {}, {}, {}
    for f in sorted(os.listdir(LOF_DIR)):
        if not f.endswith("_px.csv"):
            continue
        code = f[:6]
        p = pd.read_csv(os.path.join(LOF_DIR, f))
        p["date"] = pd.to_datetime(p["date"])
        p = p.set_index("date").sort_index()
        px[code] = p["close"]
        amt[code] = p["amount"]
        nf = os.path.join(LOF_DIR, f"{code}_nav.csv")
        if os.path.exists(nf):
            n = pd.read_csv(nf)
            n["净值日期"] = pd.to_datetime(n["净值日期"])
            nav[code] = n.set_index("净值日期")["单位净值"].sort_index()
    P = pd.DataFrame(px).sort_index()
    A = pd.DataFrame(amt).sort_index()
    N = pd.DataFrame(nav).sort_index()
    return P, A, N


def main():
    print("[1] 加载 LOF 面板...", flush=True)
    P, A, N = load_panels()
    print(f"    价格面板 {P.shape} | 净值面板 {N.shape} | 区间 {P.index.min().date()} ~ {P.index.max().date()}",
          flush=True)

    # 净值对齐到行情交易日并 ffilled; 折价率用 T-1 净值 (PIT)
    Nf = N.reindex(P.index).ffill()
    D = P / Nf.shift(1) - 1          # 折价率(负=折价)
    days = A.notna().resample("ME").sum()          # 月内有效交易日数

    me = P.resample("ME").last()
    d_me = D.resample("ME").last()
    fwd = me.shift(-1) / me - 1                    # 次月场内收益
    valid = (days >= MIN_DAYS).reindex(d_me.index).fillna(False)

    # ---- R0: 折价率分布 ----
    dv = d_me.where(valid).stack()
    print(f"\n[R0] 折价率分布 (月度样本 n={len(dv)}):", flush=True)
    print(f"    中位 {dv.median():.2%} | 25% {dv.quantile(.25):.2%} | 75% {dv.quantile(.75):.2%} "
          f"| 5% {dv.quantile(.05):.2%} | 95% {dv.quantile(.95):.2%}", flush=True)
    print(f"    折价(<-0.5%)占比 {np.mean(dv < -0.005):.1%} | 溢价(>0.5%)占比 {np.mean(dv > 0.005):.1%}",
          flush=True)

    # ---- R1: 横截面 3 分组 → 次月收益 ----
    recs = []
    for t in d_me.index:
        if t not in fwd.index or pd.isna(d_me.loc[t]).all():
            continue
        x = d_me.loc[t][valid.loc[t] if t in valid.index else []]
        y = fwd.loc[t].reindex(x.index) if t in fwd.index else None
        ok = x.notna() & y.notna()
        if ok.sum() < 20:
            continue
        x, y = x[ok], y[ok]
        q = pd.qcut(x, 3, labels=["deep", "mid", "prem"], duplicates="drop")
        recs.append({"date": t, "n": int(ok.sum()),
                     "deep": y[q == "deep"].mean(),
                     "mid": y[q == "mid"].mean(),
                     "prem": y[q == "prem"].mean(),
                     "ls": y[q == "deep"].mean() - y[q == "prem"].mean()})
    R1 = pd.DataFrame(recs).set_index("date")
    print(f"\n[R1] 横截面折价三分组 → 次月场内收益 (n月={len(R1)}):", flush=True)
    for c, name in [("deep", "深折价组"), ("mid", "中间组"), ("prem", "溢价组")]:
        m, s = R1[c].mean(), R1[c].std()
        print(f"    {name}: 月均 {m:>7.2%} | t = {m / s * np.sqrt(len(R1)):>5.2f} "
              f"| 月胜率 {(R1[c] > 0).mean():.0%}", flush=True)
    m, s = R1["ls"].mean(), R1["ls"].std()
    print(f"    多空差(深-溢): 月均 {m:>7.2%} | t = {m / s * np.sqrt(len(R1)):>5.2f} "
          f"| 月胜率 {(R1['ls'] > 0).mean():.0%}", flush=True)
    R1["cum_deep"] = (1 + R1["deep"]).cumprod()
    R1["cum_prem"] = (1 + R1["prem"]).cumprod()
    print(f"    累计(多空分组均, 不叠加): 深折价 {R1['cum_deep'].iloc[-1] - 1:.1%} vs "
          f"溢价 {R1['cum_prem'].iloc[-1] - 1:.1%}", flush=True)

    # ---- R2: 时序 自身z分位 异常折价 ----
    z = d_me.where(valid)
    zz = (z - z.rolling(24, min_periods=18).mean()) / z.rolling(24, min_periods=18).std()
    rows = []
    for t in zz.index:
        if t not in fwd.index:
            continue
        x = zz.loc[t]
        y = fwd.loc[t].reindex(x.index)
        ok = x.notna() & y.notna()
        if ok.sum() < 20:
            continue
        x, y = x[ok], y[ok]
        deep, normal = x < -1.5, (x >= -1.5) & (x <= 1.5)
        rows.append({"date": t, "n": int(ok.sum()),
                     "deep": y[deep].mean() if deep.sum() >= 3 else np.nan,
                     "deep_n": int(deep.sum()),
                     "normal": y[normal].mean()})
    R2 = pd.DataFrame(rows).set_index("date").dropna()
    print(f"\n[R2] 自身异常折价 (z<-1.5) → 次月收益 (n月={len(R2)}):", flush=True)
    m, s = R2["deep"].mean(), R2["deep"].std()
    print(f"    异常折价样本: 月均深折价只数 {R2['deep_n'].mean():.1f} | 次月收益均 {m:>7.2%} "
          f"| t = {m / s * np.sqrt(len(R2)):>5.2f}", flush=True)
    m2, s2 = R2["normal"].mean(), R2["normal"].std()
    print(f"    正常样本同期: 次月收益均 {m2:>7.2%} | 差 {m - m2:>7.2%} "
          f"| 胜率 {(R2['deep'] > R2['normal']).mean():.0%}", flush=True)

    # ---- 年度拆解 (R1 多空) ----
    print("\n[R3] R1 多空差 分年:", flush=True)
    R1["year"] = R1.index.year
    for yr, g in R1.groupby("year"):
        print(f"    {yr}: 多空 {g['ls'].mean():>7.2%} | 月数 {len(g)}", flush=True)

    out = os.path.join(config.OUT_DIR, "lof_report.txt")
    with open(out, "w") as f:
        f.write("LOF折溢价研究(方向2, 研究版)\n")
        f.write(f"样本: 成交额前80只, {P.index.min().date()} ~ {P.index.max().date()}\n\n")
        f.write(f"[R0] 折价率中位 {dv.median():.2%} | 折价占比 {np.mean(dv < -0.005):.1%}\n\n")
        f.write("[R1] 横截面(月度) 深折价/中间/溢价 次月收益:\n")
        f.write(R1[["deep", "mid", "prem", "ls", "n"]].to_string())
        f.write(f"\n\n[R2] 自身异常折价(z<-1.5) 次月收益:\n")
        f.write(R2.to_string())
    print(f"\n[+] 报告 → {out}", flush=True)


if __name__ == "__main__":
    main()
