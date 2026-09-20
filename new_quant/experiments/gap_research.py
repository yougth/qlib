#!/usr/bin/env python3
"""experiments.gap_research —— 方向8(QVIX)/9(回购)/1(期货) 研究版检验
================================================================================
方向8 QVIX波动率择时:
  · 统计: QVIX分位(过去252d) → 下月中证500/沪深300收益 (恐慌是否有预测力)
  · 叠加: QVIX>80分位 时G0降半仓 (文章"市场发疯时少下注"; 安全带非预测器)
方向9 回购事件研究:
  · 事件=回购起始时间; 检验事件后1/3/6月相对中证500超额
  · 披露: 当期快照的覆盖偏差 (久远已完成回购缺席)
方向1 期货主力连续:
  · 数据已落盘(6品种); 本轮只做趋势性统计 (期限结构需单合约重建, 留待下轮)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")


def load_qvix():
    q = pd.read_csv(os.path.join(DATA, "index_option_50etf_qvix.csv"))
    q["date"] = pd.to_datetime(q["date"])
    return q.set_index("date")["close"].astype(float).sort_index()


def qvix_research(rets_g0_path):
    q = load_qvix()
    print(f"[CHECK] QVIX: {q.index[0].date()} ~ {q.index[-1].date()} "
          f"({len(q)}天) | 中位 {q.median():.1f}", flush=True)

    # 指数月收益
    def _idx(stem):
        d = pd.read_parquet(os.path.join(config.TENCENT_DIR, stem + ".parquet"))
        d["date"] = pd.to_datetime(d["date"])
        s = d.set_index("date")["close"].astype(float).sort_index()
        return s

    c500 = _idx("sh000905")
    m500 = (1 + c500.pct_change().fillna(0)).resample("ME").prod() - 1

    # QVIX月末分位 (过去252交易日, PIT)
    pct = q.rolling(252).rank(pct=True)
    q_m = pct.resample("ME").last()
    df = pd.concat({"qvix_pct": q_m, "fwd_500": m500.shift(-1)}, axis=1).dropna()
    df = df[df.index >= "2019-01-01"]
    print(f"\n  == 方向8: QVIX月末分位 → 下月中证500收益 "
          f"({df.index[0].date()}~{df.index[-1].date()}, {len(df)}月) ==", flush=True)
    bins = [(-0.01, 0.2, "0-20% (平静)"), (0.2, 0.4, "20-40%"),
            (0.4, 0.6, "40-60%"), (0.6, 0.8, "60-80%"),
            (0.8, 1.01, "80-100% (恐慌)")]
    for lo, hi, name in bins:
        g = df[(df["qvix_pct"] > lo) & (df["qvix_pct"] <= hi)]["fwd_500"]
        if len(g):
            print(f"    {name:<14} n={len(g):3d} | 下月均值 {g.mean():+7.2%} | "
                  f"胜率 {(g > 0).mean():5.0%} | 中位 {g.median():+7.2%}",
                  flush=True)

    # QVIX极值月后的反弹
    hi_m = df[df["qvix_pct"] > 0.9].index
    if len(hi_m):
        nxt = df.loc[hi_m, "fwd_500"]
        print(f"    QVIX>90分位月 (n={len(nxt)}): 下月均值 {nxt.mean():+.2%} "
              f"| 胜率 {(nxt > 0).mean():.0%}", flush=True)

    # ---- 叠加到 G0: QVIX>80分位 → 当月半仓 ----
    if os.path.exists(rets_g0_path):
        r = pd.read_csv(rets_g0_path, index_col=0, parse_dates=True)["G0"]
        q_d = pct.reindex(r.index).ffill()
        half = (q_d > 0.8).astype(float) * 0.5 + 0.5   # 0.5(恐慌)~1.0(正常)
        r_t = r * half
        n_h = int((q_d > 0.8).sum())
        def _m(x):
            ar = (1 + x).prod() ** (244 / len(x)) - 1
            vol = x.std() * np.sqrt(244)
            nav = (1 + x).cumprod()
            dd = (nav / nav.cummax() - 1).min()
            return ar, vol, ar / vol, dd
        a0, v0, s0, d0 = _m(r)
        a1, v1, s1, d1 = _m(r_t)
        print(f"\n    G0 + QVIX恐慌半仓 (触发日 {n_h}/{len(r)} = "
              f"{n_h/len(r):.0%}):", flush=True)
        print(f"      年化 {a0:.2%} → {a1:.2%} | Sharpe {s0:.2f} → {s1:.2f} | "
              f"回撤 {d0:.1%} → {d1:.1%}", flush=True)


def repurchase_research():
    rp = pd.read_csv(os.path.join(DATA, "repurchase.csv"), dtype={"股票代码": str})
    rp["start"] = pd.to_datetime(rp["回购起始时间"], errors="coerce")
    rp["ann"] = pd.to_datetime(rp["最新公告日期"], errors="coerce")
    rp["amt"] = pd.to_numeric(rp["已回购金额"], errors="coerce")
    rp = rp[rp["start"].notna()].copy()
    rp["sym"] = np.where(rp["股票代码"].str[:2].isin(["60", "68"]),
                         "SH" + rp["股票代码"], "SZ" + rp["股票代码"])
    print(f"\n  == 方向9: 回购事件研究 (快照 {len(rp)}条含起始时间) ==", flush=True)
    print(f"    起始时间分布: {rp['start'].dt.year.value_counts().sort_index().to_dict()}",
          flush=True)
    print(f"    披露: 当期快照, 久远已完成回购可能缺席 (覆盖偏差, 结论只作方向参考)",
          flush=True)

    # 指数与个股行情: 用 tencent parquet
    def _px(stem):
        f = os.path.join(config.TENCENT_DIR, stem + ".parquet")
        if not os.path.exists(f):
            return None
        d = pd.read_parquet(f, columns=["date", "close"])
        d["date"] = pd.to_datetime(d["date"])
        return d.set_index("date")["close"].astype(float).sort_index()

    c500 = _px("sh000905")

    # 事件后超额: 事件日取该股起始日之后的第一个交易日
    px = {}
    events = rp[["sym", "start"]].dropna()
    for s in events["sym"].unique():
        if s not in px:
            px[s] = _px(s.lower())
    res = {21: [], 63: [], 126: []}
    for s, st in events.itertuples(index=False):
        p = px.get(s)
        if p is None:
            continue
        idx = p.index
        pos = idx.searchsorted(st)
        if pos + 126 >= len(idx):
            continue
        for h in res:
            fwd_s = p.iloc[pos + h] / p.iloc[pos] - 1
            fwd_b = (c500.iloc[pos + h] / c500.iloc[pos] - 1
                     if pos + h < len(c500) else np.nan)
            if fwd_b == fwd_b:
                res[h].append(fwd_s - fwd_b)
    for h, v in res.items():
        if v:
            v = np.array(v)
            print(f"    事件后{h:>3}日超额: n={len(v):4d} | 均值 {v.mean():+6.2%} | "
                  f"中位 {np.median(v):+6.2%} | 胜率 {(v > 0).mean():5.0%} | "
                  f"t值 {v.mean()/(v.std()/np.sqrt(len(v))):+.2f}", flush=True)


def futures_research():
    print(f"\n  == 方向1: 期货主力连续 (数据层完成, 期限结构留待下轮) ==", flush=True)
    for sym, name in [("RB0", "螺纹钢"), ("CU0", "沪铜"), ("AU0", "沪金")]:
        p = os.path.join(DATA, "futures_main", f"{sym}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["date"] = pd.to_datetime(d["日期"])
        d = d.set_index("date")["收盘价"].astype(float).sort_index().to_frame("close")
        mom = d["close"] / d["close"].shift(252) - 1
        fwd = d["close"].shift(-21) / d["close"] - 1
        df = pd.concat({"mom": mom, "fwd": fwd}, axis=1).dropna()
        df = df[df.index >= "2016-01-01"]
        up = df[df["mom"] > 0]["fwd"]
        dn = df[df["mom"] <= 0]["fwd"]
        print(f"    {name}: 12月动量>0 → 未来1月均值 {up.mean():+6.2%} (n={len(up)}) "
              f"| ≤0 → {dn.mean():+6.2%} (n={len(dn)})", flush=True)


if __name__ == "__main__":
    rets_path = os.path.join(config.OUT_DIR, "lowatt_combo_returns.csv")
    qvix_research(rets_path)
    repurchase_research()
    futures_research()
