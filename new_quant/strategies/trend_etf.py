"""strategies.trend_etf —— 方向7 跨资产趋势跟踪 (ETF篮子)

规则 (v1, 月频):
  · 信号: 月末; 入场条件 = 12-1 动量 > 0 (P[t-21]/P[t-250]-1, 后复权)
  · 权重: 入场资产按 1/σ(60日) 逆波动率配权; 组合估计波动(对角近似)超过
    目标波动则整体降杠杆(剩余为现金), 无杠杆
  · 执行: T+1; ETF往返成本 0.1%
基线: 全篮子月度等权再平衡 (ETF_EW); 基准: 沪深300。
注意: ETF hfq 序列含分红再投资; 若某 ETF 无 hfq(接口不支持)则回退 raw 并
  显式记录 —— 有分红的ETF(红利ETF等)回退口径会低估其收益, 报告中披露。
"""
import os

import numpy as np
import pandas as pd

from core import config


def load_etf_panel(start, end):
    files = sorted(os.listdir(config.ETF_DIR)) if os.path.isdir(config.ETF_DIR) else []
    if not files:
        raise RuntimeError("[CHECK] ETF 数据缺失, 请先运行 tools/fetch_etf_ohlcv.py")
    frames, modes = [], {}
    for f in files:
        if not f.endswith(".parquet"):
            continue
        sym = f.replace(".parquet", "")
        d = pd.read_parquet(os.path.join(config.ETF_DIR, f))
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]
        if not len(d):
            continue
        d["sym"] = sym
        frames.append(d[["date", "sym", "close"]])
        modes[sym] = d["mode"].iloc[0] if "mode" in d.columns and len(d) else "?"
    px = pd.concat(frames, ignore_index=True).pivot(
        index="date", columns="sym", values="close").sort_index().ffill()
    raw_syms = [s for s, m in modes.items() if m != "hfq"]
    print(f"[CHECK] ETF面板: {px.shape[1]} 只 | 覆盖 {px.index[0].date()} ~ "
          f"{px.index[-1].date()} | raw回退口径: "
          f"{raw_syms if raw_syms else '无'}", flush=True)
    return px


def _month_end_positions(idx):
    out = []
    for i, t in enumerate(idx):
        if i + 1 < len(idx) and idx[i + 1].month != t.month:
            out.append(i)
    return out


def trend_rebalances(px, target_vol=None):
    target_vol = target_vol or config.TREND_TARGET_VOL
    idx = px.index
    out = []
    for i in _month_end_positions(idx):
        if i < 250:
            continue
        mom = px.iloc[i - 21] / px.iloc[i - 250] - 1
        vol = px.iloc[i - 60:i].pct_change().std() * np.sqrt(244)
        ok = mom[(mom > 0) & (vol > 0) & vol.notna()]
        if not len(ok):
            out.append((idx[i + 1], {}))
            continue
        iv = 1.0 / vol[ok.index]
        w = iv / iv.sum()
        est = float(np.sqrt(((w * vol[ok.index]) ** 2).sum()))
        k = min(1.0, target_vol / est) if est > 0 else 1.0
        w = w * k
        out.append((idx[i + 1], {s: float(x) for s, x in w.items()}))
    return out


def ew_rebalances(px):
    """全篮子等权基线 —— 与 trend_rebalances 同样跳过前250个交易日,
    保证两者回测区间完全一致 (否则基线早半年起跑, 对比不公平)"""
    idx = px.index
    out = []
    for i in _month_end_positions(idx):
        if i < 250:
            continue
        cols = px.iloc[i].dropna().index
        w = {s: 1.0 / len(cols) for s in cols} if len(cols) else {}
        out.append((idx[i + 1], w))
    return out
