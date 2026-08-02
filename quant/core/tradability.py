"""
core.tradability —— 可交易性 / 流动性 + 月末信号日
================================================================================
- build_tradability: 一字涨停 / 停牌 / 20日均成交额 (仅测试段, model-agnostic 过滤)
- get_month_end_dates: 月末交易日 (信号日)
"""
import numpy as np
import pandas as pd
from qlib.data import D

from . import config


def build_tradability(universe, start, end):
    # $factor: 复权因子。$close 是复权价, 真实成交价 = $close / $factor。
    # 流动性口径必须用真实价 —— 复权价与真实价可以差几十倍 (老数据里广汇能源
    # 复权后 0.12 元 vs 真实 ~4.5 元), 用复权价算成交额会把它低估 37 倍, 直接
    # 把本该被流动性过滤掉/留下的股票搞反。
    px = D.features(list(universe), ["$close", "$open", "$high", "$low", "$volume", "$factor"],
                    start_time=pd.Timestamp(start) - pd.Timedelta(days=60), end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[CHECK] 可交易性数据拉取失败 ({start}~{end}), 拒绝静默跳过!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "open", "high", "low", "volume", "factor"]
    px = px.sort_values(["instrument", "datetime"])
    px["prev_close"] = px.groupby("instrument")["close"].shift(1)
    ret = (px["close"] - px["prev_close"]) / px["prev_close"]
    one_line = (px["open"] == px["high"]) & (px["high"] == px["low"]) & \
               (px["low"] == px["close"]) & (ret > 0.09)
    limit_up = set(zip(px.loc[one_line, "datetime"], px.loc[one_line, "instrument"]))
    susp = px[(px["volume"].isna()) | (px["volume"] == 0)]
    suspension = set(zip(susp["datetime"], susp["instrument"]))
    # 流动性: 真实价 = close/factor, $volume单位为手 → 成交额 = 真实价*volume*100
    fac = px["factor"].where(px["factor"] > 0)
    px["amount"] = (px["close"] / fac) * px["volume"] * 100
    px["avg20"] = px.groupby("instrument")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    liq = px.set_index(["datetime", "instrument"])["avg20"]
    # ---- check: 单位自检, 池内中位数成交额应在合理量级 (1e6~1e11) ----
    med = liq.dropna().median()
    if not (1e6 < med < 1e11):
        raise RuntimeError(f"[CHECK] 流动性单位异常: 池内20日均成交额中位数={med:.3g}元!")
    # ---- check: 复权因子缺失率 (缺了就退化成复权价口径, 必须让人看见) ----
    fac_miss = float(fac.isna().mean())
    if fac_miss > 0.01:
        raise RuntimeError(f"[CHECK] $factor 缺失率 {fac_miss:.1%} > 1%, "
                           f"流动性口径不可信!")
    print(f"    [流动性] 中位数20日均成交额={med/1e8:.2f}亿, 一字涨停{len(limit_up)}条, "
          f"停牌{len(suspension)}条", flush=True)
    return limit_up, suspension, liq


def get_month_end_dates(cal, start, end):
    dates = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    out = []
    for i, d in enumerate(dates):
        if i + 1 == len(dates) or dates[i + 1].month != d.month:
            out.append(d)
    return out


def build_candidates(base_idx, sig_dt, limit_up, susp, liq):
    """model-agnostic 可交易候选集: 过滤一字涨停/停牌/流动性不足"""
    cand = []
    for inst in base_idx:
        if (sig_dt, inst) in limit_up or (sig_dt, inst) in susp:
            continue
        a = liq.get((sig_dt, inst), np.nan)
        if pd.isna(a) or a < config.LIQ_THRESHOLD:
            continue
        cand.append(inst)
    return pd.Index(cand)
