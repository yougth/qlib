"""
行情数据 —— 价量矩阵 / 流动性 / 涨停停牌 / 基准
==================================================================
函数体逐字搬自:
  load_price_volume   ← v17_residual_model
  load_price_matrix   ← v14_ablation
  build_liquidity_table ← v14_ablation
  build_limit_up_set  ← v5_validation
  load_bench_returns  ← v15_model_fixed.get_bench_returns_ak

几个必须知道的口径细节:
  · close 矩阵 **ffill**: 停牌日沿用最后已知价, 组合估值才连续
  · volume 矩阵 **不 ffill**: 停牌判定依赖 volume 为 NaN/0
  · 流动性 = close * volume 的 20日均值 (min_periods=10), 前置回看60自然日让首日就有值
  · 一字涨停 = open==high==low==close 且日涨幅>9%  → 执行日买不进, 必须剔除
  · 基准用 akshare 缓存的沪深300 (qlib 的 SH000300 只到 2019 年)
"""
import numpy as np
import pandas as pd
from qlib.data import D

from . import config


def load_price_volume(insts, start, end):
    """价格+成交量矩阵, 用于流动性/组合估值。返回 (close_mat, volume_mat), close 已ffill"""
    px = D.features(list(insts), ["$close", "$volume"], start_time=start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[价量] 拉取失败 ({start}~{end})!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "volume"]
    close = px.pivot(index="datetime", columns="instrument", values="close").sort_index().ffill()
    vol = px.pivot(index="datetime", columns="instrument", values="volume").sort_index()
    return close, vol


def load_price_matrix(insts, start, end):
    px = D.features(list(insts), ["$close"], start_time=start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[价格] 数据拉取失败 ({start}~{end})!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close"]
    mat = px.pivot(index="datetime", columns="instrument", values="close").sort_index()
    return mat.ffill()   # 停牌用最后已知价


def build_liquidity_table(universe, start, end):
    """返回 Series, index=(datetime, instrument), 值=20日均成交额(qlib口径)"""
    lookback_start = pd.Timestamp(start) - pd.Timedelta(days=60)
    px = D.features(list(universe), ["$close", "$volume"],
                    start_time=lookback_start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[流动性] 数据拉取失败 ({start}~{end}), 拒绝静默跳过!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "volume"]
    px["amount"] = px["close"] * px["volume"]
    px = px.sort_values(["instrument", "datetime"])
    px["avg20"] = px.groupby("instrument")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    return px.set_index(["datetime", "instrument"])["avg20"]


def build_limit_up_set(universe, cal, verbose=True):
    """构建一字涨停集合与停牌集合: (date, instrument) — 当日不可买入"""
    if verbose:
        print("  --- 构建涨跌停/停牌集合 ---")
    price_df = D.features(list(universe), ["$close", "$open", "$high", "$low"],
                          start_time=cal[0], end_time=cal[-1])
    if price_df is None or len(price_df) == 0:
        return set(), set()
    price_df = price_df.reset_index()
    price_df.columns = ["instrument", "datetime", "close", "open", "high", "low"]

    # 日收益率
    price_df = price_df.sort_values(["instrument", "datetime"])
    price_df["prev_close"] = price_df.groupby("instrument")["close"].shift(1)
    price_df["daily_ret"] = (price_df["close"] - price_df["prev_close"]) / price_df["prev_close"]

    # 一字涨停: open == high == low == close, 且涨幅 > 9%
    limit_up_mask = (
        (price_df["open"] == price_df["high"]) &
        (price_df["high"] == price_df["low"]) &
        (price_df["low"] == price_df["close"]) &
        (price_df["daily_ret"] > 0.09)
    )
    limit_up_set = set(zip(price_df.loc[limit_up_mask, "datetime"], price_df.loc[limit_up_mask, "instrument"]))

    # 停牌: volume 为 NaN 或 0
    vol_df = D.features(list(universe), ["$volume"], start_time=cal[0], end_time=cal[-1])
    suspension_set = set()
    if vol_df is not None and len(vol_df) > 0:
        vol_df = vol_df.reset_index()
        vol_df.columns = ["instrument", "datetime", "volume"]
        susp = vol_df[(vol_df["volume"].isna()) | (vol_df["volume"] == 0)]
        suspension_set = set(zip(susp["datetime"], susp["instrument"]))

    if verbose:
        print(f"  一字涨停: {len(limit_up_set)} 条, 停牌: {len(suspension_set)} 条")
    return limit_up_set, suspension_set


def load_bench_returns(start, end):
    """沪深300日收益 (akshare 缓存, qlib 的 SH000300 只到2019年)"""
    df = pd.read_csv(config.BENCH_CACHE, sep='\t', parse_dates=["date"])
    s = df.set_index("date")["close"].sort_index().loc[start:end]
    return s.pct_change().fillna(0)
