"""
core.data —— 数据访问层 (qlib 初始化 / 日历 / 价格矩阵 / csi300 基准)
================================================================================
统一入口, 保证所有引擎(滚动/基准)使用同一份修补后行情 cn_data_fixed 与同一基准。
"""
import os
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from . import config

_INITED = False


def init_qlib():
    """初始化 qlib 到修补后的 cn_data_fixed (幂等)"""
    global _INITED
    if _INITED:
        return
    if not os.path.exists(f"{config.QLIB_PROVIDER_FIXED}/features"):
        raise RuntimeError("[CHECK] cn_data_fixed 不存在, 请先运行 fix_qlib_seam.py!")
    qlib.init(provider_uri=config.QLIB_PROVIDER_FIXED, region=REG_CN)
    _INITED = True


def get_calendar(start="2014-01-01", end="2026-07-23"):
    cal = D.calendar(start_time=start, end_time=end)
    print(f"[CHECK] 交易日历: {cal[0].date()} ~ {cal[-1].date()} ({len(cal)}天)", flush=True)
    if cal[-1] < pd.Timestamp("2026-07-01"):
        raise RuntimeError("[CHECK] qlib日历未覆盖2026H1!")
    return cal


def load_price_matrix(all_insts, start="2019-11-01", end=None):
    """收盘价矩阵 date×instrument (ffill), 用于连续 NAV 回测"""
    end = end or config.BT_END
    px = D.features(all_insts, ["$close"], start_time=start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError("[CHECK] 回测价格矩阵拉取失败!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close"]
    return px.pivot(index="datetime", columns="instrument",
                    values="close").sort_index().ffill()


def load_benchmark():
    """沪深300基准: qlib 内 SH000300 止于 2020-09-25(断点后全0), 改用外部干净缓存"""
    if not os.path.exists(config.BENCH_CACHE):
        raise RuntimeError("[CHECK] csi300_cache.csv 不存在, 沪深300基准缺失!")
    braw = pd.read_csv(config.BENCH_CACHE, sep="\t")
    braw.columns = ["date", "close"]
    braw["date"] = pd.to_datetime(braw["date"])
    bclose = braw.set_index("date")["close"].astype(float).sort_index()
    if bclose.index[0] > pd.Timestamp("2019-12-01") or bclose.index[-1] < pd.Timestamp("2026-07-01"):
        raise RuntimeError(f"[CHECK] csi300_cache 覆盖不足: {bclose.index[0]}~{bclose.index[-1]}")
    if bclose.pct_change().abs().max() > 0.12:
        raise RuntimeError("[CHECK] csi300_cache 存在异常日收益(>12%), 数据可疑!")
    bench = bclose.pct_change().fillna(0)
    print(f"[CHECK] 沪深300基准(csi300_cache): {bclose.index[0].date()} ~ "
          f"{bclose.index[-1].date()} ({len(bclose)}天)", flush=True)
    return bench


def forward_return_matrix(universe, xs, xe, horizon=None):
    """OOS IC 用: 信号段 horizon 日前瞻真实收益矩阵 (date × instrument)"""
    horizon = horizon or config.LABEL_HORIZON
    icpx = D.features(universe, ["$close"],
                      start_time=pd.Timestamp(xs) - pd.Timedelta(days=10),
                      end_time=pd.Timestamp(xe) + pd.Timedelta(days=60))
    icclose = icpx["$close"].unstack(level=0).sort_index()
    return icclose.shift(-horizon) / icclose - 1
