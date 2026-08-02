"""
core.data —— 数据访问层 (qlib 初始化 / 日历 / 价格矩阵 / csi300 基准)
================================================================================
统一入口, 保证所有引擎(滚动/基准)使用同一份全市场行情与同一基准。

数据源选择 (环境变量 QLIB_PROVIDER):
  · 新全市场 bin (默认): tools/fetch_ohlcv.py 抓腾讯后复权 → tools/build_qlib_bin.py
    转成 qlib bin, 覆盖 5400+ 只 (含退市), 2012~2026。这是唯一可信的行情源 ——
    老的 cn_data/cn_data_fixed 只有 358 只有 2020-09-25 后的行情, 其余截断在拼接点。
  · 老数据 (fallback): 仍可用 QLIB_PROVIDER_FIXED 环境变量切回旧的 cn_data_fixed。
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
    """初始化 qlib 到全市场新 bin (幂等)。可用环境变量覆盖数据源。"""
    global _INITED
    if _INITED:
        return
    provider = os.environ.get("QLIB_PROVIDER") or config.QLIB_PROVIDER
    if not os.path.exists(f"{provider}/features"):
        raise RuntimeError(f"[CHECK] qlib 数据目录不存在: {provider}\n"
                           f"    请先运行 tools/fetch_ohlcv.py + tools/build_qlib_bin.py\n"
                           f"    或设 QLIB_PROVIDER 环境变量指向已有数据目录")
    qlib.init(provider_uri=provider, region=REG_CN)
    _INITED = True
    print(f"[qlib] 数据源: {provider}", flush=True)


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


def assert_market_coverage(universe, start, end, min_ratio=0.95, tag=""):
    """行情覆盖率门禁: 池内股票必须在整个回测段有行情, 否则静默消失 = 选择偏差。

    历史教训 (2026-07-31 发现): cn_data 里 3943 只股票中只有 358 只有 2020-09-25 之后
    的行情, 其余 3585 只全部截断在拼接点。而那 358 只恰好≈用最新财务数据算出的近年
    股票池 (W2025/W2026 覆盖 100%, W2020 仅 54.7%) —— 等于用“今天才知道的名单”决定
    了历史回测宇宙。缺行情的池成员在 build_candidates 里因 liq 为 NaN 被静默 continue,
    既不报错也不计数, 所以跑了几十轮回测都没人发现。

    改进 (2026-08-01): 区分“正常退市”和“系统性截断”。新全市场数据有 5400+ 只
    含退市股, 某些池成员在回测段内退市是正常的 (不等于偏差); 但如果大量股票在同一
    天齐刷刷终止, 那是数据被截断, 不是退市。
    """
    px = D.features(list(universe), ["$close"], start_time=start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[CHECK] {tag} 行情拉取失败 ({start}~{end})!")
    close = px["$close"].unstack(level=0)                    # date × instrument
    n_all = len(universe)
    # 段末仍有行情 = 真正可全程持有; 只看段首会漏掉中途截断
    tail = close.tail(20)
    covered = [c for c in close.columns if tail[c].notna().any()]
    ratio = len(covered) / max(n_all, 1)
    missing = sorted(set(map(str, universe)) - set(map(str, covered)))
    print(f"[CHECK] {tag} 行情覆盖率 {len(covered)}/{n_all} = {ratio:.1%}"
          + (f", 缺失例: {missing[:5]}" if missing else ""), flush=True)

    # ---- 区分退市 vs 截断: 检查缺失股票的最后行情日是否扎堆 ----
    if missing:
        last_dates = []
        for m in missing:
            col = m if m in close.columns else None
            if col is not None:
                last = close[col].last_valid_index()
                if last is not None:
                    last_dates.append(last.date())
        if last_dates:
            from collections import Counter
            c = Counter(last_dates)
            top_date, top_n = c.most_common(1)[0]
            if top_n > 20:
                raise RuntimeError(
                    f"[CHECK] {tag} {top_n} 只股票行情齐刷刷终止于 {top_date}, 疑似数据截断\n"
                    f"    (退市不会扎堆在同一天), 拒绝回测! 请检查 qlib 数据源。")
    if ratio < min_ratio:
        raise RuntimeError(
            f"[CHECK] {tag} 行情覆盖率仅 {ratio:.1%} (<{min_ratio:.0%}), 拒绝回测!\n"
            f"    {n_all - len(covered)} 只池内股票在 {start}~{end} 段末无行情, 会被\n"
            f"    build_candidates 静默剔除 → 回测宇宙 = 有数据的那部分股票, 而数据\n"
            f"    是按今天的名单下载的 → 存活/前视选择偏差。请先补全市场行情数据。\n"
            f"    缺失(前20): {missing[:20]}")
    return ratio, missing


def forward_return_matrix(universe, xs, xe, horizon=None):
    """OOS IC 用: 信号段 horizon 日前瞻真实收益矩阵 (date × instrument)"""
    horizon = horizon or config.LABEL_HORIZON
    icpx = D.features(universe, ["$close"],
                      start_time=pd.Timestamp(xs) - pd.Timedelta(days=10),
                      end_time=pd.Timestamp(xe) + pd.Timedelta(days=60))
    icclose = icpx["$close"].unstack(level=0).sort_index()
    return icclose.shift(-horizon) / icclose - 1
