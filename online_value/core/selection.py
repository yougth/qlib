"""
选股 —— 可交易过滤 + Top20 取仓
==================================================================
这是**生产与回测唯一共用的选股路径**。上线目录里任何地方要选股, 都只能调这里的函数,
不允许再手写一遍 mask —— 手抄两份 mask 是"实盘悄悄偏离回测"最常见的来源。

三道过滤 (顺序无关, 逐股独立):
  1) 流动性: 信号日的20日均成交额 >= LIQ_THRESHOLD (真实500万元)
  2) 一字涨停: 判定日涨停 → 买不进, 剔除
  3) 停牌: 判定日无量 → 买不进, 剔除

判定日 (check_dt) 的选择:
  · 回测 / 历史复算: 用**执行日** —— 历史上执行日的涨停停牌是已知事实
  · 生产出清单时  : 执行日还在未来不可知, 用**信号日**状态代理,
                    实际下单时若某只当天真涨停, 用备选顺延兜底
"""
import numpy as np
import pandas as pd

from . import config


def liquidity_at(liq_table, sig, index, fallback_prev=False):
    """取信号日那一横截面的20日均成交额; 信号日不在表内则全 NaN (=全部不可交易)

    fallback_prev: 生产出清单时若信号日恰好不在流动性表内(极少见), 退到最近的前一个交易日。
    回测路径**必须**保持 False —— 静默回退会让历史结果无声漂移。
    """
    lv = liq_table.index.get_level_values(0)
    if sig in lv:
        return liq_table.xs(sig, level=0).reindex(index)
    if fallback_prev:
        prev = [d for d in lv.unique() if d <= sig]
        if prev:
            return liq_table.xs(max(prev), level=0).reindex(index)
    return pd.Series(np.nan, index=index)


def tradable_universe(index, liq, check_dt, limit_up_set, susp_set):
    """返回可交易的 instrument Index"""
    idx = pd.Series(list(index), index=index)
    return index[(liq.notna()) & (liq >= config.LIQ_THRESHOLD)
                 & (~idx.map(lambda i: (check_dt, i) in limit_up_set))
                 & (~idx.map(lambda i: (check_dt, i) in susp_set))]


def pick_topk(vc, tradable, topk=None):
    """可交易集合内按 value_comp 降序取前 topk (NaN 自动排除), 返回 Series(instrument→score)"""
    return vc.reindex(tradable).nlargest(topk if topk is not None else config.TOPK)


def select_holdings(vc, liq_table, sig, check_dt, limit_up_set, susp_set, topk=None):
    """一步到位: value_comp + 三道过滤 → Top-k 持仓。返回 (holdings_series, n_dropped)"""
    trd = tradable_universe(vc.index, liquidity_at(liq_table, sig, vc.index),
                            check_dt, limit_up_set, susp_set)
    return pick_topk(vc, trd, topk), len(vc) - len(trd)
