"""
回测引擎 —— 精确份额组合回测 + 指标
==================================================================
函数体逐字搬自 v14_ablation.portfolio_backtest / calc_metrics 与 v18_full_stack.full_metrics /
trade_count。这套引擎是"份额级"的: 记录每只股票的持有股数, 按日收盘估值,
换仓日按当日收盘再平衡并扣成本 —— 不是简化的"等权收益平均", 所以能反映真实的
权重漂移与交易摩擦。

成本口径: cost = 组合市值 x 单边换手 x FEE_ROUNDTRIP(0.4%)
T+1 已体现在 exec_dt 的选取上(信号日打分, 次一交易日成交), 引擎本身不再延迟。
"""
import numpy as np
import pandas as pd

from . import config


def portfolio_backtest(rebalances, price_mat, cal_list=None):
    """
    rebalances: [(exec_dt, [目标等权持仓列表]), ...] 按时间排序
    返回: (日收益序列, 平均单边换手)
    """
    dates = [d for d in price_mat.index if d >= rebalances[0][0]]
    shares = {}
    value = 1.0
    nav, nav_dates = [], []
    turnovers = []
    ri = 0
    for d in dates:
        px = price_mat.loc[d]
        # 先按当日收盘估值
        if shares:
            value = sum(sh * px[inst] for inst, sh in shares.items() if pd.notna(px.get(inst)))
        # 调仓日: 当日收盘价再平衡
        if ri < len(rebalances) and d == rebalances[ri][0]:
            targets = [t for t in rebalances[ri][1] if pd.notna(px.get(t)) and px[t] > 0]
            if targets:
                w_tgt = {t: 1.0 / len(targets) for t in targets}
                w_cur = {inst: sh * px[inst] / value for inst, sh in shares.items()
                         if pd.notna(px.get(inst))} if shares and value > 0 else {}
                all_inst = set(w_tgt) | set(w_cur)
                turnover = 0.5 * sum(abs(w_tgt.get(i, 0) - w_cur.get(i, 0)) for i in all_inst)
                cost = value * turnover * config.FEE_ROUNDTRIP
                value -= cost
                shares = {t: w_tgt[t] * value / px[t] for t in targets}
                turnovers.append(turnover)
            ri += 1
        nav.append(value)
        nav_dates.append(d)
    nav = pd.Series(nav, index=nav_dates)
    rets = nav.pct_change().fillna(0)
    return rets, (np.mean(turnovers) if turnovers else 0)


def calc_metrics(returns):
    if len(returns) == 0:
        return {"ar": 0, "sharpe": 0, "max_dd": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd}


def full_metrics(returns):
    """组合层指标: 年化收益/年化波动/Sharpe/最大回撤/Calmar"""
    if len(returns) == 0:
        return dict(ar=0, vol=0, sharpe=0, max_dd=0, calmar=0)
    m = calc_metrics(returns)
    vol = returns.std() * np.sqrt(252)
    calmar = m["ar"] / abs(m["max_dd"]) if m["max_dd"] < 0 else 0.0
    return dict(ar=m["ar"], vol=vol, sharpe=m["sharpe"], max_dd=m["max_dd"], calmar=calmar)


def trade_count(rebal_list):
    """总买入次数(容量压力代理): 每次换仓相对上期新进的名字数之和"""
    prev, total = set(), 0
    for _, hold in rebal_list:
        cur = set(hold); total += len(cur - prev); prev = cur
    return total
