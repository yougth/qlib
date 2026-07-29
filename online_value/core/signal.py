"""
信号编排 —— 把"池子/估值/流动性/涨停"拼成逐月目标持仓
==================================================================
这一层是 run_backtest / gen_holdings / preflight 三个入口的共同上游:
它们必须走同一条路径, 否则"回测通过但实盘选出别的股票"这类问题无法被发现。

年度上下文 (year_context) 的取数区间刻意与 v18 回测逐字一致:
  · close 矩阵起点 = 首个信号日 - 400 自然日  (让 ffill 的初值状态一致)
  · close 矩阵终点 = min(次年3月31日, BT_END) (给年末月份留未来窗口)
  · 流动性表起点   = 首个信号日 - 10 自然日   (内部再回看60日算20日均)
  · 涨停/停牌集合  = 整条日历范围
改动任一区间都可能让历史结果发生微小漂移, 因此不要"顺手优化"。
"""
import pandas as pd

from . import config
from .universe import get_universe
from .valuation import compute_value_comp
from .market import load_price_volume, build_liquidity_table, build_limit_up_set
from .calendar_rules import monthly_signal_exec_dates
from .selection import select_holdings


def year_bt_end(year):
    """该年回测终点: 年末, 但不超过冻结的 BT_END"""
    return min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(config.BT_END))


def year_context(year, cal_list, fcf_df=None, profit_df=None, need_close=True, verbose=True):
    """准备一年的全部输入。返回 dict(universe/sig_exec/close/liq_table/lu/sus/bt_end)"""
    universe = get_universe(year, fcf_df, profit_df)
    bt_end = year_bt_end(year)
    sig_exec = monthly_signal_exec_dates(year, cal_list)
    if not sig_exec:
        raise RuntimeError(f"[{year}] 日历中无有效换仓点, 检查行情数据是否已更新")
    close = None
    if need_close:
        ext_start = sig_exec[0][0] - pd.Timedelta(days=400)
        ext_end = min(pd.Timestamp(f"{year+1}-03-31"), pd.Timestamp(config.BT_END))
        close, _ = load_price_volume(universe, ext_start, ext_end)
    liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
    lu, sus = build_limit_up_set(universe, cal_list, verbose=verbose)
    if verbose:
        print(f"  [{year}] 池{len(universe)}只 月度换仓点{len(sig_exec)}个 就绪", flush=True)
    return dict(year=year, universe=universe, sig_exec=sig_exec, close=close,
                liq_table=liq_table, lu=lu, sus=sus, bt_end=bt_end)


def year_rebalances(ctx, val_piv, topk=None):
    """逐月算目标持仓。返回 [(exec_dt, holdings_series), ...]

    判定日用**执行日**: 历史上执行日的涨停/停牌是已知事实, 与回测口径一致。
    """
    out = []
    for sig, exec_dt in ctx["sig_exec"]:
        vc = compute_value_comp(ctx["universe"], sig, val_piv)
        hold, _ = select_holdings(vc, ctx["liq_table"], sig, exec_dt,
                                  ctx["lu"], ctx["sus"], topk)
        out.append((exec_dt, hold))
    return out


def holdings_frame(rebalances, nmap=None, topk=None):
    """[(exec_dt, holdings_series)] → 长表 DataFrame(exec_date/rank/instrument/name/value_comp/weight)"""
    k = topk if topk is not None else config.TOPK
    rows = []
    for exec_dt, hold in rebalances:
        for rk, (inst, score) in enumerate(hold.items(), 1):
            rows.append(dict(exec_date=str(pd.Timestamp(exec_dt).date()), rank=rk,
                             instrument=inst,
                             name=(nmap or {}).get(inst[2:], ""),
                             value_comp=round(float(score), 4), weight=1.0 / k))
    df = pd.DataFrame(rows)
    if nmap is None and len(df):
        df = df.drop(columns=["name"])
    return df
