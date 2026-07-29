"""
换仓日历 —— 月度换仓日规则
==================================================================
规则 (逐字搬自 v18_full_stack.monthly_signal_exec_dates 的逻辑):
  信号日 = 该自然月首日之前的最后一个交易日  (用它的收盘估值/价格打分)
  执行日 = 该自然月首个交易日                (T+1, 尾盘按目标持仓再平衡)
  两者严格 信号日 < 执行日 —— 这是"不偷看未来"的结构保证。

与上游的唯一差别: 上游把 2026 写死成 range(1,8)。这里改为统一 range(1,13),
月份是否成立完全由日历自然裁定 (月首之后无交易日 → 该月不成立)。
在日历末端为 2026-07-31 时两者输出**完全相同**, 但新写法在数据延伸后能自动多出一期,
不需要每月改代码 —— 改代码才是真正的上线风险。
"""
import pandas as pd
from qlib.data import D

from . import config


def get_calendar(start=None, end=None):
    """交易日列表。end=None 表示用数据的自然末端(生产用); 回测请显式传 config.BT_END"""
    cal = D.calendar(start_time=start or config.CAL_START, end_time=end)
    return list(cal)


def monthly_signal_exec_dates(year, cal_list):
    """返回[(信号日, 执行日), ...]。信号=月首前最后交易日, 执行=月首交易日"""
    out = []
    for m in range(1, 13):
        q = pd.Timestamp(f"{year}-{m:02d}-01")
        prev = [d for d in cal_list if d < q]
        nxt = [d for d in cal_list if d >= q]
        if not prev or not nxt:
            continue
        out.append((max(prev), min(nxt)))
    return out


def next_rebalance(cal_list, today=None):
    """下一个换仓点 (信号日, 执行日) —— 用于每月运维确认"今晚该不该出清单"。

    注意: 执行日在未来, 历史交易日历里**不可能**有它, 所以这里不能去日历里"查",
    只能推导。判断本月的执行日是否已经发生:
      · 已发生 → 下一换仓点在下月: 信号日=本月最后一个工作日, 执行日=下月首个工作日
      · 未发生 → 下一换仓点就是本月: 信号日=日历末端(上月末最后交易日), 执行日=本月首个工作日
    未来日期用工作日(bdate_range)近似, 不含节假日, 因此遇到长假(如国庆)给出的执行日
    会偏早几天; 它只用于人工核对, 不参与任何选股或回测计算。
    """
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now().normalize()
    if not cal_list:
        return None, None
    last = cal_list[-1]
    ref = max(last, today)
    m1 = pd.Timestamp(ref.year, ref.month, 1)                    # 本月1号
    if not [d for d in cal_list if d >= m1]:                     # 本月还没有任何交易日
        return last, _first_bday(m1)                             # → 信号日已就位, 等执行
    nm = m1 + pd.offsets.MonthBegin(1)                           # 下月1号
    rest = pd.bdate_range(max(last, m1), nm - pd.Timedelta(days=1))
    return (rest[-1] if len(rest) else last), _first_bday(nm)


def _first_bday(day):
    """day 当天或之后的第一个工作日"""
    return pd.bdate_range(day, day + pd.Timedelta(days=12))[0]
