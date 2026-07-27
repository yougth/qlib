"""月频Top-K调仓策略 — 独立模块避免__main__查找问题"""
import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO


class MonthlyTopk(TopkDropoutStrategy):
    """仅在月末最后一个交易日调仓的TopkDropoutStrategy"""
    def __init__(self, **kw):
        super().__init__(**kw)
        self._lp = None

    def generate_trade_decision(self, er=None):
        ts = self.trade_calendar.get_trade_step()
        cs, _ = self.trade_calendar.get_step_time(ts)
        cd = pd.Timestamp(cs)
        cp = cd.to_period("M")
        rb = False
        try:
            ns, _ = self.trade_calendar.get_step_time(ts + 1)
            if pd.Timestamp(ns).to_period("M") != cp:
                rb = True
        except Exception:
            rb = True
        if self._lp is None:
            rb = True
        if not rb:
            return TradeDecisionWO([], self)
        self._lp = cp
        return super().generate_trade_decision(er)
