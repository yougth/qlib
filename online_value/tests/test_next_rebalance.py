"""next_rebalance 的时点单测 (纯逻辑, 不需要 qlib 数据)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from core.calendar_rules import next_rebalance

# 模拟交易日历: 到 2026-07-29 为止的工作日 (不含节假日, 够用)
def cal(end):
    return list(pd.bdate_range("2026-06-01", end))

cases = [
    ("月中(7/29), 数据到7/29", cal("2026-07-29"), "2026-07-29", "2026-07-31", "2026-08-03"),
    ("月末当晚(7/31), 数据到7/31", cal("2026-07-31"), "2026-07-31", "2026-07-31", "2026-08-03"),
    ("周末(8/1), 数据到7/31", cal("2026-07-31"), "2026-08-01", "2026-07-31", "2026-08-03"),
    ("执行日当天(8/3), 数据未补", cal("2026-07-31"), "2026-08-03", "2026-07-31", "2026-08-03"),
    ("执行后(8/4), 数据到8/3", cal("2026-08-03"), "2026-08-04", "2026-08-31", "2026-09-01"),
]
bad = 0
for desc, c, today, exp_sig, exp_ex in cases:
    s, e = next_rebalance(c, today=today)
    ok = str(s.date()) == exp_sig and str(e.date()) == exp_ex
    bad += not ok
    print(f"  [{'OK  ' if ok else 'FAIL'}] {desc:<26} → 信号{s.date()} 执行{e.date()}  (期望 {exp_sig}/{exp_ex})")
print("FAIL数 =", bad)
sys.exit(1 if bad else 0)
