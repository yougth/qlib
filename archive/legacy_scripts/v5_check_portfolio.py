
"""
月度持仓演变与换手率审查脚本
=========================================
打印指定年份每个月的 Top 10 股票池，并计算真实的月度换手率。
"""
import os
import sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from v5_new_baseline_xgb import WINDOWS, get_month_end_dates, apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, load_value_factors

def run_portfolio_review():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # 选择审查 2024 年 (W4窗口) 的月度持仓
    win = [w for w in WINDOWS if w["name"] == "W4"][0]
    bs, be = win["backtest"]
    month_ends = get_month_end_dates(cal, bs, be)

    print(f"\n[+] 正在加载 {win['name']} ({bs} ~ {be}) 的预测结果与持仓数据...")

    # 注：这里假设你已经运行过主脚本或通过 MLflow 能够拿到预测数据。
    # 为保证独立运行，我们这里直接重新跑一遍 W4 的预测逻辑获取 pred 和 vf
    from v5_new_baseline_xgb import train_window
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    pred, universe, vf, fund = train_window(win, fcf_df, profit_df, cal, cal_set)

    # 应用 alpha=0.3 价值融合（如果你想看融合后的持仓）
    best_alpha = 0.3
    if best_alpha > 0 and vf is not None:
        pred = apply_value_fusion(pred.copy(), vf, alpha=best_alpha)

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

    print(f"\n{'='*75}")
    print(f"  {win['year']} 年各月末 Top 10 持仓与换手率演变 (Alpha={best_alpha})")
    print(f"{'='*75}")

    prev_topk = set()
    turnovers = []

    for i, dt in enumerate(month_ends):
        all_dates = sorted(pred.index.get_level_values(0).unique())
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]
            if not earlier:
                continue
            dt = earlier[-1]

        day_pred = pred.xs(dt, level=0)
        topk_stocks = day_pred["score"].nlargest(10).index.tolist()
        curr_topk = set(topk_stocks)

        if i == 0:
            turnover = 1.0  # 首月全仓买入，换手率100%
        else:
            # 换手率计算：1 - (当月与上月重合的股票数 / 10)
            intersection = len(curr_topk.intersection(prev_topk))
            turnover = 1.0 - (intersection / 10.0)

        turnovers.append(turnover)
        print(f"\n【调仓日: {pd.Timestamp(dt).strftime('%Y-%m-%d')}】 换手率: {turnover*100:.1f}%")
        print(f"  Top 10 标的: {', '.join(topk_stocks)}")

        prev_topk = curr_topk

    avg_turnover = np.mean(turnovers[1:]) if len(turnovers) > 1 else 1.0
    print(f"\n{'-'*75}")
    print(f"  排查结论：剔除首月后，平均月度换手率为: {avg_turnover*100:.1f}%")
    print(f"{'='*75}")

if __name__ == "__main__":
    run_portfolio_review()
