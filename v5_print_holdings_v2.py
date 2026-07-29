"""
打印 2026 H1 真实持仓并计算单票与月度总收益
=========================================
提取 Qlib 真实收盘价，还原每一次换仓的真实盈亏
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

try:
    import akshare as ak
    print("[+] 正在加载名称映射...")
    stock_info = ak.stock_info_a_code_name()
    NAME_MAP = dict(zip(stock_info['symbol'], stock_info['name']))
except:
    NAME_MAP = {}

from v5_validation import search_alpha_on_valid, build_limit_up_set, filter_pred_by_tradability
from v5_xgb_turnover_comparative import WINDOWS, get_month_end_dates, train_window

def get_stock_name(qlib_code):
    pure_code = qlib_code[2:] if len(qlib_code) == 8 else qlib_code
    return f"{NAME_MAP.get(pure_code, '未知')}({qlib_code})"

def run_returns_calculator():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    
    # 仅计算 2026H1 (W6)
    win = [w for w in WINDOWS if w["name"] == "W6"][0]
    print(f"\n[+] 正在后台提取 {win['name']} 窗口预测与价格数据 ...\n")
    
    pred, universe, vf, _ = train_window(win, fcf_df, profit_df, cal, set(cal))
    
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

    vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
    if vol_data is not None and len(vol_data) > 0:
        vol_data = vol_data.reindex(pred.index)
        pred.loc[vol_data["$amount"] < 20_000_000, "score"] = -np.inf

    bs, be = win["backtest"]
    month_ends = get_month_end_dates(cal, bs, be)
    all_dates = sorted(pred.index.get_level_values(0).unique())

    # 拉取全量价格数据用于计算收益
    price_data = D.features(universe, ["$close"], start_time=all_dates[0], end_time=all_dates[-1])
    price_data = price_data.reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {inst: grp.set_index("datetime")["close"] for inst, grp in price_data.groupby("instrument")}

    print(f"{'='*80}")
    print(f"  2026 H1 纯 Top10 实盘持仓与月度收益核算 (不含手续费滑点)")
    print(f"{'='*80}")

    total_compound_return = 1.0

    for i in range(len(month_ends)):
        dt = month_ends[i]
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]
            if not earlier: continue
            dt = earlier[-1]
            
        # 确定卖出日期 (下个月的调仓日)
        next_dt = month_ends[i+1] if i+1 < len(month_ends) else all_dates[-1]
        if dt == next_dt: continue

        day_pred = pred.xs(dt, level=0)
        top10_codes = day_pred["score"].nlargest(10).index.tolist()
        
        date_str = pd.Timestamp(dt).strftime('%Y-%m-%d')
        next_date_str = pd.Timestamp(next_dt).strftime('%Y-%m-%d')
        print(f"\n【调仓周期: {date_str} 买入 -> {next_date_str} 卖出】")
        
        monthly_returns = []
        for code in top10_codes:
            name_str = get_stock_name(code)
            try:
                buy_price = price_dict[code].loc[dt]
                sell_price = price_dict[code].loc[next_dt]
                ret = (sell_price / buy_price) - 1
                monthly_returns.append(ret)
                
                # 打印单票收益，大于0显示红色(终端支持的话)，这里直接打印
                sign = "+" if ret > 0 else ""
                print(f"  - {name_str:<18} | 买入: {buy_price:6.2f} -> 卖出: {sell_price:6.2f} | 收益: {sign}{ret*100:5.2f}%")
            except KeyError:
                print(f"  - {name_str:<18} | 价格数据缺失")
        
        if monthly_returns:
            port_ret = np.mean(monthly_returns)
            total_compound_return *= (1 + port_ret)
            sign = "+" if port_ret > 0 else ""
            print(f"  >>> 当月组合净收益: {sign}{port_ret*100:.2f}%")

    print(f"\n{'='*80}")
    print(f"  [汇总] 2026 H1 累计毛收益率: {(total_compound_return - 1)*100:.2f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_returns_calculator()
