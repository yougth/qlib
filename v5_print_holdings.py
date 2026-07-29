"""
打印 2025-2026 年模型月度真实持仓 (XGBoost 纯 Top10 方案)
=========================================================
利用 AkShare 拉取真实股票名称，输出无缓冲方案下每月的最终决选 Top10
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# 尝试导入 akshare 获取股票真实名称映射
try:
    import akshare as ak
    print("[+] 正在从网络获取 A 股最新名称映射表，请稍候...")
    stock_info = ak.stock_info_a_code_name()
    NAME_MAP = dict(zip(stock_info['symbol'], stock_info['name']))
    print("[+] 股票名称映射加载成功！\n")
except ImportError:
    print("[-] 未检测到 akshare。请运行 'pip install akshare' 以获取真实股票名称。")
    NAME_MAP = {}
except Exception as e:
    print(f"[-] 获取股票名称失败: {e}")
    NAME_MAP = {}

from v5_validation import apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, load_value_factors, search_alpha_on_valid
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import WINDOWS, TOP80_FEATURES, MODEL_CONFIG, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates, train_window

def get_stock_name(qlib_code):
    """将 SH600000 转换为 浦发银行(SH600000)"""
    pure_code = qlib_code[2:] if len(qlib_code) == 8 else qlib_code
    name = NAME_MAP.get(pure_code, "未知名称")
    return f"{name}({qlib_code})"

def run_holdings_printer():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)
    
    # 我们只关心 2025 (W5) 和 2026H1 (W6)
    target_windows = [w for w in WINDOWS if w["name"] in ["W5", "W6"]]

    print(f"{'='*80}")
    print(f"  2025 - 2026 年 XGBoost 纯 Top10 月度实盘买入名单审查")
    print(f"{'='*80}")

    for win in target_windows:
        print(f"\n[+] 正在后台计算 {win['name']} 窗口 (回测年份: {win['year']}) ...")
        
        # 1. 训练模型并获取预测分数
        pred, universe, vf, _ = train_window(win, fcf_df, profit_df, cal, cal_set)
        
        # 2. 获取当期最佳 Alpha 并进行后置融合
        vs_w, ve_w = win["valid"]
        best_alpha, _, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
        if best_alpha > 0 and vf is not None:
            pred = apply_value_fusion(pred.copy(), vf, alpha=best_alpha)

        # 3. 施加真实交易环境约束 (剔除一字涨停、停牌)
        limit_up_set, suspension_set = build_limit_up_set(universe, cal)
        pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

        # 4. 剔除日均成交额 < 2000万 的流动性枯竭标的
        vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
        if vol_data is not None and len(vol_data) > 0:
            vol_data = vol_data.reindex(pred.index)
            pred.loc[vol_data["$amount"] < 20_000_000, "score"] = -np.inf

        # 5. 按月切片，提取最终买入的 Top 10
        bs, be = win["backtest"]
        month_ends = get_month_end_dates(cal, bs, be)
        all_dates = sorted(pred.index.get_level_values(0).unique())

        print(f"\n{'-'*80}")
        print(f"  {win['year']} 年度换仓明细 (Alpha融合 = {best_alpha})")
        print(f"{'-'*80}")
        
        for dt in month_ends:
            if dt not in pred.index.get_level_values(0):
                earlier = [d for d in all_dates if d <= dt]
                if not earlier: continue
                dt = earlier[-1]
                
            day_pred = pred.xs(dt, level=0)
            top10_codes = day_pred["score"].nlargest(10).index.tolist()
            
            # 翻译真实名称
            top10_names = [get_stock_name(c) for c in top10_codes]
            
            date_str = pd.Timestamp(dt).strftime('%Y-%m-%d')
            print(f"【{date_str}】")
            
            # 每行打印 5 个，方便阅读
            print(f"  买入: {', '.join(top10_names[:5])}")
            print(f"        {', '.join(top10_names[5:])}\n")

if __name__ == "__main__":
    run_holdings_printer()
