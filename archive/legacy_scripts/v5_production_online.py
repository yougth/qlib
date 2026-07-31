"""
V8 实盘上线交付版 (Online Deployment Ready)
=========================================================
1. 流动性过滤阈值下调至 500 万 (适配单票 10 万本金)
2. 工业标准模型落盘：保存 W6 模型权重 (.pkl) 与 特征列表
3. 输出下一交易日实盘跟单信号 (v8_live_signals.csv)
"""
import os, sys, joblib, json
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
import logging
warnings.filterwarnings("ignore")

# 强行静音 Qlib 底层关于 NaN 的啰嗦警告
logging.getLogger('qlib.online operator').setLevel(logging.ERROR)
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

try:
    import akshare as ak
    logging.getLogger('akshare').setLevel(logging.ERROR)
    print("[+] 正在加载 A 股真实名称映射表...")
    stock_info = ak.stock_info_a_code_name()
    NAME_MAP = dict(zip(stock_info['symbol'], stock_info['name']))
except Exception as e:
    print("[-] akshare 加载失败，将仅显示股票代码。")
    NAME_MAP = {}

from v5_validation import apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, load_value_factors, search_alpha_on_valid
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import TOP80_FEATURES, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates

# ==================== XGBoost M1 实盘训练配置 ====================
MODEL_CONFIG = {
    "class": "XGBModel", 
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "objective": "reg:squarederror", 
        "learning_rate": 0.005,
        "max_depth": 4,                   
        "colsample_bytree": 0.8879,
        "subsample": 0.8789,
        "reg_alpha": 10.0,                
        "reg_lambda": 50.0,               
        "tree_method": "hist",
        "n_jobs": 4,
        "early_stopping_rounds": 100,
        "n_estimators": 1000              
    }
}

WINDOWS = [
    {"train": ("2014-01-01","2017-12-31"), "valid": ("2018-01-01","2018-12-31"), "backtest": ("2019-01-01","2019-12-31"), "name":"W0", "year": 2019},
    {"train": ("2015-01-01","2018-12-31"), "valid": ("2019-01-01","2019-12-31"), "backtest": ("2020-01-01","2020-12-31"), "name":"W0_5", "year": 2020},
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1", "year": 2021},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2", "year": 2022},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3", "year": 2023},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4", "year": 2024},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5", "year": 2025},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "year": 2026},
]

def get_stock_name(qlib_code):
    pure_code = qlib_code[2:] if len(qlib_code) == 8 else qlib_code
    return f"{NAME_MAP.get(pure_code, '未知')}({qlib_code})"

def train_window(win, fcf_df, profit_df, cal, cal_set):
    by = win["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    ts, te, vs, ve, bs, be = win["train"][0], win["train"][1], win["valid"][0], win["valid"][1], win["backtest"][0], win["backtest"][1]

    dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
        "instruments": universe,
        "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                              {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label": ["Ref($close, -20) / $close - 1"]}
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"v5_validation","kwargs":dhc},
                  "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}
    dataset = init_instance_by_config(dsc)

    vf = load_value_factors(universe, cal, cal_set)
    fund = load_fundamental_features_fixed(universe, cal)
    if fund is not None: inject_features_fixed(dataset, fund, "fund")
    if vf is not None: inject_features_fixed(dataset, vf, "vf")
    prune_features(dataset, TOP80_FEATURES)

    model = init_instance_by_config(MODEL_CONFIG)
    model.fit(dataset)
    
    # === [实盘工程侧] 仅导出最新 W6 窗口的模型权重与特征列 ===
    if win["name"] == "W6":
        work_dir = "/Users/11164591/Documents/Qoder目录/qlib"
        # 1. 序列化导出 Qlib 模型对象 (供线上 Python 环境直接加载)
        joblib.dump(model, f"{work_dir}/online_xgb_model_w6.pkl")
        
        # 2. 导出纯净的 XGBoost 底层 JSON 权重 (供 C++/Java 或其他语言部署备用)
        model.model.save_model(f"{work_dir}/online_xgb_booster_w6.json")
        
        # 3. 导出模型严格依赖的输入特征顺序，线上构建数据时必须严格对齐
        train_features = dataset.prepare("train", col_set="feature")
        feature_names = [c[1] if isinstance(c, tuple) else c for c in train_features.columns]
        with open(f"{work_dir}/online_feature_columns.json", "w") as f:
            json.dump(feature_names, f, indent=4)
        print(f"\n[+] 生产环境模型组件 (W6) 已落盘至 {work_dir}")

    valid_features = dataset.prepare("valid", col_set="feature")
    pred = None
    if valid_features is not None and len(valid_features) > 0:
        dvalid = xgb.DMatrix(valid_features.values)
        valid_pred = pd.DataFrame(model.model.predict(dvalid), index=valid_features.index, columns=["score"])
        
    test_features = dataset.prepare("test", col_set="feature")
    if test_features is not None and len(test_features) > 0:
        dtest = xgb.DMatrix(test_features.values)
        test_pred = pd.DataFrame(model.model.predict(dtest), index=test_features.index, columns=["score"])
        pred = pd.concat([valid_pred, test_pred])

    return pred, universe, vf, fund

def backtest_engine_net(pred, universe, cal, month_ends, alpha=0.0, vf=None, fee_rate=0.0025, is_w6=False):
    if pred is None or len(pred) == 0: return pd.Series()
    if alpha > 0 and vf is not None: pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)
    
    # === [关键修复] 适配 100万 本金的流动性过滤：降至 500 万 ===
    vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
    if vol_data is not None and len(vol_data) > 0:
        pred.loc[vol_data.reindex(pred.index)["$amount"] < 5_000_000, "score"] = -np.inf

    all_dates = sorted(pred.index.get_level_values(0).unique())
    price_data = D.features(list(set(pred.index.get_level_values(1))), ["$close"],
                           start_time=all_dates[0] - pd.Timedelta(days=10), end_time=all_dates[-1] + pd.Timedelta(days=40)).reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {inst: grp.set_index("datetime")["close"] for inst, grp in price_data.groupby("instrument")}

    net_returns, portfolio_dates = [], []
    prev_topk = []

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]; dt = earlier[-1] if earlier else dt
            
        curr_topk = pred.xs(dt, level=0)["score"].nlargest(10).index.tolist()
        intersection = len(set(curr_topk).intersection(set(prev_topk)))
        turnover = 1.0 if not prev_topk else 1.0 - (intersection / 10.0)

        next_dt = month_ends[i + 1] if i + 1 < len(month_ends) else all_dates[-1]
        period_dates = [d for d in all_dates if dt < d <= next_dt]
        prev_prices = {inst: price_dict[inst][dt] for inst in curr_topk if inst in price_dict and dt in price_dict[inst].index}

        for p_idx, pd_dt in enumerate(period_dates):
            day_ret = 0
            n_valid = 0
            for inst in curr_topk:
                if inst in price_dict and pd_dt in price_dict[inst].index and inst in prev_prices:
                    cur_price = price_dict[inst][pd_dt]
                    if pd.notna(cur_price) and prev_prices[inst] > 0:
                        day_ret += (cur_price / prev_prices[inst] - 1)
                        prev_prices[inst] = cur_price
                        n_valid += 1
            
            daily_gross = day_ret / 10.0 if n_valid > 0 else 0
            cost = (turnover * fee_rate) if p_idx == 0 else 0.0
            net_returns.append(daily_gross - cost)
            portfolio_dates.append(pd_dt)
            
        prev_topk = curr_topk

    s_net = pd.Series(net_returns, index=pd.DatetimeIndex(portfolio_dates))
    return s_net[~s_net.index.duplicated(keep="last")]

def calc_metrics(returns):
    if len(returns) == 0: return {"ar": 0, "sharpe": 0, "max_dd": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd}

def get_annual_returns(returns):
    if len(returns) == 0: return {}
    return returns.groupby(returns.index.year).apply(lambda x: (1+x).prod()-1).to_dict()

def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    print(f"\n{'='*75}\n  启动 V8 生产环境最终版 (单票 10 万级资金量 500 万流动性测试)\n{'='*75}")
    rets_alpha = []
    final_pred_scores = None

    for win in WINDOWS:
        print(f"[+] 训练 {win['name']} ...")
        pred, universe, vf, _ = train_window(win, fcf_df, profit_df, cal, cal_set)
        vs_w, ve_w = win["valid"]
        best_alpha, _, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
        month_ends = get_month_end_dates(cal, win["backtest"][0], win["backtest"][1])
        
        if win["name"] == "W6": 
            final_pred_scores = pred
            
        rn_alpha = backtest_engine_net(pred, universe, cal, month_ends, alpha=best_alpha, vf=vf, fee_rate=0.0025, is_w6=(win["name"]=="W6"))
        rets_alpha.append(rn_alpha)

    s_alpha = pd.concat(rets_alpha).sort_index(); s_alpha = s_alpha[~s_alpha.index.duplicated(keep="last")]
    m_a = calc_metrics(s_alpha); ann_a = get_annual_returns(s_alpha)
    
    print(f"\n{'='*115}")
    print(f"{'V8 交付指标 (已降阈至500万)':<22} | {'2019':>7} {'2020':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'2026H1':>7} | {'全期净年化':>10} {'净夏普':>6} {'最大回撤':>9}")
    print(f"{'-'*115}")
    print(f"{'Alpha价值融合 (实盘优选)':<20} | {ann_a.get(2019,0)*100:>6.2f}% {ann_a.get(2020,0)*100:>6.2f}% {ann_a.get(2021,0)*100:>6.2f}% {ann_a.get(2022,0)*100:>6.2f}% {ann_a.get(2023,0)*100:>6.2f}% {ann_a.get(2024,0)*100:>6.2f}% {ann_a.get(2025,0)*100:>6.2f}% {ann_a.get(2026,0)*100:>6.2f}% | {m_a['ar']*100:>9.2f}% {m_a['sharpe']:>6.2f} {m_a['max_dd']*100:>8.1f}%")
    print(f"{'='*115}\n")

    work_dir = "/Users/11164591/Documents/Qoder目录/qlib"
    if final_pred_scores is not None:
        last_date = final_pred_scores.index.get_level_values(0).max()
        latest_scores = final_pred_scores.xs(last_date, level=0).sort_values("score", ascending=False).reset_index()
        latest_scores["stock_name"] = latest_scores["instrument"].apply(get_stock_name)
        
        # 提取真实可买的前 10 名（已剔除 500 万以下流动性和停牌股）
        limit_up_set, suspension_set = build_limit_up_set(latest_scores["instrument"].tolist(), cal)
        vol_data = D.features(latest_scores["instrument"].tolist(), ["$amount"], start_time=last_date, end_time=last_date)
        valid_scores = latest_scores.copy()
        
        if vol_data is not None and len(vol_data) > 0:
            vol_data = vol_data.reset_index().set_index("instrument")
            valid_scores = valid_scores[valid_scores["instrument"].map(lambda x: vol_data.get("$amount", {}).get(x, 1e8) >= 5_000_000)]
        valid_scores = valid_scores[~valid_scores["instrument"].isin(suspension_set.get(last_date, set()))]
        valid_scores = valid_scores[~valid_scores["instrument"].isin(limit_up_set.get(last_date, set()))]
        
        valid_scores.head(10).to_csv(f"{work_dir}/v8_live_signals.csv", sep='\t', index=False)
        print(f"[+] 实盘买入信号已生成: {work_dir}/v8_live_signals.csv (请直接依据此表挂单入场)")

if __name__ == "__main__":
    run()
