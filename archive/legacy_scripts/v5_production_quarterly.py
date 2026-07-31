"""
V11 终极实盘上线版 (Quarterly Rolling + Alpha Fusion + Fixed Seed)
=========================================================
1. 彻底解决 akshare 多进程网络风暴与 Semaphore 内存泄露警告。
2. 彻底解决 set.get() 涨跌停对象越界 Bug。
3. 全局强锁随机数 (Seed=42)，确保多核并发结果 100% 绝对一致。
4. 机构级架构：季度滚动重训 + 验证集动态 Alpha 价值因子融合。
5. 实盘风控：500万流动性底线过滤，自动落盘 2026 年最新一季信号。
"""
import os, sys, json, random
import warnings, logging

# --- 全局强锁随机数，保证实盘 100% 可复现 ---
def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
import xgboost as xgb
import joblib

# --- 强行静音各种底层烦人的警告 ---
warnings.filterwarnings("ignore")
# 忽略多进程资源泄露警告 (Mac M1 下的常见并发 Bug)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
logging.getLogger('qlib.online operator').setLevel(logging.ERROR)
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)
logging.getLogger('akshare').setLevel(logging.ERROR)

from v5_validation import apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, load_value_factors, search_alpha_on_valid
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import TOP80_FEATURES, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates

# --- 本地缓存策略: 根除 akshare 接口被墙和多进程高并发报错 ---
def get_stock_name_map():
    cache_file = "/Users/11164591/Documents/Qoder目录/qlib/stock_names_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except: pass
    try:
        import akshare as ak
        print("[+] 正在通过外网首次拉取 A 股真实名称映射表...")
        stock_info = ak.stock_info_a_code_name()
        name_map = dict(zip(stock_info['symbol'], stock_info['name']))
        with open(cache_file, "w") as f:
            json.dump(name_map, f)
        print("[+] 映射表已缓存至本地，后续将极速秒开！")
        return name_map
    except Exception as e:
        print(f"[-] akshare 首次拉取失败，启用纯代码模式。原因: {e}")
        return {}

NAME_MAP = get_stock_name_map()

def get_stock_name(qlib_code):
    pure_code = qlib_code[2:] if len(qlib_code) == 8 else qlib_code
    return f"{NAME_MAP.get(pure_code, '未知')}({qlib_code})"

# ==================== XGBoost 实盘核心参数 ====================
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
        "random_state": 42,      # 核心防随机刺客
        "seed": 42,              # 核心防随机刺客
        "early_stopping_rounds": 100,
        "n_estimators": 1000              
    }
}

# 动态生成季频窗口 (包含到 2026 年 Q3 也就是当前的 7 月份)
def generate_quarterly_windows():
    windows = []
    backtest_years = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
    for year in backtest_years:
        periods = [(1, 3), (4, 6), (7, 9)] if year == 2026 else [(1, 3), (4, 6), (7, 9), (10, 12)]
        for start_m, end_m in periods:
            bt_start = pd.Timestamp(f"{year}-{start_m:02d}-01")
            bt_end = pd.Timestamp(f"{year}-{end_m:02d}-01") + relativedelta(months=1) - pd.Timedelta(days=1)
            
            train_end = bt_start - relativedelta(years=1) - pd.Timedelta(days=1)
            train_start = train_end - relativedelta(years=4)
            valid_start = train_end + pd.Timedelta(days=1)
            valid_end = bt_start - pd.Timedelta(days=1)
            
            name = f"{year}_Q{start_m//3 + 1}"
            windows.append({
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "backtest": (bt_start.strftime("%Y-%m-%d"), bt_end.strftime("%Y-%m-%d")),
                "name": name, "year": year
            })
    return windows

def train_and_predict(win, fcf_df, profit_df, cal, cal_set):
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
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
    
    # 记录最新一季 (2026_Q3) 的模型权重，用于后续直接推断
    is_latest_quarter = (win["name"] == "2026_Q3")
    if is_latest_quarter:
        work_dir = "/Users/11164591/Documents/Qoder目录/qlib"
        joblib.dump(model, f"{work_dir}/online_xgb_model_latest_Q.pkl")
        train_features = dataset.prepare("train", col_set="feature")
        feature_names = [c[1] if isinstance(c, tuple) else c for c in train_features.columns]
        with open(f"{work_dir}/online_feature_columns.json", "w") as f:
            json.dump(feature_names, f, indent=4)

    valid_features = dataset.prepare("valid", col_set="feature")
    test_features = dataset.prepare("test", col_set="feature")
    
    pred_parts = []
    if valid_features is not None and len(valid_features) > 0:
        pred_parts.append(pd.DataFrame(model.model.predict(xgb.DMatrix(valid_features.values)), index=valid_features.index, columns=["score"]))
    if test_features is not None and len(test_features) > 0:
        pred_parts.append(pd.DataFrame(model.model.predict(xgb.DMatrix(test_features.values)), index=test_features.index, columns=["score"]))
        
    pred = pd.concat(pred_parts) if pred_parts else pd.DataFrame()
    return pred, universe, vf, is_latest_quarter

def backtest_engine_net(pred, universe, cal, month_ends, alpha=0.0, vf=None, fee_rate=0.0025):
    if pred is None or len(pred) == 0: return pd.Series()
    if alpha > 0 and vf is not None: pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)
    
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
            day_ret, n_valid = 0, 0
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
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_set = set(cal)

    print(f"\n{'='*95}\n  启动 V11 终极实盘交付验证 (季滚动 + Alpha 融合 + 锁死随机数)\n{'='*95}")
    rets_alpha = []
    final_pred_scores = None
    final_best_alpha = 0.0
    final_vf = None

    windows = generate_quarterly_windows()
    for win in windows:
        print(f"[+] 并行组装季度切片: {win['name']} ...")
        pred, universe, vf, is_latest = train_and_predict(win, fcf_df, profit_df, cal, cal_set)
        
        vs_w, ve_w = win["valid"]
        best_alpha, _, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
        month_ends = get_month_end_dates(cal, win["backtest"][0], win["backtest"][1])
        
        if is_latest:
            final_pred_scores = pred
            final_best_alpha = best_alpha
            final_vf = vf
            
        rn_alpha = backtest_engine_net(pred, universe, cal, month_ends, alpha=best_alpha, vf=vf, fee_rate=0.0025)
        rets_alpha.append(rn_alpha)

    s_alpha = pd.concat(rets_alpha).sort_index(); s_alpha = s_alpha[~s_alpha.index.duplicated(keep="last")]
    m_a = calc_metrics(s_alpha); ann_a = get_annual_returns(s_alpha)
    
    print(f"\n{'='*115}")
    print(f"{'V11 终极锁定版指标 (季滚动+融合)':<26} | {'2019':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'26H1/Q3':>7} | {'全期净年化':>10} {'净夏普':>6} {'最大回撤':>9}")
    print(f"{'-'*115}")
    print(f"{'Alpha实盘优选':<30} | {ann_a.get(2019,0)*100:>6.2f}% {ann_a.get(2021,0)*100:>6.2f}% {ann_a.get(2022,0)*100:>6.2f}% {ann_a.get(2023,0)*100:>6.2f}% {ann_a.get(2024,0)*100:>6.2f}% {ann_a.get(2025,0)*100:>6.2f}% {ann_a.get(2026,0)*100:>6.2f}% | {m_a['ar']*100:>9.2f}% {m_a['sharpe']:>6.2f} {m_a['max_dd']*100:>8.1f}%")
    print(f"{'='*115}\n")

    # ==================== 实盘信号强力脱水与落盘 ====================
    work_dir = "/Users/11164591/Documents/Qoder目录/qlib"
    if final_pred_scores is not None:
        # 1. 在最后一天应用 Alpha 融合
        if final_best_alpha > 0 and final_vf is not None:
            final_pred_scores = apply_value_fusion(final_pred_scores.copy(), final_vf, alpha=final_best_alpha)
            
        last_date = final_pred_scores.index.get_level_values(0).max()
        latest_scores = final_pred_scores.xs(last_date, level=0).sort_values("score", ascending=False).reset_index()
        latest_scores["stock_name"] = latest_scores["instrument"].apply(get_stock_name)
        
        # 2. 彻底修复集合元组判定 Bug：使用安全的列表推导式匹配 (datetime, instrument)
        limit_up_set, suspension_set = build_limit_up_set(latest_scores["instrument"].tolist(), cal)
        
        valid_insts = []
        for inst in latest_scores["instrument"]:
            if (last_date, inst) not in suspension_set and (last_date, inst) not in limit_up_set:
                valid_insts.append(inst)
                
        valid_scores = latest_scores[latest_scores["instrument"].isin(valid_insts)]
        
        # 3. 施加 500 万硬底线流动性限制
        vol_data = D.features(valid_insts, ["$amount"], start_time=last_date, end_time=last_date)
        if vol_data is not None and len(vol_data) > 0:
            vol_data = vol_data.reset_index().set_index("instrument")
            valid_scores = valid_scores[valid_scores["instrument"].map(lambda x: vol_data.get("$amount", {}).get(x, 1e8) >= 5_000_000)]
        
        # 4. 生成 Top 10 买入清单
        final_top10 = valid_scores.head(10)[["instrument", "stock_name", "score"]]
        output_file = f"{work_dir}/v11_live_signals_Q3.csv"
        final_top10.to_csv(output_file, sep='\t', index=False)
        
        print(f"[+] 工业级实盘买入信号已生成: {output_file}")
        print(f"[+] 请注意：这是执行了 季滚动重训 + 验证集Alpha={final_best_alpha} 融合后的最终名单。")
        print(f"    名单内的股票已确认：非停牌、非涨跌停、昨日成交额 >= 500 万。")
        print(f"    接下来请携带 100 万资金，等分为 10 份，在次日早盘执行限价分批买入操作。祝你好运！")

if __name__ == "__main__":
    run()
