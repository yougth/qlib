"""
终极实盘压测版: XGBoost + 真实特征解码 + Top20换手率缓冲
=========================================
1. 修复 DMatrix 预测报错与 Series/DataFrame 降维崩溃问题
2. 解码并打印真实的特征重要性 (Feature Importance)
3. 引入缓冲换仓逻辑 (Buffer Rule): 持仓跌出Top20才卖，新进Top10才买
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

from v5_validation import Alpha158Enhanced, apply_value_fusion, \
    build_limit_up_set, filter_pred_by_tradability, load_value_factors, \
    search_alpha_on_valid
from v5_pipeline_fix import load_fundamental_features_fixed, \
    inject_features_fixed, filter_by_fundamental_deterioration

# ==================== XGBoost 模型配置 ====================
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
        "n_jobs": 4,                      
        "early_stopping_rounds": 100,
        "n_estimators": 1000              
    }
}

# 7个窗口
WINDOWS = [
    {"train": ("2014-01-01","2017-12-31"), "valid": ("2018-01-01","2018-12-31"), "backtest": ("2019-01-01","2019-12-31"), "name":"W0", "year": 2019},
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1", "year": 2021},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2", "year": 2022},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3", "year": 2023},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4", "year": 2024},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5", "year": 2025},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "year": 2026},
]

TOP80_FEATURES = {
    "roe_annual", "pb_pct_3y", "pe_pct_3y", "div_yield_est",
    "fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y",
    "MAX30", "ROC60", "STD240", "STD120", "ROC240", "MAX60", "MA120", "STD30",
    "ROC30", "CNTN60", "ROC120", "STD60", "MA20", "CORR30", "CORR60", "CORD60",
    "CORD30", "VRATIO_5_120", "CNTD60", "IMXD60", "QTLD60", "VSTD30", "WVMA30",
    "VMA240", "MA240", "BETA30", "CNTP60", "SUMP60", "BETA60", "IMAX60", "BOLL120",
    "STD20", "RSQR60", "WVMA60", "CORD20", "IMIN60", "CNTN30", "MIN60", "RESI60",
    "CORR_PV60", "MIN20", "VSTD60", "RSQR30", "QTLD20", "IMXD30", "STD10", "MIN5",
    "CORR20", "SUMD60", "CNTD20", "KLEN", "IMXD20", "CNTD30", "SUMD30", "MIN10",
    "CORR10", "CORR_PV20", "WVMA20", "MAX10", "RSV60", "SUMP20", "SUMN60", "VSUMN30",
    "VRATIO_5_60", "CORD5", "WVMA10", "CNTN10", "MAX20", "CNTN20", "CNTP20", "SUMN30",
}

def prune_features(dataset, keep_features):
    handler = dataset.handler
    for attr_name in ["_infer", "_learn", "_data"]:
        if not hasattr(handler, attr_name): continue
        df = getattr(handler, attr_name)
        if df is None: continue
        is_multi = isinstance(df.columns, pd.MultiIndex)
        if is_multi:
            keep_cols = [c for c in df.columns if c[0] != "feature" or c[1] in keep_features]
        else:
            keep_cols = [c for c in df.columns if c in keep_features]
        setattr(handler, attr_name, df[keep_cols])

def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"

def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    fcf_start = backtest_year - 11; fcf_end = backtest_year - 2
    target_fcf_years = list(range(fcf_start, fcf_end + 1))
    fcf_filtered = fcf_df[fcf_df["year"].isin(target_fcf_years)]
    fcf_positive = fcf_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_positive["code"].unique())
    
    profit_start = max(backtest_year - 11, 2016); profit_end = backtest_year - 2
    if profit_end >= 2016:
        target_profit_years = list(range(profit_start, profit_end + 1))
        profit_filtered = profit_df[profit_df["year"].isin(target_profit_years)]
        profit_positive = profit_filtered.groupby("code").filter(
            lambda g: len(g) >= len(target_profit_years) * 0.8 and (g["net_profit"] > 0).all())
        profit_codes = set(profit_positive["code"].unique())
    else:
        profit_codes = fcf_codes
    return fcf_codes & profit_codes

def train_window(win, fcf_df, profit_df, cal, cal_set):
    by = win["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    
    ts, te = win["train"]
    vs, ve = win["valid"]
    bs, be = win["backtest"]

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
    
    # 【修复Bug】强制使用 pd.DataFrame，且列名为 "score" 适配 apply_value_fusion
    valid_features = dataset.prepare("valid", col_set="feature")
    pred = None
    if valid_features is not None and len(valid_features) > 0:
        dvalid = xgb.DMatrix(valid_features.values)
        valid_pred_vals = model.model.predict(dvalid)
        valid_pred = pd.DataFrame(valid_pred_vals, index=valid_features.index, columns=["score"])
        
    test_features = dataset.prepare("test", col_set="feature")
    if test_features is not None and len(test_features) > 0:
        dtest = xgb.DMatrix(test_features.values)
        test_pred_vals = model.model.predict(dtest)
        test_pred = pd.DataFrame(test_pred_vals, index=test_features.index, columns=["score"])
        pred = pd.concat([valid_pred, test_pred])

    # ================= 真实特征重要性解码 =================
    if win["name"] == "W6":
        train_features = dataset.prepare("train", col_set="feature")
        # 提取真实列名列表 (剥离MultiIndex)
        real_feature_names = [c[1] if isinstance(c, tuple) else c for c in train_features.columns]
        
        importance_dict = model.model.get_score(importance_type='gain')
        
        # 映射回真实名称
        decoded_imp = []
        for f_id, gain in importance_dict.items():
            idx = int(f_id.replace('f', ''))
            real_name = real_feature_names[idx] if idx < len(real_feature_names) else f_id
            decoded_imp.append({"feature": real_name, "gain": gain})
            
        imp_df = pd.DataFrame(decoded_imp).sort_values(by="gain", ascending=False).reset_index(drop=True)
        imp_df["gain_pct"] = imp_df["gain"] / imp_df["gain"].sum() * 100
        
        print(f"\n{'='*60}")
        print(f"  W6 窗口真实特征重要性 Top 20 (解码后，按 Gain 排序)")
        print(f"{'='*60}")
        for idx, row in imp_df.head(20).iterrows():
            is_fund = row["feature"] in {"roe_annual", "pb_pct_3y", "pe_pct_3y", "div_yield_est", "fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"}
            tag = " [★ 核心基本面/价值]" if is_fund else ""
            print(f"  {(idx+1):<3} {row['feature']:<20} {row['gain_pct']:>6.2f}%{tag}")
        print(f"{'='*60}\n")
    # ====================================================

    return pred, universe, vf, fund

def get_month_end_dates(cal, start, end):
    dates = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not dates: return []
    monthly = pd.DatetimeIndex(dates).to_period("M")
    month_ends = []
    seen = set()
    for i, d in enumerate(dates):
        m = monthly[i]
        if m not in seen: seen.add(m)
        if i + 1 < len(dates) and monthly[i + 1] != m: month_ends.append(d)
        elif i + 1 == len(dates): month_ends.append(d)
    return month_ends

def vectorized_backtest_with_buffer(pred, universe, cal, month_ends, topk=10, alpha=0.0, vf=None):
    """
    带 Buffer Rule 的向量化回测:
    - 跌出 Top20 才卖，新进 Top10 才买
    """
    if pred is None or len(pred) == 0: return pd.Series()

    if alpha > 0 and vf is not None:
        pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

    vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
    if vol_data is not None and len(vol_data) > 0:
        vol_data = vol_data.reindex(pred.index)
        # 【修复Bug】精确赋值到 "score" 列
        pred.loc[vol_data["$amount"] < 20_000_000, "score"] = -np.inf

    all_dates = sorted(pred.index.get_level_values(0).unique())
    all_instruments = list(set(pred.index.get_level_values(1)))
    price_data = D.features(all_instruments, ["$close"],
                           start_time=all_dates[0] - pd.Timedelta(days=10),
                           end_time=all_dates[-1] + pd.Timedelta(days=40))
    price_data = price_data.reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {inst: grp.sort_values("datetime").set_index("datetime")["close"] 
                  for inst, grp in price_data.groupby("instrument")}

    portfolio_returns = []
    portfolio_dates = []
    prev_topk = []
    turnovers = []

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]
            if not earlier: continue
            dt = earlier[-1]

        day_pred = pred.xs(dt, level=0)
        
        # ================= Buffer Rule 核心逻辑 =================
        top20_set = set(day_pred["score"].nlargest(20).index)
        top10_list = day_pred["score"].nlargest(10).index.tolist()

        curr_topk = []
        if not prev_topk:
            curr_topk = top10_list[:topk]
            turnover = 1.0
        else:
            # 规则1: 持有老仓中依然在 Top 20 里的票 (坚决不卖)
            for stock in prev_topk:
                if stock in top20_set:
                    curr_topk.append(stock)
            
            # 规则2: 剩余空位用新榜单的 Top 10 顺位填补 (新买入)
            for stock in top10_list:
                if len(curr_topk) >= topk:
                    break
                if stock not in curr_topk:
                    curr_topk.append(stock)
            
            # 计算真实换手率
            intersection = len(set(curr_topk).intersection(set(prev_topk)))
            turnover = 1.0 - (intersection / float(topk))
            turnovers.append(turnover)
        # ========================================================

        next_dt = month_ends[i + 1] if i + 1 < len(month_ends) else all_dates[-1]
        period_dates = [d for d in all_dates if dt < d <= next_dt]

        prev_prices = {inst: price_dict[inst][dt] for inst in curr_topk 
                       if inst in price_dict and dt in price_dict[inst].index}

        for pd_dt in period_dates:
            day_ret = 0
            n_valid = 0
            for inst in curr_topk:
                if inst in price_dict and pd_dt in price_dict[inst].index and inst in prev_prices:
                    cur_price = price_dict[inst][pd_dt]
                    if pd.notna(cur_price) and prev_prices[inst] > 0:
                        day_ret += (cur_price / prev_prices[inst] - 1)
                        prev_prices[inst] = cur_price
                        n_valid += 1
            daily_ret = day_ret / topk if n_valid > 0 else 0
            portfolio_returns.append(daily_ret)
            portfolio_dates.append(pd_dt)

        prev_topk = curr_topk

    avg_turnover = np.mean(turnovers) if turnovers else 1.0
    print(f"    --> 平均月度换手率 (带Buffer): {avg_turnover*100:.1f}%")

    returns = pd.Series(portfolio_returns, index=pd.DatetimeIndex(portfolio_dates))
    return returns[~returns.index.duplicated(keep="last")]

def calc_metrics(returns):
    if len(returns) == 0: return {"ar": 0, "sharpe": 0, "max_dd": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd}

def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    print(f"\n{'='*70}\n  执行全量回测 (含 Top20 Buffer Rule) \n{'='*70}")
    
    all_returns_alpha = []

    for win in WINDOWS:
        print(f"\n--- {win['name']} (回测{win['year']}) ---")
        pred, universe, vf, _ = train_window(win, fcf_df, profit_df, cal, cal_set)
        
        vs_w, ve_w = win["valid"]
        best_alpha, _, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
        
        bs, be = win["backtest"]
        month_ends = get_month_end_dates(cal, bs, be)

        ret_alpha = vectorized_backtest_with_buffer(
            pred, universe, cal, month_ends, topk=10, alpha=best_alpha, vf=vf)
        
        m_alpha = calc_metrics(ret_alpha)
        all_returns_alpha.append(ret_alpha)
        
        print(f"    Buffer回测结果: α={best_alpha}, 年化{m_alpha['ar']*100:.2f}%, 夏普{m_alpha['sharpe']:.2f}")

    print(f"\n{'='*70}\n  最终聚合结果\n{'='*70}")
    final_returns = pd.concat(all_returns_alpha).sort_index()
    final_returns = final_returns[~final_returns.index.duplicated(keep="last")]
    
    m_final = calc_metrics(final_returns)
    print(f"  策略: XGBoost + α融合 + Top20容忍池")
    print(f"  全期年化收益 : {m_final['ar']*100:.2f}%")
    print(f"  全期夏普比率 : {m_final['sharpe']:.2f}")
    print(f"  最大回撤     : {m_final['max_dd']*100:.1f}%")

if __name__ == "__main__":
    run()
