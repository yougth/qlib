"""
V7 生产环境实盘发车版 (M1 极速版 + 全景输出)
=========================================================
1. 包含 2019-2026 全量窗口 (补齐 2020)
2. 剔除无效的宏观阀门，聚焦 [纯Top10基线] 与 [Alpha后置融合]
3. 强制扣除 0.25% 交易摩擦，计算真实净收益
4. 输出实盘必需品: 特征重要性、年度净收益矩阵、2026真实换仓单、全市场打分CSV
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

try:
    import akshare as ak
    print("[+] 正在加载 A 股真实名称映射表...")
    stock_info = ak.stock_info_a_code_name()
    NAME_MAP = dict(zip(stock_info['symbol'], stock_info['name']))
except:
    print("[-] 未检测到 akshare 或网络超时，将仅显示股票代码。")
    NAME_MAP = {}

from v5_validation import apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, load_value_factors, search_alpha_on_valid
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import TOP80_FEATURES, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates

# ==================== XGBoost M1 极速实盘配置 ====================
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
        "tree_method": "hist",  # M1 直方图极速引擎
        "n_jobs": 4,            # 4核并发防过热
        "early_stopping_rounds": 100,
        "n_estimators": 1000              
    }
}

# ==================== 完整 8 年滚动窗口 (含 2020) ====================
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

    # === [输出] 打印最新一期的特征重要性 ===
    if win["name"] == "W6":
        train_features = dataset.prepare("train", col_set="feature")
        real_feature_names = [c[1] if isinstance(c, tuple) else c for c in train_features.columns]
        importance_dict = model.model.get_score(importance_type='gain')
        
        decoded_imp = []
        for f_id, gain in importance_dict.items():
            idx = int(f_id.replace('f', ''))
            real_name = real_feature_names[idx] if idx < len(real_feature_names) else f_id
            decoded_imp.append({"feature": real_name, "gain": gain})
            
        imp_df = pd.DataFrame(decoded_imp).sort_values(by="gain", ascending=False).reset_index(drop=True)
        imp_df["gain_pct"] = imp_df["gain"] / imp_df["gain"].sum() * 100
        
        print(f"\n{'='*70}\n  实盘监控: 最新 W6 窗口 XGBoost 特征重要性 Top 15\n{'='*70}")
        for idx, row in imp_df.head(15).iterrows():
            is_fund = row["feature"] in {"roe_annual", "pb_pct_3y", "pe_pct_3y", "div_yield_est", "fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"}
            tag = " [★ 基本面因子]" if is_fund else ""
            print(f"  {(idx+1):<2} {row['feature']:<20} 贡献度: {row['gain_pct']:>5.2f}% {tag}")
        print(f"{'='*70}\n")

    return pred, universe, vf, fund

def backtest_engine_net(pred, universe, cal, month_ends, alpha=0.0, vf=None, fee_rate=0.0025, print_holdings=False):
    """带实盘扣费和真实交割单打印的回测引擎"""
    if pred is None or len(pred) == 0: return pd.Series()
    
    # 1. 估值因子融合
    if alpha > 0 and vf is not None: pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    # 2. 真实流动性与涨跌停约束
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)
    vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
    if vol_data is not None and len(vol_data) > 0:
        pred.loc[vol_data.reindex(pred.index)["$amount"] < 20_000_000, "score"] = -np.inf

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

        # === [输出] 打印真实股票换仓名单 ===
        if print_holdings:
            date_str = pd.Timestamp(dt).strftime('%Y-%m-%d')
            top10_names = [get_stock_name(c) for c in curr_topk]
            print(f"【{date_str} 换仓执行 (月换手率: {turnover*100:.0f}%)】")
            print(f"  买入/持有: {', '.join(top10_names[:5])}")
            print(f"             {', '.join(top10_names[5:])}\n")

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
            cost = (turnover * fee_rate) if p_idx == 0 else 0.0  # 仅在调仓首日扣除当月所有摩擦费
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

    print(f"\n{'='*95}\n  启动 V7 实盘生产线 (强制 0.25% 扣费校验)\n{'='*95}")
    
    rets_base, rets_alpha = [], []
    final_pred_scores = None

    for win in WINDOWS:
        print(f"[+] 正在并行训练 {win['name']} 窗口 (回测 {win['year']}) ...")
        pred, universe, vf, _ = train_window(win, fcf_df, profit_df, cal, cal_set)
        
        vs_w, ve_w = win["valid"]
        best_alpha, _, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
        month_ends = get_month_end_dates(cal, win["backtest"][0], win["backtest"][1])
        
        # 记录最后一期的预测打分，用于实盘输出
        if win["name"] == "W6": 
            final_pred_scores = pred
            print(f"\n{'='*70}\n  实盘监控: 2026 H1 Alpha融合方案 真实调仓交割单\n{'='*70}")
            
        rn_base = backtest_engine_net(pred, universe, cal, month_ends, alpha=0.0, vf=vf, fee_rate=0.0025, print_holdings=False)
        rn_alpha = backtest_engine_net(pred, universe, cal, month_ends, alpha=best_alpha, vf=vf, fee_rate=0.0025, print_holdings=(win["name"]=="W6"))
        
        rets_base.append(rn_base)
        rets_alpha.append(rn_alpha)

    # ====== 收益汇总与指标拆解 ======
    s_base = pd.concat(rets_base).sort_index(); s_base = s_base[~s_base.index.duplicated(keep="last")]
    s_alpha = pd.concat(rets_alpha).sort_index(); s_alpha = s_alpha[~s_alpha.index.duplicated(keep="last")]
    
    m_b = calc_metrics(s_base); ann_b = get_annual_returns(s_base)
    m_a = calc_metrics(s_alpha); ann_a = get_annual_returns(s_alpha)
    
    print(f"\n{'='*115}")
    print(f"  V7 生产环境 核心策略年度净收益矩阵 (已严格扣除交易摩擦)")
    print(f"{'='*115}")
    print(f"{'策略分支':<18} | {'2019':>7} {'2020':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'2026H1':>7} | {'全期净年化':>10} {'净夏普':>6} {'净最大回撤':>9}")
    print(f"{'-'*115}")
    print(f"{'纯Top10 (绝对基线)':<16} | {ann_b.get(2019,0)*100:>6.2f}% {ann_b.get(2020,0)*100:>6.2f}% {ann_b.get(2021,0)*100:>6.2f}% {ann_b.get(2022,0)*100:>6.2f}% {ann_b.get(2023,0)*100:>6.2f}% {ann_b.get(2024,0)*100:>6.2f}% {ann_b.get(2025,0)*100:>6.2f}% {ann_b.get(2026,0)*100:>6.2f}% | {m_b['ar']*100:>9.2f}% {m_b['sharpe']:>6.2f} {m_b['max_dd']*100:>8.1f}%")
    print(f"{'Alpha价值融合 (优选)':<14} | {ann_a.get(2019,0)*100:>6.2f}% {ann_a.get(2020,0)*100:>6.2f}% {ann_a.get(2021,0)*100:>6.2f}% {ann_a.get(2022,0)*100:>6.2f}% {ann_a.get(2023,0)*100:>6.2f}% {ann_a.get(2024,0)*100:>6.2f}% {ann_a.get(2025,0)*100:>6.2f}% {ann_a.get(2026,0)*100:>6.2f}% | {m_a['ar']*100:>9.2f}% {m_a['sharpe']:>6.2f} {m_a['max_dd']*100:>8.1f}%")
    print(f"{'='*115}\n")

    # ====== 实盘数据落盘 ======
    work_dir = "/Users/11164591/Documents/Qoder目录/qlib"
    (1 + s_base).cumprod().to_csv(f"{work_dir}/v7_nav_baseline_net.csv", sep='\t', header=False)
    (1 + s_alpha).cumprod().to_csv(f"{work_dir}/v7_nav_alpha_net.csv", sep='\t', header=False)
    
    if final_pred_scores is not None:
        # 提取最新一天（或当月最后一天）的所有候选股票打分并保存
        last_date = final_pred_scores.index.get_level_values(0).max()
        latest_scores = final_pred_scores.xs(last_date, level=0).sort_values("score", ascending=False).reset_index()
        latest_scores["stock_name"] = latest_scores["instrument"].apply(get_stock_name)
        latest_scores = latest_scores[["instrument", "stock_name", "score"]]
        latest_scores.to_csv(f"{work_dir}/v7_live_scores_latest.csv", sep='\t', index=False)
        
    print("[+] 实盘部署物料导出完毕:")
    print("    1. [回测净值] v7_nav_baseline_net.csv, v7_nav_alpha_net.csv")
    print("    2. [下月跟单] v7_live_scores_latest.csv (包含了模型对当前所有备选股票的最新打分排名，实盘买入前 10 名即可)")

if __name__ == "__main__":
    run()
