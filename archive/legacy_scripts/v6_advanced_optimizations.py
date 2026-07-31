"""
V6 引擎进阶优化压测: Label重构 / 衍生特征 / 动态加权
=========================================================
隔离测试 3 种前沿调优手段对 XGBoost 纯 Top10 实盘净收益的影响
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

from v5_validation import apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, load_value_factors, search_alpha_on_valid
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import WINDOWS, TOP80_FEATURES, MODEL_CONFIG, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates

# ==================== 核心修改逻辑 ====================

def inject_cross_features(dataset):
    """注入基本面与量价的高阶交叉衍生特征"""
    handler = dataset.handler
    for attr in ["_infer", "_learn"]:
        if not hasattr(handler, attr): continue
        df = getattr(handler, attr)
        if df is None: continue

        def get_col(name):
            return df[("feature", name)] if ("feature", name) in df.columns else None

        roe, pe = get_col("roe_annual"), get_col("pe_pct_3y")
        fcf, std = get_col("fcf_avg_3y_norm"), get_col("STD20")
        roc240 = get_col("ROC240")

        # 1. GARP 估值性价比: 盈利能力 / 估值分位
        if roe is not None and pe is not None:
            df[("feature", "CROSS_GARP")] = roe / (pe + 0.01)
        # 2. 风险调整后现金流: 自由现金流 / 短期波动率
        if fcf is not None and std is not None:
            df[("feature", "CROSS_FCF_RISK")] = fcf / (std + 0.01)
        # 3. 长期动量夏普: 240日收益 / 20日波动
        if roc240 is not None and std is not None:
            df[("feature", "CROSS_SHARPE_240")] = roc240 / (std + 0.01)

        setattr(handler, attr, df)

def train_window(win, fcf_df, profit_df, cal, cal_set, mode="baseline"):
    by = win["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    ts, te, vs, ve, bs, be = win["train"][0], win["train"][1], win["valid"][0], win["valid"][1], win["backtest"][0], win["backtest"][1]

    # 根据实验模式动态修改 Label 处理器
    label_processor = {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}
    if mode == "label_rank":
        label_processor = {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}

    dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
        "instruments": universe,
        "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                              {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors": [{"class":"DropnaLabel"}, label_processor],
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
    
    # 根据实验模式注入衍生特征
    if mode == "feature_cross":
        inject_cross_features(dataset)

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

    return pred, universe, vf, fund

def backtest_engine(pred, universe, cal, month_ends, alpha=0.3, vf=None, fee_rate=0.0025, mode="baseline"):
    if pred is None or len(pred) == 0: return pd.Series()
    if alpha > 0 and vf is not None: pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

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

        next_dt = month_ends[i + 1] if i + 1 < len(month_ends) else all_dates[-1]
        period_dates = [d for d in all_dates if dt < d <= next_dt]
        prev_prices = {inst: price_dict[inst][dt] for inst in curr_topk if inst in price_dict and dt in price_dict[inst].index}

        # 根据实验模式动态分配持仓权重
        actual_k = len(curr_topk)
        if mode == "dynamic_weight" and actual_k > 0:
            w_arr = np.arange(actual_k, 0, -1)
            weights = w_arr / w_arr.sum()  # 排名越靠前权重越大
        else:
            weights = np.ones(actual_k) / actual_k if actual_k > 0 else []

        for p_idx, pd_dt in enumerate(period_dates):
            day_ret = 0
            for idx, inst in enumerate(curr_topk):
                if inst in price_dict and pd_dt in price_dict[inst].index and inst in prev_prices:
                    cur_price = price_dict[inst][pd_dt]
                    if pd.notna(cur_price) and prev_prices[inst] > 0:
                        day_ret += (cur_price / prev_prices[inst] - 1) * weights[idx]
                        prev_prices[inst] = cur_price
            
            cost = (turnover * fee_rate) if p_idx == 0 else 0.0
            net_returns.append(day_ret - cost)
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
    """提取每一年的独立收益率"""
    if len(returns) == 0: return {}
    return returns.groupby(returns.index.year).apply(lambda x: (1+x).prod()-1).to_dict()

def run_experiments():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    EXPERIMENTS = [
        {"name": "1. 纯Top10 基线", "mode": "baseline"},
        {"name": "2. Label 重构 (RankNorm)", "mode": "label_rank"},
        {"name": "3. 衍生特征扩充 (Cross)", "mode": "feature_cross"},
        {"name": "4. 动态打分加权 (Rank WT)", "mode": "dynamic_weight"}
    ]

    results_table = []
    
    print(f"\n{'='*90}\n  启动 V6 引擎独立进阶压测 (包含 0.25% 真实交易摩擦费用)\n{'='*90}")

    for exp in EXPERIMENTS:
        print(f"\n>>> 正在运行实验: {exp['name']} ...")
        exp_rets = []
        for win in WINDOWS:
            pred, universe, vf, _ = train_window(win, fcf_df, profit_df, cal, cal_set, mode=exp["mode"])
            vs_w, ve_w = win["valid"]
            best_alpha, _, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
            month_ends = get_month_end_dates(cal, win["backtest"][0], win["backtest"][1])
            
            rn = backtest_engine(pred, universe, cal, month_ends, alpha=best_alpha, vf=vf, fee_rate=0.0025, mode=exp["mode"])
            exp_rets.append(rn)
        
        s_net = pd.concat(exp_rets).sort_index(); s_net = s_net[~s_net.index.duplicated(keep="last")]
        m_net = calc_metrics(s_net)
        ann_rets = get_annual_returns(s_net)
        
        results_table.append({
            "实验分支": exp["name"],
            "2019": ann_rets.get(2019, 0), "2021": ann_rets.get(2021, 0),
            "2022": ann_rets.get(2022, 0), "2023": ann_rets.get(2023, 0),
            "2024": ann_rets.get(2024, 0), "2025": ann_rets.get(2025, 0),
            "2026H1": ann_rets.get(2026, 0),
            "全期净年化": m_net['ar'], "净夏普": m_net['sharpe'], "净最大回撤": m_net['max_dd']
        })

    # 打印精美的年度横评表
    df_res = pd.DataFrame(results_table)
    print(f"\n{'='*110}")
    print(f"  各项优化策略 年度净收益与全局指标拆解 (扣费后)")
    print(f"{'='*110}")
    print(f"{'策略分支':<22} | {'2019':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'2026H1':>7} | {'全期净年化':>10} {'夏普':>6} {'最大回撤':>9}")
    print(f"{'-'*110}")
    for _, r in df_res.iterrows():
        print(f"{r['实验分支']:<20} | {r['2019']*100:>6.2f}% {r['2021']*100:>6.2f}% {r['2022']*100:>6.2f}% {r['2023']*100:>6.2f}% {r['2024']*100:>6.2f}% {r['2025']*100:>6.2f}% {r['2026H1']*100:>6.2f}% | {r['全期净年化']*100:>9.2f}% {r['净夏普']:>6.2f} {r['净最大回撤']*100:>8.1f}%")
    print(f"{'='*110}\n")

if __name__ == "__main__":
    run_experiments()
