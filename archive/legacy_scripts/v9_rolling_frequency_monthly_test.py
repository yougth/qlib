"""
V10 终极月度重训消融实验 (单独压测版)
=========================================================
实现两种月度重训方案，验证极其耗时的细粒度迭代是否有超额收益：
1. 方案 A1 (滑动窗口): 永远保持过去 4年训练+1年验证，按月往后滑动。
2. 方案 A2 (追加扩展): 训练起点锁死在年初，每月把最新数据追加进训练集，满一年重置。
(包含全面 9 大量化指标，强制剔除 2020，流动性过滤 500 万)
"""
import os, sys, warnings, logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from dateutil.relativedelta import relativedelta

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.getLogger('qlib.online operator').setLevel(logging.ERROR)
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set, filter_pred_by_tradability
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import TOP80_FEATURES, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates

# ==================== XGBoost M1 配置 ====================
MODEL_CONFIG = {
    "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "objective": "reg:squarederror", "learning_rate": 0.005, "max_depth": 4,
        "colsample_bytree": 0.8879, "subsample": 0.8789, "reg_alpha": 10.0,
        "reg_lambda": 50.0, "tree_method": "hist", "n_jobs": 4,
        "early_stopping_rounds": 100, "n_estimators": 1000
    }
}

# 动态生成按月滚动的窗口
def generate_monthly_windows(mode='A1'):
    windows = []
    backtest_years = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
    
    for year in backtest_years:
        end_m = 6 if year == 2026 else 12
        for m in range(1, end_m + 1):
            # 回测区间: 严格只有当前这 1 个月
            bt_start = pd.Timestamp(year, m, 1)
            bt_end = bt_start + relativedelta(months=1) - pd.Timedelta(days=1)
            
            # 验证集: 测试月往前推 1 年
            valid_end = bt_start - pd.Timedelta(days=1)
            valid_start = valid_end - relativedelta(years=1) + pd.Timedelta(days=1)
            
            # 训练集终点: 验证集起点前一天
            train_end = valid_start - pd.Timedelta(days=1)
            
            if mode == 'A1': 
                # 方案 A1: 严格滑动，训练集永远保持过去 4 年长度
                train_start = train_end - relativedelta(years=4) + pd.Timedelta(days=1)
            else:
                # 方案 A2: 追加扩展，每年初锁定起点(Y-5)，每月追加数据
                train_start = pd.Timestamp(year - 5, 1, 1)
                
            windows.append({
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "backtest": (bt_start.strftime("%Y-%m-%d"), bt_end.strftime("%Y-%m-%d")),
                "name": f"{year}_M{m:02d}", "year": year
            })
    return windows

def train_and_predict(win, fcf_df, profit_df, cal):
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

    fund = load_fundamental_features_fixed(universe, cal)
    if fund is not None: inject_features_fixed(dataset, fund, "fund")
    prune_features(dataset, TOP80_FEATURES)

    model = init_instance_by_config(MODEL_CONFIG)
    model.fit(dataset)
    
    # 提取 Valid 和 Test 数据，确保能在月底生成下月信号
    valid_feat = dataset.prepare("valid", col_set="feature")
    test_feat = dataset.prepare("test", col_set="feature")
    test_label = dataset.prepare("test", col_set="label")
    
    pred_parts = []
    if valid_feat is not None and len(valid_feat) > 0:
        pred_parts.append(pd.DataFrame(model.model.predict(xgb.DMatrix(valid_feat.values)), index=valid_feat.index, columns=["score"]))
    if test_feat is not None and len(test_feat) > 0:
        test_pred = pd.DataFrame(model.model.predict(xgb.DMatrix(test_feat.values)), index=test_feat.index, columns=["score"])
        if test_label is not None: test_pred["label"] = test_label.values
        pred_parts.append(test_pred)
        
    pred = pd.concat(pred_parts) if pred_parts else pd.DataFrame()
    
    # 只返回当月测试集的预测数据供 IC 计算
    test_pred_only = pred_parts[1] if len(pred_parts) > 1 else pd.DataFrame()
    return pred, test_pred_only, universe

def calculate_ic_metrics(pred_df):
    if pred_df is None or "label" not in pred_df.columns or len(pred_df) == 0: return 0, 0
    df = pred_df.dropna()
    ic_list, rank_ic_list = [], []
    for dt, group in df.groupby(level=0):
        if len(group) > 5 and group['score'].std() > 1e-5 and group['label'].std() > 1e-5:
            ic_list.append(pearsonr(group['score'], group['label'])[0])
            rank_ic_list.append(spearmanr(group['score'], group['label'])[0])
    return np.nanmean(ic_list) if ic_list else 0, np.nanmean(rank_ic_list) if rank_ic_list else 0

def backtest_engine_monthly(pred, universe, cal, win, fee_rate=0.0025):
    if pred is None or len(pred) == 0: return pd.Series(), 0, 0
    
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

    # 获取当前测试月所需的所有调仓节点（通常是上月末）
    prior_start = (pd.Timestamp(win["backtest"][0]) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    month_ends = get_month_end_dates(cal, prior_start, win["backtest"][1])
    
    bs_ts, be_ts = pd.Timestamp(win["backtest"][0]), pd.Timestamp(win["backtest"][1])
    net_returns, portfolio_dates = [], []
    turnover = 0.0
    trade_count = 0

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]; dt = earlier[-1] if earlier else dt
            
        curr_topk = pred.xs(dt, level=0)["score"].nlargest(10).index.tolist()
        
        # 只在持仓期与当前测试月(bs_ts 到 be_ts)有交集时才记录收益
        next_dt = month_ends[i + 1] if i + 1 < len(month_ends) else all_dates[-1]
        period_dates = [d for d in all_dates if dt < d <= next_dt and bs_ts <= d <= be_ts]
        
        if not period_dates: continue
        
        # 为了极简，单月测试集假定月初买入时 100% 换手
        turnover = 1.0
        trade_count = 10
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

    s_net = pd.Series(net_returns, index=pd.DatetimeIndex(portfolio_dates))
    return s_net[~s_net.index.duplicated(keep="last")], turnover, trade_count

def fetch_benchmark(cal, start, end):
    bench = D.features(["SH000300"], ["$close"], start_time=start, end_time=end)
    if bench is not None and len(bench) > 0:
        return bench.xs("SH000300", level="instrument")["$close"].pct_change().dropna()
    return pd.Series()

def calc_all_metrics(returns):
    if len(returns) == 0: return {"ar": 0, "vol": 0, "sharpe": 0, "max_dd": 0, "calmar": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    calmar = ar / abs(max_dd) if max_dd < 0 else 0
    return {"ar": ar, "vol": vol, "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar}

def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    
    schemes = {
        "A1: 月滑动 (每月抛弃老数据)": "A1", 
        "A2: 月追加 (每年重置追加新数据)": "A2"
    }

    print(f"\n{'='*140}\n  启动 V10 月度极高频重训消融实验 (将并行训练 156 个模型，请耐心等待)\n{'='*140}")
    
    results = []
    bench_returns = fetch_benchmark(cal, "2019-01-01", "2026-07-21")
    bench_ann = bench_returns.groupby(bench_returns.index.year).apply(lambda x: (1+x).prod()-1).to_dict()
    bench_all_metrics = calc_all_metrics(bench_returns)

    for s_name, mode in schemes.items():
        print(f"\n[>>>] 正在执行极其耗时的 {s_name} 引擎...")
        windows = generate_monthly_windows(mode)
        
        preds_test_only = []
        rets = []
        total_trades = 0
        
        for win in windows:
            print(f"  -> 训练: {win['name']} (Train: {win['train'][0]} ~ {win['train'][1]})")
            pred_full, pred_test, universe = train_and_predict(win, fcf_df, profit_df, cal)
            s_net, t_avg, t_count = backtest_engine_monthly(pred_full, universe, cal, win)
            
            if not pred_test.empty: preds_test_only.append(pred_test)
            if not s_net.empty: rets.append(s_net)
            total_trades += t_count
            
        full_pred = pd.concat(preds_test_only)
        full_ret = pd.concat(rets).sort_index(); full_ret = full_ret[~full_ret.index.duplicated(keep="last")]
        
        m_all = calc_all_metrics(full_ret)
        ic, rank_ic = calculate_ic_metrics(full_pred)
        ann_ret = full_ret.groupby(full_ret.index.year).apply(lambda x: (1+x).prod()-1).to_dict()
        
        results.append({
            "方案": s_name,
            "19年": ann_ret.get(2019, 0), "21年": ann_ret.get(2021, 0), "22年": ann_ret.get(2022, 0),
            "23年": ann_ret.get(2023, 0), "24年": ann_ret.get(2024, 0), "25年": ann_ret.get(2025, 0), "26H1": ann_ret.get(2026, 0),
            "全期年化": m_all['ar'], "超额基准": m_all['ar'] - bench_all_metrics['ar'],
            "年化波动": m_all['vol'], "Sharpe": m_all['sharpe'], "最大回撤": m_all['max_dd'], "Calmar": m_all['calmar'],
            "IC": ic, "RankIC": rank_ic, "月均换手": 1.0, "总交易次": total_trades
        })

    results.append({
        "方案": "沪深300 (基准)",
        "19年": bench_ann.get(2019, 0), "21年": bench_ann.get(2021, 0), "22年": bench_ann.get(2022, 0),
        "23年": bench_ann.get(2023, 0), "24年": bench_ann.get(2024, 0), "25年": bench_ann.get(2025, 0), "26H1": bench_ann.get(2026, 0),
        "全期年化": bench_all_metrics['ar'], "超额基准": 0.0,
        "年化波动": bench_all_metrics['vol'], "Sharpe": bench_all_metrics['sharpe'], "最大回撤": bench_all_metrics['max_dd'], "Calmar": bench_all_metrics['calmar'],
        "IC": 0, "RankIC": 0, "月均换手": 0, "总交易次": 0
    })

    df_res = pd.DataFrame(results)
    
    print(f"\n{'='*150}")
    print(f"{'策略对比':<22} | {'19年':>7} {'21年':>7} {'22年':>7} {'23年':>7} {'24年':>7} {'25年':>7} {'26H1':>7} | {'年化收益':>8} {'超额收益':>8} {'波动率':>7} {'Sharpe':>6} {'最大回撤':>8} {'Calmar':>6} {'IC':>6} {'RankIC':>6}")
    print(f"{'-'*150}")
    for _, r in df_res.iterrows():
        print(f"{r['方案']:<24} | {r['19年']*100:>6.1f}% {r['21年']*100:>6.1f}% {r['22年']*100:>6.1f}% {r['23年']*100:>6.1f}% {r['24年']*100:>6.1f}% {r['25年']*100:>6.1f}% {r['26H1']*100:>6.1f}% | {r['全期年化']*100:>7.1f}% {r['超额基准']*100:>7.1f}% {r['年化波动']*100:>6.1f}% {r['Sharpe']:>6.2f} {r['最大回撤']*100:>7.1f}% {r['Calmar']:>6.2f} {r['IC']:>6.3f} {r['RankIC']:>6.3f}")
    print(f"{'='*150}\n")
    
    df_res.to_csv("/Users/11164591/Documents/Qoder目录/qlib/v10_monthly_rolling_results.csv", index=False)
    print("[+] 终极月度报告已保存至 v10_monthly_rolling_results.csv")

if __name__ == "__main__":
    run()
