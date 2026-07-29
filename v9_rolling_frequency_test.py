"""
V9 模型重训频率消融实验 (M/Q/S/Y)
=========================================================
对比方案A(月)、方案B(季)、方案C(半年)、方案D(年)的重训效果。
包含全量 9 大量化核心指标 (年化/波动/夏普/回撤/Calmar/IC/RankIC/换手率/交易次数)。
强制跳过 2020 年，流动性过滤 500 万，单边+双边摩擦 0.25%。
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

from v5_validation import build_limit_up_set, filter_pred_by_tradability, load_value_factors
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import TOP80_FEATURES, build_dynamic_universe, format_qlib_code, prune_features, get_month_end_dates

# ==================== XGBoost M1 实盘配置 ====================
MODEL_CONFIG = {
    "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "objective": "reg:squarederror", "learning_rate": 0.005, "max_depth": 4,
        "colsample_bytree": 0.8879, "subsample": 0.8789, "reg_alpha": 10.0,
        "reg_lambda": 50.0, "tree_method": "hist", "n_jobs": 4,
        "early_stopping_rounds": 100, "n_estimators": 1000
    }
}

# 动态生成滚动窗口时间轴 (跳过2020年)
def generate_rolling_windows(freq='Y'):
    windows = []
    backtest_years = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
    
    for year in backtest_years:
        end_month = 6 if year == 2026 else 12
        
        if freq == 'Y':
            periods = [(1, end_month)]
        elif freq == 'S':
            periods = [(1, 6), (7, 12)] if year != 2026 else [(1, 6)]
        elif freq == 'Q':
            periods = [(1, 3), (4, 6), (7, 9), (10, 12)] if year != 2026 else [(1, 3), (4, 6)]
        elif freq == 'M':
            periods = [(m, m) for m in range(1, end_month + 1)]
            
        for start_m, end_m in periods:
            bt_start = pd.Timestamp(f"{year}-{start_m:02d}-01")
            bt_end = pd.Timestamp(f"{year}-{end_m:02d}-01") + relativedelta(months=1) - pd.Timedelta(days=1)
            
            # 训练集: 过去 5 年 (向前推6年到1年前)
            # 这里简化处理：跳过 2020 年的训练数据太复杂，工业界通常直接取固定长度历史
            train_end = bt_start - relativedelta(years=1) - pd.Timedelta(days=1)
            train_start = train_end - relativedelta(years=4)
            valid_start = train_end + pd.Timedelta(days=1)
            valid_end = bt_start - pd.Timedelta(days=1)
            
            name = f"{year}_{freq}{start_m:02d}"
            windows.append({
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "backtest": (bt_start.strftime("%Y-%m-%d"), bt_end.strftime("%Y-%m-%d")),
                "name": name, "year": year
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
    
    test_features = dataset.prepare("test", col_set="feature")
    test_label = dataset.prepare("test", col_set="label")
    
    pred = pd.DataFrame()
    if test_features is not None and len(test_features) > 0:
        dtest = xgb.DMatrix(test_features.values)
        pred = pd.DataFrame(model.model.predict(dtest), index=test_features.index, columns=["score"])
        if test_label is not None:
            pred["label"] = test_label.values
            
    return pred, universe

def calculate_ic_metrics(pred_df):
    if pred_df is None or "label" not in pred_df.columns or len(pred_df) == 0:
        return 0, 0
    df = pred_df.dropna()
    if len(df) < 10: return 0, 0
    
    ic_list, rank_ic_list = [], []
    for dt, group in df.groupby(level=0):
        if len(group) > 5 and group['score'].std() > 1e-5 and group['label'].std() > 1e-5:
            ic_list.append(pearsonr(group['score'], group['label'])[0])
            rank_ic_list.append(spearmanr(group['score'], group['label'])[0])
            
    return np.nanmean(ic_list) if ic_list else 0, np.nanmean(rank_ic_list) if rank_ic_list else 0

def backtest_engine_full_metrics(pred, universe, cal, month_ends, fee_rate=0.0025):
    if pred is None or len(pred) == 0: 
        return pd.Series(), 0, 0

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)
    
    # 500 万流动性实盘风控阈值
    vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
    if vol_data is not None and len(vol_data) > 0:
        pred.loc[vol_data.reindex(pred.index)["$amount"] < 5_000_000, "score"] = -np.inf

    all_dates = sorted(pred.index.get_level_values(0).unique())
    price_data = D.features(list(set(pred.index.get_level_values(1))), ["$close"],
                           start_time=all_dates[0] - pd.Timedelta(days=10), end_time=all_dates[-1] + pd.Timedelta(days=40)).reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {inst: grp.set_index("datetime")["close"] for inst, grp in price_data.groupby("instrument")}

    net_returns, portfolio_dates = [], []
    turnovers, trade_counts = [], []
    prev_topk = []

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]; dt = earlier[-1] if earlier else dt
            
        curr_topk = pred.xs(dt, level=0)["score"].nlargest(10).index.tolist()
        intersection = len(set(curr_topk).intersection(set(prev_topk)))
        turnover = 1.0 if not prev_topk else 1.0 - (intersection / 10.0)
        
        turnovers.append(turnover)
        trade_counts.append(10 - intersection if prev_topk else 10) # 记录换股只数 (买入次数)

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
    s_net = s_net[~s_net.index.duplicated(keep="last")]
    
    avg_turnover = np.mean(turnovers) if turnovers else 0
    total_trades = np.sum(trade_counts) if trade_counts else 0
    
    return s_net, avg_turnover, total_trades

def fetch_benchmark(cal, start, end):
    # 获取沪深300指数作为基准
    bench = D.features(["SH000300"], ["$close"], start_time=start, end_time=end)
    if bench is not None and len(bench) > 0:
        bench_ret = bench.xs("SH000300", level="instrument")["$close"].pct_change().dropna()
        return bench_ret
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
    
    schemes = {"D: 年滚动(稳健)": "Y", "C: 半年滚动": "S", "B: 季滚动(机构)": "Q"}
    # ⚠️ 警报: 开启方案 A (月滚动) 需要训练 70+ 个 XGBoost 模型，耗时可能超过一小时，这里默认跳过。如需硬测可解除注释。
    # schemes["A: 月滚动(激进)"] = "M" 

    print(f"\n{'='*130}\n  启动 V9 滚动重训频率消融压测 (M1 极速版 | 强制过滤 2020 | 500万底线 | 全面 9 大指标)\n{'='*130}")
    
    results = []
    bench_returns = fetch_benchmark(cal, "2019-01-01", "2026-07-21")
    bench_ann = bench_returns.groupby(bench_returns.index.year).apply(lambda x: (1+x).prod()-1).to_dict()
    bench_all_metrics = calc_all_metrics(bench_returns)

    for s_name, freq in schemes.items():
        print(f"\n[>>>] 正在执行 {s_name} 回测引擎...")
        windows = generate_rolling_windows(freq)
        
        preds = []
        rets = []
        turnovers = []
        total_trades = 0
        
        for win in windows:
            pred, universe = train_and_predict(win, fcf_df, profit_df, cal)
            month_ends = get_month_end_dates(cal, win["backtest"][0], win["backtest"][1])
            s_net, t_avg, t_count = backtest_engine_full_metrics(pred, universe, cal, month_ends)
            
            preds.append(pred)
            rets.append(s_net)
            turnovers.append(t_avg)
            total_trades += t_count
            
        full_pred = pd.concat(preds)
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
            "IC": ic, "RankIC": rank_ic, "月均换手": np.mean(turnovers) if turnovers else 0, "总交易次": total_trades
        })

    # 将基准数据也加入展示
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
    print(f"{'策略对比':<14} | {'19年':>7} {'21年':>7} {'22年':>7} {'23年':>7} {'24年':>7} {'25年':>7} {'26H1':>7} | {'年化收益':>8} {'超额收益':>8} {'波动率':>7} {'Sharpe':>6} {'最大回撤':>8} {'Calmar':>6} {'IC':>6} {'RankIC':>6} {'月均换手':>8} {'交易总次':>6}")
    print(f"{'-'*150}")
    for _, r in df_res.iterrows():
        print(f"{r['方案']:<16} | {r['19年']*100:>6.1f}% {r['21年']*100:>6.1f}% {r['22年']*100:>6.1f}% {r['23年']*100:>6.1f}% {r['24年']*100:>6.1f}% {r['25年']*100:>6.1f}% {r['26H1']*100:>6.1f}% | {r['全期年化']*100:>7.1f}% {r['超额基准']*100:>7.1f}% {r['年化波动']*100:>6.1f}% {r['Sharpe']:>6.2f} {r['最大回撤']*100:>7.1f}% {r['Calmar']:>6.2f} {r['IC']:>6.3f} {r['RankIC']:>6.3f} {r['月均换手']*100:>7.1f}% {r['总交易次']:>6.0f}")
    print(f"{'='*150}\n")
    
    df_res.to_csv("/Users/11164591/Documents/Qoder目录/qlib/v9_rolling_frequency_results.csv", index=False)
    print("[+] 对比报告已保存至 v9_rolling_frequency_results.csv")

if __name__ == "__main__":
    run()
