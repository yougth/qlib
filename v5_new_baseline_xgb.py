"""
V5 新基线 (XGBoost版) + Alpha权重检查 + 终极E8压测
=========================================
1. 新V5基线: 5年滚动 + Top10 + 无阀门 + 81特征 (XGBoost)
2. Alpha权重检查: 直接输出 vs α=0.3后置价值融合
3. 终极E8: 新基线 + 方案D宏观阀门 + 基本面硬过滤, 8年(排除2020)

所有CSV读写均使用 sep='\t' (制表符分隔)
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
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
import warnings
warnings.filterwarnings("ignore")

from v5_validation import Alpha158Enhanced, apply_value_fusion, \
    build_limit_up_set, filter_pred_by_tradability, load_value_factors, \
    search_alpha_on_valid, MonthlyTopkStrategy
from v5_pipeline_fix import load_fundamental_features_fixed, \
    inject_features_fixed, filter_by_fundamental_deterioration

# ==================== XGBoost 模型配置 ====================
MODEL_CONFIG = {
    "class": "XGBModel", 
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "objective": "reg:squarederror",  # 对应 mse loss
        "learning_rate": 0.005,
        "max_depth": 4,                   # 对应 LightGBM num_leaves=15 (2^4=16)
        "colsample_bytree": 0.8879,
        "subsample": 0.8789,
        "reg_alpha": 10.0,                # 对应 LightGBM lambda_l1
        "reg_lambda": 50.0,               # 对应 LightGBM lambda_l2
        "n_jobs": 4,                      # 对应 num_threads
        "early_stopping_rounds": 100,
        "n_estimators": 1000              # 对应 num_boost_round
    }
}

# 7个窗口: W0(2019) + W1-W6(2021-2026H1), 排除2020
WINDOWS = [
    {"train": ("2014-01-01","2017-12-31"), "valid": ("2018-01-01","2018-12-31"), "backtest": ("2019-01-01","2019-12-31"), "name":"W0", "year": 2019},
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1", "year": 2021},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2", "year": 2022},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3", "year": 2023},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4", "year": 2024},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5", "year": 2025},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "year": 2026},
]

# ==================== 特征硬剪枝: Top80 ====================
TOP80_FEATURES = {
    # 基本面+价值因子 (9个)
    "roe_annual", "pb_pct_3y", "pe_pct_3y", "div_yield_est",
    "fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y",
    # Alpha158 Top71 (按gain排序)
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
    """硬剪枝: 只保留keep_features中的特征列+所有非feature列(如label)"""
    handler = dataset.handler
    for attr_name in ["_infer", "_learn", "_data"]:
        if not hasattr(handler, attr_name):
            continue
        df = getattr(handler, attr_name)
        if df is None:
            continue
        is_multi = isinstance(df.columns, pd.MultiIndex)
        if is_multi:
            keep_cols = [c for c in df.columns
                         if c[0] != "feature" or c[1] in keep_features]
        else:
            keep_cols = [c for c in df.columns if c in keep_features]
        n_before = len(df.columns)
        setattr(handler, attr_name, df[keep_cols])
        n_after = len(keep_cols)
        if attr_name == "_infer":
            print(f"    特征硬剪枝: {n_before}→{n_after}列 (剔除{n_before-n_after}个零增益特征)")

def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"

def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    fcf_start = backtest_year - 11; fcf_end = backtest_year - 2
    profit_start = max(backtest_year - 11, 2016); profit_end = backtest_year - 2
    target_fcf_years = list(range(fcf_start, fcf_end + 1))
    target_profit_years = list(range(profit_start, profit_end + 1))
    fcf_filtered = fcf_df[fcf_df["year"].isin(target_fcf_years)]
    fcf_positive = fcf_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_positive["code"].unique())
    if profit_end >= 2016 and len(target_profit_years) > 0:
        profit_filtered = profit_df[profit_df["year"].isin(target_profit_years)]
        profit_positive = profit_filtered.groupby("code").filter(
            lambda g: len(g) >= len(target_profit_years) * 0.8 and (g["net_profit"] > 0).all())
        profit_codes = set(profit_positive["code"].unique())
    else:
        profit_codes = fcf_codes
    return fcf_codes & profit_codes

def train_window(win, fcf_df, profit_df, cal, cal_set):
    """训练单个窗口的模型, 返回 (pred, universe, vf, fund)"""
    by = win["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"  {win['name']}: 股票池{len(universe)}只, 回测{win['backtest'][0]}~{win['backtest'][1]}")

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
    if fund is not None:
        inject_features_fixed(dataset, fund, "fund")
    if vf is not None:
        inject_features_fixed(dataset, vf, "vf")

    prune_features(dataset, TOP80_FEATURES)

    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_xgb_baseline"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    import xgboost as xgb  # 引入 xgboost
    
    valid_features = dataset.prepare("valid", col_set="feature")
    if valid_features is not None and len(valid_features) > 0:
        # 强制转换为 DMatrix 格式
        dvalid = xgb.DMatrix(valid_features.values)
        valid_pred_vals = model.model.predict(dvalid)
        valid_pred = pd.Series(valid_pred_vals, index=valid_features.index, name="score")
        pred = pd.concat([valid_pred, pred])
        print(f"    valid预测: {len(valid_pred)}条, 合并后预测: {len(pred)}条")

    train_data = dataset.prepare("train", col_set="feature")
    n_feat = train_data.shape[1]
    print(f"    特征数: {n_feat}, 预测: {len(pred)}条")

    return pred, universe, vf, fund

def compute_macro_signals_all(universe, cal, lookback=300):
    """计算全A等权200日均线信号 (所有日期)"""
    start = cal[0] - pd.Timedelta(days=lookback)
    prices = D.features(universe, ["$close"], start_time=start, end_time=cal[-1])
    if prices is None or len(prices) == 0:
        return {}
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "close"]
    prices = prices.sort_values(["instrument", "datetime"])
    prices["ret"] = prices.groupby("instrument")["close"].pct_change()
    ew_ret = prices.groupby("datetime")["ret"].mean().dropna()
    ew_cum = (1 + ew_ret).cumprod()
    ew_ma200 = ew_cum.rolling(200, min_periods=60).mean()
    ew_cum.index = pd.to_datetime(ew_cum.index)
    ew_ma200.index = ew_cum.index

    signals = {}
    for dt in cal:
        dt_ts = pd.Timestamp(dt)
        if dt_ts in ew_cum.index and not pd.isna(ew_ma200[dt_ts]):
            signals[dt_ts] = 0.7 if ew_cum[dt_ts] < ew_ma200[dt_ts] else 1.0
    return signals

def get_month_end_dates(cal, start, end):
    """获取月末交易日列表"""
    dates = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not dates:
        return []
    monthly = pd.DatetimeIndex(dates).to_period("M")
    month_ends = []
    seen = set()
    for i, d in enumerate(dates):
        m = monthly[i]
        if m not in seen:
            seen.add(m)
        if i + 1 < len(dates) and monthly[i + 1] != m:
            month_ends.append(d)
        elif i + 1 == len(dates):
            month_ends.append(d)
    return month_ends

def vectorized_backtest(pred, universe, cal, prices_dict, month_ends,
                        topk=10, macro_signals=None, fund_filter_func=None,
                        alpha=0.0, vf=None):
    """
    向量化月频TopK回测
    Returns: (daily_returns Series, portfolio_nav Series)
    """
    if pred is None or len(pred) == 0:
        return pd.Series(), pd.Series()

    if alpha > 0 and vf is not None:
        pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    # 涨跌停、停牌处理
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

    # ================= 修复：重新加入2000万/日流动性限制 =================
    vol_data = D.features(universe, ["$amount"], start_time=pred.index.get_level_values(0).min(), end_time=pred.index.get_level_values(0).max())
    if vol_data is not None and len(vol_data) > 0:
        # 将成交金额小于 20,000,000 的预测分强行置为极小值
        vol_data = vol_data.reindex(pred.index)
        pred.loc[vol_data["$amount"] < 20_000_000] = -np.inf
    # ====================================================================

    if fund_filter_func is not None:
        pred = fund_filter_func(pred)

    all_dates = sorted(pred.index.get_level_values(0).unique())
    start_dt = all_dates[0]
    end_dt = all_dates[-1]

    all_instruments = list(set(pred.index.get_level_values(1)))
    price_data = D.features(all_instruments, ["$close"],
                           start_time=start_dt - pd.Timedelta(days=10),
                           end_time=end_dt + pd.Timedelta(days=40))
    if price_data is None or len(price_data) == 0:
        return pd.Series(), pd.Series()
    price_data = price_data.reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {}
    for inst, grp in price_data.groupby("instrument"):
        grp = grp.sort_values("datetime").set_index("datetime")
        price_dict[inst] = grp["close"]

    portfolio_returns = []
    portfolio_dates = []

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]
            if not earlier:
                continue
            dt = earlier[-1]

        day_pred = pred.xs(dt, level=0)
        topk_stocks = day_pred["score"].nlargest(topk).index.tolist()

        position = 1.0
        if macro_signals and dt in macro_signals:
            position = macro_signals[dt]

        if i + 1 < len(month_ends):
            next_dt = month_ends[i + 1]
        else:
            next_dt = end_dt

        period_dates = [d for d in all_dates if dt < d <= next_dt]
        if not period_dates:
            continue

        daily_rets = []
        prev_prices = {}
        for inst in topk_stocks:
            if inst in price_dict and dt in price_dict[inst].index:
                prev_prices[inst] = price_dict[inst][dt]

        for pd_dt in period_dates:
            day_ret = 0
            n_valid = 0
            for inst in topk_stocks:
                if inst in price_dict and pd_dt in price_dict[inst].index and inst in prev_prices:
                    cur_price = price_dict[inst][pd_dt]
                    if pd.notna(cur_price) and prev_prices[inst] > 0:
                        ret = cur_price / prev_prices[inst] - 1
                        day_ret += ret
                        prev_prices[inst] = cur_price
                        n_valid += 1
            if n_valid > 0:
                daily_rets.append(day_ret / topk * position)
                portfolio_dates.append(pd_dt)
            else:
                daily_rets.append(0)
                portfolio_dates.append(pd_dt)

        portfolio_returns.extend(daily_rets)

    returns = pd.Series(portfolio_returns, index=pd.DatetimeIndex(portfolio_dates))
    returns = returns[~returns.index.duplicated(keep="last")]
    return returns, (1 + returns).cumprod()

def calc_metrics(returns, benchmark=None):
    if len(returns) == 0:
        return {"ar": 0, "sharpe": 0, "max_dd": 0, "ir": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    ir = 0
    if benchmark is not None and len(benchmark) > 0:
        common = returns.index.intersection(benchmark.index)
        if len(common) > 10:
            excess = returns.loc[common] - benchmark.loc[common]
            ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd, "ir": ir,
            "vol": vol, "n_days": len(returns)}

def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    print(f"\n{'='*70}")
    print(f"  Phase 1: 训练7个窗口 XGBoost (W0-W6, 排除2020)")
    print(f"{'='*70}")

    all_preds = {}
    all_universes = set()
    for win in WINDOWS:
        print(f"\n--- {win['name']} (回测{win['year']}) ---")
        result = train_window(win, fcf_df, profit_df, cal, cal_set)
        all_preds[win["name"]] = result
        all_universes.update(result[1])

    print(f"\n{'='*70}")
    print(f"  Phase 2: 计算全A等权200日均线信号")
    print(f"{'='*70}")
    macro_signals = compute_macro_signals_all(sorted(all_universes), cal)

    print(f"\n{'='*70}")
    print(f"  Phase 3: 三组实验回测 (XGBoost版)")
    print(f"{'='*70}")

    results = {"baseline": {}, "alpha03": {}, "e8": {}}
    all_returns = {"baseline": [], "alpha03": [], "e8": []}

    for win in WINDOWS:
        name = win["name"]
        pred, universe, vf, fund = all_preds[name]
        bs, be = win["backtest"]
        month_ends = get_month_end_dates(cal, bs, be)

        print(f"\n  {name}: {bs}~{be}, 月末调仓日{len(month_ends)}个")

        vs_w, ve_w = win["valid"]
        best_alpha, best_ir, _ = search_alpha_on_valid(pred, vf, vs_w, ve_w)
        print(f"    valid集α搜索: 最优α={best_alpha} (IR={best_ir:.4f})")

        def make_fund_filter(signal_date=None):
            return lambda p: filter_by_fundamental_deterioration(
                p, fcf_df, profit_df, threshold=0.30, signal_date=signal_date)

        ret1, nav1 = vectorized_backtest(
            pred, universe, cal, {}, month_ends, topk=10,
            macro_signals=None, fund_filter_func=None, alpha=0.0, vf=None)
        m1 = calc_metrics(ret1)
        results["baseline"][name] = m1
        all_returns["baseline"].append(ret1)
        print(f"    基线: 年化{m1['ar']*100:.2f}%, 夏普{m1['sharpe']:.2f}, 回撤{m1['max_dd']*100:.1f}%")

        ret2, nav2 = vectorized_backtest(
            pred, universe, cal, {}, month_ends, topk=10,
            macro_signals=None, fund_filter_func=None, alpha=best_alpha, vf=vf)
        m2 = calc_metrics(ret2)
        results["alpha03"][name] = m2
        all_returns["alpha03"].append(ret2)
        print(f"    α={best_alpha}: 年化{m2['ar']*100:.2f}%, 夏普{m2['sharpe']:.2f}, 回撤{m2['max_dd']*100:.1f}%")

        ret3, nav3 = vectorized_backtest(
            pred, universe, cal, {}, month_ends, topk=10,
            macro_signals=macro_signals, fund_filter_func=make_fund_filter(),
            alpha=best_alpha, vf=vf)
        m3 = calc_metrics(ret3)
        results["e8"][name] = m3
        all_returns["e8"].append(ret3)
        print(f"    E8:   年化{m3['ar']*100:.2f}%, 夏普{m3['sharpe']:.2f}, 回撤{m3['max_dd']*100:.1f}%")

    print(f"\n{'='*70}")
    print(f"  Phase 4: 汇总结果")
    print(f"{'='*70}")

    for exp_name in ["baseline", "alpha03", "e8"]:
        all_returns[exp_name] = pd.concat(all_returns[exp_name]).sort_index()
        all_returns[exp_name] = all_returns[exp_name][~all_returns[exp_name].index.duplicated(keep="last")]

    print(f"\n  {'窗口':<6} {'基线年化':>10} {'α融合年化':>10} {'E8年化':>10} {'基线夏普':>10} {'α融合夏普':>10} {'E8夏普':>10}")
    print(f"  {'-'*66}")
    for win in WINDOWS:
        name = win["name"]
        b = results["baseline"][name]
        a = results["alpha03"][name]
        e = results["e8"][name]
        print(f"  {name:<6} {b['ar']*100:>9.2f}% {a['ar']*100:>9.2f}% {e['ar']*100:>9.2f}% {b['sharpe']:>10.2f} {a['sharpe']:>10.2f} {e['sharpe']:>10.2f}")

    print(f"\n  全期汇总 (排除2020):")
    print(f"  {'实验':<16} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'波动率':>10} {'交易日':>8}")
    print(f"  {'-'*64}")
    for exp_name, label in [("baseline", "新V5基线(XGB)"), ("alpha03", "α融合(valid)"), ("e8", "E8/方案D")]:
        m = calc_metrics(all_returns[exp_name])
        print(f"  {label:<16} {m['ar']*100:>9.2f}% {m['sharpe']:>8.2f} {m['max_dd']*100:>9.1f}% {m['vol']*100:>9.2f}% {m['n_days']:>8}")

    summary_data = []
    for win in WINDOWS:
        name = win["name"]
        for exp, label in [("baseline","基线"), ("alpha03","α融合"), ("e8","E8")]:
            m = results[exp][name]
            summary_data.append({"window": name, "experiment": label,
                "ar": m["ar"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]})
    pd.DataFrame(summary_data).to_csv(
        "/Users/11164591/Documents/Qoder目录/qlib/v5_xgb_baseline_results.csv",
        sep='\t', index=False)
    print(f"\n  已保存: v5_xgb_baseline_results.csv")

    for exp, label in [("baseline","baseline"), ("alpha03","alpha03"), ("e8","e8")]:
        nav = (1 + all_returns[exp]).cumprod()
        nav.to_csv(f"/Users/11164591/Documents/Qoder目录/qlib/v5_xgb_nav_{label}.csv",
                   sep='\t', header=False)
    print(f"  已保存: v5_xgb_nav_*.csv (3条净值曲线)")

if __name__ == "__main__":
    run()
