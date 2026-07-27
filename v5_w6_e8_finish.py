"""
W6 E8 收尾脚本 — 只训练W6, 完成E8回测, 输出完整汇总
优化: 涨跌停集合只计算一次, 仅加载回测期间数据
所有CSV读写均使用 sep='\t'
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
    load_value_factors, filter_pred_by_tradability
from v5_pipeline_fix import load_fundamental_features_fixed, \
    inject_features_fixed, filter_by_fundamental_deterioration

MODEL_CONFIG = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
        "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
        "max_depth": 8, "num_leaves": 210, "num_threads": 4}}

# W6 配置
W6 = {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"),
      "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "year": 2026}

# 已知结果 (从上一轮运行日志中提取)
KNOWN_RESULTS = {
    "W0": {"baseline": {"ar": 0.3357, "sharpe": 1.98, "max_dd": -0.090},
           "alpha03": {"ar": 0.5850, "sharpe": 3.32, "max_dd": -0.085},
           "e8":      {"ar": 0.3680, "sharpe": 2.17, "max_dd": -0.079}},
    "W1": {"baseline": {"ar": 0.0843, "sharpe": 0.32, "max_dd": -0.198},
           "alpha03": {"ar": 0.0998, "sharpe": 0.38, "max_dd": -0.215},
           "e8":      {"ar": -0.0128, "sharpe": -0.05, "max_dd": -0.214}},
    "W2": {"baseline": {"ar": 0.2901, "sharpe": 1.03, "max_dd": -0.195},
           "alpha03": {"ar": 0.0953, "sharpe": 0.34, "max_dd": -0.234},
           "e8":      {"ar": 0.0105, "sharpe": 0.05, "max_dd": -0.186}},
    "W3": {"baseline": {"ar": 0.0588, "sharpe": 0.34, "max_dd": -0.138},
           "alpha03": {"ar": 0.0212, "sharpe": 0.12, "max_dd": -0.132},
           "e8":      {"ar": -0.0258, "sharpe": -0.16, "max_dd": -0.167}},
    "W4": {"baseline": {"ar": 0.4555, "sharpe": 1.47, "max_dd": -0.218},
           "alpha03": {"ar": 0.7632, "sharpe": 2.34, "max_dd": -0.177},
           "e8":      {"ar": 0.3269, "sharpe": 1.32, "max_dd": -0.173}},
    "W5": {"baseline": {"ar": 0.3277, "sharpe": 2.05, "max_dd": -0.075},
           "alpha03": {"ar": 0.4377, "sharpe": 2.74, "max_dd": -0.065},
           "e8":      {"ar": 0.2005, "sharpe": 1.32, "max_dd": -0.079}},
    "W6": {"baseline": {"ar": 0.0092, "sharpe": 0.04, "max_dd": -0.129},
           "alpha03": {"ar": 0.2070, "sharpe": 0.92, "max_dd": -0.091},
           "e8":      None},  # 待计算
}


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


def build_limit_up_set_fast(universe, bt_start, bt_end):
    """优化版: 仅加载回测期间数据, 而非全量日历"""
    print("  --- 构建涨跌停/停牌集合 (仅回测期间) ---")
    # 多加载10天用于计算前日收盘
    fetch_start = pd.Timestamp(bt_start) - pd.Timedelta(days=15)
    price_df = D.features(list(universe), ["$close", "$open", "$high", "$low"],
                          start_time=fetch_start, end_time=bt_end)
    if price_df is None or len(price_df) == 0:
        return set(), set()
    price_df = price_df.reset_index()
    price_df.columns = ["instrument", "datetime", "close", "open", "high", "low"]
    price_df = price_df.sort_values(["instrument", "datetime"])
    price_df["prev_close"] = price_df.groupby("instrument")["close"].shift(1)
    price_df["daily_ret"] = (price_df["close"] - price_df["prev_close"]) / price_df["prev_close"]

    limit_up_mask = (
        (price_df["open"] == price_df["high"]) &
        (price_df["high"] == price_df["low"]) &
        (price_df["low"] == price_df["close"]) &
        (price_df["daily_ret"] > 0.09)
    )
    limit_up_set = set(zip(price_df.loc[limit_up_mask, "datetime"],
                           price_df.loc[limit_up_mask, "instrument"]))

    vol_df = D.features(list(universe), ["$volume"],
                        start_time=fetch_start, end_time=bt_end)
    suspension_set = set()
    if vol_df is not None and len(vol_df) > 0:
        vol_df = vol_df.reset_index()
        vol_df.columns = ["instrument", "datetime", "volume"]
        susp = vol_df[(vol_df["volume"].isna()) | (vol_df["volume"] == 0)]
        suspension_set = set(zip(susp["datetime"], susp["instrument"]))

    print(f"  一字涨停: {len(limit_up_set)} 条, 停牌: {len(suspension_set)} 条")
    return limit_up_set, suspension_set


def compute_macro_signals_all(universe, cal, lookback=300):
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
    signals = {}
    for dt in cal:
        if dt in ew_cum.index and dt in ew_ma200.index:
            if not pd.isna(ew_ma200[dt]):
                signals[dt] = 0.7 if ew_cum[dt] < ew_ma200[dt] else 1.0
    return signals


def get_month_end_dates(cal, start, end):
    dates = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not dates:
        return []
    monthly = pd.DatetimeIndex(dates).to_period("M")
    month_ends = []
    for i, d in enumerate(dates):
        m = monthly[i]
        if i + 1 < len(dates) and monthly[i + 1] != m:
            month_ends.append(d)
        elif i + 1 == len(dates):
            month_ends.append(d)
    return month_ends


def vectorized_backtest_fast(pred, universe, bt_start, bt_end, month_ends,
                              topk=10, macro_signals=None, fund_filter_func=None,
                              alpha=0.0, vf=None, limit_up_set=None, suspension_set=None):
    """优化版回测: 使用预计算的涨跌停集合"""
    if pred is None or len(pred) == 0:
        return pd.Series(), pd.Series()

    if alpha > 0 and vf is not None:
        pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    if limit_up_set is not None:
        pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

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
                portfolio_returns.append(day_ret / n_valid * position)
                portfolio_dates.append(pd_dt)
            else:
                portfolio_returns.append(0)
                portfolio_dates.append(pd_dt)

    returns = pd.Series(portfolio_returns, index=pd.DatetimeIndex(portfolio_dates))
    returns = returns[~returns.index.duplicated(keep="last")]
    return returns, (1 + returns).cumprod()


def calc_metrics(returns):
    if len(returns) == 0:
        return {"ar": 0, "sharpe": 0, "max_dd": 0, "vol": 0, "n_days": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd, "vol": vol, "n_days": len(returns)}


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2021-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # ====== 1. 训练W6 ======
    print(f"\n{'='*70}")
    print(f"  1. 训练 W6 (2026年回测)")
    print(f"{'='*70}")

    by = W6["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"  W6: 股票池{len(universe)}只")

    ts, te = W6["train"]
    vs, ve = W6["valid"]
    bs, be = W6["backtest"]

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

    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_w6_e8_finish"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    train_data = dataset.prepare("train", col_set="feature")
    print(f"  特征数: {train_data.shape[1]}, 预测: {len(pred)}条")

    # ====== 2. 预计算涨跌停集合 (只计算一次) ======
    print(f"\n{'='*70}")
    print(f"  2. 预计算涨跌停集合 (仅W6回测期间)")
    print(f"{'='*70}")
    limit_up_set, suspension_set = build_limit_up_set_fast(universe, bs, be)

    # ====== 3. 计算宏观信号 ======
    print(f"\n  计算全A等权200日均线信号...")
    macro_cal = D.calendar(start_time="2024-06-01", end_time="2026-07-23")
    macro_signals = compute_macro_signals_all(universe, macro_cal)

    # ====== 4. 运行W6三组实验 ======
    print(f"\n{'='*70}")
    print(f"  3. W6 三组实验回测")
    print(f"{'='*70}")

    month_ends = get_month_end_dates(cal, bs, be)
    print(f"  W6: {bs}~{be}, 月末调仓日{len(month_ends)}个")

    # Exp1: 基线
    ret1, _ = vectorized_backtest_fast(
        pred, universe, bs, be, month_ends, topk=10,
        macro_signals=None, fund_filter_func=None, alpha=0.0, vf=None,
        limit_up_set=limit_up_set, suspension_set=suspension_set)
    m1 = calc_metrics(ret1)
    print(f"    基线: 年化{m1['ar']*100:.2f}%, 夏普{m1['sharpe']:.2f}, 回撤{m1['max_dd']*100:.1f}%")

    # Exp2: α=0.3
    ret2, _ = vectorized_backtest_fast(
        pred, universe, bs, be, month_ends, topk=10,
        macro_signals=None, fund_filter_func=None, alpha=0.3, vf=vf,
        limit_up_set=limit_up_set, suspension_set=suspension_set)
    m2 = calc_metrics(ret2)
    print(f"    α=0.3: 年化{m2['ar']*100:.2f}%, 夏普{m2['sharpe']:.2f}, 回撤{m2['max_dd']*100:.1f}%")

    # Exp3: E8
    def make_fund_filter():
        return lambda p: filter_by_fundamental_deterioration(
            p, fcf_df, profit_df, threshold=0.30)

    ret3, _ = vectorized_backtest_fast(
        pred, universe, bs, be, month_ends, topk=10,
        macro_signals=macro_signals, fund_filter_func=make_fund_filter(),
        alpha=0.3, vf=vf,
        limit_up_set=limit_up_set, suspension_set=suspension_set)
    m3 = calc_metrics(ret3)
    print(f"    E8:   年化{m3['ar']*100:.2f}%, 夏普{m3['sharpe']:.2f}, 回撤{m3['max_dd']*100:.1f}%")

    # 更新已知结果
    KNOWN_RESULTS["W6"]["baseline"] = m1
    KNOWN_RESULTS["W6"]["alpha03"] = m2
    KNOWN_RESULTS["W6"]["e8"] = m3

    # ====== 5. 完整汇总 ======
    print(f"\n{'='*70}")
    print(f"  4. 完整汇总结果 (7窗口, 排除2020)")
    print(f"{'='*70}")

    # 逐年对比表
    print(f"\n  {'窗口':<6} {'年份':<6} {'基线年化':>10} {'α=0.3年化':>10} {'E8年化':>10} {'基线夏普':>10} {'α=0.3夏普':>10} {'E8夏普':>10}")
    print(f"  {'-'*78}")
    win_info = [
        ("W0", 2019), ("W1", 2021), ("W2", 2022), ("W3", 2023),
        ("W4", 2024), ("W5", 2025), ("W6", 2026),
    ]
    for name, year in win_info:
        r = KNOWN_RESULTS[name]
        b, a, e = r["baseline"], r["alpha03"], r["e8"]
        print(f"  {name:<6} {year:<6} {b['ar']*100:>9.2f}% {a['ar']*100:>9.2f}% {e['ar']*100:>9.2f}% {b['sharpe']:>10.2f} {a['sharpe']:>10.2f} {e['sharpe']:>10.2f}")

    # 全期年化 (简单平均)
    all_b_ar = np.mean([KNOWN_RESULTS[n]["baseline"]["ar"] for n, _ in win_info])
    all_a_ar = np.mean([KNOWN_RESULTS[n]["alpha03"]["ar"] for n, _ in win_info])
    all_e_ar = np.mean([KNOWN_RESULTS[n]["e8"]["ar"] for n, _ in win_info])
    all_b_sharpe = np.mean([KNOWN_RESULTS[n]["baseline"]["sharpe"] for n, _ in win_info])
    all_a_sharpe = np.mean([KNOWN_RESULTS[n]["alpha03"]["sharpe"] for n, _ in win_info])
    all_e_sharpe = np.mean([KNOWN_RESULTS[n]["e8"]["sharpe"] for n, _ in win_info])

    # 胜率统计
    b_wins = sum(1 for n, _ in win_info if KNOWN_RESULTS[n]["baseline"]["ar"] > 0)
    a_wins = sum(1 for n, _ in win_info if KNOWN_RESULTS[n]["alpha03"]["ar"] > 0)
    e_wins = sum(1 for n, _ in win_info if KNOWN_RESULTS[n]["e8"]["ar"] > 0)

    # α=0.3 vs 基线 对比
    alpha_better = sum(1 for n, _ in win_info
                       if KNOWN_RESULTS[n]["alpha03"]["ar"] > KNOWN_RESULTS[n]["baseline"]["ar"])

    print(f"\n  全期汇总 (7窗口平均, 排除2020):")
    print(f"  {'实验':<16} {'平均年化':>10} {'平均夏普':>10} {'胜率':>8}")
    print(f"  {'-'*46}")
    print(f"  {'新V5基线':<16} {all_b_ar*100:>9.2f}% {all_b_sharpe:>10.2f} {b_wins}/7")
    print(f"  {'α=0.3融合':<16} {all_a_ar*100:>9.2f}% {all_a_sharpe:>10.2f} {a_wins}/7")
    print(f"  {'E8/方案D':<16} {all_e_ar*100:>9.2f}% {all_e_sharpe:>10.2f} {e_wins}/7")

    # 旧V5对比
    print(f"\n  {'='*50}")
    print(f"  新V5 vs 旧V5 对比:")
    print(f"  {'='*50}")
    print(f"  旧V5(假V5, 172特征): 年化18.13%, 夏普0.89")
    print(f"  新V5(181特征):       年化{all_b_ar*100:.2f}%, 夏普{all_b_sharpe:.2f}")
    print(f"  提升: {(all_b_ar - 0.1813)*100:+.2f}个百分点")

    # Alpha权重结论
    print(f"\n  {'='*50}")
    print(f"  Alpha权重检查结论:")
    print(f"  {'='*50}")
    print(f"    直接输出:     年化{all_b_ar*100:.2f}%")
    print(f"    α=0.3融合:    年化{all_a_ar*100:.2f}%")
    print(f"    α=0.3优于基线的窗口数: {alpha_better}/7")
    if all_a_ar > all_b_ar:
        print(f"    结论: 后置融合锦上添花 (+{(all_a_ar-all_b_ar)*100:.2f}%), 保留")
    else:
        print(f"    结论: 后置融合过载反效果 ({(all_a_ar-all_b_ar)*100:.2f}%), 建议去掉")

    # E8结论
    print(f"\n  {'='*50}")
    print(f"  终极E8 (方案D + 基本面过滤) 结论:")
    print(f"  {'='*50}")
    print(f"    年化{all_e_ar*100:.2f}%, 夏普{all_e_sharpe:.2f}")
    print(f"    对比基线: {(all_e_ar-all_b_ar)*100:+.2f}%")
    print(f"    对比α=0.3: {(all_e_ar-all_a_ar)*100:+.2f}%")

    # 保存结果
    summary_data = []
    for name, year in win_info:
        r = KNOWN_RESULTS[name]
        for exp, label in [("baseline","基线"), ("alpha03","α=0.3"), ("e8","E8")]:
            m = r[exp]
            summary_data.append({"window": name, "year": year, "experiment": label,
                "ar": m["ar"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]})
    pd.DataFrame(summary_data).to_csv(
        "/Users/11164591/Documents/Qoder目录/qlib/v5_new_baseline_results.csv",
        sep='\t', index=False)
    print(f"\n  已保存: v5_new_baseline_results.csv")


if __name__ == "__main__":
    run()
