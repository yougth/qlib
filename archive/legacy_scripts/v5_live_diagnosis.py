"""
V5实盘落地诊断 + 防御性降级方案
==================================
1. W6持仓深挖: 导出2026H1 Top3亏损持仓 + 基本面/微观结构检查
2. Feature Importance: 拉出W6模型特征重要性, 排查高频因子带偏
3. 宏观止损阀门: 全A等权多头+波动率放大时砍半仓
4. Paper Trading: 生成次日Top10调仓列表
"""
import os, sys, json
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from scipy import stats
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.backtest import backtest as qlib_backtest
import warnings
warnings.filterwarnings("ignore")

from v5_validation import Alpha158Enhanced, MonthlyTopkStrategy, \
    apply_value_fusion, inject_features, build_limit_up_set, \
    filter_pred_by_tradability, load_value_factors, load_fundamental_features

MODEL_CONFIG = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
        "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
        "max_depth": 8, "num_leaves": 210, "num_threads": 4}}


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


# ==================== Part 1: W6 持仓深挖 ====================
def w6_deep_dive(universe, vf, fund, cal, cal_set, fcf_df, profit_df):
    """重跑W6训练+回测, 导出持仓明细和亏损归因"""
    print(f"\n{'='*70}")
    print(f"  Part 1: W6 持仓深挖 — 2026H1亏损归因")
    print(f"{'='*70}")

    ts, te = "2021-01-01", "2024-12-31"
    vs, ve = "2025-01-01", "2025-12-31"
    bs, be = "2026-01-01", "2026-07-21"

    # Dataset
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
    if fund is not None: inject_features(dataset, fund)
    if vf is not None: inject_features(dataset, vf)

    # 训练
    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_w6_diagnosis"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    # ====== Feature Importance ======
    print(f"\n  --- Feature Importance (W6 LightGBM) ---")
    booster = model.model  # lightgbm.Booster
    importance_gain = booster.feature_importance(importance_type='gain')
    importance_split = booster.feature_importance(importance_type='split')
    feature_names = booster.feature_name()
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "gain": importance_gain,
        "split": importance_split
    }).sort_values("gain", ascending=False)

    print(f"  Top 20 特征 (按gain排序):")
    print(f"  {'排名':<6} {'特征名':<20} {'Gain':>12} {'Split':>8}")
    print(f"  {'-'*50}")
    for i, (_, r) in enumerate(fi_df.head(20).iterrows()):
        print(f"  {i+1:<6} {r['feature']:<20} {r['gain']:>12.1f} {r['split']:>8}")
    fi_df.to_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_feature_importance.csv", index=False)

    # 按类别分组统计
    alpha158_cols = ["ROC", "MA", "STD", "BOLL", "RSI", "CORR", "CORD", "CNTN", "CNTP",
                     "IMBA", "SUM", "ABS", "VSTD", "WMA", "KDJ", "MACD", "CCI", "VRATIO",
                     "VMA", "CORR_PV", "PEAK", "TREND"]
    fund_cols = ["fcf", "net_profit", "profit_growth", "fcf_growth", "fcf_profit_ratio",
                 "fcf_avg_3y", "fcf_cv_3y"]
    value_cols = ["roe_annual", "pb_pct_3y", "div_yield_est", "pe_pct_3y"]

    categories = {"Alpha158技术(短周期)": [], "Alpha158增强(长周期)": [],
                  "基本面特征": [], "价值因子": [], "其他": []}
    for _, r in fi_df.iterrows():
        fn = r["feature"]
        if any(fn.startswith(c) for c in fund_cols) or fn in fund_cols:
            categories["基本面特征"].append(r)
        elif any(fn.startswith(c) for c in value_cols) or fn in value_cols:
            categories["价值因子"].append(r)
        elif fn in ["ROC120", "ROC240", "MA120", "MA240", "STD120", "STD240",
                     "BOLL60", "BOLL120", "VMA120", "VMA240", "CORR_PV20",
                     "CORR_PV60", "VRATIO_5_60", "VRATIO_5_120"]:
            categories["Alpha158增强(长周期)"].append(r)
        elif any(fn.startswith(c) for c in alpha158_cols):
            categories["Alpha158技术(短周期)"].append(r)
        else:
            categories["其他"].append(r)

    print(f"\n  按类别统计Gain占比:")
    total_gain = fi_df["gain"].sum()
    for cat, items in categories.items():
        if items:
            cat_gain = sum(r["gain"] for r in items)
            print(f"    {cat:<24} {cat_gain/total_gain*100:>6.1f}% ({len(items)}个特征)")

    # ====== 价值融合 + 回测 ======
    pred_fused = apply_value_fusion(pred.copy(), vf, alpha=0.3)
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred_fused = filter_pred_by_tradability(pred_fused, limit_up_set, suspension_set)

    # 回测获取日频收益
    executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
            "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    strategy_config = {"class":"MonthlyTopkStrategy","module_path":"v5_validation",
            "kwargs":{"topk":10,"n_drop":10,"signal":pred_fused}}
    pm, _ = qlib_backtest(start_time=bs, end_time=be, strategy=strategy_config,
        executor=executor_config, account=100000000, benchmark=None,
        exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
            "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
    report, _ = pm.get("1day", (None, None))
    daily_ret = report["return"].copy()
    daily_ret.index = pd.to_datetime(daily_ret.index)

    # ====== 持仓分析: 从predictions推导每月Top10 ======
    print(f"\n  --- 2026H1 每月持仓 + 亏损归因 ---")

    # 找到每月最后一个交易日 (调仓日)
    bt_dates = sorted(set(pred_fused.index.get_level_values(0)))
    monthly_rebalance = {}
    seen_months = set()
    for d in bt_dates:
        m = d.to_period("M")
        if m not in seen_months:
            seen_months.add(m)
            monthly_rebalance[m] = d  # 该月第一个交易日 (实际策略在月末调仓)
        else:
            monthly_rebalance[m] = max(monthly_rebalance[m], d)  # 更新为该月最后可用日

    # 对每个月, 取预测分数的Top10
    all_holdings = []
    for i, (period, rebalance_date) in enumerate(sorted(monthly_rebalance.items())):
        if rebalance_date < pd.Timestamp("2026-01-01") or rebalance_date > pd.Timestamp("2026-07-21"):
            continue
        if rebalance_date not in pred_fused.index.get_level_values(0):
            continue
        day_pred = pred_fused.loc[[rebalance_date]].copy()
        day_pred.index = day_pred.index.get_level_values(1)
        top10 = day_pred["score"].nlargest(10)

        # 计算持有期收益
        if i + 1 < len(monthly_rebalance):
            next_period = sorted(monthly_rebalance.keys())[i+1]
            next_date = monthly_rebalance[next_period]
        else:
            next_date = pd.Timestamp("2026-07-21")

        # 获取每只股票在持有期的收益
        for inst in top10.index:
            prices = D.features([inst], ["$close", "$volume"], start_time=rebalance_date, end_time=next_date)
            if prices is None or len(prices) == 0: continue
            prices = prices.reset_index()
            prices.columns = ["instrument", "datetime", "close", "volume"]
            if len(prices) < 2: continue
            ret = (prices.iloc[-1]["close"] / prices.iloc[0]["close"] - 1) * 100
            # 检查是否涨跌停或停牌
            prices["prev_close"] = prices["close"].shift(1)
            prices["daily_ret"] = (prices["close"] - prices["prev_close"]) / prices["prev_close"]
            n_limit_up = ((prices["daily_ret"] > 0.095) & (prices["close"] == prices["close"].shift(1) * 1.1)).sum()
            n_volume_zero = (prices["volume"] == 0).sum()
            max_vol = prices["volume"].max()
            avg_vol = prices["volume"].mean()
            # 微观结构: 检查是否有异常放量
            vol_ratio = prices["volume"].iloc[-1] / (avg_vol + 1e-8) if avg_vol > 0 else 1

            all_holdings.append({
                "month": str(period),
                "rebalance_date": rebalance_date.date(),
                "instrument": inst,
                "score": top10[inst],
                "hold_return_pct": ret,
                "n_limit_up_days": n_limit_up,
                "n_suspension_days": n_volume_zero,
                "max_vol_ratio": vol_ratio,
            })

    holdings_df = pd.DataFrame(all_holdings)
    holdings_df.to_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_holdings_2026h1.csv", index=False)

    # Top 3 亏损持仓
    top_losers = holdings_df.nsmallest(5, "hold_return_pct")
    print(f"\n  2026H1 Top 5 亏损持仓:")
    print(f"  {'月份':<8} {'股票':<10} {'持有收益':>10} {'分数':>8} {'涨停天':>6} {'停牌天':>6} {'量比':>6}")
    print(f"  {'-'*56}")
    for _, r in top_losers.iterrows():
        print(f"  {r['month']:<8} {r['instrument']:<10} {r['hold_return_pct']:>+9.2f}% {r['score']:>8.3f} "
              f"{r['n_limit_up_days']:>6} {r['n_suspension_days']:>6} {r['max_vol_ratio']:>6.1f}")

    # 基本面检查: 检查亏损股票的FCF/净利润是否恶化
    print(f"\n  亏损持仓基本面检查:")
    for _, r in top_losers.head(3).iterrows():
        code = r["instrument"][2:]
        stock_fcf = fcf_df[fcf_df["code"] == code].sort_values("year")
        stock_profit = profit_df[profit_df["code"] == code].sort_values("year")
        print(f"\n  {r['instrument']} (持有收益: {r['hold_return_pct']:.2f}%):")
        if len(stock_fcf) > 0:
            recent_fcf = stock_fcf.tail(5)
            fcf_trend = "恶化" if recent_fcf["fcf"].iloc[0] > recent_fcf["fcf"].iloc[-1] else "稳定/改善"
            print(f"    FCF趋势: {fcf_trend}")
            for _, fr in recent_fcf.iterrows():
                print(f"      {int(fr['year'])}: FCF={fr['fcf']/1e8:.2f}亿")
        if len(stock_profit) > 0:
            recent_profit = stock_profit.tail(5)
            profit_trend = "恶化" if recent_profit["net_profit"].iloc[0] > recent_profit["net_profit"].iloc[-1] else "稳定/改善"
            print(f"    净利润趋势: {profit_trend}")
            for _, pr in recent_profit.iterrows():
                print(f"      {int(pr['year'])}: 净利润={pr['net_profit']/1e8:.2f}亿")

        # 微观结构: 获取2026H1的量价数据
        pv = D.features([r["instrument"]], ["$close", "$volume"], start_time="2026-01-01", end_time="2026-07-21")
        if pv is not None and len(pv) > 0:
            pv = pv.reset_index()
            pv.columns = ["instrument", "datetime", "close", "volume"]
            pv["ret"] = pv["close"].pct_change() * 100
            pv["vol_ma20"] = pv["volume"].rolling(20).mean()
            pv["vol_ratio"] = pv["volume"] / (pv["vol_ma20"] + 1e-8)
            # 找到跌幅最大的几天
            big_drops = pv.nsmallest(3, "ret")
            print(f"    2026H1最大跌幅日:")
            for _, bd in big_drops.iterrows():
                vr = bd["vol_ratio"] if pd.notna(bd["vol_ratio"]) else 0
                print(f"      {bd['datetime'].date()}: 跌{bd['ret']:.2f}%, 量比{vr:.1f}x")

    return model, dataset, pred_fused, fi_df, daily_ret


# ==================== Part 3: 宏观止损阀门 ====================
def macro_valve_backtest(universe, daily_ret, pred_fused, cal):
    """全A等权多头+波动率放大时砍半仓"""
    print(f"\n{'='*70}")
    print(f"  Part 3: 宏观止损阀门 — 牛市/震荡市砍半仓")
    print(f"{'='*70}")

    # 计算全A等权指数
    bs, be = "2021-01-01", "2026-07-21"
    prices = D.features(universe, ["$close"], start_time=bs, end_time=be)
    if prices is None or len(prices) == 0:
        print("  无法获取价格数据")
        return daily_ret
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "close"]
    prices = prices.sort_values(["instrument", "datetime"])
    prices["ret"] = prices.groupby("instrument")["close"].pct_change()
    ew_ret = prices.groupby("datetime")["ret"].mean().dropna()

    # 200日均线
    ew_cum = (1 + ew_ret).cumprod()
    ew_ma200 = ew_cum.rolling(200, min_periods=60).mean()

    # 60日实现波动率
    ew_vol60 = ew_ret.rolling(60, min_periods=20).std() * np.sqrt(252) * 100

    # 波动率中位数
    vol_median = ew_vol60.median()

    # 阀门信号: 1=全仓, 0.5=半仓
    # 条件: EW > 200日均线(多头排列) 且 波动率 > 中位数 → 砍半
    regime_signal = pd.Series(1.0, index=ew_ret.index)
    bull_high_vol = (ew_cum > ew_ma200) & (ew_vol60 > vol_median)
    regime_signal[bull_high_vol] = 0.5

    # 统计
    n_full = (regime_signal == 1.0).sum()
    n_half = (regime_signal == 0.5).sum()
    print(f"  全仓天数: {n_full} ({n_full/len(regime_signal)*100:.1f}%)")
    print(f"  半仓天数: {n_half} ({n_half/len(regime_signal)*100:.1f}%)")
    print(f"  波动率中位数: {vol_median:.2f}%")

    # 逐年统计regime
    print(f"\n  逐年regime分布:")
    print(f"  {'年份':<8} {'全仓天':>8} {'半仓天':>8} {'半仓占比':>10}")
    print(f"  {'-'*34}")
    for year in range(2021, 2027):
        yr_sig = regime_signal[regime_signal.index.year == year]
        if len(yr_sig) == 0: continue
        full = (yr_sig == 1.0).sum()
        half = (yr_sig == 0.5).sum()
        label = str(year) if year < 2026 else "2026H1"
        print(f"  {label:<8} {full:>8} {half:>8} {half/len(yr_sig)*100:>9.1f}%")

    # 应用阀门到策略收益
    common = daily_ret.index.intersection(regime_signal.index)
    adjusted_ret = daily_ret.loc[common] * regime_signal.loc[common]

    # 对比
    orig_ar = ((1 + daily_ret.loc[common]).prod() ** (252/len(common)) - 1) * 100
    adj_ar = ((1 + adjusted_ret).prod() ** (252/len(adjusted_ret)) - 1) * 100
    orig_sharpe = daily_ret.loc[common].mean() / daily_ret.loc[common].std() * np.sqrt(252)
    adj_sharpe = adjusted_ret.mean() / adjusted_ret.std() * np.sqrt(252)
    orig_mdd = ((1 + daily_ret.loc[common]).cumprod() / (1 + daily_ret.loc[common]).cumprod().expanding().max() - 1).min() * 100
    adj_mdd = ((1 + adjusted_ret).cumprod() / (1 + adjusted_ret).cumprod().expanding().max() - 1).min() * 100

    print(f"\n  宏观阀门效果对比:")
    print(f"  {'指标':<16} {'无阀门':>12} {'有阀门':>12} {'差异':>12}")
    print(f"  {'-'*52}")
    print(f"  {'年化收益':<16} {orig_ar:>11.2f}% {adj_ar:>11.2f}% {adj_ar-orig_ar:>+11.2f}%")
    print(f"  {'夏普':<16} {orig_sharpe:>12.4f} {adj_sharpe:>12.4f} {adj_sharpe-orig_sharpe:>+12.4f}")
    print(f"  {'最大回撤':<16} {orig_mdd:>11.2f}% {adj_mdd:>11.2f}% {adj_mdd-orig_mdd:>+11.2f}%")

    # 逐年对比
    print(f"\n  逐年对比:")
    print(f"  {'年份':<8} {'无阀门':>10} {'有阀门':>10} {'差异':>10}")
    print(f"  {'-'*38}")
    for year in range(2021, 2027):
        yr_orig = daily_ret[daily_ret.index.year == year]
        yr_adj = adjusted_ret[adjusted_ret.index.year == year]
        if len(yr_orig) == 0: continue
        o = ((1 + yr_orig).prod() ** (252/len(yr_orig)) - 1) * 100 if len(yr_orig) > 0 else 0
        a = ((1 + yr_adj).prod() ** (252/len(yr_adj)) - 1) * 100 if len(yr_adj) > 0 else 0
        label = str(year) if year < 2026 else "2026H1"
        print(f"  {label:<8} {o:>+9.2f}% {a:>+9.2f}% {a-o:>+9.2f}%")

    # 保存信号
    regime_df = pd.DataFrame({"datetime": regime_signal.index, "signal": regime_signal.values})
    regime_df.to_csv("/Users/11164591/Documents/Qoder目录/qlib/macro_valve_signal.csv", index=False)

    return adjusted_ret


# ==================== Part 4: Paper Trading ====================
def paper_trading(universe, model, dataset, vf, cal, cal_set):
    """生成次日Top10调仓列表"""
    print(f"\n{'='*70}")
    print(f"  Part 4: Paper Trading — 次日Top10调仓列表")
    print(f"{'='*70}")

    # 获取最新交易日
    last_date = cal[-1]
    print(f"  最新数据日期: {last_date.date()}")

    # 生成预测
    # 使用dataset的test segment的最后一天
    try:
        pred_latest = model.predict(dataset, segment="test")
    except:
        # 如果失败, 用已有pred的最后一天
        print("  [WARNING] predict失败, 使用最后可用预测")
        return

    if pred_latest is None or len(pred_latest) == 0:
        print("  无预测数据")
        return

    # 取最后一个交易日的预测
    latest_dates = sorted(pred_latest.index.get_level_values(0).unique())
    latest_dt = latest_dates[-1]
    day_pred = pred_latest.loc[[latest_dt]].copy()
    day_pred.index = day_pred.index.get_level_values(1)

    # 价值融合
    if vf is not None and latest_dt in vf.index.get_level_values(0):
        vf_day = vf.loc[[latest_dt]]
        vf_day.index = vf_day.index.get_level_values(1)
        common = day_pred.index.intersection(vf_day.index)
        if len(common) >= 5:
            roe_vals = vf_day.loc[common, "roe_annual"]
            pb_val = vf_day.loc[common, "pb_pct_3y"]
            div_val = vf_day.loc[common, "div_yield_est"]
            def rn(s): return s.rank(pct=True).fillna(0.5)
            value_score = rn(roe_vals) + rn(-pb_val) + rn(div_val)
            value_score = (value_score - value_score.mean()) / (value_score.std() + 1e-8)
            lgb_score = day_pred.loc[common, "score"]
            lgb_norm = (lgb_score - lgb_score.mean()) / (lgb_score.std() + 1e-8)
            fused = (1 - 0.3) * lgb_norm + 0.3 * value_score
            for inst in common:
                if inst in fused.index:
                    day_pred.loc[inst, "score"] = fused[inst]

    # 涨跌停/停牌过滤
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    for inst in day_pred.index:
        if (latest_dt, inst) in limit_up_set:
            day_pred.loc[inst, "score"] = -999
        if (latest_dt, inst) in suspension_set:
            day_pred.loc[inst, "score"] = -999

    # Top 10
    top10 = day_pred["score"].nlargest(10)

    # 获取股票名称
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    name_map = {}
    if "name" in fcf_df.columns:
        name_map = dict(zip(fcf_df["code"].astype(str).str.zfill(6), fcf_df["name"]))

    print(f"\n  Paper Trading Top 10 (信号日: {latest_dt.date()}):")
    print(f"  {'排名':<6} {'代码':<10} {'名称':<12} {'融合分':>10}")
    print(f"  {'-'*40}")
    for i, (inst, score) in enumerate(top10.items()):
        code = inst[2:]
        name = name_map.get(code, "—")
        print(f"  {i+1:<6} {inst:<10} {name:<12} {score:>10.4f}")

    # 获取这些股票的最近价格
    top10_codes = top10.index.tolist()
    prices = D.features(top10_codes, ["$close", "$volume"], start_time=latest_dt - pd.Timedelta(days=10), end_time=latest_dt)
    if prices is not None and len(prices) > 0:
        prices = prices.reset_index()
        prices.columns = ["instrument", "datetime", "close", "volume"]
        print(f"\n  最近价格:")
        print(f"  {'代码':<10} {'最新收盘':>10} {'5日均量':>12}")
        print(f"  {'-'*32}")
        for inst in top10_codes:
            sp = prices[prices["instrument"] == inst].sort_values("datetime")
            if len(sp) > 0:
                last_close = sp.iloc[-1]["close"]
                vol_5 = sp["volume"].tail(5).mean() if len(sp) >= 5 else sp["volume"].mean()
                print(f"  {inst:<10} {last_close:>10.2f} {vol_5:>12.0f}")

    # 保存
    output = pd.DataFrame({
        "rank": range(1, len(top10) + 1),
        "instrument": top10.index,
        "name": [name_map.get(i[2:], "—") for i in top10.index],
        "fused_score": top10.values,
        "signal_date": latest_dt,
    })
    output_path = "/Users/11164591/Documents/Qoder目录/qlib/paper_trading_top10.csv"
    output.to_csv(output_path, index=False)
    print(f"\n  已保存: {output_path}")
    print(f"\n  ⚠ 注意: 这是Paper Trading信号, 不构成投资建议")
    print(f"  建议空跑1个月, 对比实际盘口滑点和因子有效性恢复情况")


# ==================== Main ====================
def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    # W6股票池
    codes = build_dynamic_universe(2026, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"W6股票池: {len(universe)}只")

    cal = D.calendar(start_time="2021-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    print("加载价值因子...")
    vf = load_value_factors(universe, cal, cal_set)
    print("加载基本面特征...")
    fund = load_fundamental_features(universe, cal)

    # Part 1: W6深挖
    model, dataset, pred_fused, fi_df, daily_ret = w6_deep_dive(
        universe, vf, fund, cal, cal_set, fcf_df, profit_df)

    # Part 3: 宏观阀门
    adjusted_ret = macro_valve_backtest(universe, daily_ret, pred_fused, cal)

    # Part 4: Paper Trading
    paper_trading(universe, model, dataset, vf, cal, cal_set)

    print(f"\n{'='*70}")
    print(f"  全部诊断完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    run()
