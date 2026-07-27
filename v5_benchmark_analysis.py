"""
V5策略 vs 基准对比分析
=======================
1. 计算股票池等权基准 (270+只股票日频等权收益)
2. 使用已知沪深300年度收益率
3. 重跑V5回测保存日频收益
4. 计算超额收益、跟踪误差、信息比率、Alpha/Beta
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from scipy import stats
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
import warnings
warnings.filterwarnings("ignore")


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


# ==================== 沪深300年度数据 (公开数据) ====================
CSI300_ANNUAL = {
    2021: -5.20,
    2022: -21.63,
    2023: -11.38,
    2024: 14.68,
    2025: 18.20,   # ~4651/3935-1 (Yahoo Finance 2025.12.25收盘4651)
    2026: 1.66,    # ~4728/4651-1 (2026.07.23收盘4728, 半年)
}

# V5已知年度收益 (from v5_validation_results.csv, dynamic_alpha列)
V5_ANNUAL = {
    2021: 36.45,
    2022: 32.33,
    2023: 5.76,
    2024: 20.90,
    2025: 25.60,
    2026: -12.28,  # 2026H1
}


# ==================== Alpha158Enhanced (复用) ====================
class Alpha158Enhanced:
    pass  # 占位, 实际从v5_validation导入


def compute_equal_weight_benchmark(universe, start, end):
    """计算股票池等权日频收益率"""
    print("  计算股票池等权基准...")
    prices = D.features(list(universe), ["$close"], start_time=start, end_time=end)
    if prices is None or len(prices) == 0:
        return None
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "close"]
    prices = prices.sort_values(["instrument", "datetime"])

    # 计算每只股票日收益率
    prices["ret"] = prices.groupby("instrument")["close"].pct_change()

    # 等权平均 (截面均值)
    daily_ret = prices.groupby("datetime")["ret"].mean()
    daily_ret = daily_ret.dropna()

    # 累计收益
    cum = (1 + daily_ret).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_days = len(daily_ret)
    ar = ((1 + total_ret) ** (252 / n_days) - 1) * 100
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    cum_max = cum.expanding().max()
    dd = ((cum - cum_max) / cum_max).min() * 100

    print(f"  等权基准: 年化 {ar:.2f}%, 夏普 {sharpe:.4f}, 回撤 {dd:.2f}%")
    print(f"  天数: {n_days}, 总收益: {total_ret*100:.2f}%")

    # 逐年
    yearly = {}
    for year in range(2021, 2027):
        if year == 2026:
            yr = daily_ret[daily_ret.index >= pd.Timestamp("2026-01-01")]
            label = "2026H1"
        else:
            yr = daily_ret[(daily_ret.index >= pd.Timestamp(f"{year}-01-01")) &
                           (daily_ret.index < pd.Timestamp(f"{year+1}-01-01"))]
            label = str(year)
        if len(yr) == 0:
            continue
        yr_ret = (1 + yr).prod() - 1
        yr_ar = ((1 + yr_ret) ** (252 / len(yr)) - 1) * 100 if yr_ret > -1 else -100
        yearly[label] = yr_ar
        print(f"    {label}: {yr_ar:+.2f}%")

    return daily_ret, ar, sharpe, dd, yearly


def run_v5_with_daily_returns(universe, fund, vf, cal, cal_set):
    """重跑V5回测, 保存日频收益"""
    from v5_validation import (
        Alpha158Enhanced, load_fundamental_features, load_value_factors,
        apply_value_fusion, search_alpha_on_valid, build_limit_up_set,
        filter_pred_by_tradability, inject_features, MonthlyTopkStrategy
    )

    MODEL_CONFIG = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
                   "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
                   "max_depth": 8, "num_leaves": 210, "num_threads": 4},
    }

    WINDOWS_5Y = [
        {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1"},
        {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2"},
        {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3"},
        {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4"},
        {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5"},
        {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6"},
    ]

    # 涨跌停/停牌
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)

    all_daily_returns = []

    for i, w in enumerate(WINDOWS_5Y):
        ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
        print(f"\n  {w['name']}: train {ts}~{te}, backtest {bs}~{be}")

        # 构建dataset
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
        with R.start(experiment_name=f"v5_benchmark_{w['name']}"):
            rec = R.get_recorder()
            model.fit(dataset)
            from qlib.workflow.record_temp import SignalRecord
            sig_rec = SignalRecord(model, dataset, rec)
            sig_rec.generate()
            pred = rec.load_object("pred.pkl")

        # 动态α搜索
        best_alpha, best_ir, _ = search_alpha_on_valid(pred, vf, vs, ve)
        print(f"    最佳α: {best_alpha}")

        # 价值融合
        pred_dyn = apply_value_fusion(pred.copy(), vf, alpha=best_alpha)
        # 涨跌停过滤
        pred_dyn = filter_pred_by_tradability(pred_dyn, limit_up_set, suspension_set)

        # 回测 - 保存日频收益
        executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
                "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
        strategy_config = {"class":"MonthlyTopkStrategy","module_path":"v5_validation",
                "kwargs":{"topk":10,"n_drop":10,"signal":pred_dyn}}
        portfolio_metric_dict, _ = qlib_backtest(
            start_time=bs, end_time=be, strategy=strategy_config, executor=executor_config,
            account=100000000, benchmark=None,
            exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
                "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
        report_normal, _ = portfolio_metric_dict.get("1day", (None, None))
        if report_normal is not None:
            daily_ret = report_normal["return"].copy()
            daily_ret.index = pd.to_datetime(daily_ret.index)
            all_daily_returns.append(daily_ret)
            # 计算窗口指标
            ar = ((1 + daily_ret).prod() ** (252/len(daily_ret)) - 1) * 100
            print(f"    年化: {ar:.2f}%, 天数: {len(daily_ret)}")

    if all_daily_returns:
        v5_daily = pd.concat(all_daily_returns)
        v5_daily = v5_daily[~v5_daily.index.duplicated(keep="last")].sort_index()
        return v5_daily
    return None


def analyze_excess_returns(strategy_ret, benchmark_ret, label="V5 vs Benchmark"):
    """计算超额收益指标"""
    # 对齐日期
    common_dates = strategy_ret.index.intersection(benchmark_ret.index)
    if len(common_dates) == 0:
        print(f"  [{label}] 无重叠日期!")
        return None
    s = strategy_ret.loc[common_dates]
    b = benchmark_ret.loc[common_dates]
    excess = s - b

    # 基本指标
    n_days = len(excess)
    total_excess = (1 + s).prod() / (1 + b).prod() - 1
    ar_excess = ((1 + total_excess) ** (252 / n_days) - 1) * 100 if total_excess > -1 else -100
    tracking_error = excess.std() * np.sqrt(252) * 100
    ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Alpha & Beta (CAPM回归)
    beta, alpha_capm, r_value, p_value, std_err = stats.linregress(b, s)
    alpha_annual = alpha_capm * 252 * 100  # 年化Alpha

    # 超额收益最大回撤
    cum_excess = (1 + excess).cumprod()
    cum_max = cum_excess.expanding().max()
    excess_dd = ((cum_excess - cum_max) / cum_max).min() * 100

    # 月度胜率
    monthly_excess = excess.resample("M").apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly_excess > 0).sum() / len(monthly_excess) * 100

    # 策略和基准各自指标
    s_ar = ((1 + s).prod() ** (252 / n_days) - 1) * 100
    b_ar = ((1 + b).prod() ** (252 / n_days) - 1) * 100
    s_sharpe = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
    b_sharpe = b.mean() / b.std() * np.sqrt(252) if b.std() > 0 else 0
    s_cum = (1 + s).cumprod()
    b_cum = (1 + b).cumprod()
    s_dd = ((s_cum - s_cum.expanding().max()) / s_cum.expanding().max()).min() * 100
    b_dd = ((b_cum - b_cum.expanding().max()) / b_cum.expanding().max()).min() * 100

    print(f"\n  === {label} ===")
    print(f"  {'指标':<24} {'策略':>12} {'基准':>12} {'超额':>12}")
    print(f"  {'-'*60}")
    print(f"  {'年化收益':<24} {s_ar:>11.2f}% {b_ar:>11.2f}% {ar_excess:>+11.2f}%")
    print(f"  {'夏普/信息比率':<24} {s_sharpe:>12.4f} {b_sharpe:>12.4f} {ir:>12.4f}")
    print(f"  {'最大回撤':<24} {s_dd:>11.2f}% {b_dd:>11.2f}%")
    print(f"  {'跟踪误差(年化)':<24} {'':>12} {'':>12} {tracking_error:>11.2f}%")
    print(f"  {'Alpha(年化)':<24} {'':>12} {'':>12} {alpha_annual:>+11.2f}%")
    print(f"  {'Beta':<24} {'':>12} {'':>12} {beta:>12.4f}")
    print(f"  {'超额最大回撤':<24} {'':>12} {'':>12} {excess_dd:>11.2f}%")
    print(f"  {'月度胜率':<24} {'':>12} {'':>12} {win_rate:>11.1f}%")
    print(f"  {'月度数':<24} {'':>12} {'':>12} {len(monthly_excess):>12}")
    print(f"  {'Beta P-value':<24} {'':>12} {'':>12} {p_value:>12.6f}")

    # 逐年超额
    print(f"\n  逐年超额:")
    print(f"  {'年份':<8} {'策略':>10} {'基准':>10} {'超额':>10} {'胜':>4}")
    print(f"  {'-'*42}")
    for year in range(2021, 2027):
        if year == 2026:
            s_yr = s[s.index >= pd.Timestamp("2026-01-01")]
            b_yr = b[b.index >= pd.Timestamp("2026-01-01")]
            label_y = "2026H1"
        else:
            s_yr = s[(s.index >= pd.Timestamp(f"{year}-01-01")) & (s.index < pd.Timestamp(f"{year+1}-01-01"))]
            b_yr = b[(b.index >= pd.Timestamp(f"{year}-01-01")) & (b.index < pd.Timestamp(f"{year+1}-01-01"))]
            label_y = str(year)
        if len(s_yr) == 0 or len(b_yr) == 0:
            continue
        s_ar_y = ((1 + s_yr).prod() - 1) * 100
        b_ar_y = ((1 + b_yr).prod() - 1) * 100
        ex_y = s_ar_y - b_ar_y
        win = "✓" if ex_y > 0 else "✗"
        print(f"  {label_y:<8} {s_ar_y:>+9.2f}% {b_ar_y:>+9.2f}% {ex_y:>+9.2f}% {win:>4}")

    return {
        "ar_excess": ar_excess, "tracking_error": tracking_error, "ir": ir,
        "alpha": alpha_annual, "beta": beta, "excess_dd": excess_dd,
        "win_rate": win_rate, "s_ar": s_ar, "b_ar": b_ar,
        "s_sharpe": s_sharpe, "b_sharpe": b_sharpe, "s_dd": s_dd, "b_dd": b_dd,
    }


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    universe = df["code"].apply(format_qlib_code).tolist()
    print(f"股票池: {len(universe)} 只")

    cal = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # ====== 1. 股票池等权基准 ======
    print(f"\n{'='*70}")
    print(f"  Part 1: 股票池等权基准")
    print(f"{'='*70}")
    ew_daily, ew_ar, ew_sharpe, ew_dd, ew_yearly = compute_equal_weight_benchmark(
        universe, "2021-01-01", "2026-07-21")

    # ====== 2. 沪深300年度对比 (已知数据) ======
    print(f"\n{'='*70}")
    print(f"  Part 2: V5 vs 沪深300 (年度对比)")
    print(f"{'='*70}")
    print(f"  {'年份':<8} {'V5':>10} {'沪深300':>10} {'超额':>10} {'股票池等权':>12}")
    print(f"  {'-'*50}")
    for year in ["2021", "2022", "2023", "2024", "2025", "2026H1"]:
        v5 = V5_ANNUAL.get(int(year[:4]) if len(year) == 4 else 2026, 0)
        csi = CSI300_ANNUAL.get(int(year[:4]) if len(year) == 4 else 2026, 0)
        ew = ew_yearly.get(year, 0)
        excess_v5_csi = v5 - csi
        excess_v5_ew = v5 - ew
        print(f"  {year:<8} {v5:>+9.2f}% {csi:>+9.2f}% {excess_v5_csi:>+9.2f}% {ew:>+11.2f}%")

    # V5全周期
    v5_total = 1.0
    csi_total = 1.0
    for year in range(2021, 2027):
        v5_total *= (1 + V5_ANNUAL[year] / 100)
        csi_total *= (1 + CSI300_ANNUAL[year] / 100)
    v5_ar = (v5_total ** (252 / (5.5 * 252)) - 1) * 100
    csi_ar = (csi_total ** (252 / (5.5 * 252)) - 1) * 100
    print(f"\n  全周期(5.5年):")
    print(f"    V5累计: {(v5_total-1)*100:.2f}%, 年化: {v5_ar:.2f}%")
    print(f"    沪深300累计: {(csi_total-1)*100:.2f}%, 年化: {csi_ar:.2f}%")
    print(f"    超额年化: {v5_ar - csi_ar:.2f}%")
    print(f"    股票池等权年化: {ew_ar:.2f}%")
    print(f"    V5 vs 等权超额: {v5_ar - ew_ar:.2f}%")

    # ====== 3. 重跑V5获取日频收益, 做详细分析 ======
    print(f"\n{'='*70}")
    print(f"  Part 3: 重跑V5回测 (保存日频收益)")
    print(f"{'='*70}")

    print("  加载价值因子...")
    from v5_validation import load_value_factors, load_fundamental_features
    vf = load_value_factors(universe, cal, cal_set)
    fund = load_fundamental_features(universe, cal)

    v5_daily = run_v5_with_daily_returns(universe, fund, vf, cal, cal_set)

    if v5_daily is not None:
        v5_daily.to_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_daily_returns.csv")
        print(f"\n  V5日频收益已保存: {len(v5_daily)} 天")

        # ====== 4. 详细超额分析 ======
        print(f"\n{'='*70}")
        print(f"  Part 4: V5 vs 股票池等权基准 (日频对比)")
        print(f"{'='*70}")
        metrics_ew = analyze_excess_returns(v5_daily, ew_daily, "V5 vs 股票池等权")

        # ====== 5. 上线可行性评估 ======
        print(f"\n{'='*70}")
        print(f"  Part 5: 上线可行性评估")
        print(f"{'='*70}")

        if metrics_ew:
            print(f"\n  评估维度:")
            print(f"  {'维度':<20} {'指标':>12} {'阈值':>12} {'结论':>8}")
            print(f"  {'-'*52}")

            checks = [
                ("超额年化收益", metrics_ew["ar_excess"], 5.0, ">%"),
                ("信息比率(IR)", metrics_ew["ir"], 0.5, ">"),
                ("跟踪误差", metrics_ew["tracking_error"], 15.0, "<%"),
                ("Alpha显著性", metrics_ew["alpha"], 3.0, ">%"),
                ("Beta范围", abs(metrics_ew["beta"] - 1.0), 0.3, "<"),
                ("月度胜率", metrics_ew["win_rate"], 55.0, ">%"),
                ("超额最大回撤", abs(metrics_ew["excess_dd"]), 10.0, "<%"),
                ("策略最大回撤", abs(metrics_ew["s_dd"]), 20.0, "<%"),
                ("策略夏普", metrics_ew["s_sharpe"], 0.8, ">"),
            ]
            pass_count = 0
            for name, val, thresh, cmp in checks:
                if cmp.startswith(">"):
                    ok = val > thresh if not cmp.endswith("%") else val > thresh
                else:
                    ok = val < thresh
                if ok: pass_count += 1
                status = "✓ PASS" if ok else "✗ FAIL"
                print(f"  {name:<20} {val:>11.2f} {thresh:>11.1f} {status:>8}")

            total = len(checks)
            print(f"\n  总计: {pass_count}/{total} 项通过")

            if pass_count >= 7:
                print(f"  结论: ✓ 建议上线 — 核心指标达标, 超额收益稳定")
            elif pass_count >= 5:
                print(f"  结论: ⚠ 谨慎上线 — 部分指标边缘, 建议小仓位试运行")
            else:
                print(f"  结论: ✗ 暂不建议上线 — 多项指标未达标")
    else:
        print("  V5回测失败!")


if __name__ == "__main__":
    run()
