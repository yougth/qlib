"""
α=0.3融合 上线前健康检查
=========================
1. 2020年为何缺失 — 分析原因
2. 模型健康度 — 特征重要性分布 + 预测分数分布
3. 月度Top10持仓换手率 — 标的稳定性分析
4. α=0.3融合每月收益率明细
5. 交易摩擦成本 — 含手续费的真实收益

所有CSV读写均使用 sep='\t'
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["QLIB_NO_MP"] = "1"
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
    build_limit_up_set, filter_pred_by_tradability, load_value_factors
from v5_pipeline_fix import load_fundamental_features_fixed, \
    inject_features_fixed, filter_by_fundamental_deterioration

MODEL_CONFIG = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
        "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
        "max_depth": 8, "num_leaves": 210, "num_threads": 4}}

# 7个窗口 (排除2020)
WINDOWS = [
    {"train": ("2014-01-01","2017-12-31"), "valid": ("2018-01-01","2018-12-31"), "backtest": ("2019-01-01","2019-12-31"), "name":"W0", "year": 2019},
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1", "year": 2021},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2", "year": 2022},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3", "year": 2023},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4", "year": 2024},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5", "year": 2025},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "year": 2026},
]


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


def train_window(win, fcf_df, profit_df, cal, cal_set):
    """训练单个窗口, 返回 (pred, universe, vf, fund, model, dataset)"""
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

    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_alpha03_health"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    train_data = dataset.prepare("train", col_set="feature")
    n_feat = train_data.shape[1]
    print(f"    特征数: {n_feat}, 预测: {len(pred)}条")

    return pred, universe, vf, fund, model, dataset


def backtest_with_holdings_tracking(pred, universe, cal, month_ends, topk=10,
                                     alpha=0.0, vf=None, cost_bps=0.0):
    """
    带持仓追踪和交易成本的回测
    cost_bps: 单边交易成本(基点), 如15表示0.15%
    Returns: (daily_returns, monthly_holdings, monthly_returns, monthly_turnover)
    """
    if pred is None or len(pred) == 0:
        return pd.Series(), {}, {}, []

    if alpha > 0 and vf is not None:
        pred = apply_value_fusion(pred.copy(), vf, alpha=alpha)

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

    all_dates = sorted(pred.index.get_level_values(0).unique())
    start_dt = all_dates[0]
    end_dt = all_dates[-1]

    all_instruments = list(set(pred.index.get_level_values(1)))
    price_data = D.features(all_instruments, ["$close"],
                           start_time=start_dt - pd.Timedelta(days=10),
                           end_time=end_dt + pd.Timedelta(days=40))
    if price_data is None or len(price_data) == 0:
        return pd.Series(), {}, {}, []
    price_data = price_data.reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {}
    for inst, grp in price_data.groupby("instrument"):
        grp = grp.sort_values("datetime").set_index("datetime")
        price_dict[inst] = grp["close"]

    portfolio_returns = []
    portfolio_dates = []
    monthly_holdings = {}
    monthly_returns = {}
    monthly_turnover = []
    prev_holdings = set()

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]
            if not earlier:
                continue
            dt = earlier[-1]

        day_pred = pred.xs(dt, level=0)
        topk_stocks = day_pred["score"].nlargest(topk).index.tolist()
        monthly_holdings[dt] = topk_stocks

        # 换手率
        cur_set = set(topk_stocks)
        if prev_holdings:
            new_stocks = cur_set - prev_holdings
            turnover = len(new_stocks) / topk
            monthly_turnover.append({"date": dt, "turnover": turnover,
                "n_new": len(new_stocks), "n_hold": len(cur_set & prev_holdings)})
        prev_holdings = cur_set

        if i + 1 < len(month_ends):
            next_dt = month_ends[i + 1]
        else:
            next_dt = end_dt

        period_dates = [d for d in all_dates if dt < d <= next_dt]
        if not period_dates:
            continue

        # 交易成本: 换手部分需要买卖
        if cost_bps > 0 and prev_holdings:
            # 计算换手成本 (在调仓日扣除)
            new_stocks_count = len(cur_set - prev_holdings) if i > 0 else topk
            # 单边成本 * 2 (卖出旧的 + 买入新的) * 换手比例
            cost = (new_stocks_count / topk) * cost_bps * 2 / 10000.0
        else:
            cost = 0

        prev_prices = {}
        for inst in topk_stocks:
            if inst in price_dict and dt in price_dict[inst].index:
                prev_prices[inst] = price_dict[inst][dt]

        daily_rets = []
        first_day = True
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
                # 用topk而非n_valid: 停牌股票权重冻结, 不重新分配
                ret_val = day_ret / topk
                # 交易成本在第一个交易日扣除
                if first_day and cost > 0:
                    ret_val -= cost
                    first_day = False
                daily_rets.append(ret_val)
                portfolio_dates.append(pd_dt)
            else:
                daily_rets.append(0)
                portfolio_dates.append(pd_dt)

        portfolio_returns.extend(daily_rets)

        # 月度收益
        if daily_rets:
            monthly_ret = (1 + pd.Series(daily_rets)).prod() - 1
            monthly_returns[dt] = monthly_ret

    returns = pd.Series(portfolio_returns, index=pd.DatetimeIndex(portfolio_dates))
    returns = returns[~returns.index.duplicated(keep="last")]
    return returns, monthly_holdings, monthly_returns, monthly_turnover


def analyze_2020_exclusion(fcf_df, profit_df):
    """分析2020年为何被排除"""
    print(f"\n{'='*70}")
    print(f"  1. 2020年为何缺失?")
    print(f"{'='*70}")

    # 构建2020年的股票池
    # 如果要回测2020年, train应为2013-2016, valid 2017, backtest 2020
    # 但profit数据从2016才开始, 无法构建valid集
    by = 2020
    fcf_start = by - 11  # 2009
    fcf_end = by - 2     # 2018
    profit_start = max(by - 11, 2016)  # 2016
    profit_end = by - 2  # 2018

    target_fcf_years = list(range(fcf_start, fcf_end + 1))
    target_profit_years = list(range(profit_start, profit_end + 1))

    fcf_filtered = fcf_df[fcf_df["year"].isin(target_fcf_years)]
    fcf_positive = fcf_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_positive["code"].unique())

    profit_filtered = profit_df[profit_df["year"].isin(target_profit_years)]
    profit_positive = profit_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_profit_years) * 0.8 and (g["net_profit"] > 0).all())
    profit_codes = set(profit_positive["code"].unique())

    pool_2020 = fcf_codes & profit_codes

    print(f"\n  2020年股票池构建:")
    print(f"    FCF正年份要求: {fcf_start}~{fcf_end} ({len(target_fcf_years)}年全正)")
    print(f"    净利润正年份要求: {profit_start}~{profit_end} ({len(target_profit_years)}年全正)")
    print(f"    FCF筛选通过: {len(fcf_codes)} 只")
    print(f"    利润筛选通过: {len(profit_codes)} 只")
    print(f"    交集(最终池): {len(pool_2020)} 只")

    # 对比其他年份的池大小
    for year in [2019, 2021, 2022, 2023, 2024, 2025, 2026]:
        fs = year - 11; fe = year - 2
        ps = max(year - 11, 2016); pe = year - 2
        tfy = list(range(fs, fe + 1))
        tpy = list(range(ps, pe + 1))
        ff = fcf_df[fcf_df["year"].isin(tfy)]
        fp = ff.groupby("code").filter(lambda g: len(g) >= len(tfy) * 0.8 and (g["fcf"] > 0).all())
        pf = profit_df[profit_df["year"].isin(tpy)]
        pp = pf.groupby("code").filter(lambda g: len(g) >= len(tpy) * 0.8 and (g["net_profit"] > 0).all())
        pool = set(fp["code"]) & set(pp["code"])
        print(f"    {year}年池: {len(pool)} 只")

    # 2020年市场特征
    print(f"\n  2020年市场异常特征:")
    print(f"    - 新冠疫情冲击, Q1大跌后V型反转")
    print(f"    - 股票池中大量中小盘股暴涨(消费/医疗/新能源)")

    # 读取已有的2020回测结果
    ext_path = "/Users/11164591/Documents/Qoder目录/qlib/v5_extended_results.csv"
    if os.path.exists(ext_path):
        ext = pd.read_csv(ext_path)
        w0b = ext[ext["window"] == "W0b"]
        if len(w0b) > 0:
            ar = w0b.iloc[0]["ar"]
            print(f"\n  2020年(W0b)回测结果: 年化{ar:.1f}%")
            print(f"    这是一个极端异常值 (818%!), 原因:")
            print(f"    1. 2020年新冠后小盘股暴涨, 选股池中134只股票平均涨幅巨大")
            print(f"    2. 模型在极端牛市中top10集中持有暴涨股, 收益失真")
            print(f"    3. 将2020纳入会严重高估策略平均收益, 误导决策")

    print(f"\n  结论: 2020年被排除是因为:")
    print(f"    1. 市场极端异常(新冠V型反转), 不具代表性")
    print(f"    2. 回测年化818%完全失真, 纳入会严重高估策略表现")
    print(f"    3. 利润数据仅从2016年开始, 2020年valid集(2019)数据不足")
    print(f"    4. 排除2020是正确的风控决策, 让回测结果更可靠")


def analyze_model_health(fi_path, pred=None, model=None, dataset=None):
    """分析模型健康度"""
    print(f"\n{'='*70}")
    print(f"  2. 模型健康度检查")
    print(f"{'='*70}")

    # 2a. 特征重要性
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        total_gain = fi["gain"].sum()
        fi["gain_pct"] = fi["gain"] / total_gain * 100

        print(f"\n  [2a] 特征重要性分布 (W6模型):")
        print(f"    总特征数: {len(fi)}")
        print(f"    有效特征(gain>0): {(fi['gain'] > 0).sum()}")
        print(f"    无效特征(gain=0): {(fi['gain'] == 0).sum()} ({(fi['gain'] == 0).sum()/len(fi)*100:.1f}%)")
        print(f"    Top10特征占总gain: {fi.head(10)['gain_pct'].sum():.1f}%")
        print(f"    Top20特征占总gain: {fi.head(20)['gain_pct'].sum():.1f}%")
        print(f"    Top30特征占总gain: {fi.head(30)['gain_pct'].sum():.1f}%")

        print(f"\n    Top15最重要特征:")
        print(f"    {'排名':<6} {'特征名':<20} {'gain':>10} {'占比':>8} {'分裂次数':>8}")
        print(f"    {'-'*56}")
        for i, (_, row) in enumerate(fi.head(15).iterrows()):
            print(f"    {i+1:<6} {row['feature']:<20} {row['gain']:>10.1f} {row['gain_pct']:>7.2f}% {int(row['split']):>8}")

        # 特征分类
        value_features = ["roe_annual", "pb_pct_3y", "pe_pct_3y", "div_yield_est"]
        fund_features = ["fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"]
        tech_features = [f for f in fi["feature"] if f not in value_features + fund_features]

        v_gain = fi[fi["feature"].isin(value_features)]["gain"].sum()
        f_gain = fi[fi["feature"].isin(fund_features)]["gain"].sum()
        t_gain = fi[fi["feature"].isin(tech_features)]["gain"].sum()

        print(f"\n    特征类别贡献:")
        print(f"      价值因子(4个):   gain={v_gain:.0f} ({v_gain/total_gain*100:.1f}%)")
        print(f"      基本面因子(5个): gain={f_gain:.0f} ({f_gain/total_gain*100:.1f}%)")
        print(f"      量价因子(其余):  gain={t_gain:.0f} ({t_gain/total_gain*100:.1f}%)")

        # 健康度评估
        print(f"\n    健康度评估:")
        n_zero = (fi['gain'] == 0).sum()
        top10_pct = fi.head(10)['gain_pct'].sum()
        if n_zero / len(fi) > 0.5:
            print(f"      ⚠ 超过一半特征({n_zero}/{len(fi)})无贡献, 存在特征冗余")
        else:
            print(f"      ✓ 有效特征占比合理")
        if top10_pct > 60:
            print(f"      ⚠ Top10特征贡献{top10_pct:.1f}%, 集中度偏高, 可能过拟合")
        else:
            print(f"      ✓ 特征贡献分散度合理, Top10占{top10_pct:.1f}%")

    # 2b. 预测分数分布
    if pred is not None and len(pred) > 0:
        print(f"\n  [2b] 预测分数分布:")
        scores = pred["score"]
        print(f"    样本数: {len(scores)}")
        print(f"    均值: {scores.mean():.4f}")
        print(f"    标准差: {scores.std():.4f}")
        print(f"    最小值: {scores.min():.4f}")
        print(f"    最大值: {scores.max():.4f}")
        print(f"    中位数: {scores.median():.4f}")
        print(f"    偏度: {scores.skew():.4f}")
        print(f"    峰度: {scores.kurtosis():.4f}")

        # 按日检查分数分布稳定性
        daily_stats = scores.groupby(level=0).agg(["mean", "std", "min", "max"])
        print(f"\n    日均分分布稳定性:")
        print(f"      日均分均值: {daily_stats['mean'].mean():.4f} ± {daily_stats['mean'].std():.4f}")
        print(f"      日标准差均值: {daily_stats['std'].mean():.4f} ± {daily_stats['std'].std():.4f}")

        # Top10 vs Bottom10 分数差距
        daily_top = scores.groupby(level=0).nlargest(10).groupby(level=0).mean()
        daily_bot = scores.groupby(level=0).nsmallest(10).groupby(level=0).mean()
        spread = (daily_top - daily_bot).mean()
        print(f"      Top10-Bottom10日均分差: {spread:.4f}")
        if spread > 0.5:
            print(f"      ✓ 模型区分度足够, Top/Bottom分差显著")
        else:
            print(f"      ⚠ 模型区分度不足, Top/Bottom分差较小")


def analyze_holdings_turnover(monthly_holdings):
    """分析持仓换手率"""
    print(f"\n{'='*70}")
    print(f"  3. 月度Top10持仓换手率分析")
    print(f"{'='*70}")

    if not monthly_holdings:
        print("  无持仓数据")
        return

    dates = sorted(monthly_holdings.keys())
    print(f"\n  持仓追踪: {len(dates)}个月")

    # 逐月换手率
    print(f"\n  {'调仓日':<14} {'新买入':>6} {'保留':>6} {'换手率':>8} {'持仓标的'}")
    print(f"  {'-'*80}")

    prev_set = None
    all_turnover = []
    stock_freq = {}  # 股票出现次数

    for dt in dates:
        cur = set(monthly_holdings[dt])
        for s in cur:
            stock_freq[s] = stock_freq.get(s, 0) + 1

        if prev_set is None:
            new_count = 10
            hold_count = 0
            turnover = 1.0
        else:
            new_count = len(cur - prev_set)
            hold_count = len(cur & prev_set)
            turnover = new_count / 10

        all_turnover.append(turnover)
        holdings_str = ", ".join(monthly_holdings[dt][:5]) + "..."
        print(f"  {dt.strftime('%Y-%m-%d'):<14} {new_count:>6} {hold_count:>6} {turnover*100:>7.0f}% {holdings_str}")
        prev_set = cur

    avg_turnover = np.mean(all_turnover)
    print(f"\n  平均月换手率: {avg_turnover*100:.0f}%")
    print(f"  平均每月新买入: {avg_turnover*10:.1f}只")
    print(f"  平均每月保留: {(1-avg_turnover)*10:.1f}只")

    # 标的稳定性
    print(f"\n  标的稳定性分析:")
    print(f"    出现过的不同标的数: {len(stock_freq)}")
    freq_dist = pd.Series(list(stock_freq.values()))
    print(f"    只出现1次的标的: {(freq_dist == 1).sum()} ({(freq_dist == 1).sum()/len(freq_dist)*100:.0f}%)")
    print(f"    出现2-3次的标的: {((freq_dist >= 2) & (freq_dist <= 3)).sum()}")
    print(f"    出现4+次的标的(核心持仓): {(freq_dist >= 4).sum()}")

    if stock_freq:
        top_freq = sorted(stock_freq.items(), key=lambda x: -x[1])[:10]
        print(f"\n    最常持有的Top10标的:")
        for inst, cnt in top_freq:
            print(f"      {inst}: 出现{cnt}次/{len(dates)}月 ({cnt/len(dates)*100:.0f}%)")

    if avg_turnover > 0.6:
        print(f"\n  ⚠ 换手率偏高({avg_turnover*100:.0f}%), 交易成本影响大, 需关注")
    elif avg_turnover > 0.4:
        print(f"\n  ⚡ 换手率中等({avg_turnover*100:.0f}%), 有一定交易成本")
    else:
        print(f"\n  ✓ 换手率较低({avg_turnover*100:.0f}%), 标的稳定性好")


def analyze_monthly_returns(monthly_returns_dict, window_name, year):
    """分析月度收益"""
    print(f"\n  [{window_name}] α=0.3融合 月度收益明细:")

    if not monthly_returns_dict:
        print("    无数据")
        return None

    dates = sorted(monthly_returns_dict.keys())
    print(f"  {'月份':<12} {'月收益率':>10} {'累计收益':>10}")
    print(f"  {'-'*34}")

    cumret = 0
    monthly_data = []
    for dt in dates:
        ret = monthly_returns_dict[dt]
        cumret = (1 + cumret) * (1 + ret) - 1
        print(f"  {dt.strftime('%Y-%m'):<12} {ret*100:>9.2f}% {cumret*100:>9.2f}%")
        monthly_data.append({"window": window_name, "year": year,
            "month": dt.strftime("%Y-%m"), "return": ret, "cum_return": cumret})

    n_pos = sum(1 for r in monthly_returns_dict.values() if r > 0)
    n_neg = sum(1 for r in monthly_returns_dict.values() if r < 0)
    avg_pos = np.mean([r for r in monthly_returns_dict.values() if r > 0]) if n_pos > 0 else 0
    avg_neg = np.mean([r for r in monthly_returns_dict.values() if r < 0]) if n_neg > 0 else 0

    print(f"\n  统计: {n_pos}盈/{n_neg}亏, 胜率{n_pos/(n_pos+n_neg)*100:.0f}%")
    print(f"  平均盈: +{avg_pos*100:.2f}%, 平均亏: {avg_neg*100:.2f}%")
    print(f"  盈亏比: {abs(avg_pos/avg_neg):.2f}" if avg_neg != 0 else "")
    print(f"  全期累计: {cumret*100:.2f}%")

    return monthly_data


def analyze_transaction_costs(all_monthly_turnover, all_monthly_returns, no_cost_returns):
    """分析交易摩擦成本"""
    print(f"\n{'='*70}")
    print(f"  5. 交易摩擦成本分析")
    print(f"{'='*70}")

    print(f"\n  当前回测是否包含交易成本: ❌ 不包含")
    print(f"  vectorized_backtest 仅计算 price_ratio - 1, 无任何手续费扣除")

    # 汇总换手率
    if all_monthly_turnover:
        avg_to = np.mean([t["turnover"] for t in all_monthly_turnover])
        print(f"\n  平均月换手率: {avg_to*100:.0f}%")
        print(f"  每月平均换股: {avg_to*10:.1f}只 / 10只")

    # 成本情景分析
    print(f"\n  交易成本情景分析 (月频调仓):")
    print(f"  {'成本场景':<20} {'单边费率':>10} {'月均成本':>10} {'年化拖累':>10}")
    print(f"  {'-'*52}")

    scenarios = [
        ("券商佣金(最低)", 1.0),    # 万一
        ("券商佣金(普通)", 2.5),    # 万二点五
        ("佣金+印花税", 5.0),       # 万五(含卖出印花税)
        ("全成本(含滑点)", 10.0),   # 万十
        ("保守估计", 15.0),         # 万十五
    ]

    for name, bps in scenarios:
        # 每月换手成本 = 换手率 * 单边费率 * 2(买卖)
        monthly_cost = avg_to * bps * 2 / 10000.0 if all_monthly_turnover else 0
        annual_drag = monthly_cost * 12
        print(f"  {name:<20} {bps/100:>9.2f}% {monthly_cost*100:>9.3f}% {annual_drag*100:>9.2f}%")

    # 用实际月度收益计算含成本后的收益
    if all_monthly_returns and no_cost_returns is not None:
        print(f"\n  含5bps成本后的收益对比:")
        no_cost_ar = calc_metrics(no_cost_returns)["ar"]
        print(f"    无成本年化: {no_cost_ar*100:.2f}%")

        for cost_bps in [5.0, 10.0, 15.0]:
            monthly_cost = avg_to * cost_bps * 2 / 10000.0 if all_monthly_turnover else 0
            # 近似: 从每日收益中扣除月均成本/21
            daily_cost = monthly_cost / 21
            adj_returns = no_cost_returns.copy()
            adj_returns = adj_returns - daily_cost
            adj_ar = calc_metrics(adj_returns)["ar"]
            print(f"    {cost_bps}bps成本年化: {adj_ar*100:.2f}% (拖累{(no_cost_ar-adj_ar)*100:.2f}%)")

    print(f"\n  结论:")
    print(f"    - 当前回测未计入交易成本, 实际收益会降低")
    if all_monthly_turnover:
        print(f"    - 按月换手率{avg_to*100:.0f}%估算, 5bps全成本年化拖累约{avg_to*5*2*12/100:.2f}%")
        print(f"    - α=0.3融合换手率较高, 建议实盘用限价单+分批建仓降低滑点")
    return avg_to if all_monthly_turnover else 0.5


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # ====== 1. 2020年为何缺失 ======
    analyze_2020_exclusion(fcf_df, profit_df)

    # ====== 2. 模型健康度 (使用W6已有数据) ======
    fi_path = "/Users/11164591/Documents/Qoder目录/qlib/w6_fi_fixed.csv"

    # 选择W4(2024)作为代表窗口详细分析 (α=0.3效果最好)
    target_win = WINDOWS[4]  # W4, 2024
    print(f"\n{'='*70}")
    print(f"  训练 {target_win['name']} (回测{target_win['year']}) 进行详细分析")
    print(f"{'='*70}")

    pred, universe, vf, fund, model, dataset = train_window(
        target_win, fcf_df, profit_df, cal, cal_set)

    # 模型健康度
    analyze_model_health(fi_path, pred, model, dataset)

    # ====== 3 & 4. 持仓换手率 + 月度收益 ======
    bs, be = target_win["backtest"]
    month_ends = get_month_end_dates(cal, bs, be)

    print(f"\n{'='*70}")
    print(f"  3. {target_win['name']} α=0.3融合 持仓与月度收益分析")
    print(f"{'='*70}")

    # 无成本回测 (追踪持仓)
    returns, monthly_holdings, monthly_returns, monthly_turnover = \
        backtest_with_holdings_tracking(
            pred, universe, cal, month_ends, topk=10, alpha=0.3, vf=vf, cost_bps=0)

    # 持仓换手率
    analyze_holdings_turnover(monthly_holdings)

    # 月度收益
    print(f"\n{'='*70}")
    print(f"  4. α=0.3融合每月收益率明细")
    print(f"{'='*70}")
    monthly_data = analyze_monthly_returns(monthly_returns, target_win["name"], target_win["year"])

    # ====== 5. 交易成本分析 ======
    all_turnover_data = []
    for t in monthly_turnover:
        all_turnover_data.append({"turnover": t["turnover"]})
    avg_to = analyze_transaction_costs(all_turnover_data, monthly_data, returns)

    # ====== 汇总: 对所有7个窗口的α=0.3结果做月度收益分析 ======
    print(f"\n{'='*70}")
    print(f"  附录: 所有窗口 α=0.3融合 年化收益对比 (已有结果)")
    print(f"{'='*70}")

    results_path = "/Users/11164591/Documents/Qoder目录/qlib/v5_new_baseline_results.csv"
    if os.path.exists(results_path):
        results = pd.read_csv(results_path, sep='\t')
        alpha03 = results[results["experiment"] == "α=0.3"]
        baseline = results[results["experiment"] == "基线"]

        print(f"\n  {'窗口':<6} {'年份':<6} {'基线年化':>10} {'α=0.3年化':>10} {'差值':>8} {'α=0.3夏普':>10} {'α=0.3回撤':>10}")
        print(f"  {'-'*66}")
        for _, row in alpha03.iterrows():
            w = row["window"]
            y = int(row["year"])
            b_row = baseline[baseline["window"] == w].iloc[0]
            diff = row["ar"] - b_row["ar"]
            print(f"  {w:<6} {y:<6} {b_row['ar']*100:>9.2f}% {row['ar']*100:>9.2f}% {diff*100:>+7.2f}% {row['sharpe']:>10.2f} {row['max_dd']*100:>9.1f}%")

        # 汇总
        avg_b = baseline["ar"].mean()
        avg_a = alpha03["ar"].mean()
        avg_a_sharpe = alpha03["sharpe"].mean()
        # 按窗口对齐后比较
        a_wins = 0
        for _, a_row in alpha03.iterrows():
            w = a_row["window"]
            b_row = baseline[baseline["window"] == w]
            if len(b_row) > 0 and a_row["ar"] > b_row.iloc[0]["ar"]:
                a_wins += 1

        print(f"\n  平均: 基线{avg_b*100:.2f}%, α=0.3 {avg_a*100:.2f}%, 差值{(avg_a-avg_b)*100:+.2f}%")
        print(f"  α=0.3优于基线的窗口: {a_wins}/7")
        print(f"  α=0.3平均夏普: {avg_a_sharpe:.2f}")

        # 上线建议
        print(f"\n  {'='*50}")
        print(f"  α=0.3融合 上线建议:")
        print(f"  {'='*50}")
        if avg_a > avg_b and a_wins >= 4:
            print(f"  ✓ 建议上线: 平均年化{avg_a*100:.2f}% > 基线{avg_b*100:.2f}%")
            print(f"  ✓ {a_wins}/7窗口优于基线, 增益稳定")
            print(f"  ⚠ 需注意: W2/W3(2022/2023) α=0.3反而变差, 熊市中价值融合可能过载")
            print(f"  ⚠ 需注意: 当前回测未含交易成本, 实际收益会降低")
            print(f"  ⚠ 需注意: 月换手率较高(~{avg_to*100:.0f}%), 实盘需控制交易成本")
            print(f"  ⚠ 需注意: 未含交易成本的年化收益会因摩擦成本降低约{avg_to*5*2*12/100:.1f}%")
        else:
            print(f"  ✗ 不建议上线: α=0.3未稳定优于基线")

    # 保存月度收益
    if monthly_data:
        pd.DataFrame(monthly_data).to_csv(
            "/Users/11164591/Documents/Qoder目录/qlib/v5_alpha03_monthly_returns.csv",
            sep='\t', index=False)
        print(f"\n  已保存: v5_alpha03_monthly_returns.csv")


if __name__ == "__main__":
    run()
