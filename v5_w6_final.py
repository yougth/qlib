"""
W6 最终收尾脚本 V3 — 从handler提取价格数据, 零额外D.features调用
核心: handler._data 已包含OHLCV, 直接提取用于回测
所有CSV读写均使用 sep='\t'
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["QLIB_NO_MP"] = "1"  # 禁用qlib多进程
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

from v5_validation import Alpha158Enhanced, apply_value_fusion, load_value_factors
from v5_pipeline_fix import load_fundamental_features_fixed, \
    inject_features_fixed, filter_by_fundamental_deterioration

MODEL_CONFIG = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
        "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
        "max_depth": 8, "num_leaves": 210, "num_threads": 4}}

W6 = {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"),
      "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "year": 2026}

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
    "W6": {"baseline": None, "alpha03": None, "e8": None},
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


def read_qlib_bin(instrument, field, cal):
    """直接读取qlib二进制数据文件, 避免D.features调用"""
    inst_lower = instrument.lower()
    bin_path = os.path.expanduser(
        f"~/.qlib/qlib_data/cn_data/features/{inst_lower}/{field}.day.bin")
    if not os.path.exists(bin_path):
        return pd.Series(dtype=float)
    with open(bin_path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    if n == 0:
        return pd.Series(dtype=float)
    values = np.frombuffer(data, dtype=np.float32, count=n)
    # 对齐到日历末尾
    if n <= len(cal):
        dates = cal[-n:]
    else:
        dates = cal
        values = values[-len(cal):]
    return pd.Series(values, index=dates)


def load_price_data_bin(universe, cal, fields=None):
    """从二进制文件批量加载价格数据, 返回DataFrame"""
    if fields is None:
        fields = ["close", "open", "high", "low", "volume"]
    all_dfs = []
    n_inst = len(universe)
    for i, inst in enumerate(universe):
        if i % 50 == 0:
            print(f"    加载: {i}/{n_inst}...")
        col_dict = {}
        for f in fields:
            s = read_qlib_bin(inst, f, cal)
            if len(s) > 0:
                col_dict[f] = s
        if not col_dict:
            continue
        inst_df = pd.DataFrame(col_dict)
        inst_df["instrument"] = inst
        inst_df = inst_df.reset_index().rename(columns={"index": "datetime"})
        all_dfs.append(inst_df)
    if not all_dfs:
        return pd.DataFrame(columns=["instrument", "datetime"] + fields)
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(["instrument", "datetime"])
    print(f"    完成: {len(result)}行, {len(result['instrument'].unique())}只股票")
    return result


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    by = W6["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"\n  W6: 股票池{len(universe)}只")

    cal = D.calendar(start_time="2021-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # ====== 1. 创建Dataset + 注入特征 + 训练 ======
    print(f"\n{'='*70}")
    print(f"  1. 训练 W6")
    print(f"{'='*70}")

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

    # 读取qlib全量日历 (用于二进制文件对齐)
    full_cal_path = os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt")
    with open(full_cal_path) as f:
        full_cal = [pd.Timestamp(line.strip()) for line in f if line.strip()]
    print(f"  全量日历: {len(full_cal)}天, {full_cal[0].date()}~{full_cal[-1].date()}")

    # 从二进制文件加载价格数据 (不调用D.features, 避免挂起)
    print("\n  从二进制文件加载OHLCV数据...")
    price_raw = load_price_data_bin(universe, full_cal)
    print(f"  价格数据: {len(price_raw)}行, 日期{price_raw['datetime'].min().date()}~{price_raw['datetime'].max().date()}")

    # 注入特征
    vf = load_value_factors(universe, cal, cal_set)
    fund = load_fundamental_features_fixed(universe, cal)
    if fund is not None:
        inject_features_fixed(dataset, fund, "fund")
    if vf is not None:
        inject_features_fixed(dataset, vf, "vf")

    # 训练
    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_w6_final_v3"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    train_data = dataset.prepare("train", col_set="feature")
    print(f"  特征数: {train_data.shape[1]}, 预测: {len(pred)}条")

    # ====== 2. 从提取的数据构建回测所需组件 ======
    print(f"\n{'='*70}")
    print(f"  2. 构建回测组件 (零D.features调用)")
    print(f"{'='*70}")

    # 2a. 价格字典
    price_dict = {}
    for inst, grp in price_raw.groupby("instrument"):
        grp = grp.sort_values("datetime").set_index("datetime")
        price_dict[inst] = grp["close"]

    # 2b. 涨跌停/停牌集合
    print("  计算涨跌停/停牌集合...")
    ohlc = price_raw.copy()
    ohlc["prev_close"] = ohlc.groupby("instrument")["close"].shift(1)
    ohlc["daily_ret"] = (ohlc["close"] - ohlc["prev_close"]) / ohlc["prev_close"]

    limit_up_mask = (
        (ohlc["open"] == ohlc["high"]) &
        (ohlc["high"] == ohlc["low"]) &
        (ohlc["low"] == ohlc["close"]) &
        (ohlc["daily_ret"] > 0.09)
    )
    limit_up_set = set(zip(ohlc.loc[limit_up_mask, "datetime"],
                           ohlc.loc[limit_up_mask, "instrument"]))
    if "volume" in ohlc.columns:
        susp_mask = (ohlc["volume"].isna()) | (ohlc["volume"] == 0)
        suspension_set = set(zip(ohlc.loc[susp_mask, "datetime"],
                                 ohlc.loc[susp_mask, "instrument"]))
    else:
        suspension_set = set()
    print(f"  一字涨停: {len(limit_up_set)} 条, 停牌: {len(suspension_set)} 条")

    # 2c. 宏观信号 (从提取的数据计算)
    print("  计算全A等权200日均线信号...")
    macro_data = price_raw[["instrument", "datetime", "close"]].copy()
    macro_data = macro_data.sort_values(["instrument", "datetime"])
    macro_data["ret"] = macro_data.groupby("instrument")["close"].pct_change()
    ew_ret = macro_data.groupby("datetime")["ret"].mean().dropna()
    ew_cum = (1 + ew_ret).cumprod()
    ew_ma200 = ew_cum.rolling(200, min_periods=60).mean()
    macro_signals = {}
    # 直接遍历ew_cum的index, 避免日历日期不匹配问题
    for dt in ew_cum.index:
        if dt >= pd.Timestamp("2025-01-01") and dt in ew_ma200.index:
            if not pd.isna(ew_ma200[dt]):
                macro_signals[dt] = 0.7 if ew_cum[dt] < ew_ma200[dt] else 1.0
    bear_days = sum(1 for v in macro_signals.values() if v < 1.0)
    print(f"  宏观信号: {len(macro_signals)} 个日期, 熊市(70%): {bear_days}天, 牛市(100%): {len(macro_signals)-bear_days}天")

    # ====== 3. 三组实验回测 ======
    print(f"\n{'='*70}")
    print(f"  3. W6 三组实验回测")
    print(f"{'='*70}")

    month_ends = get_month_end_dates(cal, bs, be)
    print(f"  W6: {bs}~{be}, 月末调仓日{len(month_ends)}个")

    all_dates = sorted(pred.index.get_level_values(0).unique())
    start_dt = all_dates[0]
    end_dt = all_dates[-1]
    bad_set = limit_up_set | suspension_set

    def run_backtest(pred_input, use_alpha, use_macro, use_fund_filter, label):
        print(f"\n  [{label}]")
        p = pred_input.copy()
        if use_alpha and vf is not None:
            print("    应用 α=0.3 价值融合...")
            p = apply_value_fusion(p, vf, alpha=0.3)

        if len(bad_set) > 0:
            mask = pd.Series(False, index=p.index)
            for dt, inst in bad_set:
                if (dt, inst) in p.index:
                    mask.loc[(dt, inst)] = True
            p.loc[mask, "score"] = -999
            print(f"    涨跌停过滤: {mask.sum()} 条")

        if use_fund_filter:
            print("    基本面滑坡硬过滤 (30%)...")
            # 快速版: 使用字典查找替代DataFrame过滤
            fcf_lookup = {}
            for _, row in fcf_df.iterrows():
                fcf_lookup[(row["code"], int(row["year"]))] = row["fcf"]
            profit_lookup = {}
            for _, row in profit_df.iterrows():
                profit_lookup[(row["code"], int(row["year"]))] = row["net_profit"]

            p = p.sort_index()
            dates_arr = p.index.get_level_values(0)
            insts_arr = p.index.get_level_values(1)
            scores = p["score"].values.copy()
            n_excluded = 0
            excluded_last = []

            for i in range(len(p)):
                dt = dates_arr[i]
                inst = insts_arr[i]
                code = inst[2:]
                ay = dt.year - 2 if dt.month < 5 else dt.year - 1

                fcf_cur = fcf_lookup.get((code, ay))
                fcf_prev = fcf_lookup.get((code, ay - 1))
                profit_cur = profit_lookup.get((code, ay))
                profit_prev = profit_lookup.get((code, ay - 1))

                should_exclude = False
                reason = ""
                if fcf_cur is not None and fcf_prev is not None and fcf_prev > 0:
                    fcf_decline = (fcf_prev - fcf_cur) / abs(fcf_prev)
                    if fcf_decline > 0.30:
                        should_exclude = True
                        reason = f"FCF下降{fcf_decline*100:.0f}%"
                if not should_exclude and profit_cur is not None and profit_prev is not None and profit_prev > 0:
                    profit_decline = (profit_prev - profit_cur) / abs(profit_prev)
                    if profit_decline > 0.30:
                        should_exclude = True
                        reason = f"利润下降{profit_decline*100:.0f}%"

                if should_exclude:
                    scores[i] = -999
                    n_excluded += 1
                    if dt == dates_arr[-1]:
                        excluded_last.append((inst, reason))

            p["score"] = scores
            print(f"    过滤: {n_excluded} 条")
            if excluded_last:
                print(f"    最后信号日剔除: {len(excluded_last)}只")
                for inst, reason in excluded_last[:5]:
                    print(f"      {inst}: {reason}")

        print("    回测中...")
        portfolio_returns = []
        portfolio_dates = []

        for i, dt in enumerate(month_ends):
            if dt not in p.index.get_level_values(0):
                earlier = [d for d in all_dates if d <= dt]
                if not earlier:
                    continue
                dt = earlier[-1]

            day_pred = p.xs(dt, level=0)
            topk_stocks = day_pred["score"].nlargest(10).index.tolist()

            position = 1.0
            if use_macro and dt in macro_signals:
                position = macro_signals[dt]

            next_dt = month_ends[i + 1] if i + 1 < len(month_ends) else end_dt
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
        m = calc_metrics(returns)
        print(f"    结果: 年化{m['ar']*100:.2f}%, 夏普{m['sharpe']:.2f}, 回撤{m['max_dd']*100:.1f}%")
        return m

    m1 = run_backtest(pred, False, False, False, "基线")
    KNOWN_RESULTS["W6"]["baseline"] = m1

    m2 = run_backtest(pred, True, False, False, "α=0.3融合")
    KNOWN_RESULTS["W6"]["alpha03"] = m2

    m3 = run_backtest(pred, True, True, True, "E8/方案D")
    KNOWN_RESULTS["W6"]["e8"] = m3

    # ====== 4. 完整汇总 ======
    print(f"\n{'='*70}")
    print(f"  4. 完整汇总结果 (7窗口, 排除2020)")
    print(f"{'='*70}")

    win_info = [("W0",2019),("W1",2021),("W2",2022),("W3",2023),("W4",2024),("W5",2025),("W6",2026)]

    print(f"\n  {'窗口':<6} {'年份':<6} {'基线年化':>10} {'α=0.3年化':>10} {'E8年化':>10} {'基线夏普':>10} {'α=0.3夏普':>10} {'E8夏普':>10}")
    print(f"  {'-'*78}")
    for name, year in win_info:
        r = KNOWN_RESULTS[name]
        b, a, e = r["baseline"], r["alpha03"], r["e8"]
        print(f"  {name:<6} {year:<6} {b['ar']*100:>9.2f}% {a['ar']*100:>9.2f}% {e['ar']*100:>9.2f}% {b['sharpe']:>10.2f} {a['sharpe']:>10.2f} {e['sharpe']:>10.2f}")

    all_b_ar = np.mean([KNOWN_RESULTS[n]["baseline"]["ar"] for n,_ in win_info])
    all_a_ar = np.mean([KNOWN_RESULTS[n]["alpha03"]["ar"] for n,_ in win_info])
    all_e_ar = np.mean([KNOWN_RESULTS[n]["e8"]["ar"] for n,_ in win_info])
    all_b_sharpe = np.mean([KNOWN_RESULTS[n]["baseline"]["sharpe"] for n,_ in win_info])
    all_a_sharpe = np.mean([KNOWN_RESULTS[n]["alpha03"]["sharpe"] for n,_ in win_info])
    all_e_sharpe = np.mean([KNOWN_RESULTS[n]["e8"]["sharpe"] for n,_ in win_info])
    b_wins = sum(1 for n,_ in win_info if KNOWN_RESULTS[n]["baseline"]["ar"] > 0)
    a_wins = sum(1 for n,_ in win_info if KNOWN_RESULTS[n]["alpha03"]["ar"] > 0)
    e_wins = sum(1 for n,_ in win_info if KNOWN_RESULTS[n]["e8"]["ar"] > 0)
    alpha_better = sum(1 for n,_ in win_info if KNOWN_RESULTS[n]["alpha03"]["ar"] > KNOWN_RESULTS[n]["baseline"]["ar"])

    print(f"\n  全期汇总 (7窗口平均, 排除2020):")
    print(f"  {'实验':<16} {'平均年化':>10} {'平均夏普':>10} {'胜率':>8}")
    print(f"  {'-'*46}")
    print(f"  {'新V5基线':<16} {all_b_ar*100:>9.2f}% {all_b_sharpe:>10.2f} {b_wins}/7")
    print(f"  {'α=0.3融合':<16} {all_a_ar*100:>9.2f}% {all_a_sharpe:>10.2f} {a_wins}/7")
    print(f"  {'E8/方案D':<16} {all_e_ar*100:>9.2f}% {all_e_sharpe:>10.2f} {e_wins}/7")

    print(f"\n  {'='*50}")
    print(f"  新V5 vs 旧V5 对比:")
    print(f"  {'='*50}")
    print(f"  旧V5(假V5, 172特征): 年化18.13%, 夏普0.89")
    print(f"  新V5(181特征):       年化{all_b_ar*100:.2f}%, 夏普{all_b_sharpe:.2f}")
    print(f"  提升: {(all_b_ar - 0.1813)*100:+.2f}个百分点")

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

    print(f"\n  {'='*50}")
    print(f"  终极E8 (方案D + 基本面过滤) 结论:")
    print(f"  {'='*50}")
    print(f"    年化{all_e_ar*100:.2f}%, 夏普{all_e_sharpe:.2f}")
    print(f"    对比基线: {(all_e_ar-all_b_ar)*100:+.2f}%")
    print(f"    对比α=0.3: {(all_e_ar-all_a_ar)*100:+.2f}%")

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
    print(f"\n  已保存: v5_new_baseline_results.csv (sep='\\t')")


if __name__ == "__main__":
    run()
