"""
V5策略衰减分析 + 拉长回测
===========================
1. 分析2021-2026衰减原因 (滚动超额/regime分析)
2. 拉长回测至2018-2020 (额外2个窗口验证)
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
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.backtest import backtest as qlib_backtest
import warnings
warnings.filterwarnings("ignore")

from v5_validation import Alpha158Enhanced, MonthlyTopkStrategy, \
    apply_value_fusion, inject_features, build_limit_up_set, filter_pred_by_tradability


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    fcf_start = backtest_year - 11
    fcf_end = backtest_year - 2
    profit_start = max(backtest_year - 11, 2016)
    profit_end = backtest_year - 2
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
    return fcf_codes & profit_codes, len(target_fcf_years), len(target_profit_years)


def load_value_factors(universe, start, end):
    vf = pd.read_csv("/Users/11164591/Documents/Qoder目录/value_factors_cache.csv")
    vf["date"] = pd.to_datetime(vf["date"])
    ann = vf[vf["quarter"] == 12].sort_values(["code", "year"]).reset_index(drop=True)
    fin = {}
    for _, r in ann.iterrows():
        c = str(r["code"]).zfill(6); y = int(r["year"])
        if c not in fin: fin[c] = {}
        fin[c][y] = {"roe": r.get("roe", np.nan), "eps": r.get("eps", np.nan),
                     "bps": r.get("bps", np.nan), "div_payout": r.get("div_payout", np.nan)}
    price = D.features(list(universe), ["$close"], start_time=start, end_time=end)
    if price is None or len(price) == 0: return None
    price = price.reset_index()
    price.columns = ["instrument", "datetime", "close"]
    cal_set = set(price["datetime"].unique())
    rows = []
    for qc in universe:
        cs = qc[2:]
        if cs not in fin: continue
        f = fin[cs]
        sp = price[price["instrument"] == qc].sort_values("datetime").set_index("datetime")
        if len(sp) == 0: continue
        years = sorted(f.keys())
        for i, y in enumerate(years):
            eps, bps, roe, dp = f[y]["eps"], f[y]["bps"], f[y]["roe"], f[y]["div_payout"]
            if pd.isna(eps) or eps == 0 or pd.isna(bps) or bps == 0: continue
            af = pd.Timestamp(f"{y+1}-05-01")
            at = pd.Timestamp(f"{years[i+1]+1}-04-30") if i+1 < len(years) else pd.Timestamp(f"{y+2}-04-30")
            w = sp[(sp.index >= af) & (sp.index <= at)].copy()
            if len(w) == 0: continue
            w["pb"] = w["close"] / bps; w["roe_val"] = roe
            w["div_yield"] = np.nan
            if not pd.isna(dp) and eps != 0:
                w["div_yield"] = (dp / 100.0) * eps / w["close"]
            hpe, hpb = [], []
            for hy in range(y - 3, y):
                if hy in fin:
                    he, hb = fin[hy]["eps"], fin[hy]["bps"]
                    if pd.isna(he) or he == 0 or pd.isna(hb) or hb == 0: continue
                    hf, ht = pd.Timestamp(f"{hy+1}-05-01"), pd.Timestamp(f"{hy+2}-04-30")
                    hw = sp[(sp.index >= hf) & (sp.index <= ht)]
                    if len(hw) > 0:
                        hpb.extend((hw["close"] / hb).tolist())
            if len(hpb) >= 30:
                hpb_a = np.array([x for x in hpb if 0 < x < 50])
                if len(hpb_a) >= 30:
                    hs_b = np.sort(hpb_a)
                    w["pb_pct_3y"] = w["pb"].apply(lambda x: np.searchsorted(hs_b, x) / len(hpb_a) if 0 < x < 50 else np.nan)
                else:
                    w["pb_pct_3y"] = np.nan
            else:
                w["pb_pct_3y"] = np.nan
            for dt, r in w.iterrows():
                if dt not in cal_set: continue
                rows.append({"instrument": qc, "datetime": dt,
                             "roe_annual": r["roe_val"], "pb_pct_3y": r["pb_pct_3y"],
                             "div_yield_est": r["div_yield"]})
    if not rows: return None
    vfd = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    vfd = vfd[~vfd.index.duplicated(keep="last")]
    for c in vfd.columns:
        g = vfd[c].groupby(level=0)
        med = g.transform("median")
        mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        vfd[c] = ((vfd[c] - med) / (1.4826 * mad)).clip(-3, 3).fillna(0)
    return vfd


def load_fund_features(universe, cal):
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    universe_codes = {c[2:] for c in universe}
    fcf_df = fcf_df[fcf_df["code"].isin(universe_codes)]
    profit_df = profit_df[profit_df["code"].isin(universe_codes)]
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner")
    merged = merged.dropna(subset=["fcf", "net_profit"]).sort_values(["code", "year"])
    rows = []
    for code, grp in merged.groupby("code"):
        qc = format_qlib_code(code)
        grp = grp.copy()
        grp["fcf_growth"] = grp["fcf"].pct_change()
        grp["profit_growth"] = grp["net_profit"].pct_change()
        grp["fcf_profit_ratio"] = grp["fcf"] / (grp["net_profit"].abs() + 1e-8)
        for _, r in grp.iterrows():
            rows.append({"instrument": qc, "datetime": pd.Timestamp(f"{int(r['year'])}-12-31"),
                         "fcf": r["fcf"], "net_profit": r["net_profit"],
                         "fcf_growth": r["fcf_growth"], "profit_growth": r["profit_growth"],
                         "fcf_profit_ratio": r["fcf_profit_ratio"]})
    if not rows: return None
    fdf = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    return fdf[~fdf.index.duplicated(keep="last")]


MODEL_CONFIG = {
    "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
               "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
               "max_depth": 8, "num_leaves": 210, "num_threads": 4},
}

# 所有窗口 (含扩展)
ALL_WINDOWS = [
    {"train": ("2014-01-01","2017-12-31"), "valid": ("2018-01-01","2018-12-31"), "backtest": ("2019-01-01","2019-12-31"), "name":"W0", "bt_year": 2019},
    {"train": ("2015-01-01","2018-12-31"), "valid": ("2019-01-01","2019-12-31"), "backtest": ("2020-01-01","2020-12-31"), "name":"W0b", "bt_year": 2020},
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1", "bt_year": 2021},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2", "bt_year": 2022},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3", "bt_year": 2023},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4", "bt_year": 2024},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5", "bt_year": 2025},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "bt_year": 2026},
]


def run_window(w, universe, vf, fund, cal, cal_set):
    """训练并回测单个窗口"""
    ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
    print(f"  {w['name']}: 股票池{len(universe)}只, backtest {bs[:4]}")

    # 涨跌停集合
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)

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

    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name=f"v5_decay_{w['name']}"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    # α=0.3 固定 (不再搜索, 加速)
    pred_fused = apply_value_fusion(pred.copy(), vf, alpha=0.3)
    pred_fused = filter_pred_by_tradability(pred_fused, limit_up_set, suspension_set)

    executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
            "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    strategy_config = {"class":"MonthlyTopkStrategy","module_path":"v5_validation",
            "kwargs":{"topk":10,"n_drop":10,"signal":pred_fused}}
    pm, _ = qlib_backtest(start_time=bs, end_time=be, strategy=strategy_config,
        executor=executor_config, account=100000000, benchmark=None,
        exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
            "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
    report, _ = pm.get("1day", (None, None))
    if report is None:
        print(f"    回测失败!")
        return None, None
    daily_ret = report["return"].copy()
    daily_ret.index = pd.to_datetime(daily_ret.index)
    ar = ((1 + daily_ret).prod() ** (252/len(daily_ret)) - 1) * 100
    ir = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    mdd = ((1 + daily_ret).cumprod() / (1 + daily_ret).cumprod().expanding().max() - 1).min() * 100
    print(f"    年化: {ar:.2f}%, IR: {ir:.4f}, 回撤: {mdd:.2f}%")
    return daily_ret, {"window": w["name"], "year": w["bt_year"], "ar": ar, "ir": ir, "mdd": mdd,
                       "n_stocks": len(universe)}


def compute_ew_benchmark(universe, start, end):
    prices = D.features(universe, ["$close"], start_time=start, end_time=end)
    if prices is None or len(prices) == 0: return None
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "close"]
    prices = prices.sort_values(["instrument", "datetime"])
    prices["ret"] = prices.groupby("instrument")["close"].pct_change()
    return prices.groupby("datetime")["ret"].mean().dropna()


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = D.calendar(start_time="2014-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # ====== Part 1: 衰减分析 (已有数据) ======
    print(f"\n{'='*70}")
    print(f"  Part 1: V5策略衰减分析 (2021-2026)")
    print(f"{'='*70}")

    v5_daily = pd.read_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_nolookahead_daily.csv", index_col=0, parse_dates=True)
    v5_ret = v5_daily["return"]

    # 沪深300年度收益
    csi300_annual = {2019: 36.07, 2020: 27.21, 2021: -5.20, 2022: -21.63,
                     2023: -11.38, 2024: 14.68, 2025: 18.20, 2026: 1.66}

    # 计算滚动6个月超额收益 (vs 等权基准)
    # 先计算等权基准
    print("  计算等权基准...")
    orig_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    orig_universe = orig_df["code"].apply(format_qlib_code).tolist()
    ew_ret = compute_ew_benchmark(orig_universe, "2021-01-01", "2026-07-21")

    # 滚动6个月超额
    common = v5_ret.index.intersection(ew_ret.index)
    excess = v5_ret.loc[common] - ew_ret.loc[common]
    rolling_excess = excess.rolling(126).apply(lambda x: (1 + x).prod() - 1) * 100  # 6个月滚动

    print(f"\n  滚动6个月超额收益 (vs 等权基准):")
    print(f"  {'月份':<12} {'滚动超额':>10} {'趋势':>6}")
    print(f"  {'-'*30}")
    for dt in pd.date_range("2021-06-30", "2026-06-30", freq="6M"):
        if dt in rolling_excess.index:
            val = rolling_excess.loc[dt]
            if pd.notna(val):
                trend = "↑" if val > 5 else ("↓" if val < -5 else "→")
                print(f"  {dt.strftime('%Y-%m'):<12} {val:>+9.1f}% {trend:>6}")

    # 市场regime分析: V5在牛市/熊市的表现
    print(f"\n  市场Regime分析:")
    print(f"  {'年份':<8} {'V5收益':>10} {'CSI300':>10} {'超额':>10} {'市场':>6} {'结论':>8}")
    print(f"  {'-'*54}")
    v5_yearly = {}
    for year in range(2021, 2027):
        yr_ret = v5_ret[(v5_ret.index >= pd.Timestamp(f"{year}-01-01")) &
                        (v5_ret.index < pd.Timestamp(f"{year+1}-01-01"))] if year < 2026 else \
                 v5_ret[v5_ret.index >= pd.Timestamp("2026-01-01")]
        if len(yr_ret) == 0: continue
        tr = (1 + yr_ret).prod() - 1
        ar = ((1 + tr) ** (252/len(yr_ret)) - 1) * 100 if tr > -1 else -100
        v5_yearly[year] = ar
        csi = csi300_annual.get(year, 0)
        ex = ar - csi
        regime = "熊市" if csi < 0 else ("牛市" if csi > 15 else "震荡")
        verdict = "跑赢" if ex > 0 else "跑输"
        label = str(year) if year < 2026 else "2026H1"
        print(f"  {label:<8} {ar:>+9.2f}% {csi:>+9.2f}% {ex:>+9.2f}% {regime:>6} {verdict:>8}")

    # 检查: V5超额与CSI300收益的相关性
    years_data = [(y, v5_yearly.get(y, 0) - csi300_annual.get(y, 0), csi300_annual.get(y, 0))
                  for y in range(2021, 2027) if y in v5_yearly]
    if len(years_data) >= 3:
        excess_vals = [x[1] for x in years_data]
        csi_vals = [x[2] for x in years_data]
        corr = np.corrcoef(csi_vals, excess_vals)[0, 1]
        print(f"\n  CSI300收益与V5超额收益的相关系数: {corr:.4f}")
        if corr < -0.3:
            print(f"  → 负相关: 熊市超额大, 牛市超额小 (策略偏防御型)")
        elif corr > 0.3:
            print(f"  → 正相关: 牛市超额大, 熊市超额小 (策略偏进攻型)")
        else:
            print(f"  → 弱相关: 超额与市场涨跌关系不大")

    # ====== Part 2: 拉长回测至2019-2020 ======
    print(f"\n{'='*70}")
    print(f"  Part 2: 拉长回测至2019-2020")
    print(f"{'='*70}")

    # 构建所有窗口的动态股票池
    all_universes = {}
    for w in ALL_WINDOWS:
        codes, n_fcf, n_prof = build_dynamic_universe(w["bt_year"], fcf_df, profit_df)
        qlib_codes = {format_qlib_code(c) for c in codes}
        all_universes[w["name"]] = qlib_codes
        print(f"  {w['name']} (回测{w['bt_year']}): FCF查{n_fcf}年, 净利润查{n_prof}年 → {len(qlib_codes)}只")

    # 加载VF和Fund (用最大股票池)
    all_codes = set()
    for s in all_universes.values():
        all_codes.update(s)
    all_universe = sorted(all_codes)
    print(f"\n  总不重复股票: {len(all_universe)}")

    print("  加载价值因子...")
    vf = load_value_factors(all_universe, "2014-01-01", "2026-07-21")
    print("  加载基本面特征...")
    fund = load_fund_features(all_universe, cal)

    # 只跑新增窗口 W0, W0b
    new_windows = [w for w in ALL_WINDOWS if w["name"] in ("W0", "W0b")]
    new_results = []
    new_daily = []

    for w in new_windows:
        universe = sorted(all_universes[w["name"]])
        print(f"\n  {w['name']}: 股票池{len(universe)}只, backtest {w['bt_year']}")
        if len(universe) < 20:
            print(f"  股票池太小, 跳过")
            continue
        daily_ret, result = run_window(w, universe, vf, fund, cal, cal_set)
        if daily_ret is not None:
            new_daily.append(daily_ret)
            new_results.append(result)

    # 已有窗口结果 (从上次运行)
    existing_results = [
        {"window": "W1", "year": 2021, "ar": 37.84, "ir": 1.4552, "mdd": -14.48, "n_stocks": 166},
        {"window": "W2", "year": 2022, "ar": 31.80, "ir": 1.2364, "mdd": -17.79, "n_stocks": 199},
        {"window": "W3", "year": 2023, "ar": 1.36, "ir": 0.1648, "mdd": -16.87, "n_stocks": 226},
        {"window": "W4", "year": 2024, "ar": 14.41, "ir": 0.7439, "mdd": -20.34, "n_stocks": 228},
        {"window": "W5", "year": 2025, "ar": 24.12, "ir": 1.5417, "mdd": -7.17, "n_stocks": 274},
        {"window": "W6", "year": 2026, "ar": -13.96, "ir": -0.7397, "mdd": -13.75, "n_stocks": 303},
    ]

    all_results = new_results + existing_results
    all_results.sort(key=lambda x: x["year"])

    # ====== Part 3: 完整时间序列分析 ======
    print(f"\n{'='*70}")
    print(f"  Part 3: 完整8年回测结果 (2019-2026)")
    print(f"{'='*70}")

    print(f"\n  {'年份':<8} {'股票数':<8} {'V5年化':>10} {'CSI300':>10} {'超额':>10} {'IR':>8} {'回撤':>8} {'市场':>6}")
    print(f"  {'-'*72}")
    for r in all_results:
        csi = csi300_annual.get(r["year"], 0)
        ex = r["ar"] - csi
        regime = "熊市" if csi < 0 else ("牛市" if csi > 15 else "震荡")
        label = str(r["year"]) if r["year"] < 2026 else "2026H1"
        print(f"  {label:<8} {r['n_stocks']:<8} {r['ar']:>+9.2f}% {csi:>+9.2f}% {ex:>+9.2f}% {r['ir']:>8.4f} {r['mdd']:>7.2f}% {regime:>6}")

    # 分阶段统计
    early = [r for r in all_results if r["year"] <= 2020]
    mid = [r for r in all_results if 2021 <= r["year"] <= 2022]
    late = [r for r in all_results if r["year"] >= 2023]

    print(f"\n  分阶段统计:")
    print(f"  {'阶段':<16} {'平均年化':>10} {'平均超额':>10} {'平均IR':>10} {'胜率':>8}")
    print(f"  {'-'*54}")
    for label, group in [("2019-2020(扩展)", early), ("2021-2022(强)", mid), ("2023-2026(弱)", late)]:
        if not group: continue
        avg_ar = np.mean([r["ar"] for r in group])
        avg_ex = np.mean([r["ar"] - csi300_annual.get(r["year"], 0) for r in group])
        avg_ir = np.mean([r["ir"] for r in group])
        wins = sum(1 for r in group if r["ar"] - csi300_annual.get(r["year"], 0) > 0)
        print(f"  {label:<16} {avg_ar:>+9.2f}% {avg_ex:>+9.2f}% {avg_ir:>10.4f} {wins}/{len(group)}")

    # 趋势分析
    print(f"\n  趋势分析:")
    years = [r["year"] for r in all_results]
    ars = [r["ar"] for r in all_results]
    excess = [r["ar"] - csi300_annual.get(r["year"], 0) for r in all_results]

    if len(years) >= 4:
        slope_ar, _, r_val_ar, _, _ = stats.linregress(years, ars)
        slope_ex, _, r_val_ex, _, _ = stats.linregress(years, excess)
        print(f"  年化收益趋势: 斜率={slope_ar:.2f}%/年, R²={r_val_ar**2:.4f}")
        print(f"  超额收益趋势: 斜率={slope_ex:.2f}%/年, R²={r_val_ex**2:.4f}")
        if slope_ex < -3:
            print(f"  → 超额收益每年下降约{abs(slope_ex):.1f}%, 呈明显衰减趋势")
        elif slope_ex < 0:
            print(f"  → 超额收益轻微下降, 衰减不显著")
        else:
            print(f"  → 超额收益未呈现衰减趋势")

    # ====== 结论 ======
    print(f"\n{'='*70}")
    print(f"  结论")
    print(f"{'='*70}")

    # 判断衰减原因
    bear_excess = [r["ar"] - csi300_annual.get(r["year"], 0) for r in all_results
                   if csi300_annual.get(r["year"], 0) < 0]
    bull_excess = [r["ar"] - csi300_annual.get(r["year"], 0) for r in all_results
                   if csi300_annual.get(r["year"], 0) > 15]
    range_excess = [r["ar"] - csi300_annual.get(r["year"], 0) for r in all_results
                    if 0 <= csi300_annual.get(r["year"], 0) <= 15]

    if bear_excess and bull_excess:
        print(f"\n  1. 策略表现与市场regime高度相关:")
        print(f"     熊市平均超额: +{np.mean(bear_excess):.1f}% ({len(bear_excess)}年)")
        if range_excess:
            print(f"     震荡市平均超额: +{np.mean(range_excess):.1f}% ({len(range_excess)}年)")
        print(f"     牛市平均超额: +{np.mean(bull_excess):.1f}% ({len(bull_excess)}年)")
        print(f"     → 策略本质是'熊市防御型', 熊市大幅跑赢, 牛市优势缩小")

    if len(years) >= 4 and slope_ex < 0:
        print(f"\n  2. 超额收益确实在衰减:")
        print(f"     每年平均下降{abs(slope_ex):.1f}个百分点")
        print(f"     但这主要因为2021-2022是极端熊市(超额+43%/+53%)")
        print(f"     去掉2021-2022后, 衰减趋势是否仍然存在?")

        # 去掉2021-2022后重新计算趋势
        remaining = [(y, e) for y, e in zip(years, excess) if y not in (2021, 2022)]
        if len(remaining) >= 3:
            ry, re = zip(*remaining)
            slope2, _, rv2, _, _ = stats.linregress(ry, re)
            print(f"     去掉2021-2022后: 斜率={slope2:.2f}%/年, R²={rv2**2:.4f}")
            if abs(slope2) < 3:
                print(f"     → 衰减不显著, 原始趋势主要被2021-2022的极端值驱动")

    print(f"\n  3. 后续展望:")
    late_excess = [r["ar"] - csi300_annual.get(r["year"], 0) for r in late]
    if late_excess:
        avg_late = np.mean(late_excess)
        print(f"     2023-2026平均超额: +{avg_late:.1f}%")
        if avg_late > 5:
            print(f"     → 即使在'弱'周期, 仍有正超额, 策略未完全失效")
        elif avg_late > 0:
            print(f"     → '弱'周期超额接近0, 策略alpha在收窄")
        else:
            print(f"     → '弱'周期超额转负, 策略可能已失效")

    pd.DataFrame(all_results).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_extended_results.csv", index=False)
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
