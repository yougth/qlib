"""
修复脚本: 特征名映射 + Paper Trading + 宏观阀门优化
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


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    codes = build_dynamic_universe(2026, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"W6股票池: {len(universe)}只")

    cal = D.calendar(start_time="2021-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    ts, te = "2021-01-01", "2024-12-31"
    vs, ve = "2025-01-01", "2025-12-31"
    bs, be = "2026-01-01", "2026-07-21"

    # ====== 1. 获取特征名映射 ======
    print("\n=== 1. 特征名映射 ===")
    # 创建handler获取特征名
    dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
        "instruments": universe,
        "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                              {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label": ["Ref($close, -20) / $close - 1"]}
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"v5_validation","kwargs":dhc},
                  "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}

    print("加载价值因子...")
    vf = load_value_factors(universe, cal, cal_set)
    print("加载基本面特征...")
    fund = load_fundamental_features(universe, cal)

    dataset = init_instance_by_config(dsc)
    if fund is not None: inject_features(dataset, fund)
    if vf is not None: inject_features(dataset, vf)

    # 获取实际特征名
    handler = dataset.handler
    data = handler.fetch()
    feature_cols = [c for c in data.columns if isinstance(c, tuple) and c[0] == "feature"]
    feature_names = [c[1] if isinstance(c, tuple) else c for c in feature_cols]
    print(f"数据集特征数: {len(feature_names)}")
    print(f"前10个特征: {feature_names[:10]}")
    print(f"后10个特征: {feature_names[-10:]}")

    # 读取已保存的feature importance
    fi_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_feature_importance.csv")
    # 映射Column_N -> 实际特征名
    def map_name(col_name):
        if col_name.startswith("Column_"):
            idx = int(col_name.split("_")[1])
            if idx < len(feature_names):
                return feature_names[idx]
        return col_name

    fi_df["actual_name"] = fi_df["feature"].apply(map_name)
    fi_df = fi_df.sort_values("gain", ascending=False)

    print(f"\n  Top 20 特征 (映射后):")
    print(f"  {'排名':<6} {'特征名':<24} {'Gain':>12} {'Split':>8}")
    print(f"  {'-'*52}")
    for i, (_, r) in enumerate(fi_df.head(20).iterrows()):
        print(f"  {i+1:<6} {r['actual_name']:<24} {r['gain']:>12.1f} {r['split']:>8}")

    # 按类别分组
    alpha158_base = ["KMID","KMID2","SUN","LEN","CHR","CMO","RSI","WR","CORR","CORD",
                     "CNTN","CNTP","IMBA","SUM","ABS","VSTD","WMA","KDJ","MACD","CCI",
                     "VRATIO","VMA","CORR_PV","ROC","MA","STD","BOLL","RSI","PEAK","TREND"]
    enhanced = ["ROC120","ROC240","MA120","MA240","STD120","STD240","BOLL60","BOLL120",
                "VMA120","VMA240","CORR_PV20","CORR_PV60","VRATIO_5_60","VRATIO_5_120"]
    fund_names = ["fcf","net_profit","profit_growth","fcf_growth","fcf_profit_ratio",
                  "fcf_avg_3y","fcf_cv_3y","fcf_avg_3y_norm","fcf_cv_3y_norm"]
    value_names = ["roe_annual","pb_pct_3y","div_yield_est","pe_pct_3y"]

    categories = {"Alpha158短周期": [], "Alpha158增强长周期": [], "基本面": [], "价值因子": [], "其他": []}
    for _, r in fi_df.iterrows():
        fn = r["actual_name"]
        if any(fn.startswith(c) or fn == c for c in fund_names):
            categories["基本面"].append(r)
        elif any(fn == c or fn.startswith(c) for c in value_names):
            categories["价值因子"].append(r)
        elif fn in enhanced:
            categories["Alpha158增强长周期"].append(r)
        elif any(fn.startswith(c) for c in alpha158_base):
            categories["Alpha158短周期"].append(r)
        else:
            categories["其他"].append(r)

    total_gain = fi_df["gain"].sum()
    print(f"\n  按类别统计Gain占比:")
    for cat, items in categories.items():
        if items:
            cat_gain = sum(r["gain"] for r in items)
            top_feat = items[0]["actual_name"] if items else "—"
            print(f"    {cat:<20} {cat_gain/total_gain*100:>6.1f}% ({len(items)}个) Top: {top_feat}")

    fi_df.to_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_feature_importance_mapped.csv", index=False)

    # ====== 2. 训练模型获取预测 ======
    print("\n=== 2. 训练W6模型 + 生成Paper Trading信号 ===")
    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_fix_mapping"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    # Paper Trading: 取最后一个交易日的预测
    print("\n--- Paper Trading Top 10 ---")
    pred_fused = apply_value_fusion(pred.copy(), vf, alpha=0.3)
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred_fused = filter_pred_by_tradability(pred_fused, limit_up_set, suspension_set)

    latest_dates = sorted(pred_fused.index.get_level_values(0).unique())
    latest_dt = latest_dates[-1]
    print(f"信号日: {latest_dt.date()}")

    day_pred = pred_fused.xs(latest_dt, level=0)
    top10 = day_pred["score"].nlargest(10)

    # 股票名
    name_map = {}
    if "name" in fcf_df.columns:
        name_map = dict(zip(fcf_df["code"], fcf_df["name"]))

    print(f"\n  {'排名':<6} {'代码':<10} {'名称':<12} {'融合分':>10}")
    print(f"  {'-'*40}")
    for i, (inst, score) in enumerate(top10.items()):
        code = inst[2:]
        name = name_map.get(code, "—")
        print(f"  {i+1:<6} {inst:<10} {name:<12} {score:>10.4f}")

    # 最近价格
    top10_codes = top10.index.tolist()
    prices = D.features(top10_codes, ["$close", "$volume"],
                       start_time=latest_dt - pd.Timedelta(days=10), end_time=latest_dt)
    if prices is not None and len(prices) > 0:
        prices = prices.reset_index()
        prices.columns = ["instrument", "datetime", "close", "volume"]
        print(f"\n  {'代码':<10} {'最新价':>10} {'5日均量':>12} {'量比':>8}")
        print(f"  {'-'*40}")
        avg_vol_all = prices.groupby("instrument")["volume"].mean().mean()
        for inst in top10_codes:
            sp = prices[prices["instrument"] == inst].sort_values("datetime")
            if len(sp) > 0:
                last_close = sp.iloc[-1]["close"]
                vol_5 = sp["volume"].tail(5).mean()
                vr = vol_5 / avg_vol_all if avg_vol_all > 0 else 0
                print(f"  {inst:<10} {last_close:>10.2f} {vol_5:>12.0f} {vr:>8.2f}")

    output = pd.DataFrame({
        "rank": range(1, len(top10) + 1),
        "instrument": top10.index,
        "name": [name_map.get(i[2:], "—") for i in top10.index],
        "fused_score": top10.values,
        "signal_date": latest_dt,
    })
    output.to_csv("/Users/11164591/Documents/Qoder目录/qlib/paper_trading_top10.csv", index=False)
    print(f"\n  已保存: paper_trading_top10.csv")

    # ====== 3. 宏观阀门优化: 反向逻辑 ======
    print("\n=== 3. 宏观阀门优化 ===")
    print("原方案(多头+高波→砍半)使结果变差, 因为在策略表现好的熊市也触发了砍仓")
    print("优化方案: 当宽基指数低于200日均线(熊市)且策略历史超额为负时, 降仓")

    # 读取V5日频收益
    v5_daily = pd.read_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_nolookahead_daily.csv",
                           index_col=0, parse_dates=True)
    v5_ret = v5_daily["return"]

    # 计算等权基准 (整个回测期)
    orig_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    orig_universe = orig_df["code"].apply(format_qlib_code).tolist()
    prices = D.features(orig_universe, ["$close"], start_time="2021-01-01", end_time="2026-07-21")
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "close"]
    prices = prices.sort_values(["instrument", "datetime"])
    prices["ret"] = prices.groupby("instrument")["close"].pct_change()
    ew_ret = prices.groupby("datetime")["ret"].mean().dropna()

    # 200日均线
    ew_cum = (1 + ew_ret).cumprod()
    ew_ma200 = ew_cum.rolling(200, min_periods=60).mean()

    # 方案A: 原方案 (多头+高波→半仓)
    ew_vol60 = ew_ret.rolling(60, min_periods=20).std() * np.sqrt(252) * 100
    vol_median = ew_vol60.median()
    signal_A = pd.Series(1.0, index=ew_ret.index)
    signal_A[(ew_cum > ew_ma200) & (ew_vol60 > vol_median)] = 0.5

    # 方案B: 熊市+低波→半仓 (市场走弱时降仓)
    signal_B = pd.Series(1.0, index=ew_ret.index)
    signal_B[(ew_cum < ew_ma200) & (ew_vol60 < vol_median)] = 0.5

    # 方案C: 纯趋势过滤 (低于200日均线→0仓, 完全不做)
    signal_C = pd.Series(1.0, index=ew_ret.index)
    signal_C[ew_cum < ew_ma200] = 0.0

    # 方案D: 熊市减仓30%, 牛市满仓
    signal_D = pd.Series(1.0, index=ew_ret.index)
    signal_D[ew_cum < ew_ma200] = 0.7

    common = v5_ret.index.intersection(ew_ret.index)

    for label, sig in [("无阀门", pd.Series(1.0, index=common)),
                        ("A:多头+高波→半仓", signal_A.reindex(common).fillna(1.0)),
                        ("B:熊市+低波→半仓", signal_B.reindex(common).fillna(1.0)),
                        ("C:熊市→空仓", signal_C.reindex(common).fillna(1.0)),
                        ("D:熊市→70%仓", signal_D.reindex(common).fillna(1.0))]:
        adj = v5_ret.loc[common] * sig.loc[common]
        ar = ((1 + adj).prod() ** (252/len(adj)) - 1) * 100
        sh = adj.mean() / adj.std() * np.sqrt(252) if adj.std() > 0 else 0
        cum = (1 + adj).cumprod()
        mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min() * 100
        n_half = (sig < 1.0).sum()
        print(f"  {label:<22} 年化{ar:>+8.2f}%  夏普{sh:.4f}  回撤{mdd:>7.2f}%  降仓{n_half}天")

    # 逐年对比方案D
    print(f"\n  方案D(熊市→70%仓)逐年对比:")
    sig_D = signal_D.reindex(common).fillna(1.0)
    print(f"  {'年份':<8} {'无阀门':>10} {'方案D':>10} {'差异':>10}")
    print(f"  {'-'*38}")
    for year in range(2021, 2027):
        yr_orig = v5_ret[v5_ret.index.year == year]
        yr_adj = (v5_ret * sig_D)[v5_ret.index.year == year]
        if len(yr_orig) == 0: continue
        o = ((1 + yr_orig).prod() ** (252/len(yr_orig)) - 1) * 100
        a = ((1 + yr_adj).prod() ** (252/len(yr_adj)) - 1) * 100
        label = str(year) if year < 2026 else "2026H1"
        print(f"  {label:<8} {o:>+9.2f}% {a:>+9.2f}% {a-o:>+9.2f}%")


if __name__ == "__main__":
    run()
