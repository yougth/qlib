"""
V5修复未来函数 — 动态股票池回测
================================
问题: 原来用2025年数据选出的277只股票回测2021年,存在未来函数
修复: 每个回测窗口按年报披露滞后,只用当时可知的信息选股
规则: 回测年Y的股票池 = {股票 | max(Y-11, 2010)~Y-2年每年FCF>0且净利润>0}
     (年报Y-1在Y年4月底才披露, Y年初只能用到Y-2年报)

注意: profit_cache.csv从2016年开始, 早期窗口净利润检查年限不足10年
      fcf_cache.csv从1996年开始, FCF可查完整10年
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
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest as qlib_backtest
import warnings
warnings.filterwarnings("ignore")


def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def build_dynamic_universe(backtest_year, fcf_df, profit_df, min_years=10):
    """
    根据回测年份构建无未来信息的股票池
    回测年Y: 使用 Y-11~Y-2 的年报数据 (Y-1年报在Y年4月才披露)
    要求: 所有可用年份 FCF>0 且 净利润>0
    """
    # 可用年报年份: Y-11 到 Y-2 (10年窗口)
    # FCF数据从1996年开始, 净利润数据从2016年开始
    fcf_start = backtest_year - 11
    fcf_end = backtest_year - 2
    profit_start = max(backtest_year - 11, 2016)  # profit数据最早2016
    profit_end = backtest_year - 2

    target_fcf_years = list(range(fcf_start, fcf_end + 1))
    target_profit_years = list(range(profit_start, profit_end + 1))

    # FCF筛选: 所有目标年份FCF>0
    fcf_filtered = fcf_df[fcf_df["year"].isin(target_fcf_years)].copy()
    fcf_positive = fcf_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_fcf_years) * 0.8 and (g["fcf"] > 0).all()
    )
    fcf_codes = set(fcf_positive["code"].unique())

    # 净利润筛选: 所有目标年份净利润>0
    if profit_end >= 2016 and len(target_profit_years) > 0:
        profit_filtered = profit_df[profit_df["year"].isin(target_profit_years)].copy()
        profit_positive = profit_filtered.groupby("code").filter(
            lambda g: len(g) >= len(target_profit_years) * 0.8 and (g["net_profit"] > 0).all()
        )
        profit_codes = set(profit_positive["code"].unique())
    else:
        profit_codes = fcf_codes  # 无净利润数据时不筛选

    eligible_codes = fcf_codes & profit_codes
    return eligible_codes, len(target_fcf_years), len(target_profit_years)


def load_value_factors(universe, cal, cal_set):
    """加载价值因子 (复用V5逻辑)"""
    vf = pd.read_csv("/Users/11164591/Documents/Qoder目录/value_factors_cache.csv")
    vf["date"] = pd.to_datetime(vf["date"])
    ann = vf[vf["quarter"] == 12].sort_values(["code", "year"]).reset_index(drop=True)
    fin = {}
    for _, r in ann.iterrows():
        c = str(r["code"]).zfill(6); y = int(r["year"])
        if c not in fin: fin[c] = {}
        fin[c][y] = {"roe": r.get("roe", np.nan), "eps": r.get("eps", np.nan),
                     "bps": r.get("bps", np.nan), "div_payout": r.get("div_payout", np.nan)}
    price = D.features(list(universe), ["$close"], start_time=cal[0], end_time=cal[-1])
    if price is None or len(price) == 0: return None
    price = price.reset_index()
    price.columns = ["instrument", "datetime", "close"]
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
            w["pe"] = w["close"] / eps; w["pb"] = w["close"] / bps; w["roe_val"] = roe
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
                        hpe.extend((hw["close"] / he).tolist())
                        hpb.extend((hw["close"] / hb).tolist())
            if len(hpe) >= 50:
                hpe_a = np.array([x for x in hpe if 0 < x < 500])
                hpb_a = np.array([x for x in hpb if 0 < x < 50])
                if len(hpe_a) >= 30 and len(hpb_a) >= 30:
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


def load_fundamental_features(universe, cal):
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    universe_codes = {c[2:] for c in universe}
    fcf_df = fcf_df[fcf_df["code"].isin(universe_codes)]
    profit_df = profit_df[profit_df["code"].isin(universe_codes)]
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner")
    merged = merged.dropna(subset=["fcf", "net_profit"])
    merged = merged.sort_values(["code", "year"])
    rows = []
    for code, grp in merged.groupby("code"):
        qc = format_qlib_code(code)
        grp = grp.copy()
        grp["fcf_growth"] = grp["fcf"].pct_change()
        grp["profit_growth"] = grp["net_profit"].pct_change()
        grp["fcf_profit_ratio"] = grp["fcf"] / (grp["net_profit"].abs() + 1e-8)
        grp["fcf_avg_3y"] = grp["fcf"].rolling(3, min_periods=1).mean()
        grp["fcf_cv_3y"] = grp["fcf"].rolling(3, min_periods=1).std() / (grp["fcf"].rolling(3, min_periods=1).mean().abs() + 1e-8)
        for _, r in grp.iterrows():
            rows.append({"instrument": qc, "datetime": pd.Timestamp(f"{int(r['year'])}-12-31"),
                         "fcf": r["fcf"], "net_profit": r["net_profit"],
                         "fcf_growth": r["fcf_growth"], "profit_growth": r["profit_growth"],
                         "fcf_profit_ratio": r["fcf_profit_ratio"],
                         "fcf_avg_3y_norm": r["fcf_avg_3y"], "fcf_cv_3y": r["fcf_cv_3y"]})
    if not rows: return None
    fdf = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    fdf = fdf[~fdf.index.duplicated(keep="last")]
    return fdf


from v5_validation import Alpha158Enhanced, MonthlyTopkStrategy


def inject_features(dataset, fdf):
    h = dataset.handler; data = h.fetch()
    al = fdf.reindex(data.index)
    if isinstance(data.columns, pd.MultiIndex):
        al.columns = pd.MultiIndex.from_tuples([("feature", c) for c in al.columns])
    else:
        al.columns = [("feature", c) for c in al.columns]
    h._data = data.join(al).fillna(0)


def apply_value_fusion(pred, vf_daily, alpha=0.3):
    if vf_daily is None or not all(c in vf_daily.columns for c in ["roe_annual", "pb_pct_3y", "div_yield_est"]):
        return pred
    pred_fused = pred.copy()
    dates = pred_fused.index.get_level_values(0).unique()
    for dt in dates:
        pred_day = pred_fused.loc[[dt]] if dt in pred_fused.index.get_level_values(0) else None
        if pred_day is None or len(pred_day) == 0: continue
        vf_day = vf_daily.loc[[dt]] if dt in vf_daily.index.get_level_values(0) else None
        if vf_day is None or len(vf_day) == 0: continue
        common = pred_day.index.get_level_values(1).intersection(vf_day.index.get_level_values(1))
        if len(common) < 5: continue
        roe_vals = vf_day.set_index(vf_day.index.get_level_values(1)).loc[common, "roe_annual"]
        pb_val = vf_day.set_index(vf_day.index.get_level_values(1)).loc[common, "pb_pct_3y"]
        div_val = vf_day.set_index(vf_day.index.get_level_values(1)).loc[common, "div_yield_est"]
        def rn(s): return s.rank(pct=True).fillna(0.5)
        value_score = rn(roe_vals) + rn(-pb_val) + rn(div_val)
        value_score = (value_score - value_score.mean()) / (value_score.std() + 1e-8)
        lgb_score = pred_day.set_index(pred_day.index.get_level_values(1)).loc[common, "score"]
        lgb_norm = (lgb_score - lgb_score.mean()) / (lgb_score.std() + 1e-8)
        fused = (1 - alpha) * lgb_norm + alpha * value_score
        for inst in common:
            if inst in fused.index:
                pred_fused.loc[(dt, inst), "score"] = fused[inst]
    return pred_fused


MODEL_CONFIG = {
    "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
               "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
               "max_depth": 8, "num_leaves": 210, "num_threads": 4},
}

WINDOWS = [
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1", "bt_year": 2021},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2", "bt_year": 2022},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3", "bt_year": 2023},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4", "bt_year": 2024},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5", "bt_year": 2025},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6", "bt_year": 2026},
]


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # 加载财务数据
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    # 原始股票池 (有未来函数)
    orig_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    orig_universe = set(orig_df["code"].apply(format_qlib_code))

    print(f"原始股票池(有未来函数): {len(orig_universe)} 只")
    print(f"FCF数据: {fcf_df['code'].nunique()} 只, 年份 {fcf_df['year'].min()}~{fcf_df['year'].max()}")
    print(f"净利润数据: {profit_df['code'].nunique()} 只, 年份 {profit_df['year'].min()}~{profit_df['year'].max()}")

    cal = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # 构建每个窗口的动态股票池
    print(f"\n{'='*70}")
    print(f"  动态股票池构建 (无未来函数)")
    print(f"{'='*70}")
    window_universes = {}
    for w in WINDOWS:
        bt_year = w["bt_year"]
        codes, n_fcf_yr, n_prof_yr = build_dynamic_universe(bt_year, fcf_df, profit_df)
        qlib_codes = {format_qlib_code(c) for c in codes}
        # 只保留qlib中有的股票
        test_data = D.features(list(qlib_codes)[:5], ["$close"], start_time=w["backtest"][0], end_time=w["backtest"][1])
        if test_data is None or len(test_data) == 0:
            # 可能编码格式问题, 尝试全部
            qlib_codes = {c for c in qlib_codes if c.startswith("SH") or c.startswith("SZ")}
        window_universes[w["name"]] = qlib_codes
        overlap = len(qlib_codes & orig_universe)
        print(f"  {w['name']} (回测{bt_year}): FCF查{n_fcf_yr}年, 净利润查{n_prof_yr}年 → "
              f"{len(qlib_codes)}只 (与原始池重叠{overlap})")

    # 加载价值因子和基本面特征 (用最大股票池)
    all_codes = set()
    for s in window_universes.values():
        all_codes.update(s)
    all_universe = sorted(all_codes)
    print(f"\n  总不重复股票: {len(all_universe)}")

    print("  加载价值因子...")
    vf = load_value_factors(all_universe, cal, cal_set)
    print("  加载基本面特征...")
    fund = load_fundamental_features(all_universe, cal)

    # 逐窗口训练+回测
    all_daily_returns = []
    window_results = []

    for w in WINDOWS:
        universe = sorted(window_universes[w["name"]])
        ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
        print(f"\n{'='*60}")
        print(f"  {w['name']}: 股票池{len(universe)}只, train {ts}~{te}, backtest {bs}~{be}")
        print(f"{'='*60}")

        if len(universe) < 20:
            print(f"  股票池太小({len(universe)}), 跳过")
            continue

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
        with R.start(experiment_name=f"v5_nolookahead_{w['name']}"):
            rec = R.get_recorder()
            model.fit(dataset)
            sig_rec = SignalRecord(model, dataset, rec)
            sig_rec.generate()
            pred = rec.load_object("pred.pkl")

        # 动态α搜索
        best_alpha = 0.3
        best_ir = -999
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_a = apply_value_fusion(pred.copy(), vf, alpha=alpha)
            executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
                    "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
            strategy_config = {"class":"MonthlyTopkStrategy","module_path":"v5_validation",
                    "kwargs":{"topk":10,"n_drop":10,"signal":pred_a}}
            try:
                pm, _ = qlib_backtest(start_time=vs, end_time=ve, strategy=strategy_config,
                    executor=executor_config, account=100000000, benchmark=None,
                    exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
                        "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
                rn, _ = pm.get("1day", (None, None))
                if rn is not None:
                    ar = ((1 + rn["return"]).prod() ** (252/len(rn["return"])) - 1) * 100
                    ir = rn["return"].mean() / rn["return"].std() * np.sqrt(252) if rn["return"].std() > 0 else 0
                    if ir > best_ir:
                        best_ir = ir; best_alpha = alpha
            except:
                continue
        print(f"  最佳α: {best_alpha} (valid IR: {best_ir:.4f})")

        # 价值融合
        pred_dyn = apply_value_fusion(pred.copy(), vf, alpha=best_alpha)

        # 回测
        executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
                "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
        strategy_config = {"class":"MonthlyTopkStrategy","module_path":"v5_validation",
                "kwargs":{"topk":10,"n_drop":10,"signal":pred_dyn}}
        pm, _ = qlib_backtest(start_time=bs, end_time=be, strategy=strategy_config,
            executor=executor_config, account=100000000, benchmark=None,
            exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
                "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
        report_normal, _ = pm.get("1day", (None, None))
        if report_normal is None:
            print(f"  回测失败!")
            continue
        daily_ret = report_normal["return"].copy()
        daily_ret.index = pd.to_datetime(daily_ret.index)
        all_daily_returns.append(daily_ret)
        ar = ((1 + daily_ret).prod() ** (252/len(daily_ret)) - 1) * 100
        mdd = ((1 + daily_ret).cumprod() / (1 + daily_ret).cumprod().expanding().max() - 1).min() * 100
        ir = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        print(f"  年化: {ar:.2f}%, IR: {ir:.4f}, 回撤: {mdd:.2f}%, 天数: {len(daily_ret)}")
        window_results.append({"window": w["name"], "year": w["bt_year"], "ar": ar, "ir": ir, "mdd": mdd,
                               "n_stocks": len(universe)})

    # 汇总
    if all_daily_returns:
        v5_daily = pd.concat(all_daily_returns)
        v5_daily = v5_daily[~v5_daily.index.duplicated(keep="last")].sort_index()
        v5_daily.to_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_nolookahead_daily.csv")

        total_ret = (1 + v5_daily).prod() - 1
        n_days = len(v5_daily)
        ar = ((1 + total_ret) ** (252/n_days) - 1) * 100
        sharpe = v5_daily.mean() / v5_daily.std() * np.sqrt(252) if v5_daily.std() > 0 else 0
        cum = (1 + v5_daily).cumprod()
        mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min() * 100

        print(f"\n{'='*70}")
        print(f"  V5修复未来函数后 — 全周期结果")
        print(f"{'='*70}")
        print(f"  年化: {ar:.2f}%, 夏普: {sharpe:.4f}, 回撤: {mdd:.2f}%")
        print(f"  累计: {total_ret*100:.2f}%, 天数: {n_days}")

        print(f"\n  逐窗口:")
        print(f"  {'窗口':<6} {'年份':<6} {'股票数':<8} {'年化':>8} {'IR':>8} {'回撤':>8}")
        print(f"  {'-'*46}")
        for r in window_results:
            print(f"  {r['window']:<6} {r['year']:<6} {r['n_stocks']:<8} {r['ar']:>7.2f}% {r['ir']:>8.4f} {r['mdd']:>7.2f}%")

        # 逐年
        print(f"\n  逐年年化:")
        v5_df = pd.DataFrame({"datetime": v5_daily.index, "return": v5_daily.values})
        v5_df["year"] = v5_df["datetime"].dt.year
        for year, g in v5_df.groupby("year"):
            rets = g.set_index("datetime")["return"]
            tr = (1 + rets).prod() - 1
            yr_ar = ((1 + tr) ** (252/len(rets)) - 1) * 100 if tr > -1 else -100
            label = str(year) if year < 2026 else "2026H1"
            print(f"    {label}: {yr_ar:+.2f}%")

        # 对比原始(有未来函数)
        print(f"\n{'='*70}")
        print(f"  对比: 修复前 vs 修复后")
        print(f"{'='*70}")
        print(f"  {'指标':<16} {'修复前(有未来函数)':>20} {'修复后(无未来函数)':>20} {'差异':>12}")
        print(f"  {'-'*68}")
        print(f"  {'年化收益':<16} {'18.50':>19}% {ar:>19.2f}% {ar-18.50:>+11.2f}%")
        print(f"  {'夏普':<16} {'1.10':>20} {sharpe:>20.4f} {sharpe-1.10:>+12.4f}")
        print(f"  {'最大回撤':<16} {'-19.20':>19}% {mdd:>19.2f}% {mdd-(-19.20):>+11.2f}%")
        print(f"  {'股票池':<16} {'277(固定)':>20} {'动态变化':>20}")

        # 等权基准 (动态股票池)
        print(f"\n  计算动态股票池等权基准...")
        all_ew_returns = []
        for w in WINDOWS:
            universe = sorted(window_universes[w["name"]])
            bs, be = w["backtest"]
            prices = D.features(universe, ["$close"], start_time=bs, end_time=be)
            if prices is None or len(prices) == 0: continue
            prices = prices.reset_index()
            prices.columns = ["instrument", "datetime", "close"]
            prices = prices.sort_values(["instrument", "datetime"])
            prices["ret"] = prices.groupby("instrument")["close"].pct_change()
            daily_ret = prices.groupby("datetime")["ret"].mean().dropna()
            all_ew_returns.append(daily_ret)
        if all_ew_returns:
            ew_daily = pd.concat(all_ew_returns)
            ew_daily = ew_daily[~ew_daily.index.duplicated(keep="last")].sort_index()
            ew_ar = ((1 + ew_daily).prod() ** (252/len(ew_daily)) - 1) * 100
            ew_sharpe = ew_daily.mean() / ew_daily.std() * np.sqrt(252) if ew_daily.std() > 0 else 0
            ew_cum = (1 + ew_daily).cumprod()
            ew_mdd = ((ew_cum - ew_cum.expanding().max()) / ew_cum.expanding().max()).min() * 100

            # 超额分析
            common = v5_daily.index.intersection(ew_daily.index)
            s = v5_daily.loc[common]
            b = ew_daily.loc[common]
            excess = s - b
            ar_excess = ((1 + s).prod() / (1 + b).prod() - 1)
            ar_excess = ((1 + ar_excess) ** (252/len(excess)) - 1) * 100 if ar_excess > -1 else -100
            te = excess.std() * np.sqrt(252) * 100
            ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            beta, alpha_capm, _, p_val, _ = stats.linregress(b, s)
            monthly_excess = excess.resample("M").apply(lambda x: (1+x).prod()-1)
            win_rate = (monthly_excess > 0).sum() / len(monthly_excess) * 100

            print(f"\n  V5修复后 vs 动态等权基准:")
            print(f"  {'指标':<16} {'V5策略':>12} {'等权基准':>12} {'超额':>12}")
            print(f"  {'-'*52}")
            print(f"  {'年化收益':<16} {ar:>11.2f}% {ew_ar:>11.2f}% {ar_excess:>+11.2f}%")
            print(f"  {'夏普/IR':<16} {sharpe:>12.4f} {ew_sharpe:>12.4f} {ir:>12.4f}")
            print(f"  {'最大回撤':<16} {mdd:>11.2f}% {ew_mdd:>11.2f}%")
            print(f"  {'跟踪误差':<16} {'':>12} {'':>12} {te:>11.2f}%")
            print(f"  {'Alpha(年化)':<16} {'':>12} {'':>12} {alpha_capm*252*100:>+11.2f}%")
            print(f"  {'Beta':<16} {'':>12} {'':>12} {beta:>12.4f}")
            print(f"  {'月度胜率':<16} {'':>12} {'':>12} {win_rate:>11.1f}%")

            # 沪深300对比
            csi300 = {2021:-5.20, 2022:-21.63, 2023:-11.38, 2024:14.68, 2025:18.20, 2026:1.66}
            print(f"\n  V5修复后 vs 沪深300:")
            v5_yearly = {}
            for year, g in v5_df.groupby("year"):
                tr = (1 + g.set_index("datetime")["return"]).prod() - 1
                v5_yearly[year] = ((1+tr)**(252/len(g)) - 1) * 100 if tr > -1 else -100
            print(f"  {'年份':<8} {'V5修复':>10} {'沪深300':>10} {'超额':>10}")
            print(f"  {'-'*38}")
            for year in range(2021, 2027):
                v = v5_yearly.get(year, 0)
                c = csi300.get(year, 0)
                print(f"  {year:<8} {v:>+9.2f}% {c:>+9.2f}% {v-c:>+9.2f}%")

            # 上线评估
            print(f"\n  上线可行性评估:")
            checks = [
                ("超额年化", ar_excess, 5.0, ">"),
                ("信息比率", ir, 0.5, ">"),
                ("跟踪误差", te, 15.0, "<"),
                ("策略夏普", sharpe, 0.8, ">"),
                ("策略回撤", abs(mdd), 25.0, "<"),
                ("月度胜率", win_rate, 55.0, ">"),
            ]
            n_pass = 0
            for name, val, thresh, cmp in checks:
                ok = val > thresh if cmp == ">" else val < thresh
                if ok: n_pass += 1
                print(f"    {name}: {val:.2f} {'✓' if ok else '✗'} (阈值{cmp}{thresh})")
            print(f"\n  结论: {n_pass}/{len(checks)} 通过 → ", end="")
            if n_pass >= 5:
                print("✓ 建议上线")
            elif n_pass >= 4:
                print("⚠ 谨慎上线")
            else:
                print("✗ 暂不建议上线")

        pd.DataFrame(window_results).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_nolookahead_results.csv", index=False)
        print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
