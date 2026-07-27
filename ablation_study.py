"""
消融实验 — 隔离窗口/TopK/价值因子各自的影响
=============================================
A: 5年 + Top10 + 无价值因子 (V2 baseline, 已有结果)
B: 3年 + Top10 + 无价值因子 → 隔离窗口效应
C: 5年 + Top30 + 无价值因子 → 隔离TopK效应
D: 3年 + Top10 + 有价值因子 → 隔离价值因子效应
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.data.handler import Alpha158
from qlib.data import D


class MonthlyTopkStrategy(TopkDropoutStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_rebalance_period = None
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        cur_start, _ = self.trade_calendar.get_step_time(trade_step)
        cur_date = pd.Timestamp(cur_start)
        cur_period = cur_date.to_period("M")
        should_rebalance = False
        try:
            next_start, _ = self.trade_calendar.get_step_time(trade_step + 1)
            next_date = pd.Timestamp(next_start)
            if next_date.to_period("M") != cur_period:
                should_rebalance = True
        except Exception:
            should_rebalance = True
        if self._last_rebalance_period is None:
            should_rebalance = True
        if not should_rebalance:
            return TradeDecisionWO([], self)
        self._last_rebalance_period = cur_period
        return super().generate_trade_decision(execute_result)


class Alpha158Enhanced(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields = [
            "Ref($close, 120)/$close", "Ref($close, 240)/$close",
            "Mean($close, 120)/$close", "Mean($close, 240)/$close",
            "Std($close, 120)/$close", "Std($close, 240)/$close",
            "($close - Mean($close, 60))/(Std($close, 60)+1e-12)",
            "($close - Mean($close, 120))/(Std($close, 120)+1e-12)",
            "Mean($volume, 120)/($volume+1e-12)",
            "Mean($volume, 240)/($volume+1e-12)",
            "Corr($close, Log($volume+1), 20)",
            "Corr($close, Log($volume+1), 60)",
            "Mean($volume, 5)/(Mean($volume, 60)+1e-12)",
            "Mean($volume, 5)/(Mean($volume, 120)+1e-12)",
        ]
        extra_names = [
            "ROC120", "ROC240", "MA120", "MA240", "STD120", "STD240",
            "BOLL60", "BOLL120", "VMA120", "VMA240",
            "CORR_PV20", "CORR_PV60", "VRATIO_5_60", "VRATIO_5_120",
        ]
        return fields + extra_fields, names + extra_names


def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"


def load_fundamental_features(universe, qlib_calendar):
    fcf_path = "/Users/11164591/Documents/Qoder目录/fcf_cache.csv"
    profit_path = "/Users/11164591/Documents/Qoder目录/profit_cache.csv"
    fcf_df = pd.read_csv(fcf_path)
    profit_df = pd.read_csv(profit_path)
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner")
    merged = merged.dropna(subset=["fcf", "net_profit"])
    merged = merged.sort_values(["code", "year"]).reset_index(drop=True)
    feature_rows = []
    for code, grp in merged.groupby("code"):
        code_str = str(code).zfill(6)
        qlib_code = f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"
        if qlib_code not in universe:
            continue
        grp = grp.sort_values("year").copy()
        grp["fcf_growth"] = grp["fcf"].pct_change()
        grp["profit_growth"] = grp["net_profit"].pct_change()
        grp["fcf_profit_ratio"] = grp["fcf"] / (grp["net_profit"].abs() + 1e-8)
        grp["fcf_avg_3y"] = grp["fcf"].rolling(3, min_periods=1).mean()
        rolling_std = grp["fcf"].rolling(3, min_periods=2).std()
        rolling_mean = grp["fcf"].rolling(3, min_periods=2).mean().abs()
        grp["fcf_cv_3y"] = rolling_std / (rolling_mean + 1e-8)
        for _, row in grp.iterrows():
            if pd.isna(row["fcf_growth"]): continue
            year = int(row["year"])
            avail_from = pd.Timestamp(f"{year + 1}-05-01")
            avail_to = pd.Timestamp(f"{year + 2}-04-30")
            mask = (qlib_calendar >= avail_from) & (qlib_calendar <= avail_to)
            dates = qlib_calendar[mask]
            if len(dates) == 0: continue
            for d in dates:
                feature_rows.append({"instrument": qlib_code, "datetime": d,
                    "fcf_growth": row["fcf_growth"], "profit_growth": row["profit_growth"],
                    "fcf_profit_ratio": row["fcf_profit_ratio"],
                    "fcf_avg_3y_norm": row["fcf_avg_3y"] / 1e8, "fcf_cv_3y": row["fcf_cv_3y"]})
    if not feature_rows: return None
    fund_df = pd.DataFrame(feature_rows).set_index(["datetime", "instrument"])
    fund_df = fund_df[~fund_df.index.duplicated(keep="last")]
    for col in fund_df.columns:
        median = fund_df[col].median()
        mad = (fund_df[col] - median).abs().median()
        if mad > 0: fund_df[col] = (fund_df[col] - median) / (1.4826 * mad)
        fund_df[col] = fund_df[col].clip(-3, 3).fillna(0)
    return fund_df


def load_value_factors(universe, qlib_calendar, qlib_calendar_set):
    vf_path = "/Users/11164591/Documents/Qoder目录/value_factors_cache.csv"
    vf_df = pd.read_csv(vf_path)
    vf_df["date"] = pd.to_datetime(vf_df["date"])
    annual = vf_df[vf_df["quarter"] == 12].sort_values(["code", "year"]).reset_index(drop=True)
    fin_data = {}
    for _, row in annual.iterrows():
        code = str(row["code"]).zfill(6)
        year = int(row["year"])
        if code not in fin_data: fin_data[code] = {}
        fin_data[code][year] = {"roe": row.get("roe", np.nan), "eps": row.get("eps", np.nan),
            "bps": row.get("bps", np.nan), "div_payout": row.get("div_payout", np.nan)}
    price_df = D.features(list(universe), ["$close"], start_time=qlib_calendar[0], end_time=qlib_calendar[-1])
    if price_df is None or len(price_df) == 0: return None
    price_df = price_df.reset_index()
    price_df.columns = ["datetime", "instrument", "close"]
    feature_rows = []
    processed = 0
    for qlib_code in universe:
        code_short = qlib_code[2:]
        if code_short not in fin_data: continue
        fin = fin_data[code_short]
        sp = price_df[price_df["instrument"] == qlib_code].sort_values("datetime").set_index("datetime")
        if len(sp) == 0: continue
        years = sorted(fin.keys())
        for i, year in enumerate(years):
            eps, bps = fin[year]["eps"], fin[year]["bps"]
            roe = fin[year]["roe"]
            div_payout = fin[year]["div_payout"]
            if pd.isna(eps) or eps == 0 or pd.isna(bps) or bps == 0: continue
            af = pd.Timestamp(f"{year + 1}-05-01")
            at = pd.Timestamp(f"{years[i+1] + 1}-04-30") if i + 1 < len(years) else pd.Timestamp(f"{year + 2}-04-30")
            w = sp[(sp.index >= af) & (sp.index <= at)].copy()
            if len(w) == 0: continue
            w["pe"] = w["close"] / eps
            w["pb"] = w["close"] / bps
            w["roe_val"] = roe
            w["div_yield"] = np.nan
            if not pd.isna(div_payout) and eps != 0:
                w["div_yield"] = (div_payout / 100.0) * eps / w["close"]
            hist_pe, hist_pb = [], []
            for hy in range(year - 3, year):
                if hy in fin:
                    he, hb = fin[hy]["eps"], fin[hy]["bps"]
                    if pd.isna(he) or he == 0 or pd.isna(hb) or hb == 0: continue
                    hf = pd.Timestamp(f"{hy + 1}-05-01")
                    ht = pd.Timestamp(f"{hy + 2}-04-30")
                    hw = sp[(sp.index >= hf) & (sp.index <= ht)]
                    if len(hw) > 0:
                        hist_pe.extend((hw["close"] / he).tolist())
                        hist_pb.extend((hw["close"] / hb).tolist())
            if len(hist_pe) >= 50:
                hpe = np.array([x for x in hist_pe if 0 < x < 500])
                hpb = np.array([x for x in hist_pb if 0 < x < 50])
                if len(hpe) >= 30 and len(hpb) >= 30:
                    hpe_s, hpb_s = np.sort(hpe), np.sort(hpb)
                    w["pe_pct_3y"] = w["pe"].apply(lambda x: np.searchsorted(hpe_s, x)/len(hpe_s) if 0<x<500 else np.nan)
                    w["pb_pct_3y"] = w["pb"].apply(lambda x: np.searchsorted(hpb_s, x)/len(hpb_s) if 0<x<50 else np.nan)
                else: w["pe_pct_3y"], w["pb_pct_3y"] = np.nan, np.nan
            else: w["pe_pct_3y"], w["pb_pct_3y"] = np.nan, np.nan
            for dt, row in w.iterrows():
                if dt not in qlib_calendar_set: continue
                feature_rows.append({"instrument": qlib_code, "datetime": dt,
                    "roe_annual": row["roe_val"], "pe_pct_3y": row["pe_pct_3y"],
                    "pb_pct_3y": row["pb_pct_3y"], "div_yield_est": row["div_yield"]})
        processed += 1
        if processed % 50 == 0: print(f"    价值因子: {processed}/{len(universe)}")
    if not feature_rows: return None
    vf_daily = pd.DataFrame(feature_rows).set_index(["datetime", "instrument"])
    vf_daily = vf_daily[~vf_daily.index.duplicated(keep="last")]
    for col in vf_daily.columns:
        grp = vf_daily[col].groupby(level=0)
        median = grp.transform("median")
        mad = grp.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        vf_daily[col] = ((vf_daily[col] - median) / (1.4826 * mad)).clip(-3, 3).fillna(0)
    return vf_daily


def inject_features(dataset, feature_df):
    handler = dataset.handler
    data = handler.fetch()
    handler_index = data.index
    aligned = feature_df.reindex(handler_index)
    if isinstance(data.columns, pd.MultiIndex):
        aligned.columns = pd.MultiIndex.from_tuples([("feature", c) for c in aligned.columns])
    else:
        aligned.columns = [("feature", c) for c in aligned.columns]
    handler._data = data.join(aligned).fillna(0)


MODEL_CONFIG = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
        "subsample": 0.8789, "lambda_l1": 205.69, "lambda_l2": 580.97,
        "max_depth": 8, "num_leaves": 210, "num_threads": 20}}

WINDOWS_5Y = [
    {"train": ("2016-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1"},
    {"train": ("2017-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2"},
    {"train": ("2018-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3"},
    {"train": ("2019-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4"},
    {"train": ("2020-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5"},
    {"train": ("2021-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6"},
]
WINDOWS_3Y = [
    {"train": ("2018-01-01","2019-12-31"), "valid": ("2020-01-01","2020-12-31"), "backtest": ("2021-01-01","2021-12-31"), "name":"W1"},
    {"train": ("2019-01-01","2020-12-31"), "valid": ("2021-01-01","2021-12-31"), "backtest": ("2022-01-01","2022-12-31"), "name":"W2"},
    {"train": ("2020-01-01","2021-12-31"), "valid": ("2022-01-01","2022-12-31"), "backtest": ("2023-01-01","2023-12-31"), "name":"W3"},
    {"train": ("2021-01-01","2022-12-31"), "valid": ("2023-01-01","2023-12-31"), "backtest": ("2024-01-01","2024-12-31"), "name":"W4"},
    {"train": ("2022-01-01","2023-12-31"), "valid": ("2024-01-01","2024-12-31"), "backtest": ("2025-01-01","2025-12-31"), "name":"W5"},
    {"train": ("2023-01-01","2024-12-31"), "valid": ("2025-01-01","2025-12-31"), "backtest": ("2026-01-01","2026-07-21"), "name":"W6"},
]


def run_experiment(exp_name, windows, top_k, use_value_factors, fund_features, value_factors, universe):
    """运行一组消融实验"""
    print(f"\n{'='*70}")
    print(f"  实验 {exp_name}: TopK={top_k}, 价值因子={'有' if use_value_factors else '无'}, {len(windows)}窗口")
    print(f"{'='*70}")
    results = []
    for w in windows:
        ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
        print(f"\n  {w['name']}: train {ts}~{te} → backtest {bs}~{be}")
        dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
            "instruments": universe,
            "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                                  {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
            "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
            "label": ["Ref($close, -20) / $close - 1"]}
        dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
            "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"__main__","kwargs":dhc},
                      "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}
        dataset = init_instance_by_config(dsc)
        if fund_features is not None: inject_features(dataset, fund_features)
        if use_value_factors and value_factors is not None: inject_features(dataset, value_factors)
        model = init_instance_by_config(MODEL_CONFIG)
        ename = f"ablation_{exp_name}_{w['name']}"
        with R.start(experiment_name=ename):
            rec = R.get_recorder()
            model.fit(dataset)
            sig_rec = SignalRecord(model, dataset, rec)
            sig_rec.generate()
            pred = rec.load_object("pred.pkl")
            pac = {"executor":{"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
                    "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}},
                "strategy":{"class":"MonthlyTopkStrategy","module_path":"__main__",
                    "kwargs":{"topk":top_k,"n_drop":top_k,"signal":pred}},
                "backtest":{"start_time":bs,"end_time":be,"account":100000000,"benchmark":None,
                    "exchange_kwargs":{"freq":"day","limit_threshold":0.095,"deal_price":"close",
                        "open_cost":0.0015,"close_cost":0.0025,"min_cost":5}}}
            PortAnaRecord(rec, pac, "day").generate()
            m = rec.list_metrics()
            def _s(k, fb=None):
                v = m.get(k)
                if v is not None and not (isinstance(v,float) and np.isnan(v)): return v
                if fb:
                    v2 = m.get(fb)
                    if v2 is not None and not (isinstance(v2,float) and np.isnan(v2)): return v2
                return 0.0
            ar = _s("1day.excess_return_with_cost.annualized_return","1day.excess_return_without_cost.annualized_return")*100
            ir = _s("1day.excess_return_with_cost.information_ratio","1day.excess_return_without_cost.information_ratio")
            mdd = _s("1day.excess_return_with_cost.max_drawdown","1day.excess_return_without_cost.max_drawdown")*100
            print(f"    年化 {ar:.2f}%, IR {ir:.4f}, 回撤 {mdd:.2f}%")
            results.append({"window": w["name"], "backtest_period": f"{bs}~{be}", "ar": ar, "ir": ir, "mdd": mdd})
    return results


if __name__ == "__main__":
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    universe = df["code"].apply(format_qlib_code).tolist()
    cal = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    cal_set = set(cal)
    print(f"股票池: {len(universe)} 只, 日历: {cal[0].date()}~{cal[-1].date()}")

    print("--- 加载基本面特征 ---")
    fund = load_fundamental_features(universe, cal)
    print("--- 加载价值因子 ---")
    vf = load_value_factors(universe, cal, cal_set)

    # A: 5年 + Top10 + 无价值因子 (baseline, 已有结果直接引用)
    # B: 3年 + Top10 + 无价值因子 → 隔离窗口效应
    res_B = run_experiment("B_3yr_Top10", WINDOWS_3Y, 10, False, fund, vf, universe)
    # C: 5年 + Top30 + 无价值因子 → 隔离TopK效应
    res_C = run_experiment("C_5yr_Top30", WINDOWS_5Y, 30, False, fund, vf, universe)
    # D: 3年 + Top10 + 有价值因子 → 隔离价值因子效应
    res_D = run_experiment("D_3yr_Top10_VF", WINDOWS_3Y, 10, True, fund, vf, universe)

    # A 的已知结果
    res_A = [
        {"window":"W1","backtest_period":"2021-01-01~2021-12-31","ar":19.97,"ir":0.89,"mdd":-23.45},
        {"window":"W2","backtest_period":"2022-01-01~2022-12-31","ar":33.80,"ir":1.40,"mdd":-15.76},
        {"window":"W3","backtest_period":"2023-01-01~2023-12-31","ar":1.80,"ir":0.12,"mdd":-16.23},
        {"window":"W4","backtest_period":"2024-01-01~2024-12-31","ar":23.68,"ir":1.09,"mdd":-21.83},
        {"window":"W5","backtest_period":"2025-01-01~2025-12-31","ar":19.52,"ir":1.29,"mdd":-9.45},
        {"window":"W6","backtest_period":"2026-01-01~2026-07-21","ar":-18.02,"ir":-0.92,"mdd":-14.00},
    ]

    def summarize(res):
        return np.mean([r["ar"] for r in res]), np.mean([r["ir"] for r in res]), np.mean([r["mdd"] for r in res])

    a_ar, a_ir, a_mdd = summarize(res_A)
    b_ar, b_ir, b_mdd = summarize(res_B)
    c_ar, c_ir, c_mdd = summarize(res_C)
    d_ar, d_ir, d_mdd = summarize(res_D)

    print(f"\n\n{'='*80}")
    print(f"  消融实验结果汇总")
    print(f"{'='*80}")
    print(f"\n  {'实验':<32} {'年化':>8} {'IR':>8} {'回撤':>8}")
    print(f"  {'-'*56}")
    print(f"  {'A: 5年+Top10+无价值 (baseline)':<32} {a_ar:>7.2f}% {a_ir:>8.4f} {a_mdd:>7.2f}%")
    print(f"  {'B: 3年+Top10+无价值 (窗口?)':<32} {b_ar:>7.2f}% {b_ir:>8.4f} {b_mdd:>7.2f}%")
    print(f"  {'C: 5年+Top30+无价值 (TopK?)':<32} {c_ar:>7.2f}% {c_ir:>8.4f} {c_mdd:>7.2f}%")
    print(f"  {'D: 3年+Top10+有价值 (因子?)':<32} {d_ar:>7.2f}% {d_ir:>8.4f} {d_mdd:>7.2f}%")

    print(f"\n  归因分析:")
    print(f"  {'-'*56}")
    print(f"  窗口效应 (A→B):  年化 {b_ar-a_ar:+.2f}%,  IR {b_ir-a_ir:+.4f}")
    print(f"  TopK效应 (A→C):  年化 {c_ar-a_ar:+.2f}%,  IR {c_ir-a_ir:+.4f}")
    print(f"  因子效应 (B→D):  年化 {d_ar-b_ar:+.2f}%,  IR {d_ir-b_ir:+.4f}")

    # 逐窗口对比
    print(f"\n{'='*80}")
    print(f"  逐窗口对比 (回测年份)")
    print(f"{'='*80}")
    # 对齐: A和C是5年窗口, B和D是3年窗口, 但回测期完全一致
    bt_periods = ["2021","2022","2023","2024","2025","2026H1"]
    print(f"\n  {'回测期':<10} {'A(baseline)':>12} {'B(3年)':>12} {'C(Top30)':>12} {'D(+价值)':>12}")
    print(f"  {'-'*58}")
    for i in range(6):
        print(f"  {bt_periods[i]:<10} {res_A[i]['ar']:>11.2f}% {res_B[i]['ar']:>11.2f}% {res_C[i]['ar']:>11.2f}% {res_D[i]['ar']:>11.2f}%")

    # Save all results
    all_res = {"A": res_A, "B": res_B, "C": res_C, "D": res_D}
    pd.DataFrame({k: [r["ar"] for r in v] for k, v in all_res.items()},
                 index=bt_periods).to_csv("/Users/11164591/Documents/Qoder目录/qlib/ablation_results.csv")
    print(f"\n  结果已保存")
