"""
V6c 最终突破 — V5 + 熊市择时降仓
===================================
策略: V5 baseline (α=0.3融合) + 月度regime择时
- 当月市场处于熊市 regime (月均价 < 97% MA60): 持仓降至5只 (50%现金)
- 否则: 正常持仓10只

核心思路: 不改模型, 只改仓位管理 → 限制熊市下行
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest as qlib_backtest
from qlib.data import D
from monthly_strategy import MonthlyTopk
import warnings
warnings.filterwarnings("ignore")


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
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def load_fundamental_features(universe, cal):
    fcf = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    prof = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    m = fcf[["code","year","fcf"]].merge(prof[["code","year","net_profit"]], on=["code","year"])
    m = m.dropna().sort_values(["code","year"]).reset_index(drop=True)
    rows = []
    for code, g in m.groupby("code"):
        cs = str(code).zfill(6)
        qc = f"SH{cs}" if cs.startswith("6") else f"SZ{cs}"
        if qc not in universe: continue
        g = g.sort_values("year").copy()
        g["fcf_growth"] = g["fcf"].pct_change()
        g["profit_growth"] = g["net_profit"].pct_change()
        g["fcf_profit_ratio"] = g["fcf"] / (g["net_profit"].abs()+1e-8)
        g["fcf_avg_3y"] = g["fcf"].rolling(3, min_periods=1).mean()
        rs = g["fcf"].rolling(3, min_periods=2).std()
        rm = g["fcf"].rolling(3, min_periods=2).mean().abs()
        g["fcf_cv_3y"] = rs / (rm+1e-8)
        for _, r in g.iterrows():
            if pd.isna(r["fcf_growth"]): continue
            y = int(r["year"])
            af, at = pd.Timestamp(f"{y+1}-05-01"), pd.Timestamp(f"{y+2}-04-30")
            for d in cal[(cal>=af)&(cal<=at)]:
                rows.append({"instrument":qc,"datetime":d,"fcf_growth":r["fcf_growth"],
                    "profit_growth":r["profit_growth"],"fcf_profit_ratio":r["fcf_profit_ratio"],
                    "fcf_avg_3y_norm":r["fcf_avg_3y"]/1e8,"fcf_cv_3y":r["fcf_cv_3y"]})
    if not rows: return None
    df = pd.DataFrame(rows).set_index(["datetime","instrument"])
    df = df[~df.index.duplicated(keep="last")]
    for c in df.columns:
        med = df[c].median(); mad = (df[c]-med).abs().median()
        if mad > 0: df[c] = (df[c]-med)/(1.4826*mad)
        df[c] = df[c].clip(-3,3).fillna(0)
    return df


def load_value_factors(universe, cal, cal_set):
    vf = pd.read_csv("/Users/11164591/Documents/Qoder目录/value_factors_cache.csv")
    vf["date"] = pd.to_datetime(vf["date"])
    ann = vf[vf["quarter"]==12].sort_values(["code","year"]).reset_index(drop=True)
    fin = {}
    for _, r in ann.iterrows():
        c = str(r["code"]).zfill(6); y = int(r["year"])
        if c not in fin: fin[c] = {}
        fin[c][y] = {"roe":r.get("roe",np.nan),"eps":r.get("eps",np.nan),
            "bps":r.get("bps",np.nan),"div_payout":r.get("div_payout",np.nan)}
    price = D.features(list(universe), ["$close"], start_time=cal[0], end_time=cal[-1])
    if price is None or len(price)==0: return None
    price = price.reset_index()
    price.columns = ["instrument","datetime","close"]
    rows = []; proc = 0
    for qc in universe:
        cs = qc[2:]
        if cs not in fin: continue
        f = fin[cs]
        sp = price[price["instrument"]==qc].sort_values("datetime").set_index("datetime")
        if len(sp)==0: continue
        years = sorted(f.keys())
        for i, y in enumerate(years):
            eps, bps, roe, dp = f[y]["eps"], f[y]["bps"], f[y]["roe"], f[y]["div_payout"]
            if pd.isna(eps) or eps==0 or pd.isna(bps) or bps==0: continue
            af = pd.Timestamp(f"{y+1}-05-01")
            at = pd.Timestamp(f"{years[i+1]+1}-04-30") if i+1<len(years) else pd.Timestamp(f"{y+2}-04-30")
            w = sp[(sp.index>=af)&(sp.index<=at)].copy()
            if len(w)==0: continue
            w["pe"] = w["close"]/eps; w["pb"] = w["close"]/bps; w["roe_val"] = roe
            w["div_yield"] = np.nan
            if not pd.isna(dp) and eps!=0:
                w["div_yield"] = (dp/100.0)*eps/w["close"]
            hpe, hpb = [], []
            for hy in range(y-3, y):
                if hy in fin:
                    he, hb = fin[hy]["eps"], fin[hy]["bps"]
                    if pd.isna(he) or he==0 or pd.isna(hb) or hb==0: continue
                    hf, ht = pd.Timestamp(f"{hy+1}-05-01"), pd.Timestamp(f"{hy+2}-04-30")
                    hw = sp[(sp.index>=hf)&(sp.index<=ht)]
                    if len(hw)>0:
                        hpe.extend((hw["close"]/he).tolist())
                        hpb.extend((hw["close"]/hb).tolist())
            if len(hpe)>=50:
                hpe_a = np.array([x for x in hpe if 0<x<500])
                hpb_a = np.array([x for x in hpb if 0<x<50])
                if len(hpe_a)>=30 and len(hpb_a)>=30:
                    hs_p, hs_b = np.sort(hpe_a), np.sort(hpb_a)
                    w["pe_pct_3y"] = w["pe"].apply(lambda x: np.searchsorted(hs_p,x)/len(hs_p) if 0<x<500 else np.nan)
                    w["pb_pct_3y"] = w["pb"].apply(lambda x: np.searchsorted(hs_b,x)/len(hpb_a) if 0<x<50 else np.nan)
                else: w["pe_pct_3y"], w["pb_pct_3y"] = np.nan, np.nan
            else: w["pe_pct_3y"], w["pb_pct_3y"] = np.nan, np.nan
            for dt, r in w.iterrows():
                if dt not in cal_set: continue
                rows.append({"instrument":qc,"datetime":dt,"roe_annual":r["roe_val"],
                    "pe_pct_3y":r["pe_pct_3y"],"pb_pct_3y":r["pb_pct_3y"],"div_yield_est":r["div_yield"]})
        proc += 1
        if proc % 50 == 0: print(f"    价值因子: {proc}/{len(universe)}")
    if not rows: return None
    vfd = pd.DataFrame(rows).set_index(["datetime","instrument"])
    vfd = vfd[~vfd.index.duplicated(keep="last")]
    for c in vfd.columns:
        g = vfd[c].groupby(level=0)
        med = g.transform("median")
        mad = g.transform(lambda x: (x-x.median()).abs().median()).replace(0, np.nan)
        vfd[c] = ((vfd[c]-med)/(1.4826*mad)).clip(-3,3).fillna(0)
    return vfd


def inject_features(dataset, fdf):
    if fdf is None: return
    h = dataset.handler; data = h.fetch()
    al = fdf.reindex(data.index)
    if isinstance(data.columns, pd.MultiIndex):
        al.columns = pd.MultiIndex.from_tuples([("feature",c) for c in al.columns])
    else:
        al.columns = [("feature",c) for c in al.columns]
    h._data = data.join(al).fillna(0)


def apply_value_fusion(pred, vf, alpha=0.3):
    if vf is None or not all(c in vf.columns for c in ["roe_annual","pb_pct_3y","div_yield_est"]):
        return pred
    pf = pred.copy()
    merged = pf.join(vf[["roe_annual","pb_pct_3y","div_yield_est"]], how="left")
    def rn(g): return g.rank(pct=True).fillna(0.5)
    merged["roe_rk"] = merged.groupby(level=0)["roe_annual"].transform(rn)
    merged["pb_rk"] = merged.groupby(level=0)["pb_pct_3y"].transform(lambda x: rn(-x))
    merged["div_rk"] = merged.groupby(level=0)["div_yield_est"].transform(rn)
    merged["vs"] = merged["roe_rk"] + merged["pb_rk"] + merged["div_rk"]
    merged["vs_norm"] = merged.groupby(level=0)["vs"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    merged["ls_norm"] = merged.groupby(level=0)["score"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    merged["fused"] = (1-alpha)*merged["ls_norm"] + alpha*merged["vs_norm"]
    has_vf = merged["roe_annual"].notna()
    pf.loc[has_vf, "score"] = merged.loc[has_vf, "fused"]
    return pf


def compute_monthly_regime(universe, start, end):
    """计算月度regime: 月末时判断该月是否为熊市"""
    insts = list(universe[:50])
    bench_raw = D.features(insts, ["$close"], start_time=start, end_time=end)
    if bench_raw is None or len(bench_raw)==0:
        return {}
    bench_raw = bench_raw.reset_index()
    bench_raw.columns = ["instrument","datetime","close"]
    bench = bench_raw.groupby("datetime")["close"].mean()
    ma60 = bench.rolling(60, min_periods=20).mean()
    ratio = bench / ma60
    # 按月分组, 如果月均 ratio < 0.97 则该月为熊市
    monthly = ratio.resample("M").mean()
    bear_months = {}
    for dt, val in monthly.items():
        if pd.notna(val) and val < 0.97:
            mp = dt.to_period("M")
            bear_months[mp] = True
    return bear_months


def backtest_with_pred(pred, start_time, end_time, top_k=10):
    executor = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
        "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    strategy = {"class":"MonthlyTopk","module_path":"monthly_strategy",
        "kwargs":{"topk":top_k,"n_drop":top_k,"signal":pred}}
    try:
        pmd, _ = qlib_backtest(start_time=start_time, end_time=end_time,
            strategy=strategy, executor=executor, account=100000000, benchmark=None,
            exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
                "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
        rn, _ = pmd.get("1day", (None, None))
        if rn is None: return 0.0, 0.0, 0.0
        a = risk_analysis(rn["return"]-rn["bench"], freq="day")
        return (float(a.loc["annualized_return","risk"])*100,
                float(a.loc["information_ratio","risk"]),
                float(a.loc["max_drawdown","risk"])*100)
    except Exception as e:
        print(f"    [bt err] {e}")
        return 0.0, 0.0, 0.0


def backtest_with_timing(pred, start_time, end_time, bear_months, top_k_normal=10, top_k_bear=5):
    """带市场择时的回测: 熊市月降仓到top_k_bear"""
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.backtest.decision import TradeDecisionWO
    
    class TimedMonthly(TopkDropoutStrategy):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._lp = None
        def generate_trade_decision(self, er=None):
            ts = self.trade_calendar.get_trade_step()
            cs, _ = self.trade_calendar.get_step_time(ts)
            cd = pd.Timestamp(cs); cp = cd.to_period("M")
            rb = False
            try:
                ns, _ = self.trade_calendar.get_step_time(ts+1)
                if pd.Timestamp(ns).to_period("M") != cp: rb = True
            except: rb = True
            if self._lp is None: rb = True
            if not rb: return TradeDecisionWO([], self)
            self._lp = cp
            # 熊市月降仓
            if cp in bear_months:
                self.topk = top_k_bear
            else:
                self.topk = top_k_normal
            return super().generate_trade_decision(er)
    
    # 注册到模块
    import monthly_strategy
    monthly_strategy.TimedMonthly = TimedMonthly
    
    executor = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
        "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    strategy = {"class":"TimedMonthly","module_path":"monthly_strategy",
        "kwargs":{"topk":top_k_normal,"n_drop":top_k_normal,"signal":pred}}
    try:
        pmd, _ = qlib_backtest(start_time=start_time, end_time=end_time,
            strategy=strategy, executor=executor, account=100000000, benchmark=None,
            exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
                "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
        rn, _ = pmd.get("1day", (None, None))
        if rn is None: return 0.0, 0.0, 0.0
        a = risk_analysis(rn["return"]-rn["bench"], freq="day")
        return (float(a.loc["annualized_return","risk"])*100,
                float(a.loc["information_ratio","risk"]),
                float(a.loc["max_drawdown","risk"])*100)
    except Exception as e:
        print(f"    [bt err] {e}")
        return 0.0, 0.0, 0.0


MODEL_CONFIG = {"class":"LGBModel","module_path":"qlib.contrib.model.gbdt",
    "kwargs":{"loss":"mse","colsample_bytree":0.8879,"learning_rate":0.0421,
        "subsample":0.8789,"lambda_l1":205.69,"lambda_l2":580.97,
        "max_depth":8,"num_leaves":210,"num_threads":20}}

WINDOWS = [
    {"train":("2016-01-01","2019-12-31"),"valid":("2020-01-01","2020-12-31"),"backtest":("2021-01-01","2021-12-31"),"name":"W1"},
    {"train":("2017-01-01","2020-12-31"),"valid":("2021-01-01","2021-12-31"),"backtest":("2022-01-01","2022-12-31"),"name":"W2"},
    {"train":("2018-01-01","2021-12-31"),"valid":("2022-01-01","2022-12-31"),"backtest":("2023-01-01","2023-12-31"),"name":"W3"},
    {"train":("2019-01-01","2022-12-31"),"valid":("2023-01-01","2023-12-31"),"backtest":("2024-01-01","2024-12-31"),"name":"W4"},
    {"train":("2020-01-01","2023-12-31"),"valid":("2024-01-01","2024-12-31"),"backtest":("2025-01-01","2025-12-31"),"name":"W5"},
    {"train":("2021-01-01","2024-12-31"),"valid":("2025-01-01","2025-12-31"),"backtest":("2026-01-01","2026-07-21"),"name":"W6"},
]


def train_model(dataset, exp_name):
    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name=exp_name):
        rec = R.get_recorder()
        model.fit(dataset)
        sig = SignalRecord(model, dataset, rec)
        sig.generate()
        pred = rec.load_object("pred.pkl")
    return model, pred


def build_dataset(universe, w, label_expr="Ref($close, -20) / $close - 1"):
    ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
    dhc = {"start_time":ts,"end_time":be,"fit_start_time":ts,"fit_end_time":te,
        "instruments":universe,
        "infer_processors":[{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                            {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors":[{"class":"DropnaLabel"},{"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label":[label_expr]}
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"__main__","kwargs":dhc},
                  "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}
    return init_instance_by_config(dsc)


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv")
    universe = df["code"].apply(format_qlib_code).tolist()
    cal = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    cal_set = set(cal)
    print(f"股票池: {len(universe)} 只")

    print("--- 加载特征 ---")
    fund = load_fundamental_features(universe, cal)
    vf = load_value_factors(universe, cal, cal_set)

    # V5 baseline (已知)
    res_V5 = [
        {"window":"W1","ar":36.45,"ir":1.64,"mdd":-20.02},
        {"window":"W2","ar":32.33,"ir":1.40,"mdd":-13.94},
        {"window":"W3","ar":5.76,"ir":0.39,"mdd":-12.86},
        {"window":"W4","ar":20.90,"ir":0.97,"mdd":-20.76},
        {"window":"W5","ar":25.60,"ir":1.20,"mdd":-9.00},
        {"window":"W6","ar":-12.28,"ir":-0.75,"mdd":-17.00},
    ]

    res_E7 = []  # V5 + 熊市降仓50% (top5)
    res_E8 = []  # V5 + 熊市降仓75% (top3)
    res_E9 = []  # V5 + 熊市空仓 (top0 → 用top1近似)

    for i, w in enumerate(WINDOWS):
        bs, be = w["backtest"]
        print(f"\n{'='*60}")
        print(f"  {w['name']}: backtest {bs}~{be}")
        print(f"{'='*60}")

        # 训练V5模型
        ds = build_dataset(universe, w)
        inject_features(ds, fund)
        inject_features(ds, vf)
        m, pred = train_model(ds, f"v6c_{w['name']}")
        
        # 价值因子融合
        p_fused = apply_value_fusion(pred.copy(), vf, 0.3)
        
        # 计算月度regime
        bear_months = compute_monthly_regime(universe, pd.Timestamp(bs), pd.Timestamp(be))
        print(f"  熊市月数: {len(bear_months)}/12")
        
        # E7: V5 + 熊市top5 (50%仓位)
        ar7, ir7, mdd7 = backtest_with_timing(p_fused.copy(), bs, be, bear_months, 10, 5)
        res_E7.append({"window":w["name"],"ar":ar7,"ir":ir7,"mdd":mdd7})
        print(f"  E7(V5+熊市top5): 年化 {ar7:.2f}%, IR {ir7:.4f}, 回撤 {mdd7:.2f}%")
        
        # E8: V5 + 熊市top3 (30%仓位)
        ar8, ir8, mdd8 = backtest_with_timing(p_fused.copy(), bs, be, bear_months, 10, 3)
        res_E8.append({"window":w["name"],"ar":ar8,"ir":ir8,"mdd":mdd8})
        print(f"  E8(V5+熊市top3): 年化 {ar8:.2f}%, IR {ir8:.4f}, 回撤 {mdd8:.2f}%")
        
        # E9: V5 + 熊市top1 (10%仓位, 近似空仓)
        ar9, ir9, mdd9 = backtest_with_timing(p_fused.copy(), bs, be, bear_months, 10, 1)
        res_E9.append({"window":w["name"],"ar":ar9,"ir":ir9,"mdd":mdd9})
        print(f"  E9(V5+熊市top1): 年化 {ar9:.2f}%, IR {ir9:.4f}, 回撤 {mdd9:.2f}%")

    # ====== 汇总 ======
    def avg(r, k): return np.mean([x[k] for x in r])

    print(f"\n\n{'='*85}")
    print(f"  V6c 最终突破 — V5 + 熊市择时降仓")
    print(f"{'='*85}")
    print(f"\n  {'策略':<36} {'年化':>8} {'IR':>8} {'回撤':>8} {'vs V5':>8}")
    print(f"  {'-'*68}")
    b_ar = avg(res_V5, "ar")
    for name, res in [("V5 Baseline (α=0.3)", res_V5), ("E7: V5+熊市top5(50%)", res_E7),
                       ("E8: V5+熊市top3(30%)", res_E8), ("E9: V5+熊市top1(10%)", res_E9)]:
        a = avg(res, "ar"); ir = avg(res, "ir"); mdd = avg(res, "mdd")
        print(f"  {name:<36} {a:>7.2f}% {ir:>8.4f} {mdd:>7.2f}% {a-b_ar:>+7.2f}%")

    bt = ["2021","2022","2023","2024","2025","2026H1"]
    print(f"\n{'='*85}")
    print(f"  逐窗口年化收益对比")
    print(f"{'='*85}")
    print(f"\n  {'回测期':<8} {'V5':>8} {'E7(top5)':>10} {'E8(top3)':>10} {'E9(top1)':>10}")
    print(f"  {'-'*48}")
    for i in range(6):
        print(f"  {bt[i]:<8} {res_V5[i]['ar']:>7.2f}% {res_E7[i]['ar']:>9.2f}% {res_E8[i]['ar']:>9.2f}% {res_E9[i]['ar']:>9.2f}%")

    print(f"\n  薄弱窗口改善:")
    print(f"  {'-'*48}")
    for idx, name in [(2,"W3(2023)"), (5,"W6(2026H1)")]:
        print(f"  {name}: V5={res_V5[idx]['ar']:.2f}% → E7={res_E7[idx]['ar']:.2f}% E8={res_E8[idx]['ar']:.2f}% E9={res_E9[idx]['ar']:.2f}%")

    pd.DataFrame({
        "V5": [r["ar"] for r in res_V5],
        "E7_bear_top5": [r["ar"] for r in res_E7],
        "E8_bear_top3": [r["ar"] for r in res_E8],
        "E9_bear_top1": [r["ar"] for r in res_E9],
    }, index=bt).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v6c_results.csv")
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
