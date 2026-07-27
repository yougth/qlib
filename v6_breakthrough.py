"""
V6 突破20%实验 — 四路叠加优化
===============================
基线: V5 动态α_t=0.3, 5年滚动, Top10, 年化18.13%

薄弱窗口: W3(2023, +5.76%), W6(2026H1, -12.28%)

E1: 特征剪枝 + 融合 (降噪+价值)
E2: Regime-aware α (牛市低α/熊市高α)
E3: 多α集成 (0.2/0.3/0.4 均值)
E4: 多标签集成 (20d+40d 双模型 → 融合)
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

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
import warnings
warnings.filterwarnings("ignore")


class MonthlyTopkStrategy:
    """简化策略: 直接用 pred signal 月频调仓 Top-K"""
    pass  # 用 qlib 内置的


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


class Alpha158Pruned(Alpha158Enhanced):
    _keep_features = None
    @classmethod
    def set_keep_features(cls, names):
        cls._keep_features = set(names)
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        if self._keep_features is not None:
            mask = [n in self._keep_features for n in names]
            fields = [f for f, m in zip(fields, mask) if m]
            names = [n for n, m in zip(names, mask) if m]
        return fields, names


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
    h = dataset.handler; data = h.fetch()
    al = fdf.reindex(data.index)
    if isinstance(data.columns, pd.MultiIndex):
        al.columns = pd.MultiIndex.from_tuples([("feature",c) for c in al.columns])
    else:
        al.columns = [("feature",c) for c in al.columns]
    h._data = data.join(al).fillna(0)


def extract_feature_importance(model, dataset):
    h = dataset.handler; data = h.fetch()
    if isinstance(data.columns, pd.MultiIndex):
        names = [c[1] for c in data.columns if c[0]=="feature"]
    else:
        names = list(data.columns)
    booster = model.model
    imp = booster.feature_importance(importance_type="split")
    fi = pd.DataFrame({"name":names[:len(imp)],"split":imp[:len(names)]})
    return fi.sort_values("split", ascending=False).reset_index(drop=True)


def apply_value_fusion(pred, vf, alpha=0.3):
    """向量化融合: pred + value_factor → fused score"""
    if vf is None or not all(c in vf.columns for c in ["roe_annual","pb_pct_3y","div_yield_est"]):
        return pred
    pf = pred.copy()
    # join pred 和 vf
    merged = pf.join(vf[["roe_annual","pb_pct_3y","div_yield_est"]], how="left")
    # 按日期分组计算 rank
    def rn(g):
        return g.rank(pct=True).fillna(0.5)
    merged["roe_rk"] = merged.groupby(level=0)["roe_annual"].transform(rn)
    merged["pb_rk"] = merged.groupby(level=0)["pb_pct_3y"].transform(lambda x: rn(-x))
    merged["div_rk"] = merged.groupby(level=0)["div_yield_est"].transform(rn)
    merged["vs"] = merged["roe_rk"] + merged["pb_rk"] + merged["div_rk"]
    # 按日期标准化
    merged["vs_norm"] = merged.groupby(level=0)["vs"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    merged["ls_norm"] = merged.groupby(level=0)["score"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    # 融合
    merged["fused"] = (1-alpha)*merged["ls_norm"] + alpha*merged["vs_norm"]
    # 只更新有 vf 数据的行
    has_vf = merged["roe_annual"].notna()
    pf.loc[has_vf, "score"] = merged.loc[has_vf, "fused"]
    n_fused = has_vf.sum()
    print(f"    [fusion] α={alpha}, pred={len(pf)}, fused={n_fused}")
    return pf


def apply_regime_alpha(pred, vf, cal, universe=None):
    """Regime-aware α: 用股票池等权均价的60日均线判断牛/熊"""
    if vf is None:
        return pred
    pred_start = pred.index.get_level_values(0).min()
    pred_end = pred.index.get_level_values(0).max()
    # 用股票池均价代替指数 (qlib无指数数据)
    insts = universe if universe else list(pred.index.get_level_values(1).unique()[:50])
    bench_raw = D.features(insts, ["$close"], start_time=pred_start, end_time=pred_end)
    if bench_raw is None or len(bench_raw)==0:
        return apply_value_fusion(pred, vf, alpha=0.3)
    bench_raw = bench_raw.reset_index()
    bench_raw.columns = ["instrument","datetime","close"]
    # 每日等权均价
    bench = bench_raw.groupby("datetime")["close"].mean()
    ma60 = bench.rolling(60, min_periods=20).mean()
    
    pf = pred.copy()
    merged = pf.join(vf[["roe_annual","pb_pct_3y","div_yield_est"]], how="left")
    def rn(g): return g.rank(pct=True).fillna(0.5)
    merged["roe_rk"] = merged.groupby(level=0)["roe_annual"].transform(rn)
    merged["pb_rk"] = merged.groupby(level=0)["pb_pct_3y"].transform(lambda x: rn(-x))
    merged["div_rk"] = merged.groupby(level=0)["div_yield_est"].transform(rn)
    merged["vs"] = merged["roe_rk"] + merged["pb_rk"] + merged["div_rk"]
    merged["vs_norm"] = merged.groupby(level=0)["vs"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    merged["ls_norm"] = merged.groupby(level=0)["score"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    
    # 对每个日期确定 regime α
    n_bull = n_bear = n_neutral = 0
    for dt in merged.index.get_level_values(0).unique():
        if dt not in bench.index: continue
        cp = bench.loc[dt]
        cm = ma60.loc[dt] if dt in ma60.index else cp
        if pd.isna(cm) or cm <= 0: cm = cp
        ratio = cp / cm
        if ratio > 1.03:
            alpha = 0.15; n_bull += 1
        elif ratio < 0.97:
            alpha = 0.5; n_bear += 1
        else:
            alpha = 0.3; n_neutral += 1
        mask = merged.index.get_level_values(0) == dt
        merged.loc[mask, "fused"] = (1-alpha)*merged.loc[mask, "ls_norm"] + alpha*merged.loc[mask, "vs_norm"]
    
    has_vf = merged["roe_annual"].notna()
    pf.loc[has_vf, "score"] = merged.loc[has_vf, "fused"]
    print(f"    [regime] bull={n_bull}, neutral={n_neutral}, bear={n_bear}")
    return pf


def apply_ensemble_alpha(pred, vf, alphas=[0.15, 0.3, 0.5]):
    """多α集成: 对每个α分别融合, 取排名均值"""
    if vf is None:
        return pred
    pf = pred.copy()
    rank_sum = pd.Series(0.0, index=pf.index)
    for a in alphas:
        fused = apply_value_fusion(pred.copy(), vf, alpha=a)
        # 按日期排名
        rank_sum += fused.groupby(level=0)["score"].rank(pct=True)
    pf["score"] = rank_sum / len(alphas)
    return pf


from monthly_strategy import MonthlyTopk


def backtest_with_pred(pred, start_time, end_time, top_k=10):
    n_pred = len(pred)
    n_non_nan = pred["score"].notna().sum() if "score" in pred.columns else 0
    n_dates = pred.index.get_level_values(0).nunique()
    print(f"    [bt] pred: {n_pred} rows, {n_non_nan} non-nan, {n_dates} dates")
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


def build_dataset(universe, w, label_expr="Ref($close, -20) / $close - 1",
                  handler_cls="Alpha158Enhanced", keep_features=None):
    ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
    if keep_features is not None and handler_cls == "Alpha158Pruned":
        Alpha158Pruned.set_keep_features(keep_features)
    dhc = {"start_time":ts,"end_time":be,"fit_start_time":ts,"fit_end_time":te,
        "instruments":universe,
        "infer_processors":[{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                            {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors":[{"class":"DropnaLabel"},{"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label":[label_expr]}
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":handler_cls,"module_path":"__main__","kwargs":dhc},
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
    res_base = [
        {"window":"W1","ar":36.45,"ir":1.64,"mdd":-20.02},
        {"window":"W2","ar":32.33,"ir":1.40,"mdd":-13.94},
        {"window":"W3","ar":5.76,"ir":0.39,"mdd":-12.86},
        {"window":"W4","ar":20.90,"ir":0.97,"mdd":-20.76},
        {"window":"W5","ar":25.60,"ir":1.20,"mdd":-9.00},
        {"window":"W6","ar":-12.28,"ir":-0.75,"mdd":-17.00},
    ]

    res_E1 = []; res_E2 = []; res_E3 = []; res_E4 = []
    keep_features = None

    for i, w in enumerate(WINDOWS):
        bs, be = w["backtest"]
        print(f"\n{'='*60}")
        print(f"  {w['name']}: backtest {bs}~{be}")
        print(f"{'='*60}")

        # ====== E2/E3/E4 共用: 全特征基线模型 ======
        ds2 = build_dataset(universe, w)
        if fund is not None: inject_features(ds2, fund)
        if vf is not None: inject_features(ds2, vf)
        m2, p2 = train_model(ds2, f"v6_base_{w['name']}")

        # ====== E1: 特征剪枝 + 融合 ======
        if i == 0:
            # W1: 从基线模型提取 FI
            fi = extract_feature_importance(m2, ds2)
            keep_n = int(len(fi) * 0.5)
            value_names = {'roe_annual','pe_pct_3y','pb_pct_3y','div_yield_est',
                           'fcf_growth','profit_growth','fcf_profit_ratio','fcf_avg_3y_norm','fcf_cv_3y'}
            keep_features = set(fi["name"].iloc[:keep_n].tolist()) | value_names
            print(f"  [E1] W1 FI: {len(fi)}→{len(keep_features)} 特征")
            # W1: 基线模型 + 融合 (= V5)
            p_fused = apply_value_fusion(p2.copy(), vf, 0.3)
            ar, ir, mdd = backtest_with_pred(p_fused, bs, be, 10)
        else:
            # W2-W6: 剪枝后重新训练 + 融合
            ds = build_dataset(universe, w, handler_cls="Alpha158Pruned", keep_features=keep_features)
            if fund is not None: inject_features(ds, fund)
            if vf is not None: inject_features(ds, vf)
            m1, p1 = train_model(ds, f"v6_E1_{w['name']}")
            p_fused = apply_value_fusion(p1.copy(), vf, 0.3)
            ar, ir, mdd = backtest_with_pred(p_fused, bs, be, 10)
        res_E1.append({"window":w["name"],"ar":ar,"ir":ir,"mdd":mdd})
        print(f"  E1(剪枝+融合): 年化 {ar:.2f}%, IR {ir:.4f}, 回撤 {mdd:.2f}%")

        # ====== E2: Regime-aware α ======
        p_regime = apply_regime_alpha(p2.copy(), vf, cal, universe)
        ar2, ir2, mdd2 = backtest_with_pred(p_regime, bs, be, 10)
        res_E2.append({"window":w["name"],"ar":ar2,"ir":ir2,"mdd":mdd2})
        print(f"  E2(Regime α): 年化 {ar2:.2f}%, IR {ir2:.4f}, 回撤 {mdd2:.2f}%")

        # ====== E3: 多α集成 ======
        p_ens = apply_ensemble_alpha(p2.copy(), vf, [0.2, 0.3, 0.4])
        ar3, ir3, mdd3 = backtest_with_pred(p_ens, bs, be, 10)
        res_E3.append({"window":w["name"],"ar":ar3,"ir":ir3,"mdd":mdd3})
        print(f"  E3(多α集成): 年化 {ar3:.2f}%, IR {ir3:.4f}, 回撤 {mdd3:.2f}%")

        # ====== E4: 多标签集成 (20d + 40d) ======
        # 训练40d标签模型 (W1不缓存, 每窗口重新训练)
        ds4 = build_dataset(universe, w, label_expr="Ref($close, -40) / $close - 1")
        if fund is not None: inject_features(ds4, fund)
        if vf is not None: inject_features(ds4, vf)
        m4, p4 = train_model(ds4, f"v6_E4_{w['name']}")
        # 两个模型预测分别融合后取均值
        p_20_fused = apply_value_fusion(p2.copy(), vf, 0.3)
        p_40_fused = apply_value_fusion(p4.copy(), vf, 0.3)
        p_multi = p_20_fused.copy()
        # 对齐
        common_idx = p_20_fused.index.intersection(p_40_fused.index)
        p_multi.loc[common_idx, "score"] = (p_20_fused.loc[common_idx, "score"] + p_40_fused.loc[common_idx, "score"]) / 2
        ar4, ir4, mdd4 = backtest_with_pred(p_multi, bs, be, 10)
        res_E4.append({"window":w["name"],"ar":ar4,"ir":ir4,"mdd":mdd4})
        print(f"  E4(多标签): 年化 {ar4:.2f}%, IR {ir4:.4f}, 回撤 {mdd4:.2f}%")

    # ====== 汇总 ======
    def avg(r, k): return np.mean([x[k] for x in r])

    print(f"\n\n{'='*85}")
    print(f"  V6 突破20%实验结果汇总")
    print(f"{'='*85}")
    print(f"\n  {'策略':<36} {'年化':>8} {'IR':>8} {'回撤':>8} {'vs Base':>8}")
    print(f"  {'-'*68}")
    b_ar = avg(res_base, "ar")
    for name, res in [("V5 Baseline (α=0.3)", res_base), ("E1: 剪枝+融合", res_E1),
                       ("E2: Regime-aware α", res_E2), ("E3: 多α集成", res_E3),
                       ("E4: 多标签集成(20d+40d)", res_E4)]:
        a = avg(res, "ar"); ir = avg(res, "ir"); mdd = avg(res, "mdd")
        print(f"  {name:<36} {a:>7.2f}% {ir:>8.4f} {mdd:>7.2f}% {a-b_ar:>+7.2f}%")

    # 逐窗口
    bt = ["2021","2022","2023","2024","2025","2026H1"]
    print(f"\n{'='*85}")
    print(f"  逐窗口年化收益对比")
    print(f"{'='*85}")
    print(f"\n  {'回测期':<8} {'V5(base)':>10} {'E1(剪枝)':>10} {'E2(regime)':>10} {'E3(集成)':>10} {'E4(多标签)':>10}")
    print(f"  {'-'*58}")
    for i in range(6):
        print(f"  {bt[i]:<8} {res_base[i]['ar']:>9.2f}% {res_E1[i]['ar']:>9.2f}% {res_E2[i]['ar']:>9.2f}% {res_E3[i]['ar']:>9.2f}% {res_E4[i]['ar']:>9.2f}%")

    # 找最优组合
    print(f"\n  薄弱窗口改善:")
    print(f"  {'-'*58}")
    for idx, name in [(2,"W3(2023)"), (5,"W6(2026H1)")]:
        print(f"  {name}: base={res_base[idx]['ar']:.2f}% → E1={res_E1[idx]['ar']:.2f}% E2={res_E2[idx]['ar']:.2f}% E3={res_E3[idx]['ar']:.2f}% E4={res_E4[idx]['ar']:.2f}%")

    pd.DataFrame({
        "V5_base": [r["ar"] for r in res_base],
        "E1_prune_fuse": [r["ar"] for r in res_E1],
        "E2_regime": [r["ar"] for r in res_E2],
        "E3_ensemble": [r["ar"] for r in res_E3],
        "E4_multilabel": [r["ar"] for r in res_E4],
    }, index=bt).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v6_results.csv")
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
