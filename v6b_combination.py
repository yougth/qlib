"""
V6b 突破20% — 组合策略
========================
E5: 剪枝+多标签 (Pruning + Multi-label)
    - 剪枝特征上训练20d和40d双模型, 融合后取均值
    
E6: Regime-aware 模型切换
    - 牛市(bull>30%): 用剪枝+20d+融合 (E1)
    - 熊市(bear>40%): 用多标签20d+40d+融合 (E4)
    - 震荡: 用剪枝+多标签 (E5)
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


def compute_regime(universe, pred_start, pred_end):
    """计算市场regime: 返回每日bull/neutral/bear标签"""
    insts = list(universe[:50])
    bench_raw = D.features(insts, ["$close"], start_time=pred_start, end_time=pred_end)
    if bench_raw is None or len(bench_raw)==0:
        return None
    bench_raw = bench_raw.reset_index()
    bench_raw.columns = ["instrument","datetime","close"]
    bench = bench_raw.groupby("datetime")["close"].mean()
    ma60 = bench.rolling(60, min_periods=20).mean()
    regime = pd.Series("neutral", index=bench.index)
    ratio = bench / ma60
    regime[ratio > 1.03] = "bull"
    regime[ratio < 0.97] = "bear"
    return regime


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

    # V5 baseline
    res_base = [
        {"window":"W1","ar":36.45,"ir":1.64,"mdd":-20.02},
        {"window":"W2","ar":32.33,"ir":1.40,"mdd":-13.94},
        {"window":"W3","ar":5.76,"ir":0.39,"mdd":-12.86},
        {"window":"W4","ar":20.90,"ir":0.97,"mdd":-20.76},
        {"window":"W5","ar":25.60,"ir":1.20,"mdd":-9.00},
        {"window":"W6","ar":-12.28,"ir":-0.75,"mdd":-17.00},
    ]
    # V6 E1 (剪枝+融合) and E4 (多标签)
    res_E1 = [
        {"window":"W1","ar":46.33},{"window":"W2","ar":32.81},
        {"window":"W3","ar":-2.97},{"window":"W4","ar":28.44},
        {"window":"W5","ar":14.84},{"window":"W6","ar":-21.70},
    ]
    res_E4 = [
        {"window":"W1","ar":14.43},{"window":"W2","ar":21.99},
        {"window":"W3","ar":5.33},{"window":"W4","ar":31.81},
        {"window":"W5","ar":11.23},{"window":"W6","ar":3.02},
    ]

    res_E5 = []; res_E6 = []
    keep_features = None

    for i, w in enumerate(WINDOWS):
        bs, be = w["backtest"]
        print(f"\n{'='*60}")
        print(f"  {w['name']}: backtest {bs}~{be}")
        print(f"{'='*60}")

        # ====== 训练全特征模型 (用于提取FI和作为fallback) ======
        ds_full = build_dataset(universe, w)
        if fund is not None: inject_features(ds_full, fund)
        if vf is not None: inject_features(ds_full, vf)
        m_full, p_full_20 = train_model(ds_full, f"v6b_full_{w['name']}")

        # W1: 提取FI确定剪枝集
        if i == 0:
            fi = extract_feature_importance(m_full, ds_full)
            keep_n = int(len(fi) * 0.5)
            value_names = {'roe_annual','pe_pct_3y','pb_pct_3y','div_yield_est',
                           'fcf_growth','profit_growth','fcf_profit_ratio','fcf_avg_3y_norm','fcf_cv_3y'}
            keep_features = set(fi["name"].iloc[:keep_n].tolist()) | value_names
            print(f"  FI: {len(fi)}→{len(keep_features)} 特征")

        # ====== E5: 剪枝+多标签 ======
        # 剪枝20d模型
        ds_p20 = build_dataset(universe, w, handler_cls="Alpha158Pruned", keep_features=keep_features)
        if fund is not None: inject_features(ds_p20, fund)
        if vf is not None: inject_features(ds_p20, vf)
        m_p20, p_p20 = train_model(ds_p20, f"v6b_p20_{w['name']}")
        # 剪枝40d模型
        ds_p40 = build_dataset(universe, w, label_expr="Ref($close, -40) / $close - 1",
                                handler_cls="Alpha158Pruned", keep_features=keep_features)
        if fund is not None: inject_features(ds_p40, fund)
        if vf is not None: inject_features(ds_p40, vf)
        m_p40, p_p40 = train_model(ds_p40, f"v6b_p40_{w['name']}")
        # 分别融合后取均值
        p_p20_f = apply_value_fusion(p_p20.copy(), vf, 0.3)
        p_p40_f = apply_value_fusion(p_p40.copy(), vf, 0.3)
        p_e5 = p_p20_f.copy()
        common_idx = p_p20_f.index.intersection(p_p40_f.index)
        p_e5.loc[common_idx, "score"] = (p_p20_f.loc[common_idx, "score"] + p_p40_f.loc[common_idx, "score"]) / 2
        ar5, ir5, mdd5 = backtest_with_pred(p_e5, bs, be, 10)
        res_E5.append({"window":w["name"],"ar":ar5,"ir":ir5,"mdd":mdd5})
        print(f"  E5(剪枝+多标签): 年化 {ar5:.2f}%, IR {ir5:.4f}, 回撤 {mdd5:.2f}%")

        # ====== E6: Regime-aware 模型切换 ======
        # 计算regime
        regime = compute_regime(universe, pd.Timestamp(bs), pd.Timestamp(be))
        if regime is not None:
            n_bull = (regime == "bull").sum()
            n_bear = (regime == "bear").sum()
            n_neutral = (regime == "neutral").sum()
            total = len(regime)
            bull_pct = n_bull / total * 100
            bear_pct = n_bear / total * 100
            print(f"  Regime: bull={bull_pct:.0f}%, neutral={n_neutral/total*100:.0f}%, bear={bear_pct:.0f}%")
        else:
            bull_pct = bear_pct = 0
            print(f"  Regime: 无法计算, 使用E5")

        # 训练40d全特征模型 (用于E4)
        ds_full_40 = build_dataset(universe, w, label_expr="Ref($close, -40) / $close - 1")
        if fund is not None: inject_features(ds_full_40, fund)
        if vf is not None: inject_features(ds_full_40, vf)
        m_full_40, p_full_40 = train_model(ds_full_40, f"v6b_full40_{w['name']}")

        # E1: 剪枝+20d+融合
        p_e1 = apply_value_fusion(p_p20.copy(), vf, 0.3)
        # E4: 全特征20d+40d+融合
        p_20_f = apply_value_fusion(p_full_20.copy(), vf, 0.3)
        p_40_f = apply_value_fusion(p_full_40.copy(), vf, 0.3)
        p_e4 = p_20_f.copy()
        common_idx = p_20_f.index.intersection(p_40_f.index)
        p_e4.loc[common_idx, "score"] = (p_20_f.loc[common_idx, "score"] + p_40_f.loc[common_idx, "score"]) / 2

        # E6: 按regime逐日选择
        p_e6 = p_e1.copy()  # 默认用E1
        if regime is not None:
            for dt in p_e6.index.get_level_values(0).unique():
                if dt not in regime.index: continue
                r = regime.loc[dt]
                if r == "bear":
                    # 熊市用E4
                    if dt in p_e4.index.get_level_values(0):
                        p_e6.loc[p_e6.index.get_level_values(0)==dt, "score"] = \
                            p_e4.loc[p_e4.index.get_level_values(0)==dt, "score"].values
                # bull和neutral保持E1
        ar6, ir6, mdd6 = backtest_with_pred(p_e6, bs, be, 10)
        res_E6.append({"window":w["name"],"ar":ar6,"ir":ir6,"mdd":mdd6})
        print(f"  E6(Regime切换): 年化 {ar6:.2f}%, IR {ir6:.4f}, 回撤 {mdd6:.2f}%")

    # ====== 汇总 ======
    def avg(r, k): return np.mean([x[k] for x in r])

    print(f"\n\n{'='*85}")
    print(f"  V6b 突破20%组合策略结果汇总")
    print(f"{'='*85}")
    print(f"\n  {'策略':<36} {'年化':>8} {'IR':>8} {'回撤':>8} {'vs V5':>8}")
    print(f"  {'-'*68}")
    b_ar = avg(res_base, "ar")
    for name, res in [("V5 Baseline (α=0.3)", res_base), ("E1: 剪枝+融合", res_E1),
                       ("E4: 多标签(全特征)", res_E4), ("E5: 剪枝+多标签", res_E5),
                       ("E6: Regime切换(E1/E4)", res_E6)]:
        a = avg(res, "ar"); ir = avg(res, "ir") if "ir" in res[0] else float("nan")
        mdd = avg(res, "mdd") if "mdd" in res[0] else float("nan")
        print(f"  {name:<36} {a:>7.2f}% {ir:>8.4f} {mdd:>7.2f}% {a-b_ar:>+7.2f}%")

    bt = ["2021","2022","2023","2024","2025","2026H1"]
    print(f"\n{'='*85}")
    print(f"  逐窗口年化收益对比")
    print(f"{'='*85}")
    print(f"\n  {'回测期':<8} {'V5':>8} {'E1(剪枝)':>10} {'E4(多标签)':>10} {'E5(剪枝+多)':>10} {'E6(regime)':>10}")
    print(f"  {'-'*58}")
    for i in range(6):
        print(f"  {bt[i]:<8} {res_base[i]['ar']:>7.2f}% {res_E1[i]['ar']:>9.2f}% {res_E4[i]['ar']:>9.2f}% {res_E5[i]['ar']:>9.2f}% {res_E6[i]['ar']:>9.2f}%")

    print(f"\n  薄弱窗口改善:")
    print(f"  {'-'*58}")
    for idx, name in [(2,"W3(2023)"), (5,"W6(2026H1)")]:
        print(f"  {name}: V5={res_base[idx]['ar']:.2f}% → E5={res_E5[idx]['ar']:.2f}% E6={res_E6[idx]['ar']:.2f}%")

    # 理论上限
    best = [max(res_E1[i]["ar"], res_E4[i]["ar"], res_E5[i]["ar"], res_E6[i]["ar"]) for i in range(6)]
    print(f"\n  理论上限(逐窗口最优): {np.mean(best):.2f}%")

    pd.DataFrame({
        "V5": [r["ar"] for r in res_base],
        "E1_prune": [r["ar"] for r in res_E1],
        "E4_multilabel": [r["ar"] for r in res_E4],
        "E5_prune_multi": [r["ar"] for r in res_E5],
        "E6_regime_switch": [r["ar"] for r in res_E6],
    }, index=bt).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v6b_results.csv")
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
