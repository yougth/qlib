"""
V4 优化实验 — 特征剪枝 + 价值因子过滤/融合
=============================================
基线: 5年滚动 + Top10 + Alpha158Enhanced + 基本面5因子

实验 E: 特征剪枝 (Feature Pruning)
  - W1 先跑基线拿 feature importance
  - 后续窗口砍掉 split 排名后 50% 的因子, 强制给价值因子腾位置

实验 F: 前置过滤 (Pre-Filter)
  - PE 分位数 > 90% 的股票直接剔除, 不参与选股
  - 剩余股票送进 LightGBM 精排

实验 G: 后置融合 (Post-Fusion)
  - LightGBM 出分后, 与价值因子得分加权融合
  - 融合后再做 Top-K 截断
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


class Alpha158Pruned(Alpha158Enhanced):
    """支持特征剪枝的 handler — 通过 keep_features 指定保留的特征名"""
    _keep_features = None

    @classmethod
    def set_keep_features(cls, feature_names):
        cls._keep_features = set(feature_names)

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        if self._keep_features is not None:
            mask = [n in self._keep_features for n in names]
            fields = [f for f, m in zip(fields, mask) if m]
            names = [n for n in names if n in self._keep_features]
        return fields, names


def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"


def load_fundamental_features(universe, qlib_calendar):
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner")
    merged = merged.dropna(subset=["fcf", "net_profit"])
    merged = merged.sort_values(["code", "year"]).reset_index(drop=True)
    feature_rows = []
    for code, grp in merged.groupby("code"):
        code_str = str(code).zfill(6)
        qlib_code = f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"
        if qlib_code not in universe: continue
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
    vf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/value_factors_cache.csv")
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
    if price_df is None or len(price_df) == 0:
        print("    [DEBUG] price_df 为空!")
        return None
    price_df = price_df.reset_index()
    price_df.columns = ["instrument", "datetime", "close"]
    print(f"    [DEBUG] price_df: {price_df.shape}, stocks: {price_df['instrument'].nunique()}")
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
    if not feature_rows:
        print("    [DEBUG] feature_rows 为空!")
        return None
    vf_daily = pd.DataFrame(feature_rows).set_index(["datetime", "instrument"])
    print(f"    [DEBUG] vf_daily: {vf_daily.shape}")
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


def extract_feature_importance(model, dataset):
    """提取 LightGBM feature importance, 返回排名"""
    handler = dataset.handler
    data = handler.fetch()
    if isinstance(data.columns, pd.MultiIndex):
        feature_names = [c[1] for c in data.columns if c[0] == "feature"]
    else:
        feature_names = list(data.columns)
    booster = model.model
    importance = booster.feature_importance(importance_type="split")
    fi_df = pd.DataFrame({"name": feature_names[:len(importance)], "split": importance[:len(feature_names)]})
    fi_df = fi_df.sort_values("split", ascending=False).reset_index(drop=True)
    return fi_df


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


def _safe_metric(m, k, fb=None):
    v = m.get(k)
    if v is not None and not (isinstance(v,float) and np.isnan(v)): return v
    if fb:
        v2 = m.get(fb)
        if v2 is not None and not (isinstance(v2,float) and np.isnan(v2)): return v2
    return 0.0


def run_train_and_backtest(dataset, model, window, exp_name, top_k=10):
    """训练 + 生成预测 + 回测，一体化"""
    bs, be = window["backtest"]
    pac = {"executor":{"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
            "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}},
        "strategy":{"class":"MonthlyTopkStrategy","module_path":"__main__",
            "kwargs":{"topk":top_k,"n_drop":top_k,"signal":"<PRED>"}},
        "backtest":{"start_time":bs,"end_time":be,"account":100000000,"benchmark":None,
            "exchange_kwargs":{"freq":"day","limit_threshold":0.095,"deal_price":"close",
                "open_cost":0.0015,"close_cost":0.0025,"min_cost":5}}}
    with R.start(experiment_name=f"v4_{exp_name}_{window['name']}"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")
        PortAnaRecord(rec, pac, "day").generate()
        m = rec.list_metrics()
        ar = _safe_metric(m,"1day.excess_return_with_cost.annualized_return","1day.excess_return_without_cost.annualized_return")*100
        ir = _safe_metric(m,"1day.excess_return_with_cost.information_ratio","1day.excess_return_without_cost.information_ratio")
        mdd = _safe_metric(m,"1day.excess_return_with_cost.max_drawdown","1day.excess_return_without_cost.max_drawdown")*100
    return ar, ir, mdd, pred


def run_backtest_with_pred(pred, window, exp_name, top_k=10):
    """用已有的 pred (可能经过修改) 跑回测, 直接调用 qlib backtest"""
    from qlib.contrib.evaluate import risk_analysis
    from qlib.backtest import backtest as qlib_backtest
    bs, be = window["backtest"]
    executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
            "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    strategy_config = {"class":"MonthlyTopkStrategy","module_path":"__main__",
            "kwargs":{"topk":top_k,"n_drop":top_k,"signal":pred}}
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        start_time=bs, end_time=be, strategy=strategy_config, executor=executor_config,
        account=100000000, benchmark=None,
        exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
            "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
    report_normal, _ = portfolio_metric_dict.get("1day", (None, None))
    if report_normal is None:
        return 0.0, 0.0, 0.0
    analysis = risk_analysis(report_normal["return"] - report_normal["bench"], freq="day")
    ar = float(analysis.loc["annualized_return", "risk"]) * 100
    ir = float(analysis.loc["information_ratio", "risk"])
    mdd = float(analysis.loc["max_drawdown", "risk"]) * 100
    return ar, ir, mdd


def apply_pe_filter(pred, vf_daily, pe_threshold=0.9):
    """前置过滤: PE分位数 > 90% 的股票得分置为极低"""
    pred_filtered = pred.copy()
    if "pe_pct_3y" not in vf_daily.columns:
        print("    [PE过滤] 无 pe_pct_3y 列, 跳过")
        return pred
    # 获取 PE 分位数 (未标准化的原始值用于过滤更合理, 但这里用标准化后的近似)
    # 由于 vf_daily 的 pe_pct_3y 已标准化, >1.28 (约90%分位) 表示极度高估
    pe_high = vf_daily[vf_daily["pe_pct_3y"] > 1.28].index
    if len(pe_high) == 0:
        print("    [PE过滤] 无高PE股票, 跳过")
        return pred
    # 对 pred 中被标记为高PE的股票, 得分设为极低
    common_idx = pred_filtered.index.intersection(pe_high)
    if len(common_idx) > 0:
        pred_filtered.loc[common_idx, "score"] = -999
        filtered_count = pred_filtered.loc[common_idx].groupby(level=0).size()
        print(f"    [PE过滤] 过滤 {len(filtered_count)} 天, 共 {len(common_idx)} 条记录")
    return pred_filtered


def apply_value_fusion(pred, vf_daily, alpha=0.3):
    """后置融合: LightGBM得分 + alpha * 价值得分
    价值得分 = ROE排名 + (1-PB分位)排名 + 股息率排名 (截面标准化后求和)
    """
    if not all(c in vf_daily.columns for c in ["roe_annual", "pb_pct_3y", "div_yield_est"]):
        print("    [融合] 缺少价值因子列, 跳过")
        return pred

    pred_fused = pred.copy()
    # 在每个月调仓日, 计算价值得分并与LGB得分融合
    dates = pred_fused.index.get_level_values(0).unique()
    fusion_dates = []

    for dt in dates:
        pred_day = pred_fused.loc[[dt]] if dt in pred_fused.index.get_level_values(0) else None
        if pred_day is None or len(pred_day) == 0: continue

        vf_day = vf_daily.loc[[dt]] if dt in vf_daily.index.get_level_values(0) else None
        if vf_day is None or len(vf_day) == 0: continue

        # 对齐
        common_instruments = pred_day.index.get_level_values(1).intersection(vf_day.index.get_level_values(1))
        if len(common_instruments) < 5: continue

        # 价值得分: ROE越高越好, PB分位越低越好(便宜), 股息率越高越好
        value_score = pd.Series(0.0, index=common_instruments)
        roe_vals = vf_day.loc[(vf_day.index.get_level_values(1)).isin(common_instruments)]
        roe_vals = roe_vals.set_index(roe_vals.index.get_level_values(1))["roe_annual"]
        pb_val = vf_day.loc[(vf_day.index.get_level_values(1)).isin(common_instruments)]
        pb_val = pb_val.set_index(pb_val.index.get_level_values(1))["pb_pct_3y"]
        div_val = vf_day.loc[(vf_day.index.get_level_values(1)).isin(common_instruments)]
        div_val = div_val.set_index(div_val.index.get_level_values(1))["div_yield_est"]

        # 截面排名标准化到 [0,1]
        def rank_norm(s):
            r = s.rank(pct=True)
            return r.fillna(0.5)

        value_score = rank_norm(roe_vals) + rank_norm(-pb_val) + rank_norm(div_val)
        value_score = (value_score - value_score.mean()) / (value_score.std() + 1e-8)

        # 融合
        lgb_score = pred_day.loc[[(dt, inst) for inst in common_instruments], "score"]
        lgb_score.index = lgb_score.index.get_level_values(1)
        lgb_norm = (lgb_score - lgb_score.mean()) / (lgb_score.std() + 1e-8)
        fused = (1 - alpha) * lgb_norm + alpha * value_score

        # 写回
        for inst in common_instruments:
            if inst in fused.index:
                pred_fused.loc[(dt, inst), "score"] = fused[inst]

        fusion_dates.append(dt)

    if fusion_dates:
        print(f"    [融合] alpha={alpha}, 融合 {len(fusion_dates)} 个交易日")
    return pred_fused


def run_all_experiments():
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

    # ========== 结果收集 ==========
    res_E = []  # 特征剪枝
    res_F = []  # 前置过滤
    res_G = []  # 后置融合

    keep_features = None  # W1 训练后确定的保留特征集

    for i, w in enumerate(WINDOWS_5Y):
        ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
        print(f"\n{'='*60}")
        print(f"  {w['name']}: train {ts}~{te} → backtest {bs}~{be}")
        print(f"{'='*60}")

        # --- 构建 dataset ---
        handler_cls = Alpha158Pruned if keep_features is not None else Alpha158Enhanced
        if keep_features is not None:
            Alpha158Pruned.set_keep_features(keep_features)
            print(f"  [剪枝] 保留 {len(keep_features)} 个特征")

        dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
            "instruments": universe,
            "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                                  {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
            "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
            "label": ["Ref($close, -20) / $close - 1"]}
        dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
            "kwargs":{"handler":{"class":handler_cls.__name__,"module_path":"__main__","kwargs":dhc},
                      "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}
        dataset = init_instance_by_config(dsc)
        if fund is not None: inject_features(dataset, fund)
        if vf is not None: inject_features(dataset, vf)

        # --- E: 训练 + 回测 ---
        model = init_instance_by_config(MODEL_CONFIG)
        if i == 0:
            # W1: 不剪枝(基线)
            ar_e, ir_e, mdd_e, pred_e = run_train_and_backtest(dataset, model, w, "E_baseline", 10)
            res_E.append({"window":w["name"], "bt":f"{bs}~{be}", "ar":ar_e, "ir":ir_e, "mdd":mdd_e})
            print(f"  E(基线):  年化 {ar_e:.2f}%, IR {ir_e:.4f}, 回撤 {mdd_e:.2f}%")

            # 提取 feature importance, 确定剪枝名单
            fi = extract_feature_importance(model, dataset)
            total_features = len(fi)
            keep_n = int(total_features * 0.5)  # 保留前 50%
            keep_features = set(fi["name"].iloc[:keep_n].tolist())
            print(f"\n  [W1 FI] 总特征: {total_features}, 保留前 {keep_n}:")
            print(f"  Top 10: {fi['name'].iloc[:10].tolist()}")
            value_names = {'roe_annual','pe_pct_3y','pb_pct_3y','div_yield_est',
                           'fcf_growth','profit_growth','fcf_profit_ratio','fcf_avg_3y_norm','fcf_cv_3y'}
            kept_value = [n for n in keep_features if n in value_names]
            print(f"  自然保留的价值因子: {kept_value}")
            keep_features = keep_features | value_names
            print(f"  强制保留价值因子后: {len(keep_features)} 个特征")
            fi.to_csv("/Users/11164591/Documents/Qoder目录/qlib/feature_importance_w1.csv", index=False)
        else:
            # W2-W6: 剪枝后训练 + 回测
            ar_e, ir_e, mdd_e, pred_e = run_train_and_backtest(dataset, model, w, "E_prune", 10)
            res_E.append({"window":w["name"], "bt":f"{bs}~{be}", "ar":ar_e, "ir":ir_e, "mdd":mdd_e})
            print(f"  E(剪枝):  年化 {ar_e:.2f}%, IR {ir_e:.4f}, 回撤 {mdd_e:.2f}%")

        # --- F: 前置过滤 (PE > 90% 剔除) ---
        if vf is not None:
            pred_f = apply_pe_filter(pred_e.copy(), vf, pe_threshold=0.9)
            ar_f, ir_f, mdd_f = run_backtest_with_pred(pred_f, w, "F_prefilter", 10)
        else:
            ar_f, ir_f, mdd_f = ar_e, ir_e, mdd_e
            print("  [F] 价值因子为空, 跳过")
        res_F.append({"window":w["name"], "bt":f"{bs}~{be}", "ar":ar_f, "ir":ir_f, "mdd":mdd_f})
        print(f"  F(PE过滤): 年化 {ar_f:.2f}%, IR {ir_f:.4f}, 回撤 {mdd_f:.2f}%")

        # --- G: 后置融合 (alpha=0.3) ---
        if vf is not None:
            pred_g = apply_value_fusion(pred_e.copy(), vf, alpha=0.3)
            ar_g, ir_g, mdd_g = run_backtest_with_pred(pred_g, w, "G_fusion", 10)
        else:
            ar_g, ir_g, mdd_g = ar_e, ir_e, mdd_e
            print("  [G] 价值因子为空, 跳过")
        res_G.append({"window":w["name"], "bt":f"{bs}~{be}", "ar":ar_g, "ir":ir_g, "mdd":mdd_g})
        print(f"  G(融合):  年化 {ar_g:.2f}%, IR {ir_g:.4f}, 回撤 {mdd_g:.2f}%")

    # ========== 汇总 ==========
    # A baseline (已知)
    res_A = [
        {"window":"W1","bt":"2021","ar":19.97,"ir":0.89,"mdd":-23.45},
        {"window":"W2","bt":"2022","ar":33.80,"ir":1.40,"mdd":-15.76},
        {"window":"W3","bt":"2023","ar":1.80,"ir":0.12,"mdd":-16.23},
        {"window":"W4","bt":"2024","ar":23.68,"ir":1.09,"mdd":-21.83},
        {"window":"W5","bt":"2025","ar":19.52,"ir":1.29,"mdd":-9.45},
        {"window":"W6","bt":"2026H1","ar":-18.02,"ir":-0.92,"mdd":-14.00},
    ]

    def avg(res, k): return np.mean([r[k] for r in res])

    a_ar, a_ir, a_mdd = avg(res_A,"ar"), avg(res_A,"ir"), avg(res_A,"mdd")
    e_ar, e_ir, e_mdd = avg(res_E,"ar"), avg(res_E,"ir"), avg(res_E,"mdd")
    f_ar, f_ir, f_mdd = avg(res_F,"ar"), avg(res_F,"ir"), avg(res_F,"mdd")
    g_ar, g_ir, g_mdd = avg(res_G,"ar"), avg(res_G,"ir"), avg(res_G,"mdd")

    print(f"\n\n{'='*80}")
    print(f"  V4 优化实验结果汇总 (5年滚动 + Top10)")
    print(f"{'='*80}")
    print(f"\n  {'策略':<36} {'年化':>8} {'IR':>8} {'回撤':>8}")
    print(f"  {'-'*60}")
    print(f"  {'A: Baseline (5年+Top10+全特征)':<36} {a_ar:>7.2f}% {a_ir:>8.4f} {a_mdd:>7.2f}%")
    print(f"  {'E: 特征剪枝 (W1后砍50%)':<36} {e_ar:>7.2f}% {e_ir:>8.4f} {e_mdd:>7.2f}%")
    print(f"  {'F: 前置过滤 (PE>90%剔除)':<36} {f_ar:>7.2f}% {f_ir:>8.4f} {f_mdd:>7.2f}%")
    print(f"  {'G: 后置融合 (LGB*0.7+价值*0.3)':<36} {g_ar:>7.2f}% {g_ir:>8.4f} {g_mdd:>7.2f}%")

    print(f"\n  增量对比 (vs Baseline A):")
    print(f"  {'-'*60}")
    print(f"  {'E(剪枝)':<36} 年化 {e_ar-a_ar:+.2f}%,  IR {e_ir-a_ir:+.4f}")
    print(f"  {'F(PE过滤)':<36} 年化 {f_ar-a_ar:+.2f}%,  IR {f_ir-a_ir:+.4f}")
    print(f"  {'G(融合)':<36} 年化 {g_ar-a_ar:+.2f}%,  IR {g_ir-a_ir:+.4f}")

    # 逐窗口
    bt_periods = ["2021","2022","2023","2024","2025","2026H1"]
    print(f"\n{'='*80}")
    print(f"  逐窗口年化收益对比")
    print(f"{'='*80}")
    print(f"\n  {'回测期':<10} {'A(baseline)':>12} {'E(剪枝)':>12} {'F(PE过滤)':>12} {'G(融合)':>12}")
    print(f"  {'-'*58}")
    for i in range(6):
        print(f"  {bt_periods[i]:<10} {res_A[i]['ar']:>11.2f}% {res_E[i]['ar']:>11.2f}% {res_F[i]['ar']:>11.2f}% {res_G[i]['ar']:>11.2f}%")

    # Save
    summary = pd.DataFrame({
        "A_baseline": [r["ar"] for r in res_A],
        "E_prune": [r["ar"] for r in res_E],
        "F_prefilter": [r["ar"] for r in res_F],
        "G_fusion": [r["ar"] for r in res_G],
    }, index=bt_periods)
    summary.to_csv("/Users/11164591/Documents/Qoder目录/qlib/v4_optimization_results.csv")
    print(f"\n  结果已保存到 v4_optimization_results.csv")


if __name__ == "__main__":
    run_all_experiments()
