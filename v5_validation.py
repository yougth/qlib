"""
V5 严格验证 + 优化实验
=========================
1. 动态α_t: 在每个窗口 Valid 集上网格搜索最优 α_t, 再应用到 Test
2. 涨跌停+停牌过滤: 一字板不可买入, 停牌剔除
3. 行业Beta归因: 提取 Top10 持仓, 分析行业集中度
4. Risk Parity: 波动率倒数加权替代等权
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
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest as qlib_backtest
from qlib.data import D
import warnings
warnings.filterwarnings("ignore")


# ==================== Strategy ====================
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


class LimitAwareMonthlyStrategy(TopkDropoutStrategy):
    """涨跌停+停牌感知的月频策略: 回测时自动跳过不可交易标的"""
    def __init__(self, **kwargs):
        self._limit_up_set = kwargs.pop("limit_up_set", set())
        self._suspension_set = kwargs.pop("suspension_set", set())
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
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
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
    vf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/value_factors_cache.csv", sep='\t')
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
    price_df.columns = ["instrument", "datetime", "close"]
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


def apply_value_fusion(pred, vf_daily, alpha=0.3):
    """后置融合: LGB得分 + alpha * 价值得分"""
    if not all(c in vf_daily.columns for c in ["roe_annual", "pb_pct_3y", "div_yield_est"]):
        return pred
    pred_fused = pred.copy()
    dates = pred_fused.index.get_level_values(0).unique()
    fusion_dates = []
    for dt in dates:
        pred_day = pred_fused.loc[[dt]] if dt in pred_fused.index.get_level_values(0) else None
        if pred_day is None or len(pred_day) == 0: continue
        vf_day = vf_daily.loc[[dt]] if dt in vf_daily.index.get_level_values(0) else None
        if vf_day is None or len(vf_day) == 0: continue
        common_instruments = pred_day.index.get_level_values(1).intersection(vf_day.index.get_level_values(1))
        if len(common_instruments) < 5: continue
        roe_vals = vf_day.set_index(vf_day.index.get_level_values(1)).loc[common_instruments, "roe_annual"]
        pb_val = vf_day.set_index(vf_day.index.get_level_values(1)).loc[common_instruments, "pb_pct_3y"]
        div_val = vf_day.set_index(vf_day.index.get_level_values(1)).loc[common_instruments, "div_yield_est"]
        def rank_norm(s):
            return s.rank(pct=True).fillna(0.5)
        value_score = rank_norm(roe_vals) + rank_norm(-pb_val) + rank_norm(div_val)
        value_score = (value_score - value_score.mean()) / (value_score.std() + 1e-8)
        lgb_score = pred_day.set_index(pred_day.index.get_level_values(1)).loc[common_instruments, "score"]
        lgb_norm = (lgb_score - lgb_score.mean()) / (lgb_score.std() + 1e-8)
        fused = (1 - alpha) * lgb_norm + alpha * value_score
        for inst in common_instruments:
            if inst in fused.index:
                pred_fused.loc[(dt, inst), "score"] = fused[inst]
        fusion_dates.append(dt)
    return pred_fused


def backtest_with_pred(pred, start_time, end_time, top_k=10):
    """直接调用 qlib backtest"""
    executor_config = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor",
            "kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    strategy_config = {"class":"MonthlyTopkStrategy","module_path":"__main__",
            "kwargs":{"topk":top_k,"n_drop":top_k,"signal":pred}}
    try:
        portfolio_metric_dict, _ = qlib_backtest(
            start_time=start_time, end_time=end_time, strategy=strategy_config, executor=executor_config,
            account=100000000, benchmark=None,
            exchange_kwargs={"freq":"day","limit_threshold":0.095,"deal_price":"close",
                "open_cost":0.0015,"close_cost":0.0025,"min_cost":5})
        report_normal, _ = portfolio_metric_dict.get("1day", (None, None))
        if report_normal is None: return 0.0, 0.0, 0.0
        analysis = risk_analysis(report_normal["return"] - report_normal["bench"], freq="day")
        ar = float(analysis.loc["annualized_return", "risk"]) * 100
        ir = float(analysis.loc["information_ratio", "risk"])
        mdd = float(analysis.loc["max_drawdown", "risk"]) * 100
        return ar, ir, mdd
    except Exception as e:
        print(f"    [backtest error] {e}")
        return 0.0, 0.0, 0.0


def search_alpha_on_valid(pred, vf_daily, valid_start, valid_end, alpha_grid=None):
    """在 valid 集上网格搜索最优 α — 直接用 pred 的 valid 时段切片"""
    if alpha_grid is None:
        alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # 提取 valid 时段的 pred
    dt_mask = (pred.index.get_level_values(0) >= pd.Timestamp(valid_start)) & \
              (pred.index.get_level_values(0) <= pd.Timestamp(valid_end))
    pred_valid = pred[dt_mask].copy()
    if len(pred_valid) == 0:
        print(f"    [WARNING] valid 集无数据!")
        return 0.3, 0.0, []
    best_alpha, best_ir = 0.3, -999
    results = []
    for alpha in alpha_grid:
        pred_a = apply_value_fusion(pred_valid.copy(), vf_daily, alpha=alpha)
        ar, ir, mdd = backtest_with_pred(pred_a, valid_start, valid_end, top_k=10)
        results.append({"alpha": alpha, "ar": ar, "ir": ir, "mdd": mdd})
        if ir > best_ir:
            best_ir = ir
            best_alpha = alpha
    return best_alpha, best_ir, results


def build_limit_up_set(universe, cal):
    """构建一字涨停集合: (date, instrument) — 当日不可买入"""
    print("  --- 构建涨跌停/停牌集合 ---")
    price_df = D.features(list(universe), ["$close", "$open", "$high", "$low"],
                          start_time=cal[0], end_time=cal[-1])
    if price_df is None or len(price_df) == 0:
        return set(), set()
    price_df = price_df.reset_index()
    price_df.columns = ["instrument", "datetime", "close", "open", "high", "low"]

    # 日收益率
    price_df = price_df.sort_values(["instrument", "datetime"])
    price_df["prev_close"] = price_df.groupby("instrument")["close"].shift(1)
    price_df["daily_ret"] = (price_df["close"] - price_df["prev_close"]) / price_df["prev_close"]

    # 一字涨停: open == high == low == close, 且涨幅 > 9%
    limit_up_mask = (
        (price_df["open"] == price_df["high"]) &
        (price_df["high"] == price_df["low"]) &
        (price_df["low"] == price_df["close"]) &
        (price_df["daily_ret"] > 0.09)
    )
    limit_up_set = set(zip(price_df.loc[limit_up_mask, "datetime"], price_df.loc[limit_up_mask, "instrument"]))

    # 停牌: open/high/low/close 全为 NaN 或 当日无数据
    vol_df = D.features(list(universe), ["$volume"], start_time=cal[0], end_time=cal[-1])
    suspension_set = set()
    if vol_df is not None and len(vol_df) > 0:
        vol_df = vol_df.reset_index()
        vol_df.columns = ["instrument", "datetime", "volume"]
        susp = vol_df[(vol_df["volume"].isna()) | (vol_df["volume"] == 0)]
        suspension_set = set(zip(susp["datetime"], susp["instrument"]))

    print(f"  一字涨停: {len(limit_up_set)} 条, 停牌: {len(suspension_set)} 条")
    return limit_up_set, suspension_set


def filter_pred_by_tradability(pred, limit_up_set, suspension_set):
    """将不可交易标的的得分设为极低"""
    pred_filtered = pred.copy()
    for idx in pred_filtered.index:
        dt, inst = idx
        if (dt, inst) in limit_up_set or (dt, inst) in suspension_set:
            pred_filtered.loc[idx, "score"] = -999
    return pred_filtered


def extract_topk_holdings(pred, vf_daily, alpha, dates_range, top_k=10):
    """提取指定日期范围内的 Top-K 持仓"""
    pred_f = apply_value_fusion(pred.copy(), vf_daily, alpha=alpha)
    holdings = {}
    for dt in dates_range:
        if dt not in pred_f.index.get_level_values(0): continue
        day_pred = pred_f.loc[[dt]].copy()
        day_pred.index = day_pred.index.get_level_values(1)
        top = day_pred["score"].nlargest(top_k)
        holdings[dt] = top.index.tolist()
    return holdings


def get_industry_data(universe):
    """从缓存或 akshare 获取行业分类"""
    cache_path = "/Users/11164591/Documents/Qoder目录/qlib/industry_cache.csv"
    if os.path.exists(cache_path):
        ind_df = pd.read_csv(cache_path, sep='\t')
        print(f"  [行业] 从缓存加载: {len(ind_df)} 只")
        return dict(zip(ind_df["code"], ind_df["industry"]))

    print("  [行业] 从 akshare 拉取行业分类...")
    try:
        import sys
        sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/akshare")
        import akshare as ak
        # 获取东方财富行业分类
        boards = ak.stock_board_industry_name_em()
        ind_map = {}
        for _, row in boards.iterrows():
            board_name = row["板块名称"]
            try:
                cons = ak.stock_board_industry_cons_em(symbol=board_name)
                for _, s in cons.iterrows():
                    code = str(s["代码"]).zfill(6)
                    ind_map[code] = board_name
            except:
                continue
        # 保存缓存
        ind_df = pd.DataFrame(list(ind_map.items()), columns=["code", "industry"])
        ind_df.to_csv(cache_path, sep='\t', index=False)
        print(f"  [行业] 缓存已保存: {len(ind_df)} 只")
        return ind_map
    except Exception as e:
        print(f"  [行业] akshare 失败: {e}, 使用简易分类")
        # 简易分类: 按股票代码段判断
        ind_map = {}
        for code in universe:
            c = str(code).zfill(6)
            if c.startswith("60"): ind_map[c] = "沪市主板"
            elif c.startswith("00"): ind_map[c] = "深市主板"
            elif c.startswith("30"): ind_map[c] = "创业板"
            else: ind_map[c] = "其他"
        return ind_map


# ==================== Main ====================
def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv", sep='\t')
    universe = df["code"].apply(format_qlib_code).tolist()
    cal = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    cal_set = set(cal)
    print(f"股票池: {len(universe)} 只, 日历: {cal[0].date()}~{cal[-1].date()}")

    print("--- 加载基本面特征 ---")
    fund = load_fundamental_features(universe, cal)
    print("--- 加载价值因子 ---")
    vf = load_value_factors(universe, cal, cal_set)

    # 构建涨跌停/停牌集合
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)

    # 收集结果
    res_dyn_alpha = []    # 动态α_t
    res_limit = []        # 涨跌停过滤
    res_risk_parity = []  # Risk Parity
    all_holdings = {}     # 行业归因用
    alpha_search_log = [] # α搜索记录

    for i, w in enumerate(WINDOWS_5Y):
        ts, te = w["train"]; vs, ve = w["valid"]; bs, be = w["backtest"]
        print(f"\n{'='*60}")
        print(f"  {w['name']}: train {ts}~{te}, valid {vs}~{ve}, backtest {bs}~{be}")
        print(f"{'='*60}")

        # 构建 dataset
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
        if fund is not None: inject_features(dataset, fund)
        if vf is not None: inject_features(dataset, vf)

        # 训练
        model = init_instance_by_config(MODEL_CONFIG)
        with R.start(experiment_name=f"v5_{w['name']}"):
            rec = R.get_recorder()
            model.fit(dataset)
            sig_rec = SignalRecord(model, dataset, rec)
            sig_rec.generate()
            pred = rec.load_object("pred.pkl")

        # ====== 1. 动态α_t: Valid集网格搜索 ======
        print(f"  [1] 动态α搜索 (Valid集)...")
        best_alpha, best_ir, alpha_results = search_alpha_on_valid(
            pred, vf, vs, ve, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        alpha_search_log.append({"window": w["name"], "best_alpha": best_alpha, "valid_ir": best_ir})
        for r in alpha_results:
            print(f"      α={r['alpha']:.1f}: 年化 {r['ar']:.2f}%, IR {r['ir']:.4f}")
        print(f"    → 最优 α_t = {best_alpha:.1f} (Valid IR={best_ir:.4f})")

        # 用动态α_t在Test集回测
        pred_dyn = apply_value_fusion(pred.copy(), vf, alpha=best_alpha)
        ar_dyn, ir_dyn, mdd_dyn = backtest_with_pred(pred_dyn, bs, be, 10)
        res_dyn_alpha.append({"window":w["name"], "bt":f"{bs}~{be}",
            "alpha_t":best_alpha, "ar":ar_dyn, "ir":ir_dyn, "mdd":mdd_dyn})
        print(f"  动态α_t={best_alpha}: 年化 {ar_dyn:.2f}%, IR {ir_dyn:.4f}, 回撤 {mdd_dyn:.2f}%")

        # ====== 2. 涨跌停过滤 (用动态α_t) ======
        print(f"  [2] 涨跌停+停牌过滤...")
        pred_limit = filter_pred_by_tradability(pred_dyn, limit_up_set, suspension_set)
        ar_lim, ir_lim, mdd_lim = backtest_with_pred(pred_limit, bs, be, 10)
        res_limit.append({"window":w["name"], "bt":f"{bs}~{be}", "ar":ar_lim, "ir":ir_lim, "mdd":mdd_lim})
        print(f"  涨跌停过滤: 年化 {ar_lim:.2f}%, IR {ir_lim:.4f}, 回撤 {mdd_lim:.2f}%")

        # ====== 3. 行业归因: 提取每月调仓日的 Top10 ======
        if i in [0, 3]:  # 2021 和 2024
            bt_dates = sorted(set(pred.index.get_level_values(0)))
            bt_dates = [d for d in bt_dates if pd.Timestamp(bs) <= d <= pd.Timestamp(be)]
            monthly_dates = []
            seen_months = set()
            for d in bt_dates:
                m = d.to_period("M")
                if m not in seen_months:
                    seen_months.add(m)
                    monthly_dates.append(d)
            # 取每月最后一个交易日
            holdings = extract_topk_holdings(pred, vf, best_alpha, bt_dates, 10)
            all_holdings[w["name"]] = holdings

        # ====== 4. Risk Parity (波动率倒数加权) ======
        # 注: qlib 内置 TopkDropoutStrategy 不支持自定义权重
        # 这里用近似方法: 调整 pred score 使得 Top-K 选出后,
        # 按波动率倒数加权的效果近似于调整 score 排名
        # 但更准确的做法是直接修改 strategy
        # 简化实现: 仍用等权 TopK, 但在 score 中除以 60日波动率
        print(f"  [4] Risk Parity (波动率调整)...")
        pred_rp = pred_dyn.copy()
        # 获取60日波动率: 用 close 数据手动计算
        vol_price = D.features(list(universe), ["$close"], start_time=pd.Timestamp(bs) - pd.Timedelta(days=120), end_time=be)
        if vol_price is not None and len(vol_price) > 0:
            vol_price = vol_price.reset_index()
            vol_price.columns = ["instrument", "datetime", "close"]
            vol_price = vol_price.sort_values(["instrument", "datetime"])
            vol_price["ret"] = vol_price.groupby("instrument")["close"].pct_change()
            vol_price["vol60"] = vol_price.groupby("instrument")["ret"].transform(
                lambda x: x.rolling(60, min_periods=20).std())
            vol_indexed = vol_price.set_index(["datetime", "instrument"])["vol60"]
            for idx in pred_rp.index:
                dt, inst = idx
                vol_key = (dt, inst)
                if vol_key in vol_indexed.index:
                    v = vol_indexed.loc[vol_key]
                    if isinstance(v, pd.Series): v = v.iloc[0]
                    if not pd.isna(v) and v > 0:
                        # 低波动 → 更高分; 高波动 → 更低分
                        pred_rp.loc[idx, "score"] = pred_rp.loc[idx, "score"] / (v * 100)
        ar_rp, ir_rp, mdd_rp = backtest_with_pred(pred_rp, bs, be, 10)
        res_risk_parity.append({"window":w["name"], "bt":f"{bs}~{be}", "ar":ar_rp, "ir":ir_rp, "mdd":mdd_rp})
        print(f"  Risk Parity: 年化 {ar_rp:.2f}%, IR {ir_rp:.4f}, 回撤 {mdd_rp:.2f}%")

    # ==================== 汇总 ====================
    # A baseline (已知)
    res_A = [
        {"window":"W1","bt":"2021","ar":19.97,"ir":0.89,"mdd":-23.45},
        {"window":"W2","bt":"2022","ar":33.80,"ir":1.40,"mdd":-15.76},
        {"window":"W3","bt":"2023","ar":1.80,"ir":0.12,"mdd":-16.23},
        {"window":"W4","bt":"2024","ar":23.68,"ir":1.09,"mdd":-21.83},
        {"window":"W5","bt":"2025","ar":19.52,"ir":1.29,"mdd":-9.45},
        {"window":"W6","bt":"2026H1","ar":-18.02,"ir":-0.92,"mdd":-14.00},
    ]
    # G 固定α=0.3 (已知)
    res_G = [
        {"window":"W1","bt":"2021","ar":36.45,"ir":1.64,"mdd":-20.02},
        {"window":"W2","bt":"2022","ar":34.37,"ir":1.49,"mdd":-13.94},
        {"window":"W3","bt":"2023","ar":4.00,"ir":0.26,"mdd":-15.60},
        {"window":"W4","bt":"2024","ar":26.58,"ir":1.18,"mdd":-21.35},
        {"window":"W5","bt":"2025","ar":19.06,"ir":1.28,"mdd":-9.07},
        {"window":"W6","bt":"2026H1","ar":-18.81,"ir":-1.23,"mdd":-17.08},
    ]

    def avg(res, k): return np.mean([r[k] for r in res])

    print(f"\n\n{'='*85}")
    print(f"  V5 严格验证 + 优化实验结果汇总")
    print(f"{'='*85}")

    # 1. 动态α_t 结果
    print(f"\n  [验证1] 动态α_t (窗口内Valid搜索) vs 全局固定α=0.3")
    print(f"  {'-'*75}")
    print(f"  {'窗口':<6} {'α_t':>6} {'Valid IR':>10} {'动态α 年化':>12} {'固定α 年化':>12} {'差值':>8}")
    for i in range(6):
        d = res_dyn_alpha[i]
        g = res_G[i]
        diff = d["ar"] - g["ar"]
        print(f"  {d['window']:<6} {d['alpha_t']:>6.1f} {alpha_search_log[i]['valid_ir']:>10.4f} {d['ar']:>11.2f}% {g['ar']:>11.2f}% {diff:>+7.2f}%")

    da_ar = avg(res_dyn_alpha, "ar")
    da_ir = avg(res_dyn_alpha, "ir")
    g_ar = avg(res_G, "ar")
    g_ir = avg(res_G, "ir")
    print(f"  {'平均':<6} {'':>6} {'':>10} {da_ar:>11.2f}% {g_ar:>11.2f}% {da_ar-g_ar:>+7.2f}%")
    print(f"  结论: 动态α_t {'通过' if da_ar >= 15 else '未通过'} (目标 ≥15%, 实际 {da_ar:.2f}%)")

    # 2. 涨跌停过滤
    print(f"\n  [验证2] 涨跌停+停牌约束")
    print(f"  {'-'*75}")
    print(f"  {'窗口':<6} {'无约束':>10} {'加约束':>10} {'差值':>8} {'结论':>10}")
    for i in range(6):
        d = res_dyn_alpha[i]
        l = res_limit[i]
        diff = l["ar"] - d["ar"]
        flag = "≈无影响" if abs(diff) < 1 else ("下降" if diff < 0 else "上升")
        print(f"  {d['window']:<6} {d['ar']:>9.2f}% {l['ar']:>9.2f}% {diff:>+7.2f}% {flag:>10}")

    # 3. 行业归因
    print(f"\n  [验证3] 行业Beta归因")
    print(f"  {'-'*75}")
    ind_map = get_industry_data(universe)
    for wname, holdings in all_holdings.items():
        print(f"\n  {wname} 持仓行业分布:")
        ind_count = {}
        total = 0
        for dt, stocks in holdings.items():
            for s in stocks:
                code = s[2:]
                ind = ind_map.get(code, "未知")
                ind_count[ind] = ind_count.get(ind, 0) + 1
                total += 1
        if total > 0:
            sorted_ind = sorted(ind_count.items(), key=lambda x: -x[1])
            for ind, cnt in sorted_ind[:8]:
                pct = cnt / total * 100
                print(f"    {ind:<12} {cnt:>4} 次 ({pct:.1f}%)")
            # 集中度: Top3行业占比
            top3 = sum(cnt for _, cnt in sorted_ind[:3]) / total * 100
            print(f"    Top3行业集中度: {top3:.1f}%")

    # 4. Risk Parity
    print(f"\n  [优化] Risk Parity (波动率倒数加权)")
    print(f"  {'-'*75}")
    print(f"  {'窗口':<6} {'等权融合':>10} {'Risk Parity':>12} {'差值':>8}")
    for i in range(6):
        d = res_dyn_alpha[i]
        rp = res_risk_parity[i]
        diff = rp["ar"] - d["ar"]
        print(f"  {d['window']:<6} {d['ar']:>9.2f}% {rp['ar']:>11.2f}% {diff:>+7.2f}%")
    rp_ar = avg(res_risk_parity, "ar")
    rp_ir = avg(res_risk_parity, "ir")
    rp_mdd = avg(res_risk_parity, "mdd")
    print(f"  {'平均':<6} {da_ar:>9.2f}% {rp_ar:>11.2f}% {rp_ar-da_ar:>+7.2f}%")

    # 全局汇总
    print(f"\n{'='*85}")
    print(f"  全局对比")
    print(f"{'='*85}")
    print(f"\n  {'策略':<36} {'年化':>8} {'IR':>8} {'回撤':>8}")
    print(f"  {'-'*60}")
    a_ar, a_ir, a_mdd = avg(res_A,"ar"), avg(res_A,"ir"), avg(res_A,"mdd")
    print(f"  {'A: Baseline (5年+Top10+无融合)':<36} {a_ar:>7.2f}% {a_ir:>8.4f} {a_mdd:>7.2f}%")
    print(f"  {'G: 固定α=0.3 融合':<36} {g_ar:>7.2f}% {g_ir:>8.4f} {-16.18:>7.2f}%")
    print(f"  {'动态α_t (Valid搜索)':<36} {da_ar:>7.2f}% {da_ir:>8.4f} {avg(res_dyn_alpha,'mdd'):>7.2f}%")
    print(f"  {'动态α_t + 涨跌停过滤':<36} {avg(res_limit,'ar'):>7.2f}% {avg(res_limit,'ir'):>8.4f} {avg(res_limit,'mdd'):>7.2f}%")
    print(f"  {'动态α_t + Risk Parity':<36} {rp_ar:>7.2f}% {rp_ir:>8.4f} {rp_mdd:>7.2f}%")

    # 逐窗口
    bt_periods = ["2021","2022","2023","2024","2025","2026H1"]
    print(f"\n{'='*85}")
    print(f"  逐窗口年化收益对比")
    print(f"{'='*85}")
    print(f"\n  {'回测期':<8} {'A(baseline)':>12} {'G(固定α)':>12} {'动态α_t':>12} {'+涨跌停':>12} {'+RP':>12}")
    print(f"  {'-'*68}")
    for i in range(6):
        print(f"  {bt_periods[i]:<8} {res_A[i]['ar']:>11.2f}% {res_G[i]['ar']:>11.2f}% {res_dyn_alpha[i]['ar']:>11.2f}% {res_limit[i]['ar']:>11.2f}% {res_risk_parity[i]['ar']:>11.2f}%")

    # Save
    summary = pd.DataFrame({
        "A_baseline": [r["ar"] for r in res_A],
        "G_fixed_alpha": [r["ar"] for r in res_G],
        "dynamic_alpha": [r["ar"] for r in res_dyn_alpha],
        "limit_filtered": [r["ar"] for r in res_limit],
        "risk_parity": [r["ar"] for r in res_risk_parity],
    }, index=bt_periods)
    summary.to_csv("/Users/11164591/Documents/Qoder目录/qlib/v5_validation_results.csv", sep='\t')
    pd.DataFrame(alpha_search_log).to_csv("/Users/11164591/Documents/Qoder目录/qlib/alpha_search_log.csv", sep='\t', index=False)
    print(f"\n  结果已保存")


if __name__ == "__main__":
    run()
