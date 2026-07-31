"""
滚动训练 V3 — 3年窗口 + Top30 + 价值因子
==========================================
改进点（相对V2滚动）：
1. 滚动窗口从5年缩短至3年（2年train + 1年valid），更快遗忘陈旧分布
2. 选股从Top10扩至Top30，降低个股集中度风险
3. 新增价值因子：ROE、PE_3Y分位、PB_3Y分位、股息率代理

窗口划分：
  W1: train 2018-2020 (train 2018-2019, valid 2020) → backtest 2021
  W2: train 2019-2021 (train 2019-2020, valid 2021) → backtest 2022
  W3: train 2020-2022 (train 2020-2021, valid 2022) → backtest 2023
  W4: train 2021-2023 (train 2021-2022, valid 2023) → backtest 2024
  W5: train 2022-2024 (train 2022-2023, valid 2024) → backtest 2025
  W6: train 2023-2025 (train 2023-2024, valid 2025) → backtest 2026H1
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


# ================================================================
# 自定义类（与 V2 一致）
# ================================================================
class MonthlyTopkStrategy(TopkDropoutStrategy):
    """月频再平衡策略：仅在每月最后一个交易日调仓。"""
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
    """Alpha158 + 长周期价格因子。"""
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


# ================================================================
# 工具函数
# ================================================================
def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"


# ================================================================
# 基本面 + 价值因子加载
# ================================================================
def load_fundamental_features(universe, qlib_calendar):
    """原有基本面特征：FCF增速/利润增速/FCF利润比/FCF均值/FCF变异系数"""
    fcf_path = "/Users/11164591/Documents/Qoder目录/fcf_cache.csv"
    profit_path = "/Users/11164591/Documents/Qoder目录/profit_cache.csv"
    fcf_df = pd.read_csv(fcf_path)
    profit_df = pd.read_csv(profit_path)
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner"
    )
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
            if pd.isna(row["fcf_growth"]):
                continue
            year = int(row["year"])
            available_from = pd.Timestamp(f"{year + 1}-05-01")
            available_to = pd.Timestamp(f"{year + 2}-04-30")
            mask = (qlib_calendar >= available_from) & (qlib_calendar <= available_to)
            dates = qlib_calendar[mask]
            if len(dates) == 0:
                continue
            for d in dates:
                feature_rows.append({
                    "instrument": qlib_code, "datetime": d,
                    "fcf_growth": row["fcf_growth"],
                    "profit_growth": row["profit_growth"],
                    "fcf_profit_ratio": row["fcf_profit_ratio"],
                    "fcf_avg_3y_norm": row["fcf_avg_3y"] / 1e8,
                    "fcf_cv_3y": row["fcf_cv_3y"],
                })
    if not feature_rows:
        return None
    fund_df = pd.DataFrame(feature_rows)
    fund_df = fund_df.set_index(["datetime", "instrument"])
    fund_df = fund_df[~fund_df.index.duplicated(keep="last")]
    for col in fund_df.columns:
        median = fund_df[col].median()
        mad = (fund_df[col] - median).abs().median()
        if mad > 0:
            fund_df[col] = (fund_df[col] - median) / (1.4826 * mad)
        fund_df[col] = fund_df[col].clip(-3, 3).fillna(0)
    return fund_df


def load_value_factors(universe, qlib_calendar):
    """
    加载价值因子：ROE、PE_3Y分位、PB_3Y分位、股息率代理

    逻辑：
    - ROE: 年报净资产收益率(%)，次年5/1起可用
    - PE_TTM: $close / EPS_annual，取3年滚动截面分位数（越低越便宜）
    - PB: $close / BPS_annual，取3年滚动截面分位数（越低越便宜）
    - 股息率代理: div_payout / PE（高派息+低PE = 高股息率）

    PE/PB 分位数计算：
    - 对每个日期，用该股票过去约750个交易日(3年)的PE/PB值
    - 计算当前PE/PB在历史分布中的分位数 (0=历史最低, 1=历史最高)
    - 分位数越低 → 估值越便宜 → 预期收益越高
    """
    vf_path = "/Users/11164591/Documents/Qoder目录/value_factors_cache.csv"
    vf_df = pd.read_csv(vf_path)
    vf_df["date"] = pd.to_datetime(vf_df["date"])

    # 只取年报数据(quarter=12) 用于 PE/PB 计算
    annual = vf_df[vf_df["quarter"] == 12].copy()
    annual = annual.sort_values(["code", "year"]).reset_index(drop=True)

    # 构建 {code: {year: {roe, eps, bps, div_payout}}} 映射
    fin_data = {}
    for _, row in annual.iterrows():
        code = str(row["code"]).zfill(6)
        year = int(row["year"])
        if code not in fin_data:
            fin_data[code] = {}
        fin_data[code][year] = {
            "roe": row.get("roe", np.nan),
            "eps": row.get("eps", np.nan),
            "bps": row.get("bps", np.nan),
            "div_payout": row.get("div_payout", np.nan),
        }

    # 获取价格数据（用于计算PE/PB）
    cal_start = qlib_calendar[0]
    cal_end = qlib_calendar[-1]

    print("  获取收盘价用于 PE/PB 计算...")
    price_df = D.features(
        list(universe), ["$close"], start_time=cal_start, end_time=cal_end
    )
    if price_df is None or len(price_df) == 0:
        print("  [WARN] 无法获取价格数据，跳过价值因子")
        return None

    price_df = price_df.reset_index()
    price_df.columns = ["datetime", "instrument", "close"]

    # 对每只股票计算 PE/PB 并映射到日频
    feature_rows = []
    processed = 0

    for qlib_code in universe:
        code_short = qlib_code[2:]  # SH600519 -> 600519
        if code_short not in fin_data:
            continue

        fin = fin_data[code_short]
        stock_prices = price_df[price_df["instrument"] == qlib_code].copy()
        if len(stock_prices) == 0:
            continue

        stock_prices = stock_prices.sort_values("datetime").set_index("datetime")

        # 构建日频财务数据（年报次年5/1起可用）
        years = sorted(fin.keys())
        for i, year in enumerate(years):
            eps = fin[year]["eps"]
            bps = fin[year]["bps"]
            roe = fin[year]["roe"]
            div_payout = fin[year]["div_payout"]

            if pd.isna(eps) or eps == 0 or pd.isna(bps) or bps == 0:
                continue

            # 可用时间窗口
            avail_from = pd.Timestamp(f"{year + 1}-05-01")
            if i + 1 < len(years):
                avail_to = pd.Timestamp(f"{years[i+1] + 1}-04-30")
            else:
                avail_to = pd.Timestamp(f"{year + 2}-04-30")

            mask = (stock_prices.index >= avail_from) & (stock_prices.index <= avail_to)
            window = stock_prices[mask]
            if len(window) == 0:
                continue

            # 计算当前年度的 PE/PB 序列
            window = window.copy()
            window["pe"] = window["close"] / eps
            window["pb"] = window["close"] / bps
            window["roe_val"] = roe
            window["div_yield"] = np.nan
            if not pd.isna(div_payout) and eps != 0:
                # 股息率 ≈ 派息率 * EPS / Price = div_payout% * eps / close
                window["div_yield"] = (div_payout / 100.0) * eps / window["close"]

            # 计算3年滚动分位数（需要之前3年的PE/PB数据）
            hist_start = year - 3
            hist_pe = []
            hist_pb = []
            for h_year in range(hist_start, year):
                if h_year in fin:
                    h_eps = fin[h_year]["eps"]
                    h_bps = fin[h_year]["bps"]
                    if pd.isna(h_eps) or h_eps == 0 or pd.isna(h_bps) or h_bps == 0:
                        continue
                    h_from = pd.Timestamp(f"{h_year + 1}-05-01")
                    h_to = pd.Timestamp(f"{h_year + 2}-04-30")
                    h_mask = (stock_prices.index >= h_from) & (stock_prices.index <= h_to)
                    h_win = stock_prices[h_mask]
                    if len(h_win) > 0:
                        hist_pe.extend((h_win["close"] / h_eps).tolist())
                        hist_pb.extend((h_win["close"] / h_bps).tolist())

            # 当前 PE/PB 在3年历史中的分位数
            if len(hist_pe) >= 50:
                hist_pe_arr = np.array(hist_pe)
                hist_pb_arr = np.array(hist_pb)
                # 去掉极端值
                hist_pe_arr = hist_pe_arr[(hist_pe_arr > 0) & (hist_pe_arr < 500)]
                hist_pb_arr = hist_pb_arr[(hist_pb_arr > 0) & (hist_pb_arr < 50)]
                if len(hist_pe_arr) >= 30 and len(hist_pb_arr) >= 30:
                    window["pe_pct_3y"] = window["pe"].apply(
                        lambda x: np.searchsorted(np.sort(hist_pe_arr), x) / len(hist_pe_arr)
                        if 0 < x < 500 else np.nan
                    )
                    window["pb_pct_3y"] = window["pb"].apply(
                        lambda x: np.searchsorted(np.sort(hist_pb_arr), x) / len(hist_pb_arr)
                        if 0 < x < 50 else np.nan
                    )
                else:
                    window["pe_pct_3y"] = np.nan
                    window["pb_pct_3y"] = np.nan
            else:
                window["pe_pct_3y"] = np.nan
                window["pb_pct_3y"] = np.nan

            # 收集日频数据
            for dt, row in window.iterrows():
                if dt not in qlib_calendar_set:
                    continue
                feature_rows.append({
                    "instrument": qlib_code,
                    "datetime": dt,
                    "roe_annual": row["roe_val"],
                    "pe_pct_3y": row["pe_pct_3y"],
                    "pb_pct_3y": row["pb_pct_3y"],
                    "div_yield_est": row["div_yield"],
                })

        processed += 1
        if processed % 50 == 0:
            print(f"    处理 {processed}/{len(universe)} 只...")

    if not feature_rows:
        print("  [WARN] 无价值因子数据")
        return None

    vf_daily = pd.DataFrame(feature_rows)
    vf_daily = vf_daily.set_index(["datetime", "instrument"])
    vf_daily = vf_daily[~vf_daily.index.duplicated(keep="last")]

    # 截面标准化（每天对所有股票做 Z-Score）
    for col in vf_daily.columns:
        grp = vf_daily[col].groupby(level=0)
        median = grp.transform("median")
        mad = grp.transform(lambda x: (x - x.median()).abs().median())
        mad = mad.replace(0, np.nan)
        vf_daily[col] = ((vf_daily[col] - median) / (1.4826 * mad))
        vf_daily[col] = vf_daily[col].clip(-3, 3).fillna(0)

    print(f"  价值因子: {vf_daily.shape[0]} 行, {vf_daily.shape[1]} 列")
    print(f"  因子列: {vf_daily.columns.tolist()}")
    return vf_daily


def inject_features(dataset, feature_df):
    """将特征注入到 dataset 的 handler 内部数据中。"""
    handler = dataset.handler
    data = handler.fetch()
    handler_index = data.index
    aligned = feature_df.reindex(handler_index)
    if isinstance(data.columns, pd.MultiIndex):
        aligned.columns = pd.MultiIndex.from_tuples(
            [("feature", c) for c in aligned.columns]
        )
    else:
        aligned.columns = [("feature", c) for c in aligned.columns]
    merged = data.join(aligned).fillna(0)
    handler._data = merged
    return merged


# ================================================================
# 滚动窗口定义（3年窗口 = 2年train + 1年valid + 1年backtest）
# ================================================================
ROLLING_WINDOWS = [
    {"train": ("2018-01-01", "2019-12-31"), "valid": ("2020-01-01", "2020-12-31"),
     "backtest": ("2021-01-01", "2021-12-31"), "name": "W1"},
    {"train": ("2019-01-01", "2020-12-31"), "valid": ("2021-01-01", "2021-12-31"),
     "backtest": ("2022-01-01", "2022-12-31"), "name": "W2"},
    {"train": ("2020-01-01", "2021-12-31"), "valid": ("2022-01-01", "2022-12-31"),
     "backtest": ("2023-01-01", "2023-12-31"), "name": "W3"},
    {"train": ("2021-01-01", "2022-12-31"), "valid": ("2023-01-01", "2023-12-31"),
     "backtest": ("2024-01-01", "2024-12-31"), "name": "W4"},
    {"train": ("2022-01-01", "2023-12-31"), "valid": ("2024-01-01", "2024-12-31"),
     "backtest": ("2025-01-01", "2025-12-31"), "name": "W5"},
    {"train": ("2023-01-01", "2024-12-31"), "valid": ("2025-01-01", "2025-12-31"),
     "backtest": ("2026-01-01", "2026-07-21"), "name": "W6"},
]

MODEL_CONFIG = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.69,
        "lambda_l2": 580.97,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}


def run_single_window(window, custom_universe, fund_features, value_factors, qlib_calendar, top_k_num):
    """执行单个滚动窗口的训练+回测。"""
    name = window["name"]
    train_start, train_end = window["train"]
    valid_start, valid_end = window["valid"]
    bt_start, bt_end = window["backtest"]

    data_start = train_start
    data_end = bt_end

    print(f"\n{'#' * 70}")
    print(f"  {name}: 训练 {train_start}~{train_end} | 验证 {valid_start}~{valid_end} | 回测 {bt_start}~{bt_end}")
    print(f"{'#' * 70}")

    data_handler_config = {
        "start_time": data_start,
        "end_time": data_end,
        "fit_start_time": train_start,
        "fit_end_time": train_end,
        "instruments": custom_universe,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -20) / $close - 1"],
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158Enhanced",
                "module_path": "__main__",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (bt_start, bt_end),
            },
        },
    }

    dataset = init_instance_by_config(dataset_config)

    # 注入基本面特征
    if fund_features is not None:
        inject_features(dataset, fund_features)

    # 注入价值因子
    if value_factors is not None:
        inject_features(dataset, value_factors)

    # 训练模型
    model = init_instance_by_config(MODEL_CONFIG)
    exp_name = f"rolling_v3_{name}"
    with R.start(experiment_name=exp_name):
        recorder = R.get_recorder()
        print(f"  训练模型 (RID={recorder.id[:12]}...)")
        model.fit(dataset)
        R.save_objects(trained_model=model)

        sig_rec = SignalRecord(model, dataset, recorder)
        sig_rec.generate()

        pred = recorder.load_object("pred.pkl")

        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
            },
            "strategy": {
                "class": "MonthlyTopkStrategy",
                "module_path": "__main__",
                "kwargs": {"topk": top_k_num, "n_drop": top_k_num, "signal": pred},
            },
            "backtest": {
                "start_time": bt_start,
                "end_time": bt_end,
                "account": 100000000,
                "benchmark": None,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0015,
                    "close_cost": 0.0025,
                    "min_cost": 5,
                },
            },
        }

        port_ana_rec = PortAnaRecord(recorder, port_analysis_config, "day")
        port_ana_rec.generate()

        metrics = recorder.list_metrics()

        def _safe(m, key, fb=None, default=0.0):
            v = m.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
            if fb:
                v2 = m.get(fb)
                if v2 is not None and not (isinstance(v2, float) and np.isnan(v2)):
                    return v2
            return default

        result = {
            "window": name,
            "train_period": f"{train_start}~{train_end}",
            "backtest_period": f"{bt_start}~{bt_end}",
            "l2_train": metrics.get("l2.train"),
            "l2_valid": metrics.get("l2.valid"),
            "ar_without_cost": _safe(metrics, "1day.excess_return_without_cost.annualized_return"),
            "ir_without_cost": _safe(metrics, "1day.excess_return_without_cost.information_ratio"),
            "mdd_without_cost": _safe(metrics, "1day.excess_return_without_cost.max_drawdown"),
            "ar_with_cost": _safe(metrics, "1day.excess_return_with_cost.annualized_return",
                                   "1day.excess_return_without_cost.annualized_return"),
            "ir_with_cost": _safe(metrics, "1day.excess_return_with_cost.information_ratio",
                                   "1day.excess_return_without_cost.information_ratio"),
            "mdd_with_cost": _safe(metrics, "1day.excess_return_with_cost.max_drawdown",
                                    "1day.excess_return_without_cost.max_drawdown"),
            "std_with_cost": _safe(metrics, "1day.excess_return_with_cost.std",
                                    "1day.excess_return_without_cost.std"),
        }

        ar_wc = (result["ar_with_cost"] or 0) * 100
        ir_wc = result["ir_with_cost"] or 0
        mdd_wc = (result["mdd_with_cost"] or 0) * 100
        print(f"\n  [{name} 结果]  扣成本后: 年化 {ar_wc:.2f}%,  IR {ir_wc:.4f},  最大回撤 {mdd_wc:.2f}%")
        return result


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    csv_path = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
    df = pd.read_csv(csv_path)
    custom_universe = df["code"].apply(format_qlib_code).tolist()
    print(f"股票池: {len(custom_universe)} 只")

    qlib_calendar = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    qlib_calendar_set = set(qlib_calendar)
    print(f"交易日历: {qlib_calendar[0].date()} ~ {qlib_calendar[-1].date()} ({len(qlib_calendar)} 天)")

    # 加载原有基本面特征
    print("--- 加载基本面特征 (FCF/利润) ---")
    fund_features = load_fundamental_features(custom_universe, qlib_calendar)
    if fund_features is not None:
        print(f"  基本面: {fund_features.shape[0]} 行, {fund_features.shape[1]} 列")

    # 加载价值因子（ROE/PE分位/PB分位/股息率）
    print("--- 加载价值因子 (ROE/PE/PB/股息率) ---")
    value_factors = load_value_factors(custom_universe, qlib_calendar)

    # ★ Top-K 从 10 扩至 30
    top_k_num = 30

    # 逐窗口滚动训练
    all_results = []
    for window in ROLLING_WINDOWS:
        result = run_single_window(window, custom_universe, fund_features, value_factors, qlib_calendar, top_k_num)
        all_results.append(result)

    # ================================================================
    # 汇总对比
    # ================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  V3 滚动训练汇总 — 3年窗口 + Top30 + 价值因子")
    print(f"{'=' * 90}")
    print(f"\n  {'窗口':<6} {'训练期':<24} {'回测期':<24} {'年化(扣成本)':>12} {'IR':>8} {'最大回撤':>10}")
    print(f"  {'-' * 84}")

    ar_list, ir_list, mdd_list = [], [], []
    for r in all_results:
        ar_wc = (r["ar_with_cost"] or 0) * 100
        ir_wc = r["ir_with_cost"] or 0
        mdd_wc = (r["mdd_with_cost"] or 0) * 100
        ar_list.append(ar_wc)
        ir_list.append(ir_wc)
        mdd_list.append(mdd_wc)
        print(f"  {r['window']:<6} {r['train_period']:<24} {r['backtest_period']:<24} "
              f"{ar_wc:>11.2f}% {ir_wc:>8.4f} {mdd_wc:>9.2f}%")

    print(f"\n  统计汇总 (扣成本后):")
    print(f"    平均年化: {np.mean(ar_list):.2f}%  (std={np.std(ar_list):.2f}%)")
    print(f"    平均 IR:  {np.mean(ir_list):.4f}  (std={np.std(ir_list):.4f})")
    print(f"    平均回撤: {np.mean(mdd_list):.2f}%  (std={np.std(mdd_list):.2f}%)")
    print(f"    正收益:   {sum(1 for a in ar_list if a > 0)}/{len(ar_list)}")
    print(f"    正 IR:    {sum(1 for i in ir_list if i > 0)}/{len(ir_list)}")

    # 对比
    print(f"\n{'=' * 90}")
    print(f"  版本对比")
    print(f"{'=' * 90}")
    print(f"  {'版本':<28} {'平均年化':>10} {'平均IR':>8} {'平均回撤':>10}")
    print(f"  {'-' * 56}")
    print(f"  {'静态V2 OOT(2020.10-2026)':<28} {'≈ -8%':>10} {'0.38':>8} {'-54%':>10}")
    print(f"  {'V2滚动 5年窗口 Top10':<28} {'13.46%':>10} {'0.65':>8} {'-16.75%':>10}")
    print(f"  {'V3滚动 3年窗口 Top30+价值':<28} {np.mean(ar_list):>9.2f}% {np.mean(ir_list):>8.4f} {np.mean(mdd_list):>9.2f}%")

    # 保存
    results_df = pd.DataFrame(all_results)
    results_path = "/Users/11164591/Documents/Qoder目录/qlib/rolling_v3_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  结果已保存至: {results_path}")
