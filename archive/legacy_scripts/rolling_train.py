"""
滚动训练脚本 — 滑动固定窗口（5年训练 + 1年回测）
====================================================
策略：
  每次使用过去5年的数据训练模型（4年train + 1年valid），
  然后在接下来1年上做样本外回测，窗口逐年滑动。

窗口划分：
  W1: train 2016-2020 (train 2016-2019, valid 2020) → backtest 2021
  W2: train 2017-2021 (train 2017-2020, valid 2021) → backtest 2022
  W3: train 2018-2022 (train 2018-2021, valid 2022) → backtest 2023
  W4: train 2019-2023 (train 2019-2022, valid 2023) → backtest 2024
  W5: train 2020-2024 (train 2020-2023, valid 2024) → backtest 2025
  W6: train 2021-2025 (train 2021-2024, valid 2025) → backtest 2026(半年)

核心逻辑：
  - 每个窗口重新训练 LightGBM，丢弃旧模型（避免陈旧分布）
  - 特征体系与 V2 完全一致：Alpha158Enhanced + 基本面特征
  - 月频再平衡策略 MonthlyTopkStrategy
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
# 自定义类（与 V2 完全一致）
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
# 基本面特征加载（与 V2 一致）
# ================================================================
def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"


def load_fundamental_features(universe, qlib_calendar):
    """从 fcf_cache.csv / profit_cache.csv 计算基本面特征并映射到日频。"""
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


def inject_fundamental_features(dataset, fund_features):
    """将基本面特征注入到 dataset 的 handler 内部数据中。"""
    handler = dataset.handler
    data = handler.fetch()
    handler_index = data.index
    fund_aligned = fund_features.reindex(handler_index)
    if isinstance(data.columns, pd.MultiIndex):
        fund_aligned.columns = pd.MultiIndex.from_tuples(
            [("feature", c) for c in fund_aligned.columns]
        )
    else:
        fund_aligned.columns = [("feature", c) for c in fund_aligned.columns]
    merged = data.join(fund_aligned).fillna(0)
    handler._data = merged
    return merged


# ================================================================
# 滚动窗口定义
# ================================================================
# 每个窗口: (train_start, train_end, valid_end, backtest_start, backtest_end)
# 固定5年窗口 = 4年train + 1年valid，然后1年backtest
ROLLING_WINDOWS = [
    # W1: 训练 2016-2020, 回测 2021
    {"train": ("2016-01-01", "2019-12-31"), "valid": ("2020-01-01", "2020-12-31"),
     "backtest": ("2021-01-01", "2021-12-31"), "name": "W1"},
    # W2: 训练 2017-2021, 回测 2022
    {"train": ("2017-01-01", "2020-12-31"), "valid": ("2021-01-01", "2021-12-31"),
     "backtest": ("2022-01-01", "2022-12-31"), "name": "W2"},
    # W3: 训练 2018-2022, 回测 2023
    {"train": ("2018-01-01", "2021-12-31"), "valid": ("2022-01-01", "2022-12-31"),
     "backtest": ("2023-01-01", "2023-12-31"), "name": "W3"},
    # W4: 训练 2019-2023, 回测 2024
    {"train": ("2019-01-01", "2022-12-31"), "valid": ("2023-01-01", "2023-12-31"),
     "backtest": ("2024-01-01", "2024-12-31"), "name": "W4"},
    # W5: 训练 2020-2024, 回测 2025
    {"train": ("2020-01-01", "2023-12-31"), "valid": ("2024-01-01", "2024-12-31"),
     "backtest": ("2025-01-01", "2025-12-31"), "name": "W5"},
    # W6: 训练 2021-2025, 回测 2026（数据到2026-07-23，回测截至7-21避免日历边界越界）
    {"train": ("2021-01-01", "2024-12-31"), "valid": ("2025-01-01", "2025-12-31"),
     "backtest": ("2026-01-01", "2026-07-21"), "name": "W6"},
]

# 模型超参（与 V2 一致）
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


def run_single_window(window, custom_universe, fund_features, qlib_calendar, top_k_num):
    """执行单个滚动窗口的训练+回测。"""
    name = window["name"]
    train_start, train_end = window["train"]
    valid_start, valid_end = window["valid"]
    bt_start, bt_end = window["backtest"]

    # 数据范围 = 训练起始 ~ 回测结束
    data_start = train_start
    data_end = bt_end

    print(f"\n{'#' * 70}")
    print(f"  {name}: 训练 {train_start}~{train_end} | 验证 {valid_start}~{valid_end} | 回测 {bt_start}~{bt_end}")
    print(f"{'#' * 70}")

    # 1) 构建数据集
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

    # 2) 注入基本面特征
    if fund_features is not None:
        inject_fundamental_features(dataset, fund_features)

    # 3) 训练模型（每个窗口重新训练）
    model = init_instance_by_config(MODEL_CONFIG)

    exp_name = f"rolling_{name}"
    with R.start(experiment_name=exp_name):
        recorder = R.get_recorder()
        print(f"  训练模型 (RID={recorder.id[:12]}...)")
        model.fit(dataset)
        R.save_objects(trained_model=model)

        # 4) 生成预测信号
        sig_rec = SignalRecord(model, dataset, recorder)
        sig_rec.generate()

        # 5) 月频再平衡回测
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

        # 6) 收集指标
        metrics = recorder.list_metrics()

        def _safe_metric(metrics, key, fallback_key=None, default=0.0):
            v = metrics.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
            if fallback_key:
                v2 = metrics.get(fallback_key)
                if v2 is not None and not (isinstance(v2, float) and np.isnan(v2)):
                    return v2
            return default
        
        result = {
            "window": name,
            "train_period": f"{train_start}~{train_end}",
            "valid_period": f"{valid_start}~{valid_end}",
            "backtest_period": f"{bt_start}~{bt_end}",
            "l2_train": metrics.get("l2.train", None),
            "l2_valid": metrics.get("l2.valid", None),
            "ar_without_cost": _safe_metric(metrics, "1day.excess_return_without_cost.annualized_return"),
            "ir_without_cost": _safe_metric(metrics, "1day.excess_return_without_cost.information_ratio"),
            "mdd_without_cost": _safe_metric(metrics, "1day.excess_return_without_cost.max_drawdown"),
            "std_without_cost": _safe_metric(metrics, "1day.excess_return_without_cost.std"),
            "ar_with_cost": _safe_metric(metrics, "1day.excess_return_with_cost.annualized_return",
                                           "1day.excess_return_without_cost.annualized_return"),
            "ir_with_cost": _safe_metric(metrics, "1day.excess_return_with_cost.information_ratio",
                                          "1day.excess_return_without_cost.information_ratio"),
            "mdd_with_cost": _safe_metric(metrics, "1day.excess_return_with_cost.max_drawdown",
                                           "1day.excess_return_without_cost.max_drawdown"),
            "std_with_cost": _safe_metric(metrics, "1day.excess_return_with_cost.std",
                                           "1day.excess_return_without_cost.std"),
            "ffr": metrics.get("1day.ffr", None),
        }

        # 打印窗口结果
        ar_wc = result["ar_with_cost"] or 0
        ir_wc = result["ir_with_cost"] or 0
        mdd_wc = result["mdd_with_cost"] or 0
        ar_nc = result["ar_without_cost"] or 0
        ir_nc = result["ir_without_cost"] or 0
        mdd_nc = result["mdd_without_cost"] or 0

        print(f"\n  [{name} 结果]")
        print(f"  训练集 L2:  {result['l2_train']:.6f}" if result['l2_train'] else "  训练集 L2:  N/A")
        print(f"  验证集 L2:  {result['l2_valid']:.6f}" if result['l2_valid'] else "  验证集 L2:  N/A")
        print(f"  扣成本前:  年化 {ar_nc*100:.2f}%,  IR {ir_nc:.4f},  最大回撤 {mdd_nc*100:.2f}%")
        print(f"  扣成本后:  年化 {ar_wc*100:.2f}%,  IR {ir_wc:.4f},  最大回撤 {mdd_wc*100:.2f}%")

        return result


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    # 初始化 qlib
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 读取股票池
    csv_path = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
    df = pd.read_csv(csv_path)
    custom_universe = df["code"].apply(format_qlib_code).tolist()
    print(f"股票池: {len(custom_universe)} 只")

    # 获取 qlib 交易日历
    qlib_calendar = D.calendar(start_time="2016-01-01", end_time="2026-07-23")
    print(f"交易日历: {qlib_calendar[0].date()} ~ {qlib_calendar[-1].date()} ({len(qlib_calendar)} 天)")

    # 加载基本面特征（全量，各窗口按需切片）
    print("--- 加载基本面特征 ---")
    fund_features = load_fundamental_features(custom_universe, qlib_calendar)
    if fund_features is not None:
        print(f"  基本面特征: {fund_features.shape[0]} 行, {fund_features.shape[1]} 列")

    top_k_num = 10

    # ================================================================
    # 逐窗口滚动训练
    # ================================================================
    all_results = []
    for window in ROLLING_WINDOWS:
        result = run_single_window(window, custom_universe, fund_features, qlib_calendar, top_k_num)
        all_results.append(result)

    # ================================================================
    # 汇总对比
    # ================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  滚动训练汇总 — 滑动固定窗口（5年训练 + 1年回测）")
    print(f"{'=' * 90}")

    # 表头
    print(f"\n  {'窗口':<6} {'训练期':<24} {'回测期':<24} "
          f"{'年化(扣成本)':>12} {'IR(扣成本)':>10} {'最大回撤':>10} {'年化(无成本)':>12} {'IR(无成本)':>10}")
    print(f"  {'-' * 108}")

    ar_list = []
    ir_list = []
    mdd_list = []

    for r in all_results:
        ar_wc = (r["ar_with_cost"] or 0) * 100
        ir_wc = r["ir_with_cost"] or 0
        mdd_wc = (r["mdd_with_cost"] or 0) * 100
        ar_nc = (r["ar_without_cost"] or 0) * 100
        ir_nc = r["ir_without_cost"] or 0
        ar_list.append(ar_wc)
        ir_list.append(ir_wc)
        mdd_list.append(mdd_wc)

        print(f"  {r['window']:<6} {r['train_period']:<24} {r['backtest_period']:<24} "
              f"{ar_wc:>11.2f}% {ir_wc:>10.4f} {mdd_wc:>9.2f}% {ar_nc:>11.2f}% {ir_nc:>10.4f}")

    # 统计
    print(f"\n  {'-' * 80}")
    print(f"  统计汇总 (扣成本后):")
    print(f"    平均年化收益:  {np.mean(ar_list):.2f}%  (std={np.std(ar_list):.2f}%)")
    print(f"    平均 IR:       {np.mean(ir_list):.4f}  (std={np.std(ir_list):.4f})")
    print(f"    平均最大回撤:  {np.mean(mdd_list):.2f}%  (std={np.std(mdd_list):.2f}%)")
    print(f"    正收益窗口:    {sum(1 for a in ar_list if a > 0)}/{len(ar_list)}")
    print(f"    正 IR 窗口:    {sum(1 for i in ir_list if i > 0)}/{len(ir_list)}")

    # 对比静态模型 OOT
    print(f"\n{'=' * 90}")
    print(f"  静态模型(V2 OOT) vs 滚动训练 对比")
    print(f"{'=' * 90}")
    print(f"  静态 V2 OOT (2020.10-2026.07):  年化 ≈ -8%, IR ≈ 0.38, 最大回撤 ≈ -54%")
    print(f"  滚动训练平均 (扣成本):          年化 {np.mean(ar_list):.2f}%, IR {np.mean(ir_list):.4f}, 最大回撤 {np.mean(mdd_list):.2f}%")

    # 判定
    if np.mean(ir_list) > 0.38:
        print(f"\n  ✓ 滚动训练 IR ({np.mean(ir_list):.4f}) > 静态模型 (0.38)，滚动训练有效缓解 Alpha 衰减")
    elif np.mean(ir_list) > 0:
        print(f"\n  △ 滚动训练 IR ({np.mean(ir_list):.4f}) > 0 但未显著超越静态模型")
    else:
        print(f"\n  ✗ 滚动训练 IR ({np.mean(ir_list):.4f}) <= 0，Alpha 衰减问题需要更深层解决方案")

    if np.mean(ar_list) > 0:
        print(f"  ✓ 滚动训练平均年化 {np.mean(ar_list):.2f}% > 0，策略方向正确")
    else:
        print(f"  ✗ 滚动训练平均年化 {np.mean(ar_list):.2f}% < 0，需要重新审视特征体系")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_path = "/Users/11164591/Documents/Qoder目录/qlib/rolling_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  结果已保存至: {results_path}")
