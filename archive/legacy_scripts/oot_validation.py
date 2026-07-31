"""
OOT 验证脚本 — 锁死 V2 模型权重，对 2020.10-2026.07 进行样本外测试
===============================================================
1. 加载 V2 保存的 LightGBM 模型（不重新训练）
2. 使用相同的 Alpha158Enhanced + 基本面特征配置
3. 在 2020-10-01 ~ 2026-07-21 期间执行月频再平衡回测
4. 分阶段分析 Alpha 衰减情况
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
    _debug_count = 0  # 类级调试计数器

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

        # 调试输出：前5次调用 + 每月第一次
        if MonthlyTopkStrategy._debug_count < 5 or should_rebalance:
            MonthlyTopkStrategy._debug_count += 1
            print(f"  [DEBUG] step={trade_step} date={cur_date.date()} rebalance={should_rebalance} "
                  f"exchange={self.trade_exchange is not None} signal={self.signal is not None}")

        if not should_rebalance:
            return TradeDecisionWO([], self)
        self._last_rebalance_period = cur_period

        # 调试：检查信号
        try:
            pred_start, pred_end = self.trade_calendar.get_step_time(trade_step, shift=1)
            pred_score = self.signal.get_signal(start_time=pred_start, end_time=pred_end)
            sig_len = len(pred_score) if pred_score is not None else 0
            if MonthlyTopkStrategy._debug_count <= 10:
                print(f"  [DEBUG] signal range={pred_start} ~ {pred_end}, scores={sig_len}")
        except Exception as e:
            print(f"  [DEBUG] signal error: {e}")

        decision = super().generate_trade_decision(execute_result)
        if MonthlyTopkStrategy._debug_count <= 10:
            print(f"  [DEBUG] decision orders={len(decision.order_list) if hasattr(decision, 'order_list') else '?'}")
        return decision


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
    """与 V2 完全一致的基本面特征加载"""
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
    handler = dataset.handler
    data = handler.fetch()
    handler_index = data.index
    fund_aligned = fund_features.reindex(handler_index)
    if isinstance(data.columns, pd.MultiIndex):
        fund_aligned.columns = pd.MultiIndex.from_tuples(
            [("feature", c) for c in fund_aligned.columns]
        )
    merged = data.join(fund_aligned).fillna(0)
    handler._data = merged
    return merged


# ================================================================
# 主流程
# ================================================================
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

# 加载基本面特征
print("--- 加载基本面特征 ---")
fund_features = load_fundamental_features(custom_universe, qlib_calendar)
if fund_features is not None:
    print(f"  基本面特征: {fund_features.shape[0]} 行, {fund_features.shape[1]} 列")

# 数据集配置（与 V2 一致，但扩展时间范围到 2026）
data_handler_config = {
    "start_time": "2016-01-01",
    "end_time": "2026-07-23",
    "fit_start_time": "2016-01-01",
    "fit_end_time": "2018-12-31",
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
            "train": ("2016-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2019-09-30"),
            "test": ("2020-10-01", "2026-07-21"),
        },
    },
}

# 回测配置
top_k_num = 10
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    },
    "strategy": {
        "class": "MonthlyTopkStrategy",
        "module_path": "__main__",
        "kwargs": {"topk": top_k_num, "n_drop": top_k_num},
    },
    "backtest": {
        "start_time": "2020-10-01",
        "end_time": "2026-07-21",
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


if __name__ == "__main__":
    # --- 1. 构建数据集 ---
    print("\n=== 1. 构建数据集 (扩展到 2026-07) ===")
    dataset = init_instance_by_config(dataset_config)

    # --- 2. 注入基本面特征 ---
    if fund_features is not None:
        print("\n=== 2. 注入基本面特征 ===")
        inject_fundamental_features(dataset, fund_features)

    # --- 3. 加载 V2 锁定的模型 ---
    print("\n=== 3. 加载 V2 模型 (锁死权重，不重新训练) ===")
    exp = R.get_exp(experiment_name="fundamental_v2_monthly")
    recorders = exp.list_recorders()
    # 取第一个 recorder（V2 训练时保存的）
    first_rid = list(recorders.keys())[0]
    recorder_v2 = recorders[first_rid]
    model = recorder_v2.load_object("trained_model")
    print(f"  模型已加载: {type(model).__name__} (RID={first_rid[:12]}...)")

    # --- 4. OOT 预测与回测 ---
    with R.start(experiment_name="oot_validation_v2"):
        # 用新的 recorder 记录 OOT 结果
        recorder = R.get_recorder()

        print("\n=== 4. 生成 OOT 预测信号 (2020-10 ~ 2026-07) ===")
        sig_rec = SignalRecord(model, dataset, recorder)
        sig_rec.generate()

        print("\n=== 5. 月频再平衡回测 (OOT 期) ===")
        pred = recorder.load_object("pred.pkl")
        port_analysis_config["strategy"]["kwargs"]["signal"] = pred
        port_analysis_config["backtest"]["benchmark"] = None
        port_ana_rec = PortAnaRecord(recorder, port_analysis_config, "day")
        port_ana_rec.generate()

        # --- 6. 输出结果 ---
        print("\n=== 6. OOT 评价指标 ===")
        metrics = recorder.list_metrics()

        print(f"\n所有 metrics ({len(metrics)} 个):")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

        # 汇总
        print(f"\n{'=' * 70}")
        print(f"  OOT 验证结果 — V2 模型锁死权重，样本外测试")
        print(f"{'=' * 70}")
        print(f"  模型:        V2 LightGBM (Alpha158Enhanced + 基本面)")
        print(f"  训练期:      2016-01-01 ~ 2018-12-31 (已锁定)")
        print(f"  OOT 测试期:  2020-10-01 ~ 2026-07-21 (约5.8年)")
        print(f"  标签:        20日累计收益率")
        print(f"  策略:        MonthlyTopkStrategy (月末调仓, TopK={top_k_num})")
        print(f"  股票池:      {len(custom_universe)} 只 (227只有完整数据)")

        print(f"\n[模型训练指标 (V2锁定)]")
        print(f"  训练集 L2:   {metrics.get('l2.train', 0):.6f}")
        print(f"  验证集 L2:   {metrics.get('l2.valid', 0):.6f}")

        print(f"\n[OOT 回测收益 — 扣除交易成本前]")
        ar_nc = metrics.get("1day.excess_return_without_cost.annualized_return", 0)
        ir_nc = metrics.get("1day.excess_return_without_cost.information_ratio", 0)
        md_nc = metrics.get("1day.excess_return_without_cost.max_drawdown", 0)
        std_nc = metrics.get("1day.excess_return_without_cost.std", 0)
        print(f"  年化收益率:  {ar_nc * 100:>8.2f}%")
        print(f"  信息比率:    {ir_nc:>8.4f}")
        print(f"  最大回撤:    {md_nc * 100:>8.2f}%")
        print(f"  日均波动:    {std_nc * 100:>8.2f}%")

        print(f"\n[OOT 回测收益 — 扣除交易成本后]")
        ar_wc = metrics.get("1day.excess_return_with_cost.annualized_return", 0)
        ir_wc = metrics.get("1day.excess_return_with_cost.information_ratio", 0)
        md_wc = metrics.get("1day.excess_return_with_cost.max_drawdown", 0)
        std_wc = metrics.get("1day.excess_return_with_cost.std", 0)
        print(f"  年化收益率:  {ar_wc * 100:>8.2f}%")
        print(f"  信息比率:    {ir_wc:>8.4f}")
        print(f"  最大回撤:    {md_wc * 100:>8.2f}%")
        print(f"  日均波动:    {std_wc * 100:>8.2f}%")

        print(f"\n[交易统计]")
        print(f"  订单成交率:  {metrics.get('1day.ffr', 0) * 100:.1f}%")

        # --- 7. 分阶段 Alpha 衰减分析 ---
        print(f"\n{'=' * 70}")
        print(f"  分阶段 Alpha 衰减分析")
        print(f"{'=' * 70}")

        # 获取日频收益率
        try:
            report_normal = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
            if report_normal is not None and len(report_normal) > 0:
                report_normal = report_normal.copy()
                report_normal["date"] = pd.to_datetime(report_normal.index if isinstance(report_normal.index, pd.DatetimeIndex) else report_normal["date"])
                report_normal["year"] = report_normal["date"].dt.year
                report_normal["month"] = report_normal["date"].dt.to_period("M")

                # 按年统计
                print(f"\n  {'年份':<8} {'累计收益':>10} {'年化收益':>10} {'最大回撤':>10} {'夏普比率':>10}")
                print(f"  {'-'*48}")
                for year in sorted(report_normal["year"].unique()):
                    yr_data = report_normal[report_normal["year"] == year]
                    if "return" in yr_data.columns:
                        ret = yr_data["return"].values
                    elif "ret" in yr_data.columns:
                        ret = yr_data["ret"].values
                    else:
                        # Try to find the return column
                        ret_cols = [c for c in yr_data.columns if "return" in c.lower() or "ret" in c.lower()]
                        if ret_cols:
                            ret = yr_data[ret_cols[0]].values
                        else:
                            print(f"  {year:<8} (无法找到收益率列)")
                            continue

                    cum_ret = (1 + ret).prod() - 1
                    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
                    # Max drawdown
                    cum = np.cumprod(1 + ret)
                    running_max = np.maximum.accumulate(cum)
                    drawdown = (cum - running_max) / running_max
                    max_dd = drawdown.min()
                    # Sharpe (annualized, rf=0)
                    if ret.std() > 0:
                        sharpe = ret.mean() / ret.std() * np.sqrt(252)
                    else:
                        sharpe = 0
                    print(f"  {year:<8} {cum_ret*100:>9.2f}% {ann_ret*100:>9.2f}% {max_dd*100:>9.2f}% {sharpe:>10.4f}")
        except Exception as e:
            print(f"  分阶段分析失败: {e}")

        # --- 8. V2 vs OOT 对比 ---
        print(f"\n{'=' * 70}")
        print(f"  V2 回测期 vs OOT 期 对比")
        print(f"{'=' * 70}")
        print(f"  {'指标':<16} {'V2(2019.10-2020.09)':>22} {'OOT(2020.10-2026.07)':>22}")
        print(f"  {'-'*60}")
        print(f"  {'年化收益(扣成本)':<16} {'23.72%':>22} {ar_wc*100:>21.2f}%")
        print(f"  {'信息比率':<16} {'1.34':>22} {ir_wc:>22.4f}")
        print(f"  {'最大回撤':<16} {'-9.44%':>22} {md_wc*100:>21.2f}%")
        print(f"  {'日均波动':<16} {'1.15%':>22} {std_wc*100:>21.2f}%")

        # 判定
        print(f"\n{'=' * 70}")
        print(f"  OOT 判定")
        print(f"{'=' * 70}")
        if md_wc > -0.15 and ir_wc > 0:
            print(f"  ✓ 最大回撤 {md_wc*100:.2f}% < 15%, IR {ir_wc:.4f} > 0")
            print(f"  ✓ 特征体系抓住了底层定价逻辑，Alpha 未严重衰减")
        elif ir_wc > 0:
            print(f"  △ IR {ir_wc:.4f} > 0 但最大回撤 {md_wc*100:.2f}% > 15%")
            print(f"  △ 存在一定 Alpha 衰减，但方向依然正确")
        else:
            print(f"  ✗ IR {ir_wc:.4f} <= 0, Alpha 严重衰减")
            print(f"  ✗ 静态模型无法适应市场风格切换，需要引入滚动训练")
