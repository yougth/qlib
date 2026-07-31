"""
多因子模型回测 V2 — 月频标签 + 月度再平衡 + 扩充低频特征
===========================================================
改动点：
1. 标签：Ref($close, -20)/$close - 1（20日累计收益率，月频）
2. 策略：MonthlyTopkStrategy（每月最后一个交易日调仓）
3. 特征：Alpha158 + 长周期动量/波动率/布林位置 + 基本面特征(FCF增速/利润增速/FCF利润比)
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import copy
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


# ================================================================
# 1. 自定义月频再平衡策略
# ================================================================
class MonthlyTopkStrategy(TopkDropoutStrategy):
    """月频再平衡策略：仅在每月最后一个交易日调仓，其余日期不交易。

    判断逻辑：若下一交易日所属月份与当前不同，则今日为月末调仓日。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_rebalance_period = None

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        cur_start, _ = self.trade_calendar.get_step_time(trade_step)
        cur_date = pd.Timestamp(cur_start)
        cur_period = cur_date.to_period("M")

        # 判断是否为月末（下一交易日跨月）
        should_rebalance = False
        try:
            next_start, _ = self.trade_calendar.get_step_time(trade_step + 1)
            next_date = pd.Timestamp(next_start)
            if next_date.to_period("M") != cur_period:
                should_rebalance = True
        except Exception:
            should_rebalance = True  # 日历最后一天

        # 首个交易日强制建仓
        if self._last_rebalance_period is None:
            should_rebalance = True

        if not should_rebalance:
            return TradeDecisionWO([], self)

        self._last_rebalance_period = cur_period
        return super().generate_trade_decision(execute_result)


# ================================================================
# 2. 扩展 Alpha158 处理器（加入长周期低频因子）
# ================================================================
class Alpha158Enhanced(Alpha158):
    """在 Alpha158 基础上注入长周期价格因子。

    新增因子：
    - ROC120 / ROC240：120日/240日动量
    - MA120 / MA240：120日/240日均线比率
    - STD120 / STD240：长周期波动率
    - BOLL60 / BOLL120：布林位置（价格相对均线的标准差位置）
    - VMA120 / VMA240：长周期成交量均线
    - CORR_PV20 / CORR_PV60：价量相关性
    - VRATIO_5_60 / VRATIO_5_120：短期vs长期成交量比（换手率代理）
    """

    def get_feature_config(self):
        fields, names = super().get_feature_config()

        extra_fields = [
            # 长周期动量
            "Ref($close, 120)/$close",
            "Ref($close, 240)/$close",
            # 长周期均线
            "Mean($close, 120)/$close",
            "Mean($close, 240)/$close",
            # 长周期波动率
            "Std($close, 120)/$close",
            "Std($close, 240)/$close",
            # 布林位置
            "($close - Mean($close, 60))/(Std($close, 60)+1e-12)",
            "($close - Mean($close, 120))/(Std($close, 120)+1e-12)",
            # 成交量长周期均线
            "Mean($volume, 120)/($volume+1e-12)",
            "Mean($volume, 240)/($volume+1e-12)",
            # 价量相关性
            "Corr($close, Log($volume+1), 20)",
            "Corr($close, Log($volume+1), 60)",
            # 换手率代理
            "Mean($volume, 5)/(Mean($volume, 60)+1e-12)",
            "Mean($volume, 5)/(Mean($volume, 120)+1e-12)",
        ]
        extra_names = [
            "ROC120", "ROC240",
            "MA120", "MA240",
            "STD120", "STD240",
            "BOLL60", "BOLL120",
            "VMA120", "VMA240",
            "CORR_PV20", "CORR_PV60",
            "VRATIO_5_60", "VRATIO_5_120",
        ]
        return fields + extra_fields, names + extra_names


# ================================================================
# 3. 基本面特征加载与注入
# ================================================================
def load_fundamental_features(universe, qlib_calendar):
    """从 fcf_cache.csv / profit_cache.csv 计算基本面特征并映射到日频。

    特征列表：
    - fcf_growth：FCF 同比增速
    - profit_growth：净利润同比增速
    - fcf_profit_ratio：FCF/净利润（盈利质量）
    - fcf_avg_3y_norm：3年平均FCF（归一化到亿元）
    - fcf_cv_3y：3年FCF变异系数（稳定性）

    点位时间映射：年报Y的数据在 Y+1年5月1日 起可用（年报披露截止4/30）。
    """
    fcf_path = "/Users/11164591/Documents/Qoder目录/fcf_cache.csv"
    profit_path = "/Users/11164591/Documents/Qoder目录/profit_cache.csv"

    fcf_df = pd.read_csv(fcf_path)
    profit_df = pd.read_csv(profit_path)

    # 合并 FCF 和净利润
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner"
    )
    merged = merged.dropna(subset=["fcf", "net_profit"])
    merged = merged.sort_values(["code", "year"]).reset_index(drop=True)

    # 逐公司计算特征
    feature_rows = []
    for code, grp in merged.groupby("code"):
        code_str = str(code).zfill(6)
        qlib_code = f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"
        if qlib_code not in universe:
            continue

        grp = grp.sort_values("year").copy()
        # 同比增速
        grp["fcf_growth"] = grp["fcf"].pct_change()
        grp["profit_growth"] = grp["net_profit"].pct_change()
        # FCF/净利润（盈利质量）
        grp["fcf_profit_ratio"] = grp["fcf"] / (grp["net_profit"].abs() + 1e-8)
        # 3年均值
        grp["fcf_avg_3y"] = grp["fcf"].rolling(3, min_periods=1).mean()
        # 3年变异系数
        rolling_std = grp["fcf"].rolling(3, min_periods=2).std()
        rolling_mean = grp["fcf"].rolling(3, min_periods=2).mean().abs()
        grp["fcf_cv_3y"] = rolling_std / (rolling_mean + 1e-8)

        for _, row in grp.iterrows():
            if pd.isna(row["fcf_growth"]):
                continue  # 第一年无增速，跳过
            year = int(row["year"])
            available_from = pd.Timestamp(f"{year + 1}-05-01")
            available_to = pd.Timestamp(f"{year + 2}-04-30")
            mask = (qlib_calendar >= available_from) & (qlib_calendar <= available_to)
            dates = qlib_calendar[mask]
            if len(dates) == 0:
                continue
            for d in dates:
                feature_rows.append({
                    "instrument": qlib_code,
                    "datetime": d,
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

    # 去重（同一天同一股票可能有多条记录，保留最新年份）
    fund_df = fund_df[~fund_df.index.duplicated(keep="last")]

    # Z-Score 标准化 + clip outlier（模拟 RobustZScoreNorm）
    for col in fund_df.columns:
        median = fund_df[col].median()
        mad = (fund_df[col] - median).abs().median()
        if mad > 0:
            fund_df[col] = (fund_df[col] - median) / (1.4826 * mad)
        fund_df[col] = fund_df[col].clip(-3, 3)
        fund_df[col] = fund_df[col].fillna(0)

    return fund_df


def inject_fundamental_features(dataset, fund_features):
    """将基本面特征注入到 dataset 的 handler 内部数据中。"""
    handler = dataset.handler
    data = handler.fetch()

    print(f"  原始特征列数: {data.shape[1]}")

    # 获取 handler 数据的索引，用于对齐
    handler_index = data.index

    # 将基本面特征 reindex 到 handler 的索引
    fund_aligned = fund_features.reindex(handler_index)

    # 给基本面特征加上 MultiIndex 列（与 Alpha158 一致）
    if isinstance(data.columns, pd.MultiIndex):
        fund_aligned.columns = pd.MultiIndex.from_tuples(
            [("feature", c) for c in fund_aligned.columns]
        )
    else:
        fund_aligned.columns = [("feature", c) for c in fund_aligned.columns]

    # 合并
    merged = data.join(fund_aligned)
    merged = merged.fillna(0)

    # 更新 handler 内部数据
    handler._data = merged

    print(f"  注入基本面特征数: {fund_features.shape[1]}")
    print(f"  合并后总特征列数: {merged.shape[1]}")
    return merged


# ================================================================
# 4. 主流程
# ================================================================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# --- 读取股票池 ---
csv_path = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
df = pd.read_csv(csv_path)

def format_qlib_code(code):
    code_str = str(code).zfill(6)
    return f"SH{code_str}" if code_str.startswith("6") else f"SZ{code_str}"

custom_universe = df["code"].apply(format_qlib_code).tolist()
print(f"股票池: {len(custom_universe)} 只 (连续10年净利润+FCF双正)")

# --- 获取 qlib 交易日历 ---
from qlib.data import D
qlib_calendar = D.calendar(start_time="2016-01-01", end_time="2020-09-25")
print(f"交易日历范围: {qlib_calendar[0].date()} ~ {qlib_calendar[-1].date()}  ({len(qlib_calendar)} 天)")

# --- 加载基本面特征 ---
print("--- 加载基本面特征 ---")
fund_features = load_fundamental_features(custom_universe, qlib_calendar)
if fund_features is not None:
    print(f"  基本面特征: {fund_features.shape[0]} 行, {fund_features.shape[1]} 列")
    print(f"  特征列表: {list(fund_features.columns)}")

# --- 数据集配置 ---
data_handler_config = {
    "start_time": "2016-01-01",
    "end_time": "2020-09-25",
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
    # ★ 核心改动1：标签从次日收益改为20日累计收益率
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
            "test": ("2019-10-01", "2020-09-23"),
        },
    },
}

# --- 模型配置 ---
model_config = {
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

# --- 回测配置 ---
top_k_num = min(10, max(1, len(custom_universe) // 5))

port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        # ★ 核心改动2：月频再平衡策略
        "class": "MonthlyTopkStrategy",
        "module_path": "__main__",
        "kwargs": {
            "topk": top_k_num,
            "n_drop": top_k_num,  # 调仓日允许全部换仓
        },
    },
    "backtest": {
        "start_time": "2019-10-01",
        "end_time": "2020-09-23",
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

# ================================================================
# 5. 执行训练与回测
# ================================================================
if __name__ == "__main__":
    # --- 5.1 构建数据集 ---
    print("\n=== 1. 构建数据集 (Alpha158Enhanced + 基本面特征) ===")
    dataset = init_instance_by_config(dataset_config)

    # --- 5.2 注入基本面特征 ---
    if fund_features is not None:
        print("\n=== 2. 注入基本面特征 ===")
        inject_fundamental_features(dataset, fund_features)
    else:
        print("\n=== 2. 跳过基本面特征注入 (无数据) ===")

    # --- 5.3 训练模型 ---
    model = init_instance_by_config(model_config)

    with R.start(experiment_name="fundamental_v2_monthly"):
        print("\n=== 3. 训练模型 (标签=20日累计收益率, 策略=月频再平衡) ===")
        model.fit(dataset)
        R.save_objects(trained_model=model)

        # --- 5.4 生成预测信号 ---
        print("\n=== 4. 生成预测信号 ===")
        recorder = R.get_recorder()
        sig_rec = SignalRecord(model, dataset, recorder)
        sig_rec.generate()

        # --- 5.5 月频再平衡回测 ---
        print("\n=== 5. 月频再平衡回测 ===")
        pred = recorder.load_object("pred.pkl")
        port_analysis_config["strategy"]["kwargs"]["signal"] = pred
        port_analysis_config["backtest"]["benchmark"] = None

        port_ana_rec = PortAnaRecord(recorder, port_analysis_config, "day")
        port_ana_rec.generate()

        # --- 5.6 输出结果 ---
        print("\n=== 6. 评价指标 ===")
        metrics = recorder.list_metrics()

        print(f"\n所有 metrics ({len(metrics)} 个):")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

        # 汇总
        print(f"\n{'=' * 70}")
        print(f"  多因子模型回测 V2 — 月频标签 + 月度再平衡 + 扩充低频特征")
        print(f"{'=' * 70}")
        print(f"  股票池:      {len(custom_universe)} 只 (连续10年净利润+FCF双正)")
        print(f"  训练期:      2016-01-01 ~ 2018-12-31")
        print(f"  验证期:      2019-01-01 ~ 2019-09-30")
        print(f"  回测期:      2019-10-01 ~ 2020-09-23")
        print(f"  标签:        20日累计收益率 Ref($close,-20)/$close - 1")
        print(f"  特征:        Alpha158 + 14个长周期价格因子 + 5个基本面因子")
        print(f"  策略:        MonthlyTopkStrategy (月末调仓, TopK={top_k_num})")
        print(f"  初始资金:    1亿元")
        print(f"  交易成本:    买0.15% / 卖0.25% / 最低5元")

        print(f"\n[模型训练指标]")
        print(f"  训练集 L2:   {metrics.get('l2.train', 0):.6f}")
        print(f"  验证集 L2:   {metrics.get('l2.valid', 0):.6f}")

        print(f"\n[回测收益 — 扣除交易成本前]")
        ar_nc = metrics.get("1day.excess_return_without_cost.annualized_return", 0)
        ir_nc = metrics.get("1day.excess_return_without_cost.information_ratio", 0)
        md_nc = metrics.get("1day.excess_return_without_cost.max_drawdown", 0)
        std_nc = metrics.get("1day.excess_return_without_cost.std", 0)
        print(f"  年化收益率:  {ar_nc * 100:>8.2f}%")
        print(f"  信息比率:    {ir_nc:>8.4f}")
        print(f"  最大回撤:    {md_nc * 100:>8.2f}%")
        print(f"  日均波动:    {std_nc * 100:>8.2f}%")

        print(f"\n[回测收益 — 扣除交易成本后]")
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

        # 与 V1 对比
        print(f"\n{'=' * 70}")
        print(f"  V1 vs V2 对比 (扣成本后年化收益)")
        print(f"{'=' * 70}")
        print(f"  V1 日频调仓:  年化 -23.48%,  IR -1.46,  最大回撤 -29.02%")
        print(f"  V2 月频调仓:  年化 {ar_wc * 100:>7.2f}%,  IR {ir_wc:>6.4f},  最大回撤 {md_wc * 100:>7.2f}%")
        improvement = ar_wc * 100 - (-23.48)
        print(f"  改善幅度:     {improvement:+.2f} 个百分点")
