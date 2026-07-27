import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

# ========== 0. 数据与环境初始化 ==========
provider_uri = "~/.qlib/qlib_data/cn_data" 
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 读取你的本地 CSV 并转换股票代码格式
csv_path = "/Users/11164591/Documents/Qoder目录/fcf_profit_all_positive.csv"
df = pd.read_csv(csv_path)

def format_qlib_code(code):
    """将数字代码转换为 Qlib 格式 (SH/SZ + 6位数字)"""
    code_str = str(code).zfill(6)
    if code_str.startswith('6'):
        return f"SH{code_str}"
    else:
        return f"SZ{code_str}"

# 生成自定义股票池列表
custom_universe = df['code'].apply(format_qlib_code).tolist()
print(f"成功加载自定义基本面股票池，共 {len(custom_universe)} 只股票。")

# ========== 1. 特征与标签工程配置 ==========
data_handler_config = {
    "start_time": "2016-01-01",
    "end_time": "2020-09-23",
    "fit_start_time": "2016-01-01",
    "fit_end_time": "2018-12-31",
    "instruments": custom_universe,  # <--- 核心修改：使用 CSV 中的股票池
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}, 
    ],
    "label": ["Ref($open, -2) / Ref($open, -1) - 1"]
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2016-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2019-09-30"),
            "test": ("2019-10-01", "2020-09-23"),
        },
    },
}

# ========== 2. 预测模型配置 ==========
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

# ========== 3. 回测与交易执行配置 ==========
# 根据基本面白马股的特性，适当调小 TopK，集中持仓
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
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "topk": top_k_num,        # 根据 CSV 股票数量动态调整买入数量
            "n_drop": max(1, top_k_num // 3), # 每天最多换仓 1/3，控制基本面股票的换手率
        },
    },
    "backtest": {
        "start_time": "2019-10-01",
        "end_time": "2020-09-23",
        "account": 100000000, 
        "benchmark": None,        # <--- 核心修改：关闭基准对比，绕过报错
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

# ========== 4. 执行主引擎 ==========
if __name__ == "__main__":
    dataset = init_instance_by_config(dataset_config)
    model = init_instance_by_config(model_config)

    with R.start(experiment_name="fundamental_universe_lgb"):
        print("--- 1. 开始训练模型 (仅在基本面优质池内训练) ---")
        model.fit(dataset)
        R.save_objects(trained_model=model)

        print("--- 2. 生成特征信号 ---")
        recorder = R.get_recorder()
        sig_rec = SignalRecord(model, dataset, recorder)
        sig_rec.generate()
        print("--- 3. 注入交易策略进行实盘仿真回测 ---")
        # 加载预测结果，直接传DataFrame给策略
        pred = recorder.load_object("pred.pkl")
        port_analysis_config["strategy"]["kwargs"]["signal"] = pred
        port_analysis_config["backtest"]["benchmark"] = None

        port_ana_rec = PortAnaRecord(recorder, port_analysis_config, "day")
        port_ana_rec.generate()
        

        print("--- 4. 输出核心评价指标 ---")
        metrics = recorder.list_metrics()
        
        # 打印所有可用的metrics key
        print(f"\n所有可用metrics ({len(metrics)} 个):")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")
        
        print(f"\n{'='*60}")
        print(f"  多因子模型回测结果 (Alpha158 + LightGBM)")
        print(f"  股票池: {len(custom_universe)} 只 (连续10年净利润+FCF双正)")
        print(f"  训练期: 2016-01-01 ~ 2018-12-31")
        print(f"  验证期: 2019-01-01 ~ 2019-09-30")
        print(f"  回测期: 2019-10-01 ~ 2020-09-23")
        print(f"  策略: TopkDropoutStrategy (TopK={top_k_num})")
        print(f"  初始资金: 1亿元")
        print(f"{'='*60}")
        
        print(f"\n[模型训练指标]")
        print(f"  训练集L2损失:  {metrics.get('l2.train', 0):.6f}")
        print(f"  验证集L2损失:  {metrics.get('l2.valid', 0):.6f}")
        
        print(f"\n[回测收益评估 - 扣除交易成本前]")
        ar_nc = metrics.get('1day.excess_return_without_cost.annualized_return', 0)
        ir_nc = metrics.get('1day.excess_return_without_cost.information_ratio', 0)
        md_nc = metrics.get('1day.excess_return_without_cost.max_drawdown', 0)
        std_nc = metrics.get('1day.excess_return_without_cost.std', 0)
        print(f"  年化收益率:   {ar_nc*100:>8.2f}%")
        print(f"  信息比率:     {ir_nc:>8.4f}")
        print(f"  最大回撤:     {md_nc*100:>8.2f}%")
        print(f"  日均波动:     {std_nc*100:>8.2f}%")
        
        print(f"\n[回测收益评估 - 扣除交易成本后]")
        ar_wc = metrics.get('1day.excess_return_with_cost.annualized_return', 0)
        ir_wc = metrics.get('1day.excess_return_with_cost.information_ratio', 0)
        md_wc = metrics.get('1day.excess_return_with_cost.max_drawdown', 0)
        std_wc = metrics.get('1day.excess_return_with_cost.std', 0)
        print(f"  年化收益率:   {ar_wc*100:>8.2f}%")
        print(f"  信息比率:     {ir_wc:>8.4f}")
        print(f"  最大回撤:     {md_wc*100:>8.2f}%")
        print(f"  日均波动:     {std_wc*100:>8.2f}%")
        
        print(f"\n[交易执行统计]")
        print(f"  订单成交率(FFR): {metrics.get('1day.ffr', 0)*100:.1f}%")
