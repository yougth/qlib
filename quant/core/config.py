"""
core.config —— 全局路径 / 常量 / 模型超参 (单点配置)
================================================================================
所有路径与口径常量集中在此, 便于后续开发维护。
- 顶层活跃数据缓存 (valuation/fcf/profit/csi300) 位于 DATA_DIR, 绝不移动。
- 行情用 fix_qlib_seam.py 修补后的 cn_data_fixed (已消除 2020-09-28 两批拼接断点)。
"""
import os

# ---- 路径 ----
DATA_DIR = "/Users/11164591/Documents/Qoder目录"          # 顶层活跃数据缓存目录 (不移动)
QUANT_DIR = "/Users/11164591/Documents/Qoder目录/qlib/quant"
OUT_DIR = f"{QUANT_DIR}/outputs"                            # 所有 csv/log 产出集中此处

QLIB_PROVIDER_FIXED = os.path.expanduser("~/.qlib/qlib_data/cn_data_fixed")

FCF_CACHE = f"{DATA_DIR}/fcf_cache_pit.csv"
PROFIT_CACHE = f"{DATA_DIR}/profit_cache_pit.csv"
VAL_CACHE = f"{DATA_DIR}/valuation_cache.csv"
BENCH_CACHE = f"{DATA_DIR}/csi300_cache.csv"

# ---- 回测口径常量 ----
TOPK = 10
FEE_ROUNDTRIP = 0.004          # A股往返成本至少0.4%
LIQ_THRESHOLD = 20_000_000     # 20日均成交额 ≥ 2000万 (close*volume*100, 手→股)
N_ROUNDS, EARLY_STOP = 1000, 100
MIN_POOL, MIN_CAND = 50, 2 * TOPK
LABEL_HORIZON = 20             # 标签/前瞻窗口(交易日), 与 label Ref($close,-20) 一致

# 参与滚动回测的策略 → 持仓数(None=全池等权). VAL* = value_comp 纯估值基准(对照系),
# 与 online_value/value_comp_M20 完全同口径, 用于回答"模型是否真能打过 value comp top20"
STRATS = {"XGB": TOPK, "LGB": TOPK, "ENS": TOPK,
          "VAL20": 20, "VAL10": 10, "POOL_EW": None}
MODEL_STRATS = ["XGB", "LGB", "ENS"]            # 需要 OOS IC 的 ML 模型
VALUE_FACTORS = ["ep", "bp", "cfp", "sp"]       # value_comp 4 个估值倒数

# ---- 模型超参 ----
XGB_PARAMS = {
    "objective": "reg:squarederror", "learning_rate": 0.005,
    "max_depth": 4, "colsample_bytree": 0.8879, "subsample": 0.8789,
    "reg_alpha": 10.0, "reg_lambda": 50.0,
    "tree_method": "hist", "nthread": 4, "seed": 42,
    "disable_default_eval_metric": 1,
}
LGB_PARAMS = {
    "objective": "regression", "metric": "None",
    "learning_rate": 0.02, "max_depth": 8, "num_leaves": 128,
    "colsample_bytree": 0.8879, "subsample": 0.8789, "subsample_freq": 1,
    "lambda_l1": 50.0, "lambda_l2": 200.0,
    "num_threads": 4, "seed": 42, "verbose": -1,
}

FUND_FEATS = ["F_fcf_growth", "F_profit_growth", "F_fcf_profit_ratio",
              "F_fcf_avg_3y_norm", "F_fcf_cv_3y"]

# ---- 回测时间边界 ----
BT_START = "2020-01-01"
ALIGN_START = "2021-01-01"     # 对齐 value_comp 冻结基线(跳过2020异常年)
BT_END = "2026-07-23"

# ================================================================================
# 基准 (run_benchmark.py) 专用配置
# ================================================================================
# Phase A 单一固定无穿越切分 (全模型头对头初筛):
#   train 2016~2021(6年) / valid 2022(截至11-30 embargo) / 信号段 2022-12~2026-06
#   股票池按 2023 规则冻结 (仅用 ≤2021 基本面, 对 2023+ 交易 PIT 安全)
BENCH_SPLIT = {
    "train": ("2016-01-01", "2021-12-31"),
    "valid": ("2022-01-01", "2022-11-30"),      # embargo: 12月留白
    "test":  ("2022-12-01", "2026-06-30"),
}
BENCH_UNIVERSE_YEAR = 2023            # 冻结股票池所用 year-2 规则年份 (数据 ≤2021)
BENCH_BT_START = "2023-01-01"         # Phase A 回测起点
BENCH_MODEL_BUDGET_SEC = 2700         # 单模型 wall-clock 预算 (默认45分钟), 超时记 timeout
BENCH_TOPB = 4                        # Phase B 取 Phase A 前 N 名进入完整滚动

# 全部候选模型 (qlib 内置 + Alpha158 handler 兼容). ds=DatasetH(截面) / TSDatasetH(序列).
# dfeat 标记需按实际特征数注入的键: "d_feat"=顶层键; "mlp"=pt_model_kwargs.input_dim。
# 所有 pytorch 模型 GPU=-1 强制走 CPU (本机无 CUDA); n_jobs 降到 4 避免过载。
MODEL_CONFIGS = {
    # ---- 树 / 表格 (DatasetH, CPU 友好, 分钟级) ----
    "XGBoost": {"ds": "DatasetH", "class": "XGBModel",
                "module_path": "qlib.contrib.model.xgboost",
                # M1 提速: hist 直方图算法(xgb>=2.0 默认, 显式声明) + 6 线程(4性能核+2能效核)
                "kwargs": {"eval_metric": "rmse", "colsample_bytree": 0.8879,
                           "eta": 0.0421, "max_depth": 8, "n_estimators": 647,
                           "subsample": 0.8789, "nthread": 6, "tree_method": "hist"}},
    "LightGBM": {"ds": "DatasetH", "class": "LGBModel",
                 "module_path": "qlib.contrib.model.gbdt",
                 # M1 提速: 6 线程 + force_col_wise 跳过行/列布局自动探测开销
                 "kwargs": {"loss": "mse", "colsample_bytree": 0.8879,
                            "learning_rate": 0.2, "subsample": 0.8789,
                            "lambda_l1": 205.6999, "lambda_l2": 580.9768,
                            "max_depth": 8, "num_leaves": 210, "num_threads": 6,
                            "force_col_wise": True}},
    "CatBoost": {"ds": "DatasetH", "class": "CatBoostModel",
                 "module_path": "qlib.contrib.model.catboost_model",
                 # bootstrap_type 由 yaml 的 Poisson(需GPU) 改 Bernoulli 以在 CPU 运行
                 "kwargs": {"loss_function": "RMSE", "learning_rate": 0.0421,
                            "subsample": 0.8789, "max_depth": 6, "thread_count": 4,
                            "grow_policy": "Lossguide", "bootstrap_type": "Bernoulli"}},
    "Linear": {"ds": "DatasetH", "class": "LinearModel",
               "module_path": "qlib.contrib.model.linear", "kwargs": {"estimator": "ols"}},
    "DoubleEnsemble": {"ds": "DatasetH", "class": "DEnsembleModel",
                       "module_path": "qlib.contrib.model.double_ensemble",
                       "kwargs": {"base_model": "gbm", "loss": "mse", "num_models": 3,
                                  "enable_sr": True, "enable_fs": True, "alpha1": 1,
                                  "alpha2": 1, "bins_sr": 10, "bins_fs": 5, "decay": 0.5,
                                  "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
                                  "sub_weights": [1, 1, 1], "epochs": 28,
                                  "colsample_bytree": 0.8879, "learning_rate": 0.2,
                                  "subsample": 0.8789, "lambda_l1": 205.6999,
                                  "lambda_l2": 580.9768, "max_depth": 8,
                                  "num_leaves": 210, "num_threads": 6,
                                  "force_col_wise": True}},
    # ---- 表格 NN (DatasetH, 需 input_dim=特征数) ----
    "MLP": {"ds": "DatasetH", "class": "DNNModelPytorch",
            "module_path": "qlib.contrib.model.pytorch_nn", "dfeat": "mlp",
            "kwargs": {"loss": "mse", "lr": 0.002, "optimizer": "adam",
                       "max_steps": 8000, "batch_size": 8192, "GPU": -1,
                       "weight_decay": 0.0002, "pt_model_kwargs": {"input_dim": 157}}},
    "TabNet": {"ds": "DatasetH", "class": "TabnetModel",
               "module_path": "qlib.contrib.model.pytorch_tabnet", "dfeat": "d_feat",
               "kwargs": {"d_feat": 158, "pretrain": False, "seed": 993}},
    # ---- DL 序列 (TSDatasetH step_len=20, GPU=-1, 需 d_feat=特征数) ----
    "GRU": {"ds": "TSDatasetH", "class": "GRU", "dfeat": "d_feat",
            "module_path": "qlib.contrib.model.pytorch_gru_ts",
            "kwargs": {"d_feat": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.0,
                       "n_epochs": 200, "lr": 2e-4, "early_stop": 10, "batch_size": 800,
                       "metric": "loss", "loss": "mse", "n_jobs": 4, "GPU": -1}},
    "LSTM": {"ds": "TSDatasetH", "class": "LSTM", "dfeat": "d_feat",
             "module_path": "qlib.contrib.model.pytorch_lstm_ts",
             "kwargs": {"d_feat": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.0,
                        "n_epochs": 200, "lr": 1e-3, "early_stop": 10, "batch_size": 800,
                        "metric": "loss", "loss": "mse", "n_jobs": 4, "GPU": -1}},
    "ALSTM": {"ds": "TSDatasetH", "class": "ALSTM", "dfeat": "d_feat",
              "module_path": "qlib.contrib.model.pytorch_alstm_ts",
              "kwargs": {"d_feat": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.0,
                         "n_epochs": 200, "lr": 1e-3, "early_stop": 10, "batch_size": 800,
                         "metric": "loss", "loss": "mse", "n_jobs": 4, "GPU": -1,
                         "rnn_type": "GRU"}},
    "GATs": {"ds": "TSDatasetH", "class": "GATs", "dfeat": "d_feat",
             "module_path": "qlib.contrib.model.pytorch_gats_ts",
             # 去掉 yaml 的 model_path(预训练 LSTM 权重不存在), 让 base_model 从头训练
             "kwargs": {"d_feat": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.7,
                        "n_epochs": 200, "lr": 1e-4, "early_stop": 10, "metric": "loss",
                        "loss": "mse", "base_model": "LSTM", "n_jobs": 4, "GPU": -1}},
    "TCN": {"ds": "TSDatasetH", "class": "TCN", "dfeat": "d_feat",
            "module_path": "qlib.contrib.model.pytorch_tcn_ts",
            "kwargs": {"d_feat": 20, "num_layers": 5, "n_chans": 32, "kernel_size": 7,
                       "dropout": 0.5, "n_epochs": 200, "lr": 1e-4, "early_stop": 20,
                       "batch_size": 2000, "metric": "loss", "loss": "mse",
                       "optimizer": "adam", "n_jobs": 4, "GPU": -1}},
    "Localformer": {"ds": "TSDatasetH", "class": "LocalformerModel", "dfeat": "d_feat",
                    "module_path": "qlib.contrib.model.pytorch_localformer_ts",
                    "kwargs": {"d_feat": 20, "seed": 0, "n_jobs": 4, "GPU": -1}},
    "Transformer": {"ds": "TSDatasetH", "class": "TransformerModel", "dfeat": "d_feat",
                    "module_path": "qlib.contrib.model.pytorch_transformer_ts",
                    "kwargs": {"d_feat": 20, "seed": 0, "n_jobs": 4, "GPU": -1}},
}
