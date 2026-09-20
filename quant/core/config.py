"""
core.config —— 全局路径 / 常量 / 模型超参 (单点配置)
================================================================================
所有路径与口径常量集中在此, 便于后续开发维护。
- 顶层活跃数据缓存 (valuation/fcf/profit/csi300) 位于 DATA_DIR, 绝不移动。
- 行情用 tools/fetch_ohlcv.py 抓取腾讯后复权 → tools/build_qlib_bin.py 转成 qlib bin
  (全市场 5400+ 只, 含退市; 老的 cn_data_fixed 只有 358 只覆盖 2020+ 行情, 已废弃)。
- 数据源可用 QLIB_PROVIDER 环境变量覆盖。
"""
import os

# ---- 路径 (可用环境变量覆盖, 便于换机/CI; 默认按本仓库相对位置推导) ----
# QUANT_DIR = 本文件所在 core/ 的父目录; DATA_DIR = 顶层活跃数据缓存目录(不移动)
QUANT_DIR = os.environ.get("QUANT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("QUANT_DATA_DIR") or os.path.abspath(os.path.join(QUANT_DIR, "..", ".."))
OUT_DIR = os.environ.get("QUANT_OUT_DIR") or f"{QUANT_DIR}/outputs"   # 所有 csv/log 产出集中此处

QLIB_PROVIDER_FIXED = os.path.expanduser("~/.qlib/qlib_data/cn_data_fixed")
# 新全市场 bin: 腾讯后复权行情 (5400+只, 含退市), 由 tools/build_qlib_bin.py 生成
QLIB_PROVIDER = os.environ.get("QLIB_PROVIDER") or \
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data_cache", "qlib_cn_tencent")

FCF_CACHE = f"{DATA_DIR}/fcf_cache_pit.csv"
PROFIT_CACHE = f"{DATA_DIR}/profit_cache_pit.csv"
VAL_CACHE = f"{DATA_DIR}/valuation_cache.csv"
BENCH_CACHE = f"{DATA_DIR}/csi300_cache.csv"

# ---- 回测口径常量 ----
TOPK = 10
FEE_ROUNDTRIP = 0.004          # A股往返成本至少0.4%
LIQ_THRESHOLD = 20_000_000     # 20日均成交额 ≥ 2000万 (close*volume*100, 手→股)
MIN_PRICE = 2.0                # 最低股价红线: 剔除<2元的壳价值特征小微盘 (2024.2微盘踩踏教训)
# 注: 规模溢价的尾部不是回撤, 是流动性消失. 个人资金可下探中盘, 但必须有流动性红线.
N_ROUNDS, EARLY_STOP = 2000, 100
MIN_POOL, MIN_CAND = 50, 2 * TOPK
LABEL_HORIZON = 20             # 标签/前瞻窗口(交易日), 与 label Ref($close,-20) 一致

# 参与滚动回测的策略 → 持仓数(None=全池等权). VAL* = value_comp 纯估值基准(对照系),
# 与 online_value/value_comp_M20 完全同口径, 用于回答"模型是否真能打过 value comp top20"
# DE = DoubleEnsemble (SR样本重加权 + FS特征选择), 与 XGB/LGB 同口径注入 FCF 因子
# ENS = XGB + LGB + DE 三模型 rank 百分位等权均值融合
# DL = DE + LGB 双模型 rank 等权融合
# ICW = DE + LGB + XGB 三模型 rank 按 valid RankIC 动态加权融合 (路径1优化)
# DL_T = DL策略 + 择时仓位管理 (路径3优化, 熊市降仓位)
# X15T = XGB Top15 + 换手缓冲 (实盘候选: 降集中度降回撤 + 降换手降成本)
# VHF = value_comp + 跨市场公平系数 (港股估值打折, 消除港股挤占A股)
# VG = value + 盈利改善 (A+H池实测最优: 年化16.20%, 剔20 15.62%, Sharpe 1.09)
# VGH = 结构化剥离: A股用value+盈利, 港股用FCF收益+盈利 (纳入腾讯/阿里独有资产)
# VGHX = VGH安全底仓(Top30) + XGB二次精排(Top10) (方向2: 基本面底线+模型弹性)
# VGC = VG Top30底仓 + ENS精排Top10 (级联: 基本面粗筛保底 + 模型弹性增强)
# VHC = VGH Top30底仓 + ENS精排Top10 (级联, 港股结构化剥离版底仓)
# ICV = VGH + ICW 横向 Rank 融合 (不截断, 保留各自 Alpha; ICW比XGB更强22.83% vs 20.31%)
# ICW_T = ICW + 趋势择时 (熊市半仓, 同 DL_T 机制; 降ICW的-30%回撤)
# ICW15 = ICW Top15 + 换手缓冲 (同 X15T 机制; 降集中度降换手)
# ICW_BW = ICW 双周频调仓 (每10交易日, 捕获短期反转特征的快速衰减alpha)
# VALX = 排雷版VAL10 (形态C): value_comp打分, 模型分最低30%候选被否决后重选Top10
# VALF = 融合版VAL10 (形态A): (1-w)*value_rank + w*model_rank, w逐窗口valid段寻优
STRATS = {"XGB": TOPK, "LGB": TOPK, "DE": TOPK, "ENS": TOPK, "DL": TOPK,
          "ICW": TOPK, "DL_T": TOPK, "X15T": 15, "VHF": 20, "VHF10": 10,
          "VAL20": 20, "VAL10": 10, "VG": 10, "VGF": 20, "VGH": 10, "VGHX": 10,
          "VGC": 10, "VHC": 10, "ICV": 10, "ICW_T": TOPK, "ICW15": 15,
          "ICW_BW": TOPK, "VALX": 10, "VALF": 10, "POOL_EW": None}
MODEL_STRATS = ["XGB", "LGB", "DE", "ENS", "DL", "ICW", "DL_T", "X15T"]
VALUE_FACTORS = ["ep", "bp", "cfp", "sp"]       # value_comp 4 个估值倒数

# 换手缓冲 (实盘候选 X15T): 调仓时保留至少 35% 上一期仍在候选中的持仓 (降换手降成本)
X15T_TOPK = 15            # 持仓数
X15T_BUFFER = 0.35        # 保留比例 (0~1), 0.35=Top15保留约5只, 平衡alpha与成本

# 滞回阈值 (hysteresis): 排名变化不显著不调仓, DFA式降换手
# 上一期持仓中排名仍在 top n_pick*(1+band) 内的, 即使跌出 topk_pick 也保留
# 0.2=Top15→排名≤18的旧持仓保留, 只换掉跌出18名之后的 → 每降10%换手≈白捡对应成本
X15T_HYSTERESIS = 0.2     # 滞回带宽 (0~1), 与 turnover_buffer 叠加使用

# 跨市场估值公平系数 (value_comp 港股增强 VHF):
# 港股估值倒数乘以系数, 消除系统性市场差异 (A股/港股 估值倒数中位数比值, 2024实测):
#   ep(1/PE)=0.52, bp(1/PB)=0.51, cfp(1/PCF)=0.66, sp(1/PS)=0.76
#   即港股ep/bp约为A股一半 → 乘0.52/0.51后跨市场可比, 只有真正更便宜的港股才被选
VHF_FACTOR = {"ep": 0.52, "bp": 0.51, "cfp": 0.66, "sp": 0.76}

# VG 策略 (value + 盈利改善) 权重: 敏感性扫描最优 (w=0.38, 见 tools/opt_value.py)
# (1-0.38)*v + 0.38*(0.85*g + 0.15*q) = 0.62*v + 0.323*g + 0.057*q
# 即 value 0.62, 盈利增速 0.323, 质量 0.057 (w在0.30-0.42均稳定13-16%)
VALUE_GROWTH_W = 0.323
VALUE_QUALITY_W = 0.057

# VG 换手缓冲: 调仓保留至少 35% 上期持仓 (降交易成本, 与 X15T 同机制)
VG_BUFFER = 0.35

# VGHX 策略: VGH 安全底仓持仓数 (二次精排前的底仓规模)
VGHX_BASE_TOPK = 30
# VGHX 横向融合权重: 0.5*Z(VGH) + 0.5*Z(XGB), 可调 (用户指定 0.5/0.5)
VGHX_BLEND_W = 0.5

# ICV 策略: VGH + ICW 横向 Rank 融合权重 (0.6=偏ICW进攻, VGH提供安全垫)
ICV_BLEND_W = 0.6

# ICW15 策略: ICW Top15 + 换手缓冲 (同 X15T 机制)
ICW15_TOPK = 15
ICW15_BUFFER = 0.35

# ---- 块2实验: VAL10 × 模型打分 (对照 = 原版VAL10 / 双腿0.4*DE+0.6*VAL10) ----
# VALX = 排雷版 (形态C): value_comp 打分后, 模型分最低 VALX_VETO 比例候选被否决
#        (模型只有否决权没有提名权; 若跑赢原版 → 模型价值在"避坑"不在"选美")
# VALF = 融合版 (形态A): (1-w)*value_rank + w*model_rank, w 逐窗口在 valid 段
#        按月度 RankIC 网格寻优 (纪律: w 不在 OOS 上扫, OOS 只验一次)
VALX_VETO = 0.30             # 排雷比例: 剔除模型分最低 30% 候选
VALF_W_GRID = (0.15, 0.20, 0.25, 0.30)   # 融合权重 w 网格 (valid 段寻优)

# ---- SUMD/SUMP 系特征剔除 (RSI类, 高方差噪声) ----
# 消融实验确认剔除后所有模型策略同向提升(Sharpe+0.07~0.24, 回撤降1~7pp), 固化为默认
# 仍可通过环境变量 ABLATE_SUMD=0 临时恢复(对照用)
ABLATE_SUMD = os.environ.get("ABLATE_SUMD", "1") == "1"

# ---- ICW_BW 滞回降换手 ----
# 滞回0.3实验: 换手省5pp但年化-0.88pp/Sharpe-0.06, 净效应为负, 默认关闭
ICW_BW_HYSTERESIS = float(os.environ.get("ICW_BW_HYSTERESIS", "0"))

# 级联策略 (方向: 基本面粗筛保底 + 模型精排增强):
#   VGC = VG (value+盈利) Top30 底仓 → ENS 精排 → Top10
#   VHC = VGH (结构化剥离) Top30 底仓 → ENS 精排 → Top10
# 逻辑: value 系实测最稳 (VG 16.20%/VGH 19.32%), ML 弹性大但独立选股不稳;
# 级联让持仓永远落在基本面安全集内, 模型只在安全集内做排序 (降尾部风险).
VGC_BASE_TOPK = 30

# 是否纳入港股进股票池 (False = 纯A股对照基线, 用于回答"加港股是否提升")
INCLUDE_HK = True

# 港股豁免通道: 不满足十年双正的港股, 若满足"近5年FCF均值正 + 最近2年FCF增速>15%"
# 则豁免入池. 解决"存活期截断"(阿里等成熟科技巨头仅上市10年).
# 已通过十年双正的港股不受影响 (豁免只额外纳入).
HK_EXEMPT_CHANNEL = True

# XGB 市场感知增强: 给训练特征注入 is_hk 市场标识列 (港股=1, A股=0)
# 实测结论(2026-08-18): XGB 从不使用该列(gain=0, 市场差异已由行情特征隐式表达),
# 加列只扰动列采样随机性导致树结构/早停剧变 → 收益下降(18.25%→16.03%)。已回滚禁用。
INJECT_MARKET = False

# ---- 择时参数 (路径3) ----
# 用沪深300 60日均线趋势判断牛熊: 收盘价 > MA60 为牛市满仓, 否则半仓
# MA60 是标准业界择时窗口 (3个月趋势), 信号日用 <= sig_dt 的数据计算, 不穿越
TIMING_MA = 60
TIMING_BEAR_POS = 0.7   # 熊市仓位 (牛市=1.0, 熊市=0.7; 原0.5→0.7: 减亏5.9%实证最优)
# 带宽过滤 + 迟滞确认 (修复纯日频 MA60 交叉噪声: 6.5年98次切换 → ~28次):
#   入熊: 收盘价 < MA60*(1-TIMING_BANDWIDTH) 且连续 TIMING_HYSTERESIS 日
#   出熊: 收盘价重新站上 MA60
TIMING_BANDWIDTH = 0.03   # 入熊需跌破均线下沿 3%
TIMING_HYSTERESIS = 3     # 入熊需连续 3 日确认
# ---- 熊市防御切换 (ICW_SW: 熊市不持现金, 切换到低相关价值策略 VGH) ----
# 牛市: 100% ICW双周; 熊市: SWITCH_W_BEAR*ICW双周 + (1-SWITCH_W_BEAR)*VGH
# 依据: VGH 与 ICW 相关性仅 0.40, 熊市年 (2022/2023) VGH 显著抗跌;
# 对比持现金择时 (bear_pos=0.6: 22.6%/-29.5%/1.14), 切VGH 24.7%/-23.9%/1.28 全面占优
SWITCH_W_BEAR = 0.3       # 熊市保留的 ICW双周 权重 (0.3 = 30%BW + 70%VGH)

# ---- 模型超参 ----
XGB_PARAMS = {
    "objective": "reg:squarederror", "learning_rate": 0.01,
    "max_depth": 5, "colsample_bytree": 0.8879, "subsample": 0.8789,
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

# DoubleEnsemble 子模型超参 (回退到原最优: depth=8, leaves=128, lr=0.03, num_models=3)
# 降复杂度(depth6/leaves48)导致欠拟合 RankIC下降, 原3模型配置已是最优
DE_LGB_PARAMS = {
    "objective": "regression", "metric": "None",
    "learning_rate": 0.03, "max_depth": 8, "num_leaves": 128,
    "colsample_bytree": 0.8879, "subsample": 0.8789, "subsample_freq": 1,
    "lambda_l1": 50.0, "lambda_l2": 200.0,
    "num_threads": 4, "seed": 42, "verbose": -1,
}
DE_CONFIG = {
    "num_models": 3,          # 子模型数 (SR/FS 轮数 = num_models-1)
    "enable_sr": True,        # 样本重加权: 高损失样本权重↑
    "enable_fs": True,        # 特征选择: shuffle 后按损失增量分箱采样
    "alpha1": 1.0,            # SR: 当前集成损失排名权重
    "alpha2": 1.0,            # SR: 训练曲线 l_end/l_start 比值权重
    "bins_sr": 10,            # SR 分箱数
    "bins_fs": 5,             # FS 分箱数
    "decay": 0.5,             # SR 权重衰减系数
    "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],  # FS 各箱采样比例
    "sub_weights": [1, 1, 1], # 子模型集成权重
}

FUND_FEATS = ["F_fcf_growth", "F_profit_growth", "F_fcf_profit_ratio",
              "F_fcf_avg_3y_norm", "F_fcf_cv_3y", "F_asset_growth"]

# ---- 回测时间边界 ----
BT_START = "2020-01-01"
ALIGN_START = "2021-01-01"     # 对齐 value_comp 冻结基线(跳过2020异常年)
BT_END = "2026-12-31"

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
                                  # 复现性: 固定 lgb 种子 + deterministic (线程数变动也会
                                  # 影响结果, num_threads 已写死, 不要随意改)
                                  "seed": 43, "deterministic": True,
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
             # n_jobs=0: 禁用 DataLoader 多进程 (sandbox 下 torch_shm_manager 不可用)
             "kwargs": {"d_feat": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.7,
                        "n_epochs": 200, "lr": 1e-4, "early_stop": 10, "metric": "loss",
                        "loss": "mse", "base_model": "LSTM", "n_jobs": 0, "GPU": -1}},
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
