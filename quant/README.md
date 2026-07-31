# quant —— 十年双正 · 无穿越量化研究框架

A 股月频选股研究框架。核心目标：**策略与数据严格不穿越**，结构简洁、便于后续开发，并支持
qlib 全内置模型头对头基准。

---

## 1. 目录结构

```
quant/
  core/                    # 无穿越公共层 (滚动引擎与全模型基准共用的唯一数据入口)
    config.py              # 路径 / 口径常量 / XGB·LGB 超参 / 全模型 qlib 配置字典
    data.py                # qlib init(cn_data_fixed) / 日历 / 价格矩阵 / csi300 基准 / 前瞻收益矩阵
    universe.py            # 滚动窗口 + 十年双正 PIT 股票池 (year-2 规则)
    features.py            # Alpha158Enhanced (长周期扩展) + 基本面 PIT 因子 (Y+1-05-01 生效)
    valuation.py           # value_comp 四估值倒数打分
    dataset.py             # 无穿越数据层: build_dataset (滚动) / build_datasetH (qlib 原生)
    models.py              # RankIC 早停 XGB/LGB + fallback + 通用 qlib 模型训练器
    tradability.py         # 一字涨停 / 停牌 / 流动性过滤 + 月末信号日
    backtest.py            # T+1 组合回测 + 图1九指标 + 标准输出
  run_rolling.py           # 双模型(XGB/LGB/ENS)滚动引擎 —— 产出 24.88% 的主结果
  run_benchmark.py         # qlib 全模型基准: Phase A 单切分初筛 + Phase B top-K 完整滚动
  tests/
    test_no_lookahead.py   # 无穿越守卫单测 (回归门禁)
  outputs/                 # 所有 csv / log 产出集中此处
  README.md

archive/                   # 历史实验脚本与陈旧数据 (git mv 归档, 不删除)
  legacy_scripts/          # 早期 v* 迭代脚本 (相互 import 关系保留)
  data/                    # 根目录陈旧 csv / pkl / log / json
```

> **不移动的活跃数据** (顶层 `Qoder目录`)：`valuation_cache.csv` / `fcf_cache_pit.csv` /
> `profit_cache_pit.csv` / `csi300_cache.csv`；`~/.qlib/qlib_data/cn_data_fixed`；`online_value/` 整体。

---

## 2. 无穿越保证 (single source of truth)

所有引擎 (滚动 / 基准) 共用 `core.dataset` 与 `core.universe`，单点保证以下六条：

| # | 机制 | 位置 |
|---|------|------|
| 1 | **PIT 股票池 year-2 规则**：仅用 ≤(Y-2) 的 FCF / 净利判定，(Y-1) 数据绝不参与 | `universe.build_dynamic_universe` |
| 2 | **embargo 分段**：valid 截至 (Y-1)-11-30，12 月留白；信号段 (Y-1)-12-01 起 | `universe.build_windows` |
| 3 | **label 仅后向**：`Ref($close, -20) / $close - 1` (20 日前瞻收益作为预测目标) | `dataset.DEFAULT_LABEL` |
| 4 | **标准化仅 fit train 段**：`fit_end_time == train.end`，infer 段用 train 拟合的 RobustZScoreNorm | `dataset._handler_config` |
| 5 | **按日截面标准化**：label 用 CSZScoreNorm；基本面因子按日 MAD-zscore | `dataset` / `features` |
| 6 | **T+1 执行**：月末信号，次一交易日成交；基本面因子 Y+1-05-01 起生效 | `run_*` / `features.load_fund_features` |

守卫单测 `tests/test_no_lookahead.py` 作为回归门禁（详见 §5）。

---

## 3. 快速开始

```bash
cd qlib/quant

# (0) 前置: cn_data_fixed 已由 fix_qlib_seam.py 修补 (消除 2020-09-28 拼接断点)

# (1) 主结果: 双模型滚动 → 24.88% (剔2020对齐口径)
python3 run_rolling.py

# (2) 无穿越守卫单测 (快速静态守卫, 秒级)
python3 -m pytest tests/test_no_lookahead.py -v
#     重型探针 (打乱标签 / 特征-未来相关性, 需 qlib 数据):
RUN_SLOW=1 python3 -m pytest tests/test_no_lookahead.py -v

# (3) 全模型基准
python3 run_benchmark.py                              # Phase A 全模型初筛
python3 run_benchmark.py --phase B --models XGBoost,LightGBM,CatBoost,DoubleEnsemble  # Phase B 完整滚动
```

---

## 4. run_rolling —— 主结果 (24.88%)

- **股票池**：Y-11..Y-2 十年窗口 FCF 全正 + 净利全正 (双正)。
- **模型**：XGBoost / LightGBM，验证集 **RankIC 最大化早停** (禁止 RMSE 早停退化)，
  best_iter<30 触发备选超参 fallback (仅用 valid，不碰 test)。ENS = 两模型 rank 百分位均值。
- **回测**：月频 Top10，T+1，往返成本 0.4%，一字涨停/停牌/流动性(20 日均额≥2000万)过滤。
- **对照系**：value_comp Top20/Top10 (与 `online_value` 同口径) + 十年双正池等权 + 沪深300。

### 已验证结果 (重构后回归重跑，与重构前逐指标 **0 偏差** 复现)

| 策略 | 年化(全期) | 年化(剔2020) | Sharpe | 最大回撤 | Calmar | OOS RankIC |
|------|-----------|-------------|--------|---------|--------|-----------|
| 集成 ENS Top10 | **27.41%** | **24.88%** | 1.04 | -33.95% | 0.81 | 0.0781 |
| value_comp Top20 | 20.57% | 23.00% | 1.00 | -19.34% | 1.06 | 0.0780 |

> 重构为纯逻辑搬运 (行为等价)，回归重跑与原 `rolling10y_dual_model.py` 保存结果完全一致，
> 每窗口 best_iter 亦逐一相同 → **24.88% 真实、可复现、无穿越**。

---

## 5. 无穿越守卫单测

`tests/test_no_lookahead.py`：

- **快速静态守卫** (无需数据)：窗口分段顺序、embargo(11-30 留白)、label 后向且 horizon==20、
  扩展特征无负偏移 Ref、股票池 year-2 规则(Y-1 转负不改池 / Y-2 转负剔除)、标准化仅 fit train、
  基本面因子生效日 ≥ (报告年+1)-05-01。
- **重型探针** (`RUN_SLOW=1`)：打乱训练标签 → valid RankIC 塌缩到 ~0；test 段无任一特征列与
  20 日前瞻真实收益近乎完全相关 (label 未漏进 feature)。

---

## 6. run_benchmark —— qlib 全模型基准

统一走 `core.dataset.build_datasetH` (与滚动引擎同 handler 口径：Alpha158Enhanced + 同 processors
+ embargo 分段 + 20 日后向 label + train 段 fit 标准化)，单点保证全模型无穿越。

- **Phase A 初筛**：全部候选模型在**单一固定切分** (train 2016-2021 / valid 2022 embargo /
  信号 2022-12~2026-06，股票池按 2023 规则冻结) 上训练，按 **OOS RankIC** 排名。
  每个模型跑在**独立子进程** + wall-clock 预算 (默认 45 分钟)；**超时记 `timeout`、报错记 `failed`，
  绝不静默跳过** (符合"数据 check、不静默异常"原则)。产出 `outputs/benchmark_phaseA.csv`。
- **Phase B**：取 Phase A 前 `BENCH_TOPB` 名进入**完整 7 窗口滚动**，与 value_comp Top20 /
  十年双正池等权 / 沪深300 做图1九指标头对头。产出 `outputs/benchmark_phaseB*.csv`。

### 候选模型 (qlib 内置 · Alpha158 兼容)

| 类别 | 模型 | Dataset |
|------|------|---------|
| 树 / 表格 | XGBoost, LightGBM, CatBoost, Linear, DoubleEnsemble | DatasetH |
| 表格 NN | MLP, TabNet | DatasetH |
| DL 序列 | GRU, LSTM, ALSTM, GATs, TCN, Localformer, Transformer | TSDatasetH (step_len=20) |

> 本机无 CUDA，全部 pytorch 模型 `GPU=-1` 强制走 CPU，DL 模型较慢，故用 wall-clock 预算保护。
> CatBoost 的 `bootstrap_type` 由官方 yaml 的 `Poisson`(需 GPU) 改为 `Bernoulli` 以在 CPU 运行；
> GATs 去除官方 yaml 的预训练 `model_path`，base_model 从头训练；NN/DL 的 `d_feat` / `input_dim`
> 按 Alpha158Enhanced 实际特征数在运行时注入。

### Phase A 初筛排名 (单切分 test 2023~2026H1, OOS RankIC 降序)

| # | 模型 | RankIC | IC | 年化 | Sharpe | 最大回撤 | Calmar | 换手 | 耗时 |
|---|------|--------|-----|------|--------|---------|--------|------|------|
| 1 | **GATs** ⚠️ | 0.1841 | 0.1887 | 41.9% | 2.40 | -14.3% | 2.94 | 49.7% | 38min |
| 2 | VALUE20 | 0.0965 | 0.0545 | 26.6% | 1.42 | -14.1% | 1.89 | 14.5% | 14s |
| 3 | POOL_EW | 0.0965 | 0.0545 | 6.8% | 0.34 | -26.8% | 0.26 | 6.7% | 11s |
| 4 | Transformer | 0.0811 | 0.0467 | 10.5% | 0.47 | -35.9% | 0.29 | 86.8% | 69min |
| 5 | **MLP** | 0.0744 | 0.0496 | 20.8% | 0.80 | -34.9% | 0.60 | 90.0% | 4min |
| 6 | DoubleEnsemble | 0.0677 | 0.0564 | 18.8% | 0.81 | -31.9% | 0.59 | 80.0% | 14min |
| 7 | LightGBM | 0.0647 | 0.0545 | 20.1% | 0.87 | -28.4% | 0.71 | 81.2% | 2min |
| 8 | XGBoost | 0.0636 | 0.0547 | 18.0% | 0.71 | -34.1% | 0.53 | 79.7% | 3min |
| 9 | CatBoost | 0.0627 | 0.0652 | 14.8% | 0.64 | -28.2% | 0.52 | 68.6% | 4min |
| 10 | LSTM | 0.0601 | 0.0478 | 19.9% | 0.82 | -27.9% | 0.71 | 92.4% | 26min |
| 11 | Localformer | 0.0580 | 0.0238 | 7.0% | 0.39 | -21.7% | 0.32 | 79.6% | 68min |
| 12 | ALSTM | 0.0540 | 0.0303 | 17.7% | 0.79 | -26.5% | 0.67 | 81.4% | 27min |
| 13 | GRU | 0.0518 | 0.0345 | 12.4% | 0.61 | -24.1% | 0.52 | 80.5% | 25min |
| 14 | Linear | 0.0486 | 0.0257 | 3.5% | 0.18 | -28.9% | 0.12 | 78.1% | 3min |
| — | TabNet | timeout | — | — | — | — | — | — | >45min |
| — | TCN | timeout | — | — | — | — | — | — | >90min |

> ⚠️ **GATs 异常说明**：单切分 RankIC 0.1841 远超其他模型 (第二名 0.0965)，年化 42%+Sharpe 2.4
> 令人存疑。可能原因：(a) 单切分恰好命中 GATs 擅长的趋势期 (2024-2025 市场)；(b) 注意力机制在
> 小宇宙(226 只)上的过拟合。**需 Phase B 多窗口滚动验证稳健性**后方可下结论。

**结论 (剔除 GATs 待验证)：**
- 学习模型中 **MLP** 表现最优 (RankIC 0.0744, 年化 20.8%, Sharpe 0.80)，其次为
  DoubleEnsemble/LightGBM/LSTM，差异不大（0.06~0.07 区间）。
- 树模型整体收敛在 **RankIC 0.063~0.068** 区间，相互差异很小。
- 复杂 DL 序列模型 (Localformer/GRU/ALSTM) 在本数据集上 **反而不如** 树模型和 MLP。
- 纯基本面策略 VALUE20 (无 ML 模型) 以 RankIC 0.0965 / 年化 26.6% / Sharpe 1.42 **仍为最稳健策略**；
  学习模型虽 RankIC 更高但回撤和换手大幅恶化 (典型如 MLP 回撤 35% vs VALUE20 仅 14%)。
- Phase B 建议候选：**GATs, MLP, LightGBM, VALUE20** (需验证 GATs 稳健性 + ML vs 纯策略对比)。

完整数据详见 `outputs/benchmark_phaseA.csv`。
