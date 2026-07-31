"""
quant.core —— 十年双正双模型量化框架 公共层
================================================================================
无穿越保证集中在 core.dataset / core.universe 单点实现, 供滚动引擎与模型基准复用。
模块:
  config      路径/常量/超参
  data        qlib初始化/日历/价格矩阵/基准/前瞻收益矩阵
  universe    滚动窗口 + 十年双正 PIT 股票池
  features    Alpha158Enhanced + 基本面 PIT 因子
  valuation   value_comp 打分
  dataset     无穿越 DatasetH 构造 (train/valid embargo/test)
  models      RankIC 早停 XGB/LGB 训练器 + 收敛检查
  tradability 可交易性/流动性/月末信号日
  backtest    T+1 组合回测 + 图1指标 + 标准输出
"""
