"""
value_comp_M20 上线核心库 (core)
==================================================================
本目录是**上线锁定代码**。它自包含: 不 import 任何 qlib/ 根目录下的 vXX 实验脚本,
因此上游继续做实验、改文件, 都不会影响实盘。

模块职责 (按功能划分, 一个模块一件事):
  config.py         口径常量与路径 —— 改这里等于改策略
  universe.py       股票池: 十年净利润&FCF双正动态池 (year-2 财务避前视)
  valuation.py      估值: 估值缓存读取 + value_comp 打分 (池内截面rank均值)
  market.py         行情: 价量矩阵 / 20日均成交额 / 一字涨停停牌 / 沪深300基准
  calendar_rules.py 换仓日历: 信号日=月首前最后交易日, 执行日=月首交易日
  selection.py      选股: 三道过滤 + Top20 —— 生产与回测唯一共用路径
  signal.py         编排: 把上面拼成逐月目标持仓
  backtest.py       回测引擎: 份额级组合回测 + 指标

入口脚本 (上一级目录):
  monthly_update.py    每月数据更新 (akshare → qlib bin + 估值缓存), 唯一写数据的脚本
  preflight_check.py   上线体检: 口径锁/指纹锁/数据鲜度/黄金回归
  gen_holdings.py      出持仓与买入清单
  run_backtest.py      终检回测 (只跑上线这一个策略)

修改纪律: core/ 任一文件改动都会让 preflight 的指纹锁 FAIL 并禁止交易。
只有"有意改策略 + 已重跑 run_backtest 确认 OOS 不下降"时才允许 --relock。
"""
from . import config          # noqa: F401
