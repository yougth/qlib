"""new_quant.core.config —— 新策略研究配置 (单点口径来源)
================================================================================
与 qlib/quant 完全隔离: 共享数据只读, 全部新产出落 new_quant/。
共享数据 (只读):
  · 行情:   qlib/data_cache/tencent/*.parquet (A股~3147只含部分退市, close=后复权)
  · 财务:   根目录 fcf_cache_pit.csv / profit_cache_pit.csv (年度, PIT=year+1-05-01)
  · 基准:   根目录 csi300_cache.csv
新数据 (本目录产出):
  · ETF行情: new_quant/data/etf/*.parquet (tools/fetch_etf_ohlcv.py 抓取)
  · 净利润补丁: new_quant/data/profit_patch_pit.csv (tools/fetch_profit_patch.py
    补齐主缓存 2012-2015 全市场缺口, 东财业绩报表口径已交叉验证)
"""
import os

NEW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB_DIR = os.path.dirname(NEW_DIR)                 # .../qlib
ROOT = os.path.dirname(QLIB_DIR)                    # 工作区根

TENCENT_DIR = os.path.join(QLIB_DIR, "data_cache", "tencent")
FCF_CACHE = os.path.join(ROOT, "fcf_cache_pit.csv")
PROFIT_CACHE = os.path.join(ROOT, "profit_cache_pit.csv")
BENCH_CACHE = os.path.join(ROOT, "csi300_cache.csv")
ETF_DIR = os.path.join(NEW_DIR, "data", "etf")
PROFIT_PATCH = os.path.join(NEW_DIR, "data", "profit_patch_pit.csv")
OUT_DIR = os.path.join(NEW_DIR, "outputs")

# ---- 回测口径 ----
BT_START = "2019-01-01"        # 受 csi300_cache 起点(2019-01-02)约束
BT_END = "2026-09-11"          # 与共享行情终点对齐 (parquet虽到09-17, 统一止于09-11)
TOPK = 15                      # 股票策略持仓数
FEE_RT_STOCK = 0.004           # A股往返0.4% (沿用项目口径)
FEE_RT_ETF = 0.001             # ETF往返0.1% (佣金+滑点, 保守)
COST_STRESS = 2.0              # 成本压力测试倍数
LIQ_THRESHOLD = 20_000_000     # 20日均成交额下限 (真实价口径)
MIN_PRICE = 2.0                # 最低股价红线 (真实价, 剔壳价值小微盘)
QUALITY_YEARS = 5              # 新策略质量门槛: 连续5年FCF+净利润双正 (区别于旧10年池)
TREND_TARGET_VOL = 0.10        # 跨资产趋势: 组合目标年化波动 10%

# ---- 跨资产趋势 ETF 篮子 (腾讯 fqkline 可抓) ----
ETF_SYMS = {
    "sh510300": "沪深300ETF",
    "sh510500": "中证500ETF",
    "sz159915": "创业板ETF",
    "sh510880": "红利ETF",
    "sh513100": "纳指ETF",
    "sh513500": "标普500ETF",
    "sz159920": "恒生ETF",
    "sh518880": "黄金ETF",
    "sh511010": "国债ETF",
    "sz159985": "豆粕ETF",
    "sh501018": "原油LOF",
}
