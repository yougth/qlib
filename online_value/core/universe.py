"""
股票池 —— 动态候选池构建 (十年净利润 & FCF 双正)
==================================================================
口径 (与回测完全一致, 函数体逐字搬自 v5_xgb_turnover_comparative.build_dynamic_universe):
  · 财务年份用 backtest_year-11 ~ backtest_year-2 —— 用 year-2 是为了避免前视
    (年报次年4月才披露, 用 year-1 会偷看未来)
  · FCF: 该区间内 >=80% 年份有数据, 且**每一年**自由现金流为正
  · 净利润: 同上, 且 profit_end >= 2016 才启用(缓存起点限制), 否则回退到 FCF 单条件
  · 两个集合取交集
数据源为 PIT 增广缓存(含退市股), 修存活偏差。
"""
import pandas as pd

from . import config


def format_qlib_code(code):
    """6位代码 → qlib instrument: 6开头为沪市 SH, 其余深市 SZ"""
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"


def load_fin_caches():
    """读 PIT 财务缓存, 返回 (fcf_df, profit_df), code 已补齐6位"""
    fcf_df = pd.read_csv(config.FCF_CACHE, sep='\t')
    profit_df = pd.read_csv(config.PROFIT_CACHE, sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    return fcf_df, profit_df


def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    fcf_start = backtest_year - 11; fcf_end = backtest_year - 2
    target_fcf_years = list(range(fcf_start, fcf_end + 1))
    fcf_filtered = fcf_df[fcf_df["year"].isin(target_fcf_years)]
    fcf_positive = fcf_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_positive["code"].unique())
    profit_start = max(backtest_year - 11, 2016); profit_end = backtest_year - 2
    if profit_end >= 2016:
        target_profit_years = list(range(profit_start, profit_end + 1))
        profit_filtered = profit_df[profit_df["year"].isin(target_profit_years)]
        profit_positive = profit_filtered.groupby("code").filter(
            lambda g: len(g) >= len(target_profit_years) * 0.8 and (g["net_profit"] > 0).all())
        profit_codes = set(profit_positive["code"].unique())
    else:
        profit_codes = fcf_codes
    return fcf_codes & profit_codes


def get_universe(year, fcf_df=None, profit_df=None):
    """便捷入口: 返回该回测年的 instrument 列表(已排序)"""
    if fcf_df is None or profit_df is None:
        fcf_df, profit_df = load_fin_caches()
    codes = build_dynamic_universe(year, fcf_df, profit_df)
    return sorted(format_qlib_code(c) for c in codes)


def name_map():
    """code(6位) → 中文名称, 用于清单可读性"""
    df = pd.read_csv(config.FCF_CACHE, sep='\t')
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df.drop_duplicates("code").set_index("code")["name"].to_dict()
