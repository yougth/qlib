"""
估值因子 —— 原始估值表读取 + value_comp 打分
==================================================================
口径 (函数体逐字搬自 v16_value_rules.load_raw_valuation / v17_residual_model.compute_value_comp):
  · load_raw_valuation: valuation_cache.csv → {ep,bp,cfp,sp} 四张 date x instrument 透视表
    - 取倒数 (1/pe_ttm 等), inf 置 NaN, 同日同股去重保留最后一条
    - **ffill**: 估值缓存非每日全覆盖, 用最近可得值 (这是回测口径, 不可改)
  · compute_value_comp: 信号日截面, 4个因子各自在**池内**取 rank 百分位再求均值
    - 池内 rank 是关键: 换了池子 → 分数会变; 因此池子构建必须先冻结
    - 分数越大 = 越便宜
"""
import numpy as np
import pandas as pd

from . import config
from .universe import format_qlib_code


def load_raw_valuation(verbose=True):
    """估值原始值(非zscore): pivot表 {factor: DataFrame(date x instrument)}"""
    v = pd.read_csv(config.VAL_CACHE, sep='\t', dtype={"code": str})
    v["date"] = pd.to_datetime(v["date"])
    for c in ["pe_ttm", "pb", "ps_ttm", "pcf"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["ep"] = 1.0 / v["pe_ttm"]; v["bp"] = 1.0 / v["pb"]
    v["sp"] = 1.0 / v["ps_ttm"]; v["cfp"] = 1.0 / v["pcf"]
    v["instrument"] = v["code"].map(format_qlib_code)
    v = v.replace([np.inf, -np.inf], np.nan)
    v = v.drop_duplicates(subset=["date", "instrument"], keep="last")
    out = {}
    for c in config.VALUE_FACTORS:
        out[c] = v.pivot(index="date", columns="instrument", values=c).sort_index().ffill()
    if verbose:
        print(f"  [估值原始表] {out['ep'].shape}", flush=True)
    return out


def compute_value_comp(universe, sig, val_piv):
    """信号日 value_comp: 4估值倒数的池内截面rank百分位均值 (越大越便宜)"""
    f = pd.DataFrame(index=pd.Index(sorted(universe), name="instrument"))
    for c in config.VALUE_FACTORS:
        pv = val_piv[c]
        row = pv.loc[:sig].iloc[-1] if len(pv.loc[:sig]) else pd.Series(dtype=float)
        f[c] = row.reindex(f.index)
    rk = lambda s: s.rank(pct=True)
    vc = pd.concat([rk(f[c]) for c in config.VALUE_FACTORS], axis=1).mean(axis=1)
    return vc
