"""
core.valuation —— value_comp 纯估值对照 (与 online_value 逐字同口径)
================================================================================
- load_valuation: valuation_cache.csv → {ep,bp,cfp,sp} date×instrument 透视表(倒数, ffill)
- value_comp_score: 信号日截面, 4估值倒数在池内取 rank 百分位再求均值 (越大越便宜)
"""
import os
import numpy as np
import pandas as pd

from . import config
from .universe import format_qlib_code


def load_valuation():
    """与 online_value/core/valuation.load_raw_valuation 完全一致口径."""
    if not os.path.exists(config.VAL_CACHE):
        raise RuntimeError(f"[CHECK] {config.VAL_CACHE} 缺失, value_comp 对照无法计算!")
    v = pd.read_csv(config.VAL_CACHE, sep="\t", dtype={"code": str})
    v["date"] = pd.to_datetime(v["date"])
    for c in ["pe_ttm", "pb", "ps_ttm", "pcf"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["ep"] = 1.0 / v["pe_ttm"]; v["bp"] = 1.0 / v["pb"]
    v["sp"] = 1.0 / v["ps_ttm"]; v["cfp"] = 1.0 / v["pcf"]
    v["instrument"] = v["code"].map(format_qlib_code)
    v = v.replace([np.inf, -np.inf], np.nan).drop_duplicates(["date", "instrument"], keep="last")
    piv = {c: v.pivot(index="date", columns="instrument", values=c).sort_index().ffill()
           for c in config.VALUE_FACTORS}
    print(f"[CHECK] 估值缓存: {piv['ep'].shape[0]}日 × {piv['ep'].shape[1]}股, "
          f"{piv['ep'].index[0].date()}~{piv['ep'].index[-1].date()}", flush=True)
    return piv


def value_comp_score(candidates, sig, val_piv):
    """信号日 value_comp: 4估值倒数在池内截面 rank 百分位均值(越大越便宜)"""
    f = pd.DataFrame(index=pd.Index(sorted(candidates), name="instrument"))
    for c in config.VALUE_FACTORS:
        pv = val_piv[c].loc[:sig]
        row = pv.iloc[-1] if len(pv) else pd.Series(dtype=float)
        f[c] = row.reindex(f.index)
    vc = pd.concat([f[c].rank(pct=True) for c in config.VALUE_FACTORS], axis=1).mean(axis=1)
    return vc
