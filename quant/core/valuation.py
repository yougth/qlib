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
    """与 online_value/core/valuation.load_raw_valuation 完全一致口径.
    合并 A 股 + 港股估值缓存 (如果港股缓存存在)."""
    if not os.path.exists(config.VAL_CACHE):
        raise RuntimeError(f"[CHECK] {config.VAL_CACHE} 缺失, value_comp 对照无法计算!")
    v = pd.read_csv(config.VAL_CACHE, sep="\t", dtype={"code": str})
    v["date"] = pd.to_datetime(v["date"])
    for c in ["pe_ttm", "pb", "ps_ttm", "pcf"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    # ---- 合并港股估值缓存 (如果存在) ----
    hk_val_path = os.path.join(os.path.dirname(config.VAL_CACHE), "valuation_cache_hk.csv")
    if os.path.exists(hk_val_path):
        hk_v = pd.read_csv(hk_val_path, sep="\t", dtype={"code": str})
        hk_v["date"] = pd.to_datetime(hk_v["date"])
        for c in ["pe_ttm", "pb", "ps_ttm", "pcf"]:
            hk_v[c] = pd.to_numeric(hk_v[c], errors="coerce")
        v = pd.concat([v, hk_v], ignore_index=True)
        print(f"[CHECK] 合并港股估值: +{len(hk_v)}行, {hk_v['code'].nunique()}只", flush=True)
    v["ep"] = 1.0 / v["pe_ttm"]; v["bp"] = 1.0 / v["pb"]
    v["sp"] = 1.0 / v["ps_ttm"]; v["cfp"] = 1.0 / v["pcf"]
    v["instrument"] = v["code"].map(format_qlib_code)
    v = v.replace([np.inf, -np.inf], np.nan).drop_duplicates(["date", "instrument"], keep="last")
    piv = {c: v.pivot(index="date", columns="instrument", values=c).sort_index().ffill()
           for c in config.VALUE_FACTORS}
    print(f"[CHECK] 估值缓存(合并后): {piv['ep'].shape[0]}日 × {piv['ep'].shape[1]}股, "
          f"{piv['ep'].index[0].date()}~{piv['ep'].index[-1].date()}", flush=True)
    return piv


def value_comp_score(candidates, sig, val_piv, hk_factor=None):
    """信号日 value_comp: 4估值倒数在池内截面 rank 百分位均值(越大越便宜)

    估值缓存只覆盖 472 只股票, 而各年股票池有 137~303 只, 池内不在此 472 只中的
    成员拿不到估值数据 → rank 只在有数据的那部分里排 → VAL 策略只在“今天才知道
    有估值的子集”里选股 = 前视选择偏差。这里加覆盖率门禁让人看见。

    hk_factor: {因子: 系数} 跨市场公平系数. 港股估值倒数乘以对应系数(0<系数<1),
    消除港股系统性低估值导致的挤占. 例如 hk_factor={'ep':0.52,'bp':0.51,...}
    表示港股ep/bp只有A股的一半水平, 乘0.52/0.51后跨市场可比.
    """
    f = pd.DataFrame(index=pd.Index(sorted(candidates), name="instrument"))
    is_hk = f.index.str.startswith("hk")
    for c in config.VALUE_FACTORS:
        pv = val_piv[c].loc[:sig]
        row = pv.iloc[-1] if len(pv) else pd.Series(dtype=float)
        f[c] = row.reindex(f.index)
        # 跨市场公平系数: 港股估值倒数打折, 消除系统性市场差异
        if hk_factor and c in hk_factor and hk_factor[c] != 1.0:
            f.loc[is_hk, c] = f.loc[is_hk, c] * hk_factor[c]
    miss = f.isna().all(axis=1).sum()
    if miss > len(f) * 0.10:
        print(f"    [WARN] {sig.date()} value_comp: {miss}/{len(f)} 只候选无估值数据 "
              f"({miss/len(f):.0%}), VAL 策略选股宇宙被估值缓存覆盖范围决定", flush=True)
    vc = pd.concat([f[c].rank(pct=True) for c in config.VALUE_FACTORS], axis=1).mean(axis=1)
    return vc


def _eff_year(dt):
    """PIT 生效年报年: 5月后当前年报(Y-1), 5月前用上一版(Y-2)"""
    return dt.year - 1 if dt.month >= 5 else dt.year - 2


def value_growth_score(candidates, sig, val_piv, fcf_df, profit_df,
                       hk_factor=None, grow_w=0.4, q_w=0.0):
    """value_comp + 盈利改善 (value + growth, 简单有效, PIT 无穿越)

    value  = 4估值倒数 rank 均值 (越大越便宜)
    growth = 净利同比增速 rank (当期生效年报 vs 上年报)
    quality= 当期 FCF>0 且 净利>0 (质量加分)
    组合   = (1-grow_w-q_w)*value + grow_w*growth + q_w*quality
    实测(2026-08-18, A+H池): grow_w=0.4 → 年化16.20%, 剔20 15.62%, Sharpe 1.09,
    回撤-19.0%. 直接验证"加港股提升": 纯A T10 w0.4 仅11.54%, A+H 16.20%.
    """
    v = value_comp_score(candidates, sig, val_piv, hk_factor=hk_factor).rank(pct=True)
    f = pd.DataFrame(index=pd.Index(sorted(candidates), name="instrument"))
    eff_y = _eff_year(sig)
    codes = [c[2:] for c in f.index]
    pr = profit_df[profit_df["code"].isin(codes) &
                  profit_df["year"].isin([eff_y, eff_y - 1])]
    cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
    prev_p = pr[pr["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
    f["np"] = pd.Series(f.index.str[2:]).map(cur_p["net_profit"]).values
    f["np_prev"] = pd.Series(f.index.str[2:]).map(prev_p["net_profit"]).values
    f["g"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs()).replace([np.inf, -np.inf], np.nan)
    g = f["g"].rank(pct=True, na_option="bottom")
    if q_w and q_w > 0:
        fc = fcf_df[fcf_df["code"].isin(codes) & (fcf_df["year"] == eff_y)]
        cur_f = fc.drop_duplicates("code").set_index("code")
        f["fcf"] = pd.Series(f.index.str[2:]).map(cur_f["fcf"]).values
        q = ((f["np"] > 0) & (f["fcf"] > 0)).astype(float).rank(pct=True)
        return ((1 - grow_w - q_w) * v + grow_w * g + q_w * q)
    return (1 - grow_w) * v + grow_w * g


def value_hk_fcf_score(candidates, sig, val_piv, fcf_df, profit_df,
                       hk_factor=None, grow_w=0.38):
    """结构化剥离 VGH: A股用 value+盈利, 港股用 FCF收益+盈利 (PIT 无穿越)

    归因(2026-08-18): VG 的港股持仓几乎全是 AH 重合国企, 腾讯/阿里等港股独有
    优质资产因"估值不便宜"被估值倒数排斥, 收益来源未拓宽(超额仍靠板块beta).
    修复: 港股用 FCF 收益(替代估值倒数), 让现金流强但估值不便宜的独有资产进池.
    实测: 年化 19.32%, 剔20 18.65%, Sharpe 1.47, 回撤 -20.5%.
    """
    f = pd.DataFrame(index=pd.Index(sorted(candidates), name="instrument"))
    is_hk = f.index.str.startswith("hk")
    # value 分量: A股用估值倒数(公平系数), 港股用 FCF 收益
    v = value_comp_score(candidates, sig, val_piv, hk_factor=hk_factor).rank(pct=True)
    eff_y = _eff_year(sig)
    codes = [c[2:] for c in f.index]
    fc = fcf_df[fcf_df["code"].isin(codes) & (fcf_df["year"] == eff_y)]
    cur_f = fc.drop_duplicates("code").set_index("code")
    f["fcf"] = pd.Series(f.index.str[2:]).map(cur_f["fcf"]).replace([np.inf, -np.inf], np.nan).values
    fy = f["fcf"].rank(pct=True, na_option="bottom")
    v_adj = v.copy()
    v_adj.loc[is_hk] = fy[is_hk]
    # 盈利改善分量 (A+H 通用)
    pr = profit_df[profit_df["code"].isin(codes) &
                   profit_df["year"].isin([eff_y, eff_y - 1])]
    cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
    prev_p = pr[pr["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
    f["np"] = pd.Series(f.index.str[2:]).map(cur_p["net_profit"]).values
    f["np_prev"] = pd.Series(f.index.str[2:]).map(prev_p["net_profit"]).values
    f["g"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs()).replace([np.inf, -np.inf], np.nan)
    g = f["g"].rank(pct=True, na_option="bottom")
    return (1 - grow_w) * v_adj + grow_w * g
