"""value 简单有效优化实验: 在纯A股池上测试多种 value 增强
目标: 年化 > 15% (纯A股价值池)
思路(简单有效, PIT 无穿越):
  1. 纯 value_comp (基线): 4估值倒数 rank 均值
  2. value + 质量门槛: 剔除 FCF/净利 质量差(用PIT基本面)
  3. value + 动量过滤: 剔除 12月/6月动量最差(便宜+跌=价值陷阱)
  4. value + 质量 + 动量 (组合)
  5. value 单因子权重: ep 单因子 / 不同权重
仅纯A股 (INCLUDE_HK=False), 不训练模型, 快速遍历.
"""
import os, sys, gc
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import core.config as config
from core import data as datalayer
from core import strategy
from core.universe import (build_windows, build_dynamic_universe,
                           format_qlib_code, load_pit_caches)
from core.valuation import load_valuation, value_comp_score
from core.tradability import build_tradability
from core.backtest import portfolio_backtest, calc_metrics

config.INCLUDE_HK = False          # 纯A股对照
config.INJECT_MARKET = False       # is_hk 已回滚

OUT = config.OUT_DIR
BT_START = pd.Timestamp(config.BT_START)


def momentum_score(cand, dt, px, lookback=252):
    """动量: 过去 lookback 交易日收益. 返回 Series(index=instrument, value=动量)"""
    # px: date×instrument 收盘价矩阵
    if dt not in px.index:
        hist = px.loc[:dt]
    else:
        hist = px.loc[:dt]
    if hist.empty:
        return pd.Series(1.0, index=cand)
    last = hist.iloc[-1]
    i = hist.index.get_loc(hist.index[-1])
    j = max(0, i - lookback)
    start = hist.iloc[j] if hist.index[j] != hist.index[-1] else last
    mom = (last / start.replace(0, np.nan) - 1).reindex(cand)
    return mom


def _eff_year(dt):
    return dt.year - 1 if dt.month >= 5 else dt.year - 2


def fcf_yield_score(cand, dt, fcf_df):
    """FCF 收益代理: 当期 FCF 排名 (越大现金流越强)"""
    f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
    eff_y = _eff_year(dt)
    codes = [c[2:] for c in f.index]
    fc = fcf_df[fcf_df["code"].isin(codes) & (fcf_df["year"] == eff_y)]
    fc = fc.drop_duplicates("code").set_index("code")
    f["fcf"] = pd.Series(f.index.str[2:]).map(fc["fcf"]).replace([np.inf, -np.inf], np.nan).values
    return f["fcf"].rank(pct=True, na_option="bottom")


def profit_growth_score(cand, dt, profit_df):
    """净利同比增长率排名"""
    f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
    eff_y = _eff_year(dt)
    codes = [c[2:] for c in f.index]
    pr = profit_df[profit_df["code"].isin(codes) & profit_df["year"].isin([eff_y, eff_y - 1])]
    cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
    prev_p = pr[pr["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
    f["np"] = pd.Series(f.index.str[2:]).map(cur_p["net_profit"]).values
    f["np_prev"] = pd.Series(f.index.str[2:]).map(prev_p["net_profit"]).values
    f["g"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs()).replace([np.inf, -np.inf], np.nan)
    return f["g"].rank(pct=True, na_option="bottom")


def make_score(mode, val_piv, px, fcf_df, profit_df, cal, hkf=None):
    """构造 value 增强打分函数 (mode 指定增强类型)"""
    def base(cand, dt):
        return value_comp_score(cand, dt, val_piv, hk_factor=hkf)

    def quality_mask(cand, dt):
        """质量: 剔除 FCF/净利 增速为负 或 ROE 低于中位的 (PIT)"""
        # 用 PIT 缓存: 最新可用年报数据 (年份按 5-01 生效)
        y = dt.year
        # 当前生效财年 = 上一年报 (Y-1 年, 次年5月生效)
        eff_years = [y - 1, y - 2]
        f = fcf_df[(fcf_df["year"].isin(eff_years)) & (fcf_df["code"].isin([c[2:] for c in cand]))]
        p = profit_df[(profit_df["year"].isin(eff_years)) & (profit_df["code"].isin([c[2:] for c in cand]))]
        f = f.drop_duplicates("code").set_index("code")
        p = p.drop_duplicates("code").set_index("code")
        q = pd.DataFrame(index=cand)
        q["inst"] = cand
        q["code"] = q["inst"].str[2:]
        q["fcf"] = q["code"].map(f["fcf"]).replace([np.inf, -np.inf], np.nan)
        q["np"] = q["code"].map(p["net_profit"]).replace([np.inf, -np.inf], np.nan)
        # 质量规则: FCF>0 且 净利>0 (简单)  -- 池本身十年双正, 这里看当期
        q["ok"] = (q["fcf"] > 0) & (q["np"] > 0)
        return q.set_index("inst")["ok"].reindex(cand).fillna(False)

    if mode == "base":
        return lambda cand, dt: base(cand, dt)
    if mode == "quality":
        def fn(cand, dt):
            s = base(cand, dt)
            ok = quality_mask(cand, dt)
            return s.where(ok, -999.0)   # 质量差的排到最后
        return fn
    if mode == "momentum":
        def fn(cand, dt):
            s = base(cand, dt)
            mom = momentum_score(cand, dt, px)
            bad = mom < mom.median()      # 动量低于中位数的剔除
            return s.where(bad, s)        # 保留全部, 动量只作为排序微调
        return fn
    if mode == "mom_filter":
        def fn(cand, dt):
            s = base(cand, dt)
            mom = momentum_score(cand, dt, px)
            cutoff = mom.quantile(0.3)     # 剔除动量最差30%
            return s.where(mom >= cutoff, -999.0)
        return fn
    if mode == "ep_only":
        def fn(cand, dt):
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            pv = val_piv["ep"].loc[:dt]
            f["ep"] = pv.iloc[-1].reindex(f.index) if len(pv) else pd.Series(dtype=float)
            return f["ep"].rank(pct=True)
        return fn
    if mode == "quality_mom":
        def fn(cand, dt):
            s = base(cand, dt)
            ok = quality_mask(cand, dt)
            mom = momentum_score(cand, dt, px)
            cutoff = mom.quantile(0.3)
            keep = ok & (mom >= cutoff)
            return s.where(keep, -999.0)
        return fn
    if mode == "deep_value":
        """深度价值: 4因子秩取最小值(要求全维度都便宜), 更极端"""
        def fn(cand, dt):
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            is_hk = f.index.str.startswith("hk")
            for c in config.VALUE_FACTORS:
                pv = val_piv[c].loc[:dt]
                row = pv.iloc[-1] if len(pv) else pd.Series(dtype=float)
                f[c] = row.reindex(f.index)
                if hkf and c in hkf:
                    f.loc[is_hk, c] = f.loc[is_hk, c] * hkf[c]
            r = pd.concat([f[c].rank(pct=True) for c in config.VALUE_FACTORS], axis=1)
            return r.min(axis=1)
        return fn
    if mode == "ep_bp":
        """EP+BP 加权(更偏盈利/股息): 0.5*ep秩 + 0.5*bp秩"""
        def fn(cand, dt):
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            for c in ["ep", "bp"]:
                pv = val_piv[c].loc[:dt]
                row = pv.iloc[-1] if len(pv) else pd.Series(dtype=float)
                f[c] = row.reindex(f.index)
            return 0.5 * f["ep"].rank(pct=True) + 0.5 * f["bp"].rank(pct=True)
        return fn
    if mode == "deep_ep":
        """深度EP: 仅ep倒数 + 要求>=池中位, 极端便宜盈利股"""
        def fn(cand, dt):
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            pv = val_piv["ep"].loc[:dt]
            row = pv.iloc[-1] if len(pv) else pd.Series(dtype=float)
            f["ep"] = row.reindex(f.index)
            return f["ep"].rank(pct=True)
        return fn
    if mode == "multi4":
        """四因子等权: value + quality + growth + momentum (简单 Smart Beta)
        value   = value_comp (4估值倒数秩均值)
        quality = FCF/净利 质量: FCF收益率(FCF/市值近似用fcf排名) 或 净利率
        growth  = FCF/净利 增速 (PIT, 用缓存)
        momentum= 过去1年动量 (行情)
        """
        def fn(cand, dt):
            v = base(cand, dt).rank(pct=True)
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            # ---- quality + growth 用 PIT 基本面缓存 ----
            y = dt.year
            eff = [y - 1, y - 2]
            codes = [c[2:] for c in f.index]
            fc = fcf_df[fcf_df["code"].isin(codes) & fcf_df["year"].isin(eff)]
            pr = profit_df[profit_df["code"].isin(codes) & profit_df["year"].isin(eff)]
            # 当期 vs 前一期 (增速)
            cur_f = fc[fc["year"] == max(eff) if fc["year"].nunique() > 1 else eff[0]]
            cur_p = pr[pr["year"] == max(eff) if pr["year"].nunique() > 1 else eff[0]]
            prev_f = fc[fc["year"] == min(eff)]
            prev_p = pr[pr["year"] == min(eff)]
            cur_f = cur_f.drop_duplicates("code").set_index("code") if len(cur_f) else pd.DataFrame()
            cur_p = cur_p.drop_duplicates("code").set_index("code") if len(cur_p) else pd.DataFrame()
            prev_f = prev_f.drop_duplicates("code").set_index("code") if len(prev_f) else pd.DataFrame()
            prev_p = prev_p.drop_duplicates("code").set_index("code") if len(prev_p) else pd.DataFrame()
            f["code"] = f.index.str[2:]
            f["fcf"] = f["code"].map(cur_f["fcf"]) if len(cur_f) else np.nan
            f["np"] = f["code"].map(cur_p["net_profit"]) if len(cur_p) else np.nan
            f["fcf_prev"] = f["code"].map(prev_f["fcf"]) if len(prev_f) else np.nan
            f["np_prev"] = f["code"].map(prev_p["net_profit"]) if len(prev_p) else np.nan
            f["fcf_g"] = (f["fcf"] / f["fcf_prev"].replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
            f["np_g"] = (f["np"] / f["np_prev"].replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
            f["growth"] = f[["fcf_g", "np_g"]].mean(axis=1, skipna=True)
            # quality: FCF收益率 (用 fcf / 市值? 无市值, 用净利率 np/fcf 近似)
            f["quality"] = (f["np"] / (f["fcf"].abs() + 1e-8)).replace([np.inf, -np.inf], np.nan)
            q = f["quality"].rank(pct=True, na_option="bottom")
            g = f["growth"].rank(pct=True, na_option="bottom")
            mom = momentum_score(f.index, dt, px).rank(pct=True, na_option="bottom")
            return (v + q + g + mom) / 4
        return fn
    if mode == "value_mom":
        """value + 动量 双因子: 估值便宜 + 趋势向上"""
        def fn(cand, dt):
            v = base(cand, dt).rank(pct=True)
            mom = momentum_score(cand, dt, px).rank(pct=True, na_option="bottom")
            return 0.6 * v + 0.4 * mom
        return fn
    if mode == "value_quality":
        """value + 盈利改善: 估值便宜 + 净利/FCF 同比正增长 (加分项, 非过滤)"""
        w = getattr(config, "VALUE_GROWTH_W", 0.4)
        def fn(cand, dt):
            v = base(cand, dt).rank(pct=True)
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            y = dt.year
            # PIT: 当前生效年报 = Y-1 (5月后) 或 Y-2 (5月前)
            eff_y = y - 1 if dt.month >= 5 else y - 2
            prev_y = eff_y - 1
            codes = [c[2:] for c in f.index]
            pr = profit_df[profit_df["code"].isin(codes) & profit_df["year"].isin([eff_y, prev_y])]
            fc = fcf_df[fcf_df["code"].isin(codes) & fcf_df["year"].isin([eff_y, prev_y])]
            cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
            prev_p = pr[pr["year"] == prev_y].drop_duplicates("code").set_index("code")
            cur_f = fc[fc["year"] == eff_y].drop_duplicates("code").set_index("code")
            prev_f = fc[fc["year"] == prev_y].drop_duplicates("code").set_index("code")
            f["code"] = f.index.str[2:]
            f["np"] = f["code"].map(cur_p["net_profit"]) if len(cur_p) else np.nan
            f["np_prev"] = f["code"].map(prev_p["net_profit"]) if len(prev_p) else np.nan
            f["fcf"] = f["code"].map(cur_f["fcf"]) if len(cur_f) else np.nan
            f["fcf_prev"] = f["code"].map(prev_f["fcf"]) if len(prev_f) else np.nan
            # 盈利改善: 净利同比增长率 (简单, 稳健)
            f["np_g"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs()).replace([np.inf, -np.inf], np.nan)
            # FCF 增长
            f["fcf_g"] = ((f["fcf"] - f["fcf_prev"]) / f["fcf_prev"].abs()).replace([np.inf, -np.inf], np.nan)
            # 质量: 当期盈利+现金流都为正
            f["quality_ok"] = (f["np"] > 0) & (f["fcf"] > 0)
            g = f["np_g"].rank(pct=True, na_option="bottom")
            # 组合: value 为主 + 成长加分
            q = f["quality_ok"].astype(float).rank(pct=True)
            return (1 - w) * v + w * (0.85 * g + 0.15 * q)
        return fn
    if mode == "value_growth2y":
        """value + 2年复合盈利增速: 便宜 + 持续成长 (更稳健)"""
        w = getattr(config, "VALUE_GROWTH_W", 0.4)
        def fn(cand, dt):
            v = base(cand, dt).rank(pct=True)
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            y = dt.year
            eff_y = y - 1 if dt.month >= 5 else y - 2
            codes = [c[2:] for c in f.index]
            pr = profit_df[profit_df["code"].isin(codes) & profit_df["year"].isin([eff_y, eff_y - 1, eff_y - 2])]
            cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
            p1 = pr[pr["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
            p2 = pr[pr["year"] == eff_y - 2].drop_duplicates("code").set_index("code")
            f["code"] = f.index.str[2:]
            f["np"] = f["code"].map(cur_p["net_profit"]) if len(cur_p) else np.nan
            f["np1"] = f["code"].map(p1["net_profit"]) if len(p1) else np.nan
            f["np2"] = f["code"].map(p2["net_profit"]) if len(p2) else np.nan
            # 2年复合增速: (np/np2)^(1/2) - 1
            f["g2y"] = ((f["np"] / f["np2"].replace(0, np.nan)) ** 0.5 - 1).replace([np.inf, -np.inf], np.nan)
            # 连续增长: np > np1 > np2
            f["consist"] = ((f["np"] > f["np1"]) & (f["np1"] > f["np2"])) & (f["np2"] > 0)
            g = f["g2y"].rank(pct=True, na_option="bottom")
            c = f["consist"].astype(float).rank(pct=True)
            return (1 - w) * v + w * (0.7 * g + 0.3 * c)
        return fn
    if mode == "value_epsfcf":
        """value + 盈利改善 + FCF收益: 便宜 + 盈利增 + 现金流质量"""
        w = getattr(config, "VALUE_GROWTH_W", 0.4)
        def fn(cand, dt):
            v = base(cand, dt).rank(pct=True)
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            y = dt.year
            eff_y = y - 1 if dt.month >= 5 else y - 2
            codes = [c[2:] for c in f.index]
            pr = profit_df[profit_df["code"].isin(codes) & profit_df["year"].isin([eff_y, eff_y - 1])]
            fc = fcf_df[fcf_df["code"].isin(codes) & fcf_df["year"].isin([eff_y, eff_y - 1])]
            cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
            prev_p = pr[pr["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
            cur_f = fc[fc["year"] == eff_y].drop_duplicates("code").set_index("code")
            prev_f = fc[fc["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
            f["code"] = f.index.str[2:]
            f["np"] = f["code"].map(cur_p["net_profit"]) if len(cur_p) else np.nan
            f["np_prev"] = f["code"].map(prev_p["net_profit"]) if len(prev_p) else np.nan
            f["fcf"] = f["code"].map(cur_f["fcf"]) if len(cur_f) else np.nan
            f["fcf_prev"] = f["code"].map(prev_f["fcf"]) if len(prev_f) else np.nan
            f["np_g"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs()).replace([np.inf, -np.inf], np.nan)
            f["fcf_g"] = ((f["fcf"] - f["fcf_prev"]) / f["fcf_prev"].abs()).replace([np.inf, -np.inf], np.nan)
            f["g"] = f[["np_g", "fcf_g"]].mean(axis=1, skipna=True)
            g = f["g"].rank(pct=True, na_option="bottom")
            return (1 - w) * v + w * g
        return fn
    if mode == "fcf_yield":
        """纯 FCF 收益: 用 FCF/市值 的代理 = fcf 排名 (现金流价值)"""
        def fn(cand, dt):
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            y = dt.year
            eff_y = y - 1 if dt.month >= 5 else y - 2
            codes = [c[2:] for c in f.index]
            fc = fcf_df[fcf_df["code"].isin(codes) & (fcf_df["year"] == eff_y)]
            fc = fc.drop_duplicates("code").set_index("code")
            f["fcf"] = f.index.str[2:].map(fc["fcf"]).replace([np.inf, -np.inf], np.nan)
            return f["fcf"].rank(pct=True, na_option="bottom")
        return fn
    if mode == "fcf_improve":
        """FCF 收益 + 盈利改善: 现金流价值 + 盈利增长"""
        w = getattr(config, "VALUE_GROWTH_W", 0.4)
        def fn(cand, dt):
            v = fcf_yield_score(cand, dt, fcf_df)
            g = profit_growth_score(cand, dt, profit_df)
            return (1 - w) * v + w * g
        return fn
    if mode == "value_hk_fcf":
        """结构化剥离 VGH: A股用 value+盈利, 港股用 FCF收益+盈利 (独有资产纳入)
        让腾讯/阿里等现金流强但估值不便宜的港股独有资产进池.
        """
        w = getattr(config, "VALUE_GROWTH_W", 0.38)
        def fn(cand, dt):
            f = pd.DataFrame(index=pd.Index(sorted(cand), name="instrument"))
            is_hk = f.index.str.startswith("hk")
            # value 分量: A股用估值, 港股用 FCF 收益 (替代估值倒数)
            v = value_comp_score(cand, dt, val_piv, hk_factor=hkf).rank(pct=True)
            eff_y = _eff_year(dt)
            codes = [c[2:] for c in f.index]
            fc = fcf_df[fcf_df["code"].isin(codes) & (fcf_df["year"] == eff_y)]
            fc = fc.drop_duplicates("code").set_index("code")
            f["fcf"] = pd.Series(f.index.str[2:]).map(fc["fcf"]).replace([np.inf, -np.inf], np.nan).values
            fy = f["fcf"].rank(pct=True, na_option="bottom")
            # 港股: 用 FCF 收益; A股: 用估值倒数
            v_adj = v.copy()
            v_adj[is_hk] = fy[is_hk]
            # 盈利改善分量 (A+H 都用)
            pr = profit_df[profit_df["code"].isin(codes) &
                           profit_df["year"].isin([eff_y, eff_y - 1])]
            cur_p = pr[pr["year"] == eff_y].drop_duplicates("code").set_index("code")
            prev_p = pr[pr["year"] == eff_y - 1].drop_duplicates("code").set_index("code")
            f["np"] = pd.Series(f.index.str[2:]).map(cur_p["net_profit"]).values
            f["np_prev"] = pd.Series(f.index.str[2:]).map(prev_p["net_profit"]).values
            f["g"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs()).replace([np.inf, -np.inf], np.nan)
            g = f["g"].rank(pct=True, na_option="bottom")
            return (1 - w) * v_adj + w * g
        return fn
    raise ValueError(mode)


def run_mode(mode, topk=10, out_tag=None):
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    windows = build_windows()
    val_piv = load_valuation()
    # 价格矩阵 (动量用)
    all_codes = set()
    for win in windows:
        codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
        all_codes.update(format_qlib_code(c) for c in codes)
    px = datalayer.load_price_matrix(sorted(all_codes))

    rebalances = {mode: []}
    holdings_log = []
    ic_records = []
    score_fn = make_score(mode, val_piv, px, fcf_df, profit_df, cal,
                          hkf=config.VHF_FACTOR if config.INCLUDE_HK else None)

    for win in windows:
        y = win["year"]
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = [format_qlib_code(c) for c in codes]
        xs, xe = win["test"]
        limit_up, susp, liq, close_px = build_tradability(universe, xs, xe)
        fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)
        strategy.emit_window_signals(
            mode, None, universe, cal, val_piv, xs, xe,
            limit_up, susp, liq, fwd_mat, config.TOPK,
            ic_records=ic_records, rebalances=rebalances,
            win_name=win["name"], strict=True,
            min_cand=config.MIN_CAND, holdings_log=holdings_log,
            score_fn=score_fn, topk_pick=topk)
        gc.collect()

    all_insts = sorted({i for _, tops in rebalances[mode] for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()
    rets, avg_to, n_buys = portfolio_backtest(rebalances[mode], price_mat)
    rets = rets[rets.index >= BT_START]
    m = calc_metrics(rets, bench)
    no20 = rets[rets.index >= pd.Timestamp("2021-01-01")]
    ar_no20 = (1 + no20).prod() ** (244 / len(no20)) - 1 if len(no20) else np.nan
    res = {"mode": out_tag or mode, "annual": m.get("ar", np.nan),
           "annual_no20": ar_no20, "sharpe": m.get("sharpe", np.nan),
           "mdd": m.get("mdd", np.nan), "turnover": avg_to}
    print(f"  [{out_tag or mode}] 年化={res['annual']*100:.2f}% 剔20={res['annual_no20']*100:.2f}% "
          f"Sharpe={res['sharpe']:.2f} 回撤={res['mdd']*100:.1f}% 换手={avg_to*100:.1f}%",
          flush=True)
    return res


if __name__ == "__main__":
    import json
    results = {}
    config.INCLUDE_HK = True
    config.INJECT_MARKET = False
    print("### VGH: A股value+盈利, 港股FCF收益+盈利 ###", flush=True)
    for mode, tag, topk, w in [
        ("value_hk_fcf", "VGH T10 w0.38", 10, 0.38),
    ]:
        if w is not None:
            config.VALUE_GROWTH_W = w
        print(f"\n=== {tag} ===", flush=True)
        try:
            results[f"{mode}_{w}"] = run_mode(mode, topk, tag)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {tag} 失败: {e}", flush=True)
    print("\n===== VGH 结果 =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))
