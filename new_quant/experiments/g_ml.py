#!/usr/bin/env python3
"""experiments.g_ml —— G(低关注质量真市值) × LightGBM 模型叠加 滚动OOS实验
================================================================================
问题: 在G规则基座(年化20.12%/Sharpe0.81/回撤-39.8%)上叠加ML模型,
      能否同时提升收益与降低风险?

经验约束 (来自本项目历史教训):
  · 滚动OOS: 每月用截至t-1的样本训练, valid=最近6个月RankIC早停, 预测t月截面
  · 特征纪律: 窗口与月频调仓同量级, 剔除SUMD/SUMP类短期噪声特征
  · 融合权重w只在valid段寻优 (VALF纪律, 不碰OOS)
  · 双腿分离: 模型作为独立腿给固定资金比例, 而非仅打分融合 (value_comp经验)

变体 (统一区间 2020-02~2026-09, 因前12个月为训练预热):
  G0  规则对照  (0.6×小市值+0.4×现金转换, Top30+缓冲12-25)
  M1  纯模型    (LGB截面分 Top30+缓冲12-25, 同缓冲同TopK只换分数)
  M2  打分融合  ((1-w)×G规则rank + w×模型rank, w在valid段按月寻优)
  M3  双腿分离  (70%资金G0 + 30%资金M1, 合并目标权重, 含两腿调仓成本)

特征 (信号日PIT, 全部≤252d窗口): 反转/动量/波动/流动性/市值/质量/成长/量价相关
标签: 21日前瞻收益 (与rank_ic口径一致)
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgbm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import (run_portfolio, calc_metrics, yearly_returns,  # noqa: E402
                           rank_ic, print_metric_table)
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402

LGB_PARAMS = dict(objective="regression", learning_rate=0.02, num_leaves=31,
                  max_depth=6, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1,
                  lambda_l2=1.0, verbose=-1, seed=42)
N_ROUNDS, EARLY_STOP = 500, 50
TRAIN_MIN, VALID_MONTHS = 12, 6
W_GRID = [0.0, 0.15, 0.3, 0.5]      # M2 融合权重网格 (valid段按月选)
TOPK, ENTRY, EXIT = 30, 12, 25      # 与G0完全一致


def build_features(panels, mvp, fin, sigs, cands):
    """信号日截面特征 (PIT): 月频友好, 无SUMD类短噪"""
    px = panels["adj_close"].ffill()
    amt = panels["amount20"].ffill()
    pc = px.pct_change()
    F = {}
    F["ret_1m"] = px / px.shift(21) - 1           # 短期反转 (A股强)
    F["ret_3m"] = px / px.shift(63) - 1
    F["ret_6m"] = px / px.shift(126) - 1
    F["mom_12_1"] = px.shift(21) / px.shift(252) - 1   # 学术动量 (剔近月)
    F["vol_60"] = pc.rolling(60).std() * np.sqrt(244)
    F["vol_120"] = pc.rolling(120).std() * np.sqrt(244)
    F["amt_log"] = np.log(amt + 1)
    F["amt_chg"] = amt / amt.rolling(120).mean() - 1   # 流动性变化
    F["dd_high"] = px / px.rolling(252).max() - 1      # 距一年高点
    F["corr_pv"] = pc.rolling(60).corr(np.log(amt + 1))  # 量价相关
    fwd = px.shift(-21) / px - 1                   # 标签: 21日前瞻

    out = {}
    for t, e in sigs:
        if t not in px.index or not cands.get(t):
            continue
        cand = cands[t]
        X = pd.DataFrame({k: v.loc[t].reindex(cand) for k, v in F.items()})
        X["cc"] = data.cash_conversion(fin, t).reindex(cand)
        g = data.growth_frame(fin, t)
        X["g_np"] = g["g_np"].reindex(cand) if len(g) else np.nan
        X["g_fcf"] = g["g_fcf"].reindex(cand) if len(g) else np.nan
        mv = mvp.loc[:t].iloc[-1].reindex(cand)
        X["mv_log"] = np.log(mv + 1)
        X["mv_pct"] = (-mv).rank(pct=True)         # 小市值→高分 (与规则同向)
        out[t] = (X, fwd.loc[t].reindex(cand))
    return out


def _rank_ic_by_month(pred, y, months):
    df = pd.DataFrame({"p": pred, "y": y, "m": months}).dropna()
    ics = []
    for m, g in df.groupby("m"):
        if len(g) >= 10:
            ics.append(g["p"].corr(g["y"], method="spearman"))
    return float(np.mean(ics)) if ics else np.nan


def rolling_oos(feat, sigs):
    """逐月滚动: train=截至t-1全部(去valid), valid=最近6月; 返回 (preds, w选, IC记录)"""
    months = [t for t, e in sigs if t in feat]
    preds, ws_chosen, ic_log = {}, [], []
    for i, t in enumerate(months):
        if i < TRAIN_MIN:
            continue
        hist = months[:i]                          # 全部 < t, PIT
        va_ms = hist[-VALID_MONTHS:]
        tr_ms = hist[:-VALID_MONTHS]
        if not len(tr_ms):
            continue
        X_tr = pd.concat([feat[m][0] for m in tr_ms], ignore_index=True)
        y_tr = pd.concat([feat[m][1] for m in tr_ms], ignore_index=True)
        X_va = pd.concat([feat[m][0] for m in va_ms], ignore_index=True)
        y_va = pd.concat([feat[m][1] for m in va_ms], ignore_index=True)
        m_va = np.repeat(va_ms, [len(feat[m][0]) for m in va_ms])
        ok = y_tr.notna().values
        X_tr, y_tr = X_tr[ok], y_tr[ok]
        ok = y_va.notna().values
        X_va, y_va, m_va = X_va[ok], y_va[ok], m_va[ok]
        if len(X_tr) < 500:
            continue

        tr = lgbm.Dataset(X_tr.values, label=y_tr.values,
                          feature_name=list(X_tr.columns))
        va = lgbm.Dataset(X_va.values, label=y_va.values, reference=tr,
                          feature_name=list(X_tr.columns))

        def feval(predt, ds):
            return "rank_ic", _rank_ic_by_month(predt, y_va.values, m_va), True

        m = lgbm.train(LGB_PARAMS, tr, num_boost_round=N_ROUNDS,
                       valid_sets=[va], valid_names=["valid"], feval=feval,
                       callbacks=[lgbm.early_stopping(EARLY_STOP,
                                                      first_metric_only=True,
                                                      verbose=False)])
        p_va = m.predict(X_va.values, num_iteration=m.best_iteration)
        vic = _rank_ic_by_month(p_va, y_va.values, m_va)

        # M2 融合权重: valid段按月寻优 (规则rank需重算——valid各月规则分)
        best_w, best_ic = 0.0, -np.inf
        if vic == vic:  # not nan
            rule_va = pd.concat([feat[m][0]["mv_pct"] * 0.6
                                 + feat[m][0]["cc"].rank(pct=True) * 0.4
                                 for m in va_ms], ignore_index=True)
            pva_rank = pd.Series(p_va).rank(pct=True)
            for w in W_GRID:
                fused = (1 - w) * rule_va.rank(pct=True) + w * pva_rank
                ic = _rank_ic_by_month(fused.values, y_va.values, m_va)
                if ic == ic and ic > best_ic:
                    best_w, best_ic = w, ic
        ws_chosen.append(best_w)

        p_t = pd.Series(m.predict(feat[t][0].values,
                                  num_iteration=m.best_iteration),
                        index=feat[t][0].index)
        preds[t] = p_t
        y_t = feat[t][1]
        oos_ic = (p_t.corr(y_t, method="spearman")
                  if y_t.notna().sum() >= 10 else np.nan)
        ic_log.append((t, vic, oos_ic, m.best_iteration, len(X_tr)))
        print(f"  [ML {t.date()}] valid IC {vic:.4f} | OOS IC {oos_ic:.4f} | "
              f"best_iter {m.best_iteration} | train {len(X_tr)} | w={best_w}",
              flush=True)
    return preds, ws_chosen, ic_log


def main():
    cal = data.load_calendar(config.BT_START, config.BT_END)
    panels = data.load_panels(config.BT_START, config.BT_END)
    fin = data.load_financials()
    mvp = load_mv_panel()
    signals = data.month_end_signals(cal, config.BT_START, config.BT_END)

    cands, sigs = {}, []
    for t, e in signals:
        cand = ss.candidates_at(panels, fin, t)
        cands[t] = cand
        s = score_mv(mvp, fin, t, cand).dropna()
        sigs.append((t, e, s.sort_values(ascending=False)))

    print("\n[1] 构建特征 (PIT, 月频友好)...", flush=True)
    feat = build_features(panels, mvp, fin, signals, cands)
    n0 = len(feat[signals[0][0]][0]) if signals[0][0] in feat else 0
    print(f"    特征月数 {len(feat)} | 首月样本 {n0} | "
          f"特征数 {feat[signals[12][0]][0].shape[1]}", flush=True)

    print("\n[2] 滚动OOS训练 (LightGBM, RankIC早停)...", flush=True)
    cache = os.path.join(config.OUT_DIR, "g_ml_preds.pkl")
    if os.path.exists(cache):
        import pickle
        with open(cache, "rb") as f:
            preds, ws, ic_log = pickle.load(f)
        print(f"    [cache] 读入 {len(preds)} 个月预测", flush=True)
    else:
        preds, ws, ic_log = rolling_oos(feat, signals)
        import pickle
        with open(cache, "wb") as f:
            pickle.dump((preds, ws, ic_log), f)
    vics = [v for _, v, _, _, _ in ic_log if v == v]
    oics = [o for _, _, o, _, _ in ic_log if o == o]
    print(f"\n[CHECK] OOS月数 {len(oics)} | valid IC均值 {np.mean(vics):.4f} | "
          f"OOS IC均值 {np.mean(oics):.4f} | OOS IC>0占比 "
          f"{np.mean([o > 0 for o in oics]):.0%} | w选择分布 "
          f"{pd.Series(ws).value_counts().to_dict()}", flush=True)

    # ---- 回测变体 (统一区间: 第TRAIN_MIN+1执行月起) ----
    ml_sigs = [(t, e, preds[t].sort_values(ascending=False))
               for t, e, _ in sigs if t in preds]
    G0 = buffered(sigs, ENTRY, EXIT, TOPK)
    M1 = buffered(ml_sigs, ENTRY, EXIT, TOPK)
    # M2: 融合分数 (w按valid逐月)
    w_map = dict(zip([t for t, e, _ in sigs if t in preds], ws))
    fused_sigs = []
    for t, e, s in sigs:
        if t not in preds:
            continue
        w = w_map[t]
        f = (1 - w) * s.rank(pct=True) + w * preds[t].rank(pct=True)
        fused_sigs.append((t, e, f.sort_values(ascending=False)))
    M2 = buffered(fused_sigs, ENTRY, EXIT, TOPK)
    # M3: 双腿 70%G0 + 30%M1 (合并目标权重; 统一区间从M1首月起)
    e0 = M1[0][0] if M1 else None
    g_map = dict(G0)
    m_map = dict(M1)
    M3 = []
    for e in sorted(m_map):          # 只遍历M1有信号的执行日 (统一区间)
        wg = g_map.get(e, {})
        wm = m_map.get(e, {})
        both = {}
        for s_, x in wg.items():
            both[s_] = both.get(s_, 0.0) + 0.7 * x
        for s_, x in wm.items():
            both[s_] = both.get(s_, 0.0) + 0.3 * x
        M3.append((e, both))

    # M4 排雷 (博文第八节: 模型只做排雷不驱动主收益):
    #   G规则选 Top40+缓冲 → 模型剔除预测最差10只 → 剖30只等权
    e2t = {e: t for t, e, _ in sigs}
    G0h = buffered(sigs, ENTRY, EXIT, TOPK + 10)
    M4 = []
    for e, w in G0h:
        t = e2t.get(e)
        if t in preds and len(w) > TOPK:
            p = preds[t].reindex(list(w)).dropna()
            n_drop = min(len(w) - TOPK, 10)
            bad = p.nsmallest(n_drop).index
            keep = {s_: x for s_, x in w.items() if s_ not in set(bad)}
            keep = {s_: 1.0 / len(keep) for s_ in keep} if keep else {}
            M4.append((e, keep))
        else:
            M4.append((e, w))
    M4 = [(e, w) for e, w in M4 if e >= e0]

    # 统一区间: M1首个执行日起
    G0c = [(e, w) for e, w in G0 if e >= e0]
    variants = {"G0 规则对照": G0c, "M1 纯模型": M1, "M2 打分融合": M2,
                "M3 双腿70/30": M3, "M4 模型排雷": M4}
    print(f"\n{'='*100}\n  G×ML 叠加实验 (区间 {e0.date()} ~ "
          f"{config.BT_END}, 成本后, Top{TOPK}+缓冲{ENTRY}-{EXIT})\n{'='*100}",
          flush=True)
    rows, rets = [], {}
    for name, rebals in variants.items():
        r, to, buys, fz = run_portfolio(panels["adj_close"], rebals,
                                        config.FEE_RT_STOCK,
                                        panels["last_date"])
        rets[name] = r
        r2, *_ = run_portfolio(panels["adj_close"], rebals,
                               config.FEE_RT_STOCK * config.COST_STRESS,
                               panels["last_date"])
        m, m2 = calc_metrics(r), calc_metrics(r2)
        src = {"G0 规则对照": {t: s for t, e, s in sigs},
               "M1 纯模型": {t: s for t, e, s in ml_sigs},
               "M2 打分融合": {t: s for t, e, s in fused_sigs},
               "M4 模型排雷": {t: s for t, e, s in sigs},
               "M3 双腿70/30": None}[name]
        ic = ric = float("nan")
        if src is not None:
            ic, ric, _ = rank_ic(panels["adj_close"], src)
        rows.append({"策略": name, **m, "IC": ic, "RankIC": ric,
                     "换手率": to, "交易次数": buys,
                     "成本x2": m2["年化收益"]})
        yr = yearly_returns(r)
        print(f"{name:<14}年化 {m['年化收益']:7.2%} 波动 {m['年化波动']:7.2%} "
              f"Sharpe {m['Sharpe']:5.2f} 回撤 {m['最大回撤']:7.1%} "
              f"Calmar {m['Calmar']:5.2f} 换手 {to:6.1%} 成本x2 "
              f"{m2['年化收益']:7.2%} RankIC {ric:.4f}", flush=True)
        print("    分年: " + "  ".join(f"{y}:{v:+.0%}" for y, v in yr.items()),
              flush=True)

    print_metric_table(
        [{k: v for k, v in r.items()
          if k in ["策略", "年化收益", "年化波动", "Sharpe", "最大回撤",
                   "Calmar", "IC", "RankIC", "换手率", "交易次数"]}
         for r in rows],
        title=f"图1口径 ({e0.date()}~{config.BT_END})")

    # 相关性
    mret = pd.concat({n: (1 + r).resample("ME").prod() - 1
                      for n, r in rets.items()}, axis=1).dropna()
    print("\n  月度收益相关:\n" + mret.corr().round(3).to_string(), flush=True)

    # 特征重要性 (最后一次训练的代理: 用全部样本重训一次仅作展示)
    months = [t for t, e in signals if t in feat]
    X_all = pd.concat([feat[m][0] for m in months], ignore_index=True)
    y_all = pd.concat([feat[m][1] for m in months], ignore_index=True)
    ok = y_all.notna().values
    tr = lgbm.Dataset(X_all[ok].values, label=y_all[ok].values,
                      feature_name=list(X_all.columns))
    m = lgbm.train({**LGB_PARAMS, "num_boost_round": 300}, tr)
    imp = pd.Series(m.feature_importance("gain"),
                    index=X_all.columns).sort_values(ascending=False)
    print("\n  特征重要性 (gain, 全样本展示用, 非OOS):\n"
          + (imp / imp.sum() * 100).round(1).to_string(), flush=True)


if __name__ == "__main__":
    main()
