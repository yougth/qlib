#!/usr/bin/env python3
"""experiments.g_ml_v2 —— 特征层增强实验 (低成本分步验证, M4-14排雷位统一回测)
================================================================================
基线: M4 净网44剔14留30, 年化23.76%/Sharpe0.98 (g_ml_preds, 15基特征+绝对收益标签)

变体:
  V2a  +回购事件特征  (近12个月公告回购: flag + 占总股本比例上限, PIT按公告日)
  V2b  标签改超额收益  (fwd - 当月末候选池截面中位数)
  V2c  a+b
附:  规则权重扫描 (score_mv 市值权重 0.5/0.6/0.8, 不重训, 复用现有preds)
缓存: outputs/g_ml_v2{a,b,c}_preds.pkl
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, data  # noqa: E402
from core.backtest import run_portfolio, calc_metrics  # noqa: E402
from strategies import stock_strategies as ss  # noqa: E402
from experiments.lowatt_opt import buffered  # noqa: E402
from experiments.lowatt_mv import load_mv_panel, score_mv  # noqa: E402
from experiments.g_ml import build_features, rolling_oos  # noqa: E402

TOPK, ENTRY, EXIT, KDROP = 30, 12, 25, 14


# ---------- 回购事件特征 ----------
def load_repurchase():
    fp = os.path.join(config.NEW_DIR, "data", "repurchase.csv")
    d = pd.read_csv(fp, dtype={"股票代码": str})
    d["code"] = d["股票代码"].str[-6:]
    first = d["code"].str[0]
    pfx = np.where(first.isin(["6", "9"]), "SH",
                   np.where(first.isin(["0", "3"]), "SZ", "BJ"))
    d["code"] = pfx + d["code"]            # 与面板统一: 前缀+大写
    d["ann"] = pd.to_datetime(d["最新公告日期"], errors="coerce")
    d["share_pct"] = pd.to_numeric(d["占公告前一日总股本比例-上限"],
                                   errors="coerce")
    d = d.dropna(subset=["ann"])
    return d


def rep_frames(rep, sigs, months_window=12):
    """每个信号日 → (rep_flag, rep_pct): 近12个月有无回购公告及最大占比"""
    by_code = {c: g.sort_values("ann") for c, g in rep.groupby("code")}
    out = {}
    for t, e in sigs:
        flag, pct = {}, {}
        lo = t - pd.DateOffset(months=months_window)
        for c, g in by_code.items():
            w = g[(g["ann"] <= t) & (g["ann"] > lo)]
            if len(w):
                flag[c] = 1.0
                pct[c] = float(w["share_pct"].max())
        out[t] = (pd.Series(flag, dtype=float),
                  pd.Series(pct, dtype=float))
    return out


# ---------- 特征构建 (基特征 + 可选回购 + 可选超额标签) ----------
def build_features_v2(panels, mvp, fin, sigs, cands, rep=None, label_exc=False):
    feat = build_features(panels, mvp, fin, sigs, cands)   # 基特征+绝对标签
    if not label_exc and rep is None:
        return feat
    px = panels["adj_close"].ffill()
    fwd = px.shift(-21) / px - 1
    out = {}
    for t, e in sigs:
        if t not in feat:
            continue
        X, y = feat[t]
        if rep is not None:
            rf, rp = rep.get(t, (pd.Series(dtype=float),
                                 pd.Series(dtype=float)))
            X = X.copy()
            X["rep_flag"] = rf.reindex(X.index).fillna(0.0)
            X["rep_pct"] = rp.reindex(X.index).fillna(0.0)
        if label_exc:
            y = y - y.median()          # 截面超额 (池内中位数)
        out[t] = (X, y)
    return out


# ---------- M4-14 排雷回测 ----------
def m4_rebals(sigs, preds):
    e2t = {e: t for t, e, _ in sigs}
    e0 = min(e for t, e, _ in sigs if t in preds)
    out = []
    for e, w in buffered(sigs, ENTRY, EXIT, TOPK + KDROP):
        if e < e0:
            continue
        t = e2t.get(e)
        if t in preds and len(w) > TOPK:
            p = preds[t].reindex(list(w)).dropna()
            nd = min(len(w) - TOPK, KDROP)
            bad = set(p.nsmallest(nd).index)
            out.append((e, {s: 1.0 / (len(w) - nd) for s in w
                            if s not in bad}))
        else:
            out.append((e, w))
    return out


def report_m4(panels, name, rebals):
    r, to, _, _ = run_portfolio(panels["adj_close"], rebals,
                                config.FEE_RT_STOCK, panels["last_date"])
    m = calc_metrics(r)
    print(f"{name:<24}年化 {m['年化收益']:7.2%} 波动 {m['年化波动']:7.2%} "
          f"Sharpe {m['Sharpe']:5.2f} 回撤 {m['最大回撤']:7.1%} "
          f"Calmar {m['Calmar']:5.2f} 换手 {to:6.1%}", flush=True)
    return m


# ---------- 规则权重扫描 (不重训) ----------
def score_mv_w(mvp, fin, t, cand, w):
    row = mvp.loc[:t].iloc[-1] if len(mvp.loc[:t]) else pd.Series(dtype=float)
    mv = row.reindex(cand)
    cc = data.cash_conversion(fin, t).reindex(cand)
    return w * (-mv).rank(pct=True) + (1 - w) * cc.rank(pct=True)


def weight_scan(panels, mvp, fin, signals, preds):
    print(f"\n{'='*100}\n  规则权重扫描 (M4-14, 复用现有preds, 仅改规则分)\n{'='*100}",
          flush=True)
    for w in [0.5, 0.6, 0.8]:
        sigs_w = []
        for t, e in signals:
            cand = ss.candidates_at(panels, fin, t)
            s = score_mv_w(mvp, fin, t, cand, w).dropna()
            sigs_w.append((t, e, s.sort_values(ascending=False)))
        report_m4(panels, f"w_mv={w} M4-14",
                  m4_rebals(sigs_w, preds))
        g0 = buffered(sigs_w, ENTRY, EXIT, TOPK)
        e0 = min(e for t, e, _ in sigs_w if t in preds)
        g0 = [(e, x) for e, x in g0 if e >= e0]
        report_m4(panels, f"w_mv={w} G0-Top30", g0)


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
        sigs.append((t, e, s.sort_values(ascending=False)))  # buffered按传入序排名, 必须降序

    with open(os.path.join(config.OUT_DIR, "g_ml_preds.pkl"), "rb") as f:
        base_preds, _, base_ic = pickle.load(f)

    # ---- 基线: 现有preds的M4-14 ----
    print(f"{'='*100}\n  特征层增强实验 (M4-14统一回测)\n{'='*100}", flush=True)
    base_oic = np.nanmean([o for _, _, o, _, _ in base_ic])
    print(f"基线: 15基特征+绝对标签 | OOS IC均值 {base_oic:.4f}", flush=True)
    report_m4(panels, "基线 M4-14", m4_rebals(sigs, base_preds))

    # ---- 规则权重扫描 (不重训, 先出结果) ----
    weight_scan(panels, mvp, fin, signals, base_preds)

    # ---- 回购特征 ----
    print("\n[1] 构建回购事件特征 (PIT: 公告日≤t, 近12个月)...", flush=True)
    rep = rep_frames(load_repurchase(), signals)
    cov = np.mean([rep[t][0].reindex(cands[t]).notna().mean()
                   for t, e in signals if t in cands and t in rep])
    print(f"    候选池平均回购覆盖率 {cov:.1%}", flush=True)

    variants = [
        ("V2a +回购特征(绝对标签)", dict(rep=rep, label_exc=False)),
        ("V2b 超额标签", dict(rep=None, label_exc=True)),
        ("V2c 回购+超额标签", dict(rep=rep, label_exc=True)),
    ]
    for name, kw in variants:
        tag = {"V2a +回购特征(绝对标签)": "a", "V2b 超额标签": "b",
               "V2c 回购+超额标签": "c"}[name]
        cache = os.path.join(config.OUT_DIR, f"g_ml_v2{tag}_preds.pkl")
        print(f"\n[{tag}] {name}: 构建特征...", flush=True)
        feat = build_features_v2(panels, mvp, fin, signals, cands, **kw)
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                preds, ws, ic_log = pickle.load(f)
            print(f"    [cache] {len(preds)} 个月", flush=True)
        else:
            preds, ws, ic_log = rolling_oos(feat, signals)
            with open(cache, "wb") as f:
                pickle.dump((preds, ws, ic_log), f)
        oics = [o for _, _, o, _, _ in ic_log if o == o]
        print(f"    OOS IC均值 {np.nanmean(oics):.4f} | >0占比 "
              f"{np.mean([o > 0 for o in oics]):.0%}", flush=True)
        report_m4(panels, name, m4_rebals(sigs, preds))

    print("\n[+] 完成", flush=True)


if __name__ == "__main__":
    main()
