#!/usr/bin/env python3
"""
run_factor_screen —— value_comp 单因子体检 (五关筛选的②③④⑤关, 独立分析脚本)
================================================================================
目的: config.VALUE_FACTORS (ep/bp/cfp/sp) 各自单独检验, 把 value_comp 从
"拍脑袋等权"升级为"证据驱动"。只输出证据, 不动生产代码。

纪律红线 (防过拟合):
  - 筛选统计只在各窗口 train 段做 (W2020~W2026 的 2015~2021);
  - test 段 (2019-12~2026-06) 只作一次性 OOS 验证, 不参与任何筛选决策;
  - 同一月份出现在多个窗口 train 段属正常 (池不同 = 跨池稳健性检验)。

输出 per 因子 (train 段为主, test 段并列对照):
  ② IC均值 / ICIR / t值(门槛≥3) / IC>0月份占比 / 分年IC符号一致性
  ③ Q1~Q5 分层月均收益 (因子值高=便宜=Q5, 看单调性与 Q5-Q1 价差)
  ④ 两两 rank 相关矩阵 (去冗余: >0.7 的因子对留一个)
  ⑤ 方向: IC 稳定为负且说得通 → 翻转候选; 忽正忽负 → 剔除候选

用法:
  cd quant && PYTHONPATH=. python3 run_factor_screen.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from core import config
from core import data as datalayer
from core import strategy
from core.universe import (build_windows, build_dynamic_universe,
                            format_qlib_code, load_pit_caches)
from core.valuation import load_valuation

N_BUCKETS = 5      # 分层数
MIN_CROSS = 30     # 截面最小样本 (低于此数不算一个月的观测)


def factor_cross(val_piv, factors, universe, dt):
    """信号日因子截面: 各因子取 <=dt 最新值 (与 value_comp_score 同口径, 不打港股系数)"""
    fac = {}
    for c in factors:
        pv = val_piv[c].loc[:dt]
        fac[c] = (pv.iloc[-1] if len(pv) else pd.Series(dtype=float)).reindex(universe)
    return pd.DataFrame(fac)


def screen_segment(seg, s, e, win, universe, cal, val_piv, factors,
                   ic_rows, bucket_rows, corr_rows):
    fwd = datalayer.forward_return_matrix(universe, s, e)
    days = [d for d in strategy.signal_days(cal, s, e) if d in fwd.index]
    n_ok = 0
    for dt in days:
        fac = factor_cross(val_piv, factors, universe, dt)
        fr = fwd.loc[dt].dropna()
        common = fac.index.intersection(fr.index)
        if len(common) < MIN_CROSS:
            continue
        n_ok += 1
        sub, fwd_sub = fac.loc[common], fr.loc[common]
        for c in factors:
            v = sub[c].dropna()
            if len(v) < MIN_CROSS:
                continue
            fwd_v = fwd_sub.loc[v.index]
            # ② 单因子 RankIC (因子值 vs 下月前瞻收益)
            ic = v.corr(fwd_v, method="spearman")
            ic_rows.append({"seg": seg, "window": win["name"], "sig": dt,
                            "year": dt.year, "factor": c, "ic": ic, "n": len(v)})
            # ③ 分层: Q1=因子值低(贵) ... Q5=因子值高(便宜), 记各组月均前瞻收益
            grp = np.ceil(v.rank(pct=True) * N_BUCKETS).clip(1, N_BUCKETS).astype(int)
            for g in range(1, N_BUCKETS + 1):
                m = fwd_v[grp == g]
                if len(m):
                    bucket_rows.append({"seg": seg, "factor": c, "bucket": g,
                                        "ret": float(m.mean())})
        # ④ 两两截面 spearman (因子值间的冗余度)
        for i, c1 in enumerate(factors):
            for c2 in factors[i + 1:]:
                pair = sub[[c1, c2]].dropna()
                if len(pair) >= MIN_CROSS:
                    corr_rows.append({"seg": seg, "window": win["name"], "f1": c1,
                                      "f2": c2,
                                      "corr": pair[c1].corr(pair[c2],
                                                            method="spearman")})
    print(f"[{win['name']}] {seg}: {n_ok}/{len(days)} 个月有效截面 "
          f"(池 {len(universe)} 只)", flush=True)


def summarize(seg, label, factors, ic_rows, bucket_rows, corr_rows):
    ic = pd.DataFrame([r for r in ic_rows if r["seg"] == seg])
    if not len(ic):
        return
    print(f"\n{'='*96}\n  {label}\n{'='*96}", flush=True)
    print("  ② 单因子检验 (池内月度截面 RankIC; t值门槛≥3, 不是2)", flush=True)
    print(f"  {'因子':<5} {'IC均值':>8} {'ICIR':>7} {'t值':>7} {'IC>0月占比':>10} "
          f"{'分年正IC占比':>12} {'分年IC':>34}", flush=True)
    yearly_stat = {}
    for c in factors:
        g = ic[ic["factor"] == c]
        mean, std = g["ic"].mean(), g["ic"].std(ddof=1)
        icir = mean / std if std else float("nan")
        t = mean / std * np.sqrt(len(g)) if std else float("nan")
        yr = g.groupby("year")["ic"].mean()
        pos_yr = (yr > 0).sum() / len(yr) if len(yr) else float("nan")
        yearly_stat[c] = yr
        yr_str = " ".join(f"{y}:{v:+.3f}" for y, v in yr.items())
        print(f"  {c:<5} {mean:+8.4f} {icir:+7.2f} {t:+7.2f} "
              f"{(g['ic'] > 0).mean():>10.1%} {pos_yr:>12.0%} {yr_str:>34}", flush=True)
        # ⑤ 处置建议 (最终裁决留给人: 结合经济学解释)
        if t >= 3:
            hint = "→ 保留"
        elif t <= -3:
            hint = "→ IC稳定为负: 有经济学理由则翻转, 无则剔除"
        else:
            hint = "→ 忽正忽负(噪声): 剔除候选"
        print(f"        {hint}", flush=True)
    # ③ 分层单调性
    bk = pd.DataFrame([r for r in bucket_rows if r["seg"] == seg])
    if len(bk):
        print("\n  ③ 分层单调性 (Q5=最便宜20% ... Q1=最贵20%; 月均收益×12≈年化)", flush=True)
        piv = bk.pivot_table(index="factor", columns="bucket", values="ret",
                             aggfunc="mean") * 12
        piv.columns = [f"Q{c}" for c in piv.columns]
        piv["Q5-Q1价差"] = piv["Q5"] - piv["Q1"]
        piv["单调(Q5>Q4>..>Q1)"] = piv.iloc[:, :N_BUCKETS].apply(
            lambda r: "是" if (r["Q5"] > r["Q4"] > r["Q3"] > r["Q2"] > r["Q1"])
            else ("近似" if r["Q5"] > r["Q1"] else "否"), axis=1)
        print(piv.round(4).to_string(), flush=True)
    # ④ 两两相关
    cr = pd.DataFrame([r for r in corr_rows if r["seg"] == seg])
    if len(cr):
        print("\n  ④ 两两截面 spearman 均值 (去冗余: >0.7 的因子对实际是同一注, 留一个)",
              flush=True)
        m = cr.pivot_table(index="f1", columns="f2", values="corr", aggfunc="mean")
        m = m.reindex(index=factors, columns=factors)
        for c in factors:
            m.loc[c, c] = 1.0
        print(m.round(3).to_string(), flush=True)
        hi = cr.groupby(["f1", "f2"])["corr"].mean()
        hi = hi[hi.abs() > 0.7]
        if len(hi):
            print("  [!] 高相关对: " + "; ".join(
                f"{a}~{b}={v:.2f}" for (a, b), v in hi.items()), flush=True)


def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    val_piv = load_valuation()
    factors = config.VALUE_FACTORS

    ic_rows, bucket_rows, corr_rows = [], [], []
    for win in build_windows():
        codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
        universe = [format_qlib_code(c) for c in codes]
        # 筛选只在 train 段做; test 段只验一次
        screen_segment("train", win["train"][0], win["train"][1], win, universe,
                       cal, val_piv, factors, ic_rows, bucket_rows, corr_rows)
        screen_segment("test", win["test"][0], win["test"][1], win, universe,
                       cal, val_piv, factors, ic_rows, bucket_rows, corr_rows)

    summarize("train", "训练窗体检 (筛选依据, 2015~2021 各窗口池)", factors,
              ic_rows, bucket_rows, corr_rows)
    summarize("test", "OOS 一次性验证 (2019-12~2026-06, 只看不筛选)", factors,
              ic_rows, bucket_rows, corr_rows)

    out = f"{config.OUT_DIR}/factor_screen_ic.csv"
    pd.DataFrame(ic_rows).to_csv(out, sep="\t", index=False)
    print(f"\n[+] 逐月 IC 明细已保存: {out}", flush=True)
    print("[提示] 三类处置: IC稳定为正→保留 | 忽正忽负→剔除 | 稳定为负且有经济学"
          "理由→翻转(如A股反转因子)。bp 的先验嫌疑: 高杠杆行业天然低PB;", flush=True)
    print("       sp 的先验嫌疑: 质量池内好公司 PS 天然高, 区分度存疑。", flush=True)
    print("[纪律] 筛后 value_comp 定义变了 → 旧 VAL10 基线不动, 新版本作为新策略"
          "并存对照, 标注口径切换点。", flush=True)


if __name__ == "__main__":
    main()
