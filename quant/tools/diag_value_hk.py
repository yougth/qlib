"""诊断: value_comp 加港股后收益为何下降? A股 vs 港股持仓收益分解
核心问题(用户质疑): 加入港股池收益"一定"提升, 但 value_comp 加港股后
年化从 8.73% 降到 7.65% (同一数据基础)。需要分解:
  1. value 选出的港股持仓在持有期收益 vs A股持仓
  2. 港股是被"更便宜"选中, 但持有期表现如何?
  3. 公平系数 VHF 是否让港股选得更好?
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import core.config as config
from core import data as datalayer
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches

def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()

    # 读持仓
    h = pd.read_csv(os.path.join(config.OUT_DIR, "rolling10y_holdings.csv"),
                    encoding="utf-8-sig", sep="\t")
    h["exec_date"] = pd.to_datetime(h["exec_date"])

    # 全市场收盘价 (用于计算持有期收益)
    all_insts = sorted({i for s in h["holdings"] for i in str(s).split(",")})
    px = datalayer.load_price_matrix(all_insts)
    px = px.sort_index()

    # 对每个持仓期: 计算执行日到下一执行日的收益
    # exec_date 买入, 下一个 exec_date 卖出
    results = []
    for tag, label in [("VAL20", "value_comp20(无系数)"), ("VHF", "VHF20(公平系数)"),
                       ("VHF10", "VHF10(公平系数)")]:
        sub = h[h["model"] == tag]
        if sub.empty:
            continue
        periods = []
        for i, row in sub.iterrows():
            ed = row["exec_date"]
            holds = [c for c in str(row["holdings"]).split(",") if c]
            # 找到下一个执行日
            nxt = sub[sub["exec_date"] > ed]["exec_date"]
            if nxt.empty:
                nxt_dt = px.index[-1]
            else:
                nxt_dt = nxt.min()
            # 该持仓期日收益 (等权, 取区间内可用)
            seg = px.loc[ed:nxt_dt, holds].dropna(how="all")
            if seg.empty or len(seg) < 2:
                continue
            # 等权组合日收益
            w = seg.notna()
            port = (seg.pct_change().fillna(0).clip(-0.5, 0.5) * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
            port = port.dropna()
            if port.empty:
                continue
            period_ret = (1 + port).prod() - 1
            n_days = len(port)
            # 年化(按交易日252)
            ann = (1 + period_ret) ** (252 / max(n_days, 1)) - 1 if period_ret > -1 else -1
            hk = [c for c in holds if c.startswith("hk")]
            a = [c for c in holds if not c.startswith("hk")]
            # 港股分组合计收益
            seg_hk = px.loc[ed:nxt_dt, hk].dropna(how="all") if hk else pd.DataFrame()
            if not seg_hk.empty and len(seg_hk) >= 2:
                w_hk = seg_hk.notna()
                p_hk = (seg_hk.pct_change().fillna(0).clip(-0.5, 0.5) * w_hk).sum(axis=1) / w_hk.sum(axis=1).replace(0, np.nan)
                p_hk = p_hk.dropna()
                ret_hk = (1 + p_hk).prod() - 1 if not p_hk.empty else np.nan
            else:
                ret_hk = np.nan
            seg_a = px.loc[ed:nxt_dt, a].dropna(how="all") if a else pd.DataFrame()
            if not seg_a.empty and len(seg_a) >= 2:
                w_a = seg_a.notna()
                p_a = (seg_a.pct_change().fillna(0).clip(-0.5, 0.5) * w_a).sum(axis=1) / w_a.sum(axis=1).replace(0, np.nan)
                p_a = p_a.dropna()
                ret_a = (1 + p_a).prod() - 1 if not p_a.empty else np.nan
            else:
                ret_a = np.nan
            periods.append({"model": tag, "exec_date": ed, "n_hold": len(holds),
                            "n_hk": len(hk), "period_ret": period_ret, "ann": ann,
                            "ret_hk": ret_hk, "ret_a": ret_a})
        df = pd.DataFrame(periods)
        n_hk_hold = int(df["n_hk"].sum())
        n_tot = int(df["n_hold"].sum())
        # 平均持有期年化 (按执行日等权)
        mean_ann = df["ann"].mean()
        geo_ann = (df["period_ret"] + 1).prod() ** (252 / df["n_days"].sum()) - 1 if "n_days" in df else np.nan
        # 港股 vs A股 平均持有期收益
        hk_ann = df["ret_hk"].dropna().apply(lambda r: (1 + r) ** (252 / 30) - 1).mean() if df["ret_hk"].notna().any() else np.nan
        a_ann = df["ret_a"].dropna().apply(lambda r: (1 + r) ** (252 / 30) - 1).mean() if df["ret_a"].notna().any() else np.nan
        # 更严谨: 用平均持有期天数
        avg_days = df["n_days"].mean() if "n_days" in df else 22
        hk_ann2 = df["ret_hk"].dropna().apply(lambda r: (1 + r) ** (252 / avg_days) - 1).mean() if df["ret_hk"].notna().any() else np.nan
        a_ann2 = df["ret_a"].dropna().apply(lambda r: (1 + r) ** (252 / avg_days) - 1).mean() if df["ret_a"].notna().any() else np.nan
        print(f"\n=== {label} ({tag}) ===")
        print(f"  持仓期: {len(df)}, 平均持仓数: {df['n_hold'].mean():.1f}, 港股占比: {n_hk_hold/max(n_tot,1)*100:.1f}%")
        print(f"  组合平均持有期年化(算术): {mean_ann*100:.2f}%")
        print(f"  港股持仓平均持有期年化: {hk_ann2*100:.2f}%  (样本 {int(df['ret_hk'].notna().sum())})")
        print(f"  A股持仓平均持有期年化: {a_ann2*100:.2f}%  (样本 {int(df['ret_a'].notna().sum())})")

if __name__ == "__main__":
    main()
