"""
V16 最终上线名单: 不含金融版 value_comp Top20(主)/Top10(备)  2025/2026 各季选股 + 换手率
"""
import os, sys, warnings, logging
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import numpy as np, pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
warnings.filterwarnings("ignore"); logging.getLogger('qlib.data.data').setLevel(logging.ERROR)
from v5_validation import build_limit_up_set
from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
from v14_ablation import (build_liquidity_table, load_price_matrix, quarter_signal_exec_dates,
                          load_finind_table, WORK_DIR, DATA_DIR)
from v16_value_rules import load_raw_valuation, compute_factors, LIQ_THRESHOLD
from v16_fin_compare import FIN_SET


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t'); fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    prof = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t'); prof["code"] = prof["code"].astype(str).str.zfill(6)
    name_map = {format_qlib_code(c): n for c, n in fcf.drop_duplicates("code")[["code", "name"]].values}
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31"); cal_list = list(cal)
    val_piv = load_raw_valuation(); fin = load_finind_table()

    rows, to_rows = [], []
    prev = {10: None, 20: None}
    for y in [2024, 2025, 2026]:   # 2024Q4->2025Q1换手需2024
        codes = build_dynamic_universe(y, fcf, prof)
        universe = sorted(u for u in (format_qlib_code(c) for c in codes) if u not in FIN_SET)
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = quarter_signal_exec_dates(y, cal_list)
        price_ext = load_price_matrix(universe, sig_exec[0][0] - pd.Timedelta(days=400),
                                      min(pd.Timestamp(bt_end) + pd.Timedelta(days=20), pd.Timestamp("2026-07-31")))
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        lu, sus = build_limit_up_set(universe, cal)
        for qi, (sig, exec_dt) in enumerate(sig_exec):
            f = compute_factors(universe, sig, val_piv, fin, price_ext, cal_list)
            liq = liq_table.xs(sig, level=0).reindex(f.index) if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=f.index)
            tradable = f.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                               & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in lu))
                               & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in sus))]
            s = f.loc[tradable, "value_comp"].dropna()
            for k in (20, 10):
                top = s.nlargest(k)
                cur = top.index.tolist()
                if prev[k] is not None:
                    n_out = len(set(prev[k]) - set(cur))
                    to_rows.append({"topk": k, "period": f"{y}Q{qi+1}", "exec_date": exec_dt.date(),
                                    "turnover": round(n_out / k, 4), "n_replaced": n_out})
                prev[k] = cur
                if y in (2025, 2026):
                    for rk, (inst, sc) in enumerate(top.items(), 1):
                        rows.append({"topk": k, "period": f"{y}Q{qi+1}", "exec_date": exec_dt.date(),
                                     "rank": rk, "instrument": inst, "name": name_map.get(inst, "?"),
                                     "value_comp": round(float(sc), 4)})
    picks = pd.DataFrame(rows); to = pd.DataFrame(to_rows)
    picks.to_csv(f"{WORK_DIR}/v16_launch_nofin_picks.csv", sep='\t', index=False)
    to.to_csv(f"{WORK_DIR}/v16_launch_nofin_turnover.csv", sep='\t', index=False)

    print("\n" + "=" * 70 + "\n  不含金融版 换手率 (季度, 出库数/K)\n" + "=" * 70, flush=True)
    for k in (20, 10):
        sub = to[(to["topk"] == k) & (to["period"].str.startswith(("2025", "2026")))]
        avg = sub["turnover"].mean()
        print(f"\n  ── Top{k} (均值 {avg*100:.1f}%) ──", flush=True)
        for _, r in sub.iterrows():
            print(f"    {r['period']} ({r['exec_date']}): 换手 {r['turnover']*100:5.1f}%  ({r['n_replaced']}/{k}只)", flush=True)

    print("\n" + "=" * 70 + "\n  Top20 各季选股名单 (不含金融)\n" + "=" * 70, flush=True)
    for pid in sorted(picks[picks["topk"] == 20]["period"].unique()):
        sub = picks[(picks["topk"] == 20) & (picks["period"] == pid)].sort_values("rank")
        names = "、".join(f"{r['name']}" for _, r in sub.iterrows())
        print(f"\n  【{pid}】{sub.iloc[0]['exec_date']} 建仓20只:\n    {names}", flush=True)
    print("\n[+] 已保存 v16_launch_nofin_picks.csv / v16_launch_nofin_turnover.csv", flush=True)


if __name__ == "__main__":
    main()
