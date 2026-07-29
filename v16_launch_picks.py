"""
V16 上线准备: value_comp Top20/Top10 在 2025/2026 各季度的选股名单 + 换手率
==========================================================================
完全复用 v16_value_rules 的因子与过滤口径 (同一函数, 非复制代码):
  - compute_factors: 信号日截面因子 (估值 loc[:sig] + ROE PIT)
  - 流动性 >= 真实500万 (qlib口径5万) / 涨停 / 停牌 过滤
  - 换手率: 相邻调仓期持仓名单差异 (单边, 按等权近似 = 换出只数/K)
额外: 用 portfolio_backtest 复算 2025/2026 年度收益, 与 v16_run2.log 对账
基线: 2024Q4 持仓 (用于计算 2025Q1 的换手)
"""
import os, sys
import warnings, logging
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set
from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
from v14_ablation import (build_liquidity_table, load_price_matrix,
                          portfolio_backtest, quarter_signal_exec_dates,
                          load_finind_table, WORK_DIR, DATA_DIR)
from v16_value_rules import load_raw_valuation, compute_factors, LIQ_THRESHOLD


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    # 股票名映射 (fcf_cache 自带 name)
    name_map = {format_qlib_code(c): n for c, n in
                fcf_df.drop_duplicates("code")[["code", "name"]].values}
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)
    val_piv = load_raw_valuation()
    fin = load_finind_table()

    # (year, [季度索引列表]): 2024只取Q4做换手基线
    plan = [(2024, [3]), (2025, [0, 1, 2, 3]), (2026, [0, 1, 2])]
    holdings_seq = {10: [], 20: []}   # [(label, exec_dt, [instrument...])]
    pick_rows = []
    year_data = {}

    for y, q_idx in plan:
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = sorted(format_qlib_code(c) for c in codes)
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = quarter_signal_exec_dates(y, cal_list)
        ext_start = sig_exec[0][0] - pd.Timedelta(days=400)
        ext_end = min(pd.Timestamp(bt_end) + pd.Timedelta(days=20), pd.Timestamp("2026-07-31"))
        price_ext = load_price_matrix(universe, ext_start, ext_end)
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        limit_up_set, suspension_set = build_limit_up_set(universe, cal)
        year_data[y] = (universe, sig_exec, price_ext)
        print(f"[{y}] 池{len(universe)}只", flush=True)

        for qi in q_idx:
            sig, exec_dt = sig_exec[qi]
            f = compute_factors(universe, sig, val_piv, fin, price_ext, cal_list)
            liq = liq_table.xs(sig, level=0).reindex(f.index)
            tradable = f.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                               & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in limit_up_set))
                               & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in suspension_set))]
            fq = f.loc[tradable]
            s = fq["value_comp"].dropna()
            label = f"{y}Q{qi+1}"
            print(f"  {label}: 信号日{sig.date()} 执行日{exec_dt.date()} "
                  f"池{len(universe)} 过滤后候选{len(s)}", flush=True)
            for k in [10, 20]:
                top = s.nlargest(k)
                holdings_seq[k].append((label, exec_dt, top.index.tolist()))
                if k == 20 and y >= 2025:
                    for rank, (inst, score) in enumerate(top.items(), 1):
                        pick_rows.append({"period": label, "signal_date": sig.date(),
                                          "exec_date": exec_dt.date(), "rank": rank,
                                          "instrument": inst,
                                          "name": name_map.get(inst, "?"),
                                          "value_comp": round(float(score), 4)})

    # ---- 换手率 (相邻期名单差异, 单边) ----
    print(f"\n{'='*70}\n  换手率 (单边 = 换出只数/K, 含2024Q4→2025Q1衔接)\n{'='*70}", flush=True)
    to_rows = []
    for k in [20, 10]:
        seq = holdings_seq[k]
        print(f"  [Top{k}]", flush=True)
        for i in range(1, len(seq)):
            prev_l, _, prev_h = seq[i-1]
            cur_l, _, cur_h = seq[i]
            n_out = len(set(prev_h) - set(cur_h))
            to = n_out / k
            to_rows.append({"topk": k, "from": prev_l, "to": cur_l, "turnover_oneside": to,
                            "n_changed": n_out})
            print(f"    {prev_l}→{cur_l}: 换出{n_out}只 单边换手 {to:.0%}", flush=True)

    # ---- 对账: 2025/2026 年度收益复算 ----
    print(f"\n{'='*70}\n  对账: portfolio_backtest 复算 (应与 v16_run2.log 一致)\n{'='*70}", flush=True)
    for y in [2025, 2026]:
        universe, sig_exec, price_ext = year_data[y]
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        price_mat = price_ext.loc[sig_exec[0][0]:pd.Timestamp(bt_end)]
        for k in [10, 20]:
            rebs = [(ex, h) for lbl, ex, h in holdings_seq[k] if lbl.startswith(str(y))]
            rets, avg_to = portfolio_backtest(rebs, price_mat, cal_list)
            ann = (1 + rets).prod() - 1
            print(f"  {y} Top{k}: 年度收益 {ann*100:+.2f}%  引擎季均单边换手 {avg_to:.1%}", flush=True)

    pd.DataFrame(pick_rows).to_csv(f"{WORK_DIR}/v16_launch_picks_2025_2026.csv",
                                   sep='\t', index=False)
    pd.DataFrame(to_rows).to_csv(f"{WORK_DIR}/v16_launch_turnover.csv", sep='\t', index=False)
    print("\n[+] 已保存: v16_launch_picks_2025_2026.csv, v16_launch_turnover.csv", flush=True)


if __name__ == "__main__":
    main()
