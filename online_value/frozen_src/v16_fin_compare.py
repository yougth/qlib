"""
V16 上线择优: value_comp Top10/Top20  含金融 vs 不含金融  完整回测对比
=====================================================================
数据已补齐(2025/2026池100%行情覆盖)。两版唯一差异 = 建池是否剔除8只金融股。
金融剔除在universe层完成 → 不参与value_comp截面rank、不占TopK名额。
完全复用 v16_value_rules 的因子/过滤/引擎口径, 输出:
  - 全期(2019-2026)年化/夏普/回撤/IS/OOS + 逐年
  - 最优版的 2025/2026 各季Top名单 + 换手率
"""
import os, sys, random
import warnings, logging
def set_seed(s=42):
    random.seed(s); import numpy as np; np.random.seed(s)
set_seed(42)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
warnings.filterwarnings("ignore")
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set
from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
from v14_ablation import (build_liquidity_table, load_price_matrix, portfolio_backtest,
                          calc_metrics, quarter_signal_exec_dates, load_finind_table,
                          WORK_DIR, DATA_DIR)
from v16_value_rules import load_raw_valuation, compute_factors, LIQ_THRESHOLD
from v15_model_fixed import get_bench_returns_ak

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
FIN_SET = {"SH601318", "SH601336", "SH601398", "SH601601",
           "SH601665", "SH601838", "SH601939", "SH601988"}   # 8只金融(银行+保险)


def prep_year(y, fcf_df, profit_df, cal, cal_list, val_piv, fin, exclude_fin):
    codes = build_dynamic_universe(y, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    if exclude_fin:
        universe = [u for u in universe if u not in FIN_SET]
    bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
    sig_exec = quarter_signal_exec_dates(y, cal_list)
    ext_start = sig_exec[0][0] - pd.Timedelta(days=400)
    ext_end = min(pd.Timestamp(bt_end) + pd.Timedelta(days=20), pd.Timestamp("2026-07-31"))
    price_ext = load_price_matrix(universe, ext_start, ext_end)
    liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
    lu, sus = build_limit_up_set(universe, cal)
    fac_store = {}
    for qi, (sig, exec_dt) in enumerate(sig_exec):
        f = compute_factors(universe, sig, val_piv, fin, price_ext, cal_list)
        liq = liq_table.xs(sig, level=0).reindex(f.index) if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=f.index)
        tradable = f.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                           & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in lu))
                           & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in sus))]
        fac_store[qi] = (f.loc[tradable], sig, exec_dt)
    return universe, sig_exec, price_ext, fac_store


def run_variant(exclude_fin, ctx, cal_list, name_map, collect_picks=False):
    """返回 {strategy: (rets_series, annual_dict, avg_turnover, picks_rows)}"""
    out = {}
    strat = [("value_comp Top10", 10), ("value_comp Top20", 20)]
    agg = {s[0]: {"rets": [], "annual": {}, "to": []} for s in strat}
    agg["E0"] = {"rets": [], "annual": {}, "to": []}
    picks = []
    for y in YEARS:
        universe, sig_exec, price_ext, fac_store = ctx[y]
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        price_mat = price_ext.loc[sig_exec[0][0]:pd.Timestamp(bt_end)]
        # E0
        rb = []
        for qi, (sig, exec_dt) in enumerate(sig_exec):
            h = [i for i in universe if i in price_mat.columns and pd.notna(price_mat.loc[exec_dt, i])]
            rb.append((exec_dt, h))
        r, to = portfolio_backtest(rb, price_mat, cal_list)
        agg["E0"]["rets"].append(r); agg["E0"]["to"].append(to); agg["E0"]["annual"][y] = (1+r).prod()-1
        for name, k in strat:
            rb = []
            for qi, (sig, exec_dt) in enumerate(sig_exec):
                f, _, _ = fac_store[qi]
                s = f["value_comp"].dropna()
                top = s.nlargest(k)
                rb.append((exec_dt, top.index.tolist()))
                if collect_picks and k == 20 and y in (2025, 2026):
                    for rk, (inst, sc) in enumerate(top.items(), 1):
                        picks.append({"variant": "no_fin" if exclude_fin else "with_fin",
                                      "period": f"{y}Q{qi+1}", "exec_date": exec_dt.date(),
                                      "rank": rk, "instrument": inst,
                                      "name": name_map.get(inst, "?"), "value_comp": round(float(sc), 4)})
            r, to = portfolio_backtest(rb, price_mat, cal_list)
            agg[name]["rets"].append(r); agg[name]["to"].append(to); agg[name]["annual"][y] = (1+r).prod()-1
    for name in agg:
        s = pd.concat(agg[name]["rets"]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        agg[name]["series"] = s
    return agg, picks


def summarize(tag, agg, bench):
    bm = calc_metrics(bench)
    print(f"\n{'='*118}\n  【{tag}】  (T+1/往返0.4%/流动性500万/涨停停牌过滤)\n{'='*118}", flush=True)
    header = "  ".join(f"{y:>6}" for y in YEARS)
    print(f"{'策略':<18} | {header} | {'年化':>7} {'夏普':>5} {'回撤':>7} {'季换手':>6} {'IS':>7} {'OOS':>7}", flush=True)
    rows = []
    for name in ["E0", "value_comp Top10", "value_comp Top20"]:
        s = agg[name]["series"]
        m = calc_metrics(s); mi = calc_metrics(s[s.index.year.isin(IS_YEARS)]); mo = calc_metrics(s[s.index.year.isin(OOS_YEARS)])
        yr = "  ".join(f"{agg[name]['annual'][y]*100:>5.1f}%" for y in YEARS)
        print(f"{name:<18} | {yr} | {m['ar']*100:>6.2f}% {m['sharpe']:>5.2f} {m['max_dd']*100:>6.1f}% "
              f"{np.mean(agg[name]['to'])*100:>5.1f}% {mi['ar']*100:>6.2f}% {mo['ar']*100:>6.2f}%", flush=True)
        rows.append({"tag": tag, "strategy": name, "ar": m["ar"], "sharpe": m["sharpe"],
                     "max_dd": m["max_dd"], "is_ar": mi["ar"], "oos_ar": mo["ar"],
                     "turnover": np.mean(agg[name]["to"]),
                     **{str(y): agg[name]["annual"][y] for y in YEARS}})
    print(f"{'沪深300(ak)':<18} | {'':>55} | {bm['ar']*100:>6.2f}% {bm['sharpe']:>5.2f} {bm['max_dd']*100:>6.1f}%", flush=True)
    return rows


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t'); fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t'); profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    name_map = {format_qlib_code(c): n for c, n in fcf_df.drop_duplicates("code")[["code", "name"]].values}
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31"); cal_list = list(cal)
    val_piv = load_raw_valuation(); fin = load_finind_table()

    bench_all = []
    for y in YEARS:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns_ak(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index(); bench = bench[~bench.index.duplicated(keep="last")]

    all_rows = []; all_picks = []; variant_agg = {}
    for exclude_fin in [False, True]:
        tag = "不含金融" if exclude_fin else "含金融(完整池)"
        print(f"\n>>> 准备 {tag} ...", flush=True)
        ctx = {}
        for y in YEARS:
            ctx[y] = prep_year(y, fcf_df, profit_df, cal, cal_list, val_piv, fin, exclude_fin)
            print(f"    [{y}] 池{len(ctx[y][0])}只", flush=True)
        agg, picks = run_variant(exclude_fin, ctx, cal_list, name_map, collect_picks=True)
        variant_agg[exclude_fin] = agg
        all_rows += summarize(tag, agg, bench)
        all_picks += picks

    res = pd.DataFrame(all_rows)
    res.to_csv(f"{WORK_DIR}/v16_fin_compare_results.csv", sep='\t', index=False)
    pd.DataFrame(all_picks).to_csv(f"{WORK_DIR}/v16_fin_compare_picks.csv", sep='\t', index=False)
    # 保存最优版收益序列
    for exclude_fin in [False, True]:
        tg = "nofin" if exclude_fin else "withfin"
        for name in ["value_comp Top10", "value_comp Top20"]:
            variant_agg[exclude_fin][name]["series"].to_csv(
                f"{WORK_DIR}/v16_cmp_{tg}_{name.split()[-1]}.csv", sep='\t', header=False)
    print("\n[+] 已保存 v16_fin_compare_results.csv / v16_fin_compare_picks.csv 及收益序列", flush=True)


if __name__ == "__main__":
    main()
