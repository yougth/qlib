"""
V18 月度纯估值(value_comp_M20) 核查: 鲜度/穿越/涨停过滤/大年依赖/2026诊断
============================================================================
Part1 鲜度   : 72个月度信号日, 估值缓存最后可用日 vs 信号日 gap 分布 (防前视/陈旧)
Part2 过滤   : M20实际持仓在执行日的 涨停买入/停牌/流动性 实测 (防"买不进"虚增收益)
Part3 素超额 : 等权/无成本/无漂移 的素月度Top20 vs 素季度Top20 vs 池子等权, 逐年对比
              → 月度增益是结构性的, 还是只集中在2023-25价值大年
Part4 因子   : value_comp 月度RankIC(vs未来1个月超额) 按年时序 + 2026逐月诊断
"""
import os, sys, warnings, logging
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
from v14_ablation import build_liquidity_table, quarter_signal_exec_dates, DATA_DIR
from v16_value_rules import load_raw_valuation, LIQ_THRESHOLD
from v17_residual_model import load_price_volume, compute_value_comp
from v18_full_stack import monthly_signal_exec_dates

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
TOPK = 20


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_path = f"{DATA_DIR}/fcf_cache_pit.csv"
    prof_path = f"{DATA_DIR}/profit_cache_pit.csv"
    fcf_df = pd.read_csv(fcf_path, sep='\t'); fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df = pd.read_csv(prof_path, sep='\t'); profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)
    val_piv = load_raw_valuation()

    gap_stats = []          # (sig, 字段, gap天数, 有效数)
    filt_stats = []         # (exec, 涨停被剔数, 停牌被剔数, 流动性被剔数, 持仓当日涨幅>9.7%数)
    naive = {}              # naive[(freq, y)] = (策略年收益, 池子年收益)
    ic_rows = []            # (y, m, rank_ic)
    m2026 = []              # 2026逐月: (月, top20收益, 池子收益)

    for y in YEARS:
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = sorted(format_qlib_code(c) for c in codes)
        bt_end = pd.Timestamp(f"{y}-07-31" if y == 2026 else f"{y}-12-31")
        sig_exec_m = monthly_signal_exec_dates(y, cal_list)
        sig_exec_q = quarter_signal_exec_dates(y, cal_list)
        ext_start = sig_exec_m[0][0] - pd.Timedelta(days=60)
        close, vol = load_price_volume(universe, ext_start, min(bt_end + pd.Timedelta(days=20), pd.Timestamp("2026-07-31")))
        liq_table = build_liquidity_table(universe, sig_exec_m[0][0] - pd.Timedelta(days=10), str(bt_end.date()))
        lu, sus = build_limit_up_set(universe, cal)
        print(f"  [{y}] 池{len(universe)} 月度{len(sig_exec_m)}点 数据就绪", flush=True)

        def tradable_of(sig, exec_dt, idx):
            liq = (liq_table.xs(sig, level=0).reindex(idx)
                   if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=idx))
            m_liq = liq.notna() & (liq >= LIQ_THRESHOLD)
            m_lu = ~pd.Series(idx, index=idx).map(lambda i: (exec_dt, i) in lu)
            m_su = ~pd.Series(idx, index=idx).map(lambda i: (exec_dt, i) in sus)
            return idx[m_liq & m_lu & m_su], int((~m_liq).sum()), int((~m_lu).sum()), int((~m_su).sum())

        def naive_run(sig_exec, freq):
            """素回测: 每期Top20等权持有到下一期执行日(无成本无漂移), 池子等权对照"""
            ret_s, ret_p = 1.0, 1.0
            for i, (sig, exec_dt) in enumerate(sig_exec):
                nxt = sig_exec[i + 1][1] if i + 1 < len(sig_exec) else None
                end_d = nxt if nxt is not None else max(d for d in close.index if d <= bt_end)
                if exec_dt not in close.index or end_d not in close.index or end_d <= exec_dt:
                    continue
                vc = compute_value_comp(universe, sig, val_piv)
                trd, n_liq, n_lu, n_su = tradable_of(sig, exec_dt, vc.dropna().index)
                hold = vc.reindex(trd).nlargest(TOPK).index
                r = (close.loc[end_d] / close.loc[exec_dt] - 1)
                r_hold = r.reindex(hold).dropna().mean()
                r_pool = r.reindex(trd).dropna().mean()
                if pd.notna(r_hold):
                    ret_s *= (1 + r_hold)
                if pd.notna(r_pool):
                    ret_p *= (1 + r_pool)
                if freq == "M":
                    # Part1 鲜度
                    for c in ["ep", "bp", "cfp", "sp"]:
                        sub = val_piv[c].loc[:sig]
                        if len(sub):
                            gap_stats.append(dict(sig=sig, col=c, gap=(sig - sub.index[-1]).days,
                                                  n=int(sub.iloc[-1].notna().sum())))
                    # Part2 过滤实测 + 执行日涨幅
                    prev_d = [d for d in close.index if d < exec_dt]
                    chg = (close.loc[exec_dt] / close.loc[prev_d[-1]] - 1).reindex(hold) if prev_d else pd.Series(dtype=float)
                    filt_stats.append(dict(exec=exec_dt, n_liq=n_liq, n_lu=n_lu, n_su=n_su,
                                           hi_chg=int((chg > 0.097).sum())))
                    # Part4 因子IC
                    m_ic = pd.concat([vc.reindex(trd).rename("v"),
                                      (r - r.reindex(trd).mean()).reindex(trd).rename("f")], axis=1).dropna()
                    if len(m_ic) >= 10:
                        ic_rows.append(dict(y=y, m=exec_dt.month,
                                            ric=m_ic["v"].rank().corr(m_ic["f"].rank())))
                    if y == 2026:
                        m2026.append(dict(m=exec_dt.month, top=r_hold, pool=r_pool))
            return ret_s - 1, ret_p - 1

        rs_m, rp_m = naive_run(sig_exec_m, "M")
        rs_q, rp_q = naive_run(sig_exec_q, "Q")
        naive[("M", y)] = (rs_m, rp_m); naive[("Q", y)] = (rs_q, rp_q)

    # ================= 报告 =================
    g = pd.DataFrame(gap_stats)
    print(f"\n{'='*90}\nPart1 鲜度: 月度信号日估值gap  max={g['gap'].max()}天  均值={g['gap'].mean():.2f}天  "
          f"gap>5天次数={int((g['gap']>5).sum())}/{len(g)}  截面有效数 min={g['n'].min()}", flush=True)

    f = pd.DataFrame(filt_stats)
    print(f"\nPart2 过滤实测(72个月度执行日): 流动性剔除均值{f['n_liq'].mean():.1f}只/期  "
          f"涨停剔除均值{f['n_lu'].mean():.2f}  停牌剔除均值{f['n_su'].mean():.2f}", flush=True)
    print(f"      持仓执行日当日涨幅>9.7%(疑似追板)总计 {int(f['hi_chg'].sum())} 股次 / {len(f)*TOPK} 股次", flush=True)

    print(f"\nPart3 素超额逐年 (等权/无成本/无漂移):", flush=True)
    print(f"{'年':>6} | {'素月Top20':>9} {'素季Top20':>9} {'月-季差':>8} | {'池子等权':>9} {'月超额':>8} {'季超额':>8}", flush=True)
    for y in YEARS:
        rm, pm = naive[("M", y)]; rq, pq = naive[("Q", y)]
        print(f"{y:>6} | {rm*100:>8.2f}% {rq*100:>8.2f}% {(rm-rq)*100:>7.2f}pp | "
              f"{pm*100:>8.2f}% {(rm-pm)*100:>7.2f}pp {(rq-pq)*100:>7.2f}pp", flush=True)

    icd = pd.DataFrame(ic_rows)
    print(f"\nPart4 value_comp月度RankIC按年:", flush=True)
    for y in YEARS:
        sub = icd[icd["y"] == y]
        if len(sub):
            print(f"  {y}: 均值{sub['ric'].mean():>7.3f}  中位{sub['ric'].median():>7.3f}  "
                  f">0比例{(sub['ric']>0).mean():.2f}  n={len(sub)}", flush=True)
    print(f"  全期: 均值{icd['ric'].mean():.3f}  >0比例{(icd['ric']>0).mean():.2f}", flush=True)

    print(f"\n2026逐月诊断 (素Top20收益 vs 池子等权):", flush=True)
    for r in m2026:
        print(f"  {r['m']}月: Top20 {r['top']*100:>6.2f}%  池子 {r['pool']*100:>6.2f}%  超额 {(r['top']-r['pool'])*100:>6.2f}pp", flush=True)

    pd.DataFrame(ic_rows).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v18_verify_ic.csv", sep='\t', index=False)
    print("\n[+] 核查完成, IC明细已存 v18_verify_ic.csv", flush=True)


if __name__ == "__main__":
    main()
