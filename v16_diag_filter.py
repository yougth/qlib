"""诊断: 为什么2021-2023不同因子Top20收益完全相同 —— 过滤后剩余候选数检查"""
import sys, warnings, logging
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

def main():
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    logging.getLogger('qlib.data.data').setLevel(logging.ERROR)
    from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
    from v14_ablation import build_liquidity_table, quarter_signal_exec_dates, LIQ_THRESHOLD
    from v5_validation import build_limit_up_set

    fcf = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    prof = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    prof["code"] = prof["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)

    for y in [2021, 2022, 2023]:
        codes = build_dynamic_universe(y, fcf, prof)
        uni = sorted(format_qlib_code(c) for c in codes)
        se = quarter_signal_exec_dates(y, cal_list)
        lu, sus = build_limit_up_set(uni, cal)
        for qi, (sig, exec_dt) in enumerate(se):
            liq = build_liquidity_table(uni, sig - pd.Timedelta(days=10), sig + pd.Timedelta(days=5))
            day_liq = liq.xs(sig, level=0).reindex(uni)
            n_liq_ok = int(((day_liq.notna()) & (day_liq >= LIQ_THRESHOLD)).sum())
            n_final = sum(1 for i in uni
                          if pd.notna(day_liq.get(i, np.nan)) and day_liq[i] >= LIQ_THRESHOLD
                          and (exec_dt, i) not in sus and (exec_dt, i) not in lu)
            print(f"{y}Q{qi+1}: 池{len(uni)} | liq≥500万: {n_liq_ok} | 过滤后: {n_final} "
                  f"| liq中位数={day_liq.median():,.0f} P25={day_liq.quantile(0.25):,.0f}", flush=True)

if __name__ == "__main__":
    main()
