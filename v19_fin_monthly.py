"""
V19: 月度 value_comp Top20  含金融 vs 剔金融  (回答"上线版是否该剔银行保险")
======================================================================
背景: v16_fin_compare 是季度口径, 结论矛盾(剔金融全期赢2.1pp但OOS输3.5pp,
      优势全来自2019 IS)。上线版已改月度+PIT池, 必须在月度口径下重判。
两版唯一差异 = universe层是否剔除8只金融股(剔除后不参与截面rank、不占TopK)。
其余口径完全等同 online_value/gen_holdings.py 与 v18_full_stack.py。
"""
import os, sys, random, warnings, logging
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
from v14_ablation import build_liquidity_table, portfolio_backtest, WORK_DIR, DATA_DIR
from v16_value_rules import load_raw_valuation, LIQ_THRESHOLD
from v16_fin_compare import FIN_SET
from v17_residual_model import load_price_volume, compute_value_comp
from v18_full_stack import monthly_signal_exec_dates, full_metrics, trade_count

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
TOPK = 20


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf = pd.read_csv(f"{DATA_DIR}/fcf_cache_pit.csv", sep='\t'); fcf["code"] = fcf["code"].astype(str).str.zfill(6)
    prof = pd.read_csv(f"{DATA_DIR}/profit_cache_pit.csv", sep='\t'); prof["code"] = prof["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31"); cal_list = list(cal)
    val_piv = load_raw_valuation()

    res = {v: dict(rets=[], annual={}, to=[], trades=0, ic=[], fin_slots=[]) for v in ["with_fin", "no_fin"]}
    for y in YEARS:
        codes = build_dynamic_universe(y, fcf, prof)
        uni_all = sorted(format_qlib_code(c) for c in codes)
        bt_end = pd.Timestamp(f"{y}-07-31" if y == 2026 else f"{y}-12-31")
        sig_exec = monthly_signal_exec_dates(y, cal_list)
        close, _ = load_price_volume(uni_all, sig_exec[0][0] - pd.Timedelta(days=40),
                                     min(bt_end + pd.Timedelta(days=20), pd.Timestamp("2026-07-31")))
        liq_table = build_liquidity_table(uni_all, sig_exec[0][0] - pd.Timedelta(days=10), str(bt_end.date()))
        lu, sus = build_limit_up_set(uni_all, cal)
        price_mat = close.loc[sig_exec[0][0]:bt_end]

        for variant in ["with_fin", "no_fin"]:
            universe = uni_all if variant == "with_fin" else [u for u in uni_all if u not in FIN_SET]
            rebal = []
            for sig, exec_dt in sig_exec:
                vc = compute_value_comp(universe, sig, val_piv).dropna()
                liq = (liq_table.xs(sig, level=0).reindex(vc.index)
                       if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=vc.index))
                trd = vc.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                               & (~pd.Series(vc.index, index=vc.index).map(lambda i: (exec_dt, i) in lu))
                               & (~pd.Series(vc.index, index=vc.index).map(lambda i: (exec_dt, i) in sus))]
                s = vc.reindex(trd)
                hold = s.nlargest(TOPK)
                rebal.append((exec_dt, hold.index.tolist()))
                res[variant]["fin_slots"].append(len(set(hold.index) & FIN_SET))
                # 因子IC: 打分 vs 持有到下期的超额
                nxt = [e for _, e in sig_exec if e > exec_dt]
                end_d = nxt[0] if nxt else (price_mat.index[-1] if len(price_mat) else None)
                if end_d is not None and exec_dt in close.index and end_d in close.index:
                    fwd = close.loc[end_d] / close.loc[exec_dt] - 1
                    fwd = (fwd - fwd.reindex(trd).mean()).reindex(trd)
                    pair = pd.concat([s.rename("s"), fwd.rename("f")], axis=1).dropna()
                    if len(pair) >= 10:
                        res[variant]["ic"].append((pair["s"].corr(pair["f"]),
                                                   pair["s"].rank().corr(pair["f"].rank())))
            rets, avg_to = portfolio_backtest(rebal, price_mat, cal_list)
            res[variant]["rets"].append(rets); res[variant]["to"].append(avg_to)
            res[variant]["annual"][y] = (1 + rets).prod() - 1
            res[variant]["trades"] += trade_count(rebal)
        print(f"  [{y}] 池{len(uni_all)}只(金融{len(set(uni_all)&FIN_SET)}只) 月度{len(sig_exec)}期 完成", flush=True)

    line = "=" * 100
    print(f"\n{line}\n  月度 value_comp Top20: 含金融 vs 剔金融 (PIT池 / T+1 / 成本0.4% / 流动性500万 / 涨停停牌过滤)\n{line}", flush=True)
    rows = []
    for v, tag in [("with_fin", "含金融(上线版)"), ("no_fin", "剔金融8只")]:
        d = res[v]
        s = pd.concat(d["rets"]).sort_index(); s = s[~s.index.duplicated(keep="last")]
        m = full_metrics(s)
        m_is = full_metrics(s[s.index.year.isin(IS_YEARS)])
        m_oos = full_metrics(s[s.index.year.isin(OOS_YEARS)])
        ic = np.mean([a for a, _ in d["ic"]]); ric = np.mean([b for _, b in d["ic"]])
        rows.append(dict(variant=tag, ar=m["ar"], vol=m["vol"], sharpe=m["sharpe"], max_dd=m["max_dd"],
                         calmar=m["calmar"], ic=ic, rank_ic=ric, turnover=np.mean(d["to"]),
                         trades=d["trades"], is_ar=m_is["ar"], oos_ar=m_oos["ar"],
                         fin_slot_avg=np.mean(d["fin_slots"]),
                         **{str(y): d["annual"][y] for y in YEARS}))
    df = pd.DataFrame(rows)
    hdr = f"  {'版本':<16}{'年化':>9}{'波动':>9}{'夏普':>7}{'回撤':>9}{'Calmar':>8}{'IC':>8}{'RankIC':>8}{'换手':>8}{'交易':>7}{'IS':>9}{'OOS':>9}{'金融占位':>9}"
    print(hdr, flush=True)
    for _, r in df.iterrows():
        print(f"  {r['variant']:<16}{r['ar']*100:>8.2f}%{r['vol']*100:>8.2f}%{r['sharpe']:>7.2f}"
              f"{r['max_dd']*100:>8.2f}%{r['calmar']:>8.2f}{r['ic']:>8.3f}{r['rank_ic']:>8.3f}"
              f"{r['turnover']*100:>7.1f}%{r['trades']:>7.0f}{r['is_ar']*100:>8.2f}%{r['oos_ar']*100:>8.2f}%"
              f"{r['fin_slot_avg']:>8.1f}只", flush=True)

    print(f"\n  【逐年收益】(2026为1-7月)", flush=True)
    print(f"  {'版本':<16}" + "".join(f"{y:>10}" for y in YEARS), flush=True)
    for _, r in df.iterrows():
        print(f"  {r['variant']:<16}" + "".join(f"{r[str(y)]*100:>9.2f}%" for y in YEARS), flush=True)
    d = df.set_index("variant")
    a, b = d.index[0], d.index[1]
    print(f"\n  差值(剔金融 - 含金融): 全期{(d.loc[b,'ar']-d.loc[a,'ar'])*100:+.2f}pp  "
          f"IS{(d.loc[b,'is_ar']-d.loc[a,'is_ar'])*100:+.2f}pp  OOS{(d.loc[b,'oos_ar']-d.loc[a,'oos_ar'])*100:+.2f}pp  "
          f"夏普{d.loc[b,'sharpe']-d.loc[a,'sharpe']:+.2f}", flush=True)
    df.to_csv(f"{WORK_DIR}/v19_fin_monthly.csv", sep='\t', index=False)
    print(f"\n[+] 已保存 v19_fin_monthly.csv", flush=True)


if __name__ == "__main__":
    main()
