"""
V16 规则化价值选股: 池内因子分层检验 + TopK组合回测 (无ML)
=================================================================
背景: V14/V15消融证明XGB排序为负增量(-10.69pp), 但估值因子是唯一正增量方向
      (E2-E1 = +2.78~+5.41pp)。用户要求: 选股必须超过股票池等权(E0, 12.24%年化)。
思路: 放弃ML, 用先验因子直接排序 —— "人肉价值选择"的系统化。
方法(抗过拟合):
  Part A: 池内因子五分位分层检验(等权,无成本), IS=2019-2022选因子, OOS=2023-2026验证
  Part B: 候选策略用与E0完全一致的引擎回测(T+1/往返0.4%/500万流动性/涨停停牌过滤)
因子(全部有文献先验, 无挖掘): ep bp cfp sp roe value_comp(4值rank均值)
                              vq(value+roe) mom_12_1 rev_60 low_vol_60
"""
import os, sys, random
import warnings, logging

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
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
from v14_ablation import (build_liquidity_table, load_price_matrix,
                          portfolio_backtest, calc_metrics,
                          quarter_signal_exec_dates, load_finind_table,
                          WORK_DIR, DATA_DIR)

# 流动性阈值口径修复: qlib $volume单位为手(x100股), 实测真实成交额/(close*volume)≈111
# 之前的500万阈值实际过滤了日成交<5亿的全部股票, 池子只剩8~21只, 排序失效
LIQ_THRESHOLD = 5_000_000 / 100   # 真实500万元 → qlib口径5万
from v15_model_fixed import get_bench_returns_ak

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
N_Q = 5   # 分层数
FACTORS = ["ep", "bp", "cfp", "sp", "roe", "value_comp", "vq",
           "mom_12_1", "rev_60", "low_vol_60"]


def load_raw_valuation():
    """估值原始值(非zscore): pivot表 {factor: DataFrame(date x instrument)}"""
    v = pd.read_csv(f"{DATA_DIR}/valuation_cache.csv", sep='\t', dtype={"code": str})
    v["date"] = pd.to_datetime(v["date"])
    for c in ["pe_ttm", "pb", "ps_ttm", "pcf"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["ep"] = 1.0 / v["pe_ttm"]; v["bp"] = 1.0 / v["pb"]
    v["sp"] = 1.0 / v["ps_ttm"]; v["cfp"] = 1.0 / v["pcf"]
    v["instrument"] = v["code"].map(format_qlib_code)
    v = v.replace([np.inf, -np.inf], np.nan)
    v = v.drop_duplicates(subset=["date", "instrument"], keep="last")
    out = {}
    for c in ["ep", "bp", "cfp", "sp"]:
        out[c] = v.pivot(index="date", columns="instrument", values=c).sort_index().ffill()
    print(f"  [估值原始表] {out['ep'].shape}", flush=True)
    return out


def roe_asof(fin, universe, sig):
    """PIT: avail<=sig 的最新ROE"""
    sub = fin[fin["avail"] <= sig]
    sub = sub[sub["instrument"].isin(universe)]
    return sub.groupby("instrument")["roe"].last()


def compute_factors(universe, sig, val_piv, fin, price_ext, cal_list):
    """信号日截面因子, 返回 DataFrame(index=instrument, columns=FACTORS), 越大越好"""
    f = pd.DataFrame(index=pd.Index(sorted(universe), name="instrument"))
    for c in ["ep", "bp", "cfp", "sp"]:
        pv = val_piv[c]
        row = pv.loc[:sig].iloc[-1] if len(pv.loc[:sig]) else pd.Series(dtype=float)
        f[c] = row.reindex(f.index)
    f["roe"] = roe_asof(fin, universe, sig).reindex(f.index)

    # 量价因子 (price_ext: 含sig前一年历史的价格矩阵)
    px = price_ext.loc[:sig]
    if len(px) > 260:
        p_sig = px.iloc[-1]; p21 = px.iloc[-22]; p252 = px.iloc[-253]
        f["mom_12_1"] = (p21 / p252 - 1).reindex(f.index)
    else:
        f["mom_12_1"] = np.nan
    if len(px) > 61:
        p60 = px.iloc[-61]
        f["rev_60"] = -(px.iloc[-1] / p60 - 1).reindex(f.index)      # 反转: 跌多的好
        rets60 = px.iloc[-61:].pct_change().std()
        f["low_vol_60"] = -rets60.reindex(f.index)                    # 低波动好
    else:
        f["rev_60"] = np.nan; f["low_vol_60"] = np.nan

    # 复合: 截面rank百分位均值 (rank对outlier稳健)
    rk = lambda s: s.rank(pct=True)
    f["value_comp"] = pd.concat([rk(f[c]) for c in ["ep", "bp", "cfp", "sp"]], axis=1).mean(axis=1)
    f["vq"] = pd.concat([f["value_comp"].rank(pct=True), rk(f["roe"])], axis=1).mean(axis=1)
    return f


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)
    val_piv = load_raw_valuation()
    fin = load_finind_table()

    print(f"\n{'='*100}\n  V16 规则化价值选股 (无ML)  Part A: 因子分层  Part B: TopK回测\n{'='*100}", flush=True)

    # ============ 逐年准备: 因子截面 + 未来一季收益 ============
    # layer_rows: (year, factor, quantile, 季收益);  fac_store[(y,qi)] = (因子df, sig, exec, 可交易集)
    layer_rows = []
    fac_store = {}
    year_ctx = {}
    for y in YEARS:
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = sorted(format_qlib_code(c) for c in codes)
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = quarter_signal_exec_dates(y, cal_list)
        # 价格矩阵: 起点前推400天(动量), 终点后延20天(Q4持有到次年首交易日)
        ext_start = sig_exec[0][0] - pd.Timedelta(days=400)
        ext_end = pd.Timestamp(bt_end) + pd.Timedelta(days=20)
        price_ext = load_price_matrix(universe, ext_start, min(ext_end, pd.Timestamp("2026-07-31")))
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        limit_up_set, suspension_set = build_limit_up_set(universe, cal)
        year_ctx[y] = (universe, sig_exec, price_ext, liq_table, limit_up_set, suspension_set)

        # 各季度: 因子 + 持有期收益
        for qi, (sig, exec_dt) in enumerate(sig_exec):
            f = compute_factors(universe, sig, val_piv, fin, price_ext, cal_list)
            # 流动性&可交易过滤 (与E0/E15一致的可交易口径)
            liq = liq_table.xs(sig, level=0).reindex(f.index) if sig in liq_table.index.get_level_values(0) else pd.Series(np.nan, index=f.index)
            tradable = f.index[(liq.notna()) & (liq >= LIQ_THRESHOLD)
                               & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in limit_up_set))
                               & (~pd.Series(f.index, index=f.index).map(lambda i: (exec_dt, i) in suspension_set))]
            f = f.loc[tradable]
            fac_store[(y, qi)] = (f, sig, exec_dt)

            # 持有期收益: exec_dt → 下一调仓执行日(或期末)
            if qi + 1 < len(sig_exec):
                nxt = sig_exec[qi + 1][1]
            else:
                after = [d for d in price_ext.index if d > pd.Timestamp(bt_end)]
                nxt = after[0] if after else price_ext.index[-1]
            if exec_dt not in price_ext.index or nxt not in price_ext.index:
                continue
            fwd = price_ext.loc[nxt] / price_ext.loc[exec_dt] - 1
            pool_ret = fwd.reindex(f.index).mean()
            for fac in FACTORS:
                s = f[fac].dropna()
                if len(s) < N_Q * 4:
                    continue
                q = pd.qcut(s.rank(method="first"), N_Q, labels=False)
                for qq in range(N_Q):
                    grp = s.index[q == qq]
                    layer_rows.append({"year": y, "factor": fac, "q": qq,
                                       "ret": fwd.reindex(grp).mean(),
                                       "pool": pool_ret})
        print(f"  [{y}] 池{len(universe)}只 因子/收益就绪", flush=True)

    # ============ Part A 汇总 ============
    L = pd.DataFrame(layer_rows)
    L["excess"] = L["ret"] - L["pool"]
    L["seg"] = L["year"].map(lambda y: "IS" if y in IS_YEARS else "OOS")
    print(f"\n{'='*100}\n  Part A: 因子五分位季均超额(池内, Q4=因子值最高层)  IS=19/21/22  OOS=23-26\n{'='*100}", flush=True)
    print(f"{'因子':<12} | {'IS Q4超额':>9} {'IS单调性':>8} | {'OOS Q4超额':>9} {'OOS单调性':>8} | {'全期Q4超额':>9}", flush=True)
    summary = []
    for fac in FACTORS:
        g = L[L["factor"] == fac]
        row = {"factor": fac}
        for seg in ["IS", "OOS"]:
            gs = g[g["seg"] == seg]
            qm = gs.groupby("q")["excess"].mean()
            row[f"{seg}_top"] = qm.get(N_Q - 1, np.nan)
            row[f"{seg}_mono"] = qm.corr(pd.Series(range(N_Q), index=qm.index), method="spearman") if len(qm) == N_Q else np.nan
        row["ALL_top"] = g[g["q"] == N_Q - 1]["excess"].mean()
        summary.append(row)
        print(f"{fac:<12} | {row['IS_top']*100:>8.2f}% {row['IS_mono']:>8.2f} | "
              f"{row['OOS_top']*100:>8.2f}% {row['OOS_mono']:>8.2f} | {row['ALL_top']*100:>8.2f}%", flush=True)
    pd.DataFrame(summary).to_csv(f"{WORK_DIR}/v16_factor_layers.csv", sep='\t', index=False)

    # ============ Part B: 候选策略完整回测 ============
    # 候选: 先验固定的价值系 + IS期Top层超额最高的因子 (透明呈现全部)
    sdf = pd.DataFrame(summary).set_index("factor")
    is_best = sdf["IS_top"].idxmax()
    candidates = list(dict.fromkeys(["value_comp", "vq", "ep", is_best]))
    strategies = [("E0 等权基准", None, None)]
    for fac in candidates:
        for k in [10, 20, 30]:
            strategies.append((f"{fac} Top{k}", fac, k))

    print(f"\n{'='*100}\n  Part B: 完整回测 (T+1, 往返0.4%, 500万流动性, 涨停停牌过滤)  IS最优因子={is_best}\n{'='*100}", flush=True)
    all_rets = {s[0]: [] for s in strategies}
    annual = {s[0]: {} for s in strategies}
    to_stats = {s[0]: [] for s in strategies}

    for y in YEARS:
        universe, sig_exec, price_ext, liq_table, lu, sus = year_ctx[y]
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        price_mat = price_ext.loc[sig_exec[0][0]:pd.Timestamp(bt_end)]
        row = f"  {y}:"
        for name, fac, k in strategies:
            rebalances = []
            for qi, (sig, exec_dt) in enumerate(sig_exec):
                f, _, _ = fac_store[(y, qi)]
                if fac is None:
                    holdings = [i for i in universe if i in price_mat.columns
                                and pd.notna(price_mat.loc[exec_dt, i])]
                else:
                    s = f[fac].dropna()
                    holdings = s.nlargest(k).index.tolist()
                rebalances.append((exec_dt, holdings))
            rets, avg_to = portfolio_backtest(rebalances, price_mat, cal_list)
            all_rets[name].append(rets)
            to_stats[name].append(avg_to)
            annual[name][y] = (1 + rets).prod() - 1
        print(row + "  " + "  ".join(f"{n.split()[0][:2]}{n.split()[-1][-2:]} {annual[n][y]*100:>6.1f}%"
                                     for n in list(all_rets)[:6]), flush=True)

    # ============ 汇总 ============
    bench_all = []
    for y in YEARS:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns_ak(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]

    print(f"\n{'='*130}", flush=True)
    header = "  ".join(f"{y:>7}" for y in YEARS)
    print(f"{'策略':<18} | {header} | {'全期年化':>8} {'夏普':>5} {'回撤':>7} {'季换手':>6} {'IS年化':>7} {'OOS年化':>7}", flush=True)
    print(f"{'-'*130}", flush=True)
    out_rows = []
    for name in all_rets:
        s = pd.concat(all_rets[name]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        m = calc_metrics(s)
        m_is = calc_metrics(s[s.index.year.isin(IS_YEARS)])
        m_oos = calc_metrics(s[s.index.year.isin(OOS_YEARS)])
        row = "  ".join(f"{annual[name][y]*100:>6.2f}%" for y in YEARS)
        print(f"{name:<18} | {row} | {m['ar']*100:>7.2f}% {m['sharpe']:>5.2f} {m['max_dd']*100:>6.1f}% "
              f"{np.mean(to_stats[name])*100:>5.1f}% {m_is['ar']*100:>6.2f}% {m_oos['ar']*100:>6.2f}%", flush=True)
        out_rows.append({"strategy": name, "ar": m["ar"], "sharpe": m["sharpe"],
                         "max_dd": m["max_dd"], "is_ar": m_is["ar"], "oos_ar": m_oos["ar"],
                         **{str(y): annual[name][y] for y in YEARS}})
        safe = name.replace(" ", "_")
        s.to_csv(f"{WORK_DIR}/v16_returns_{safe}.csv", sep='\t', header=False)
    bm = calc_metrics(bench)
    row_b = "  ".join(f"{(1+bench[bench.index.year==y]).prod()-1:>7.2%}" for y in YEARS)
    print(f"{'沪深300(ak)':<18} | {row_b} | {bm['ar']*100:>7.2f}% {bm['sharpe']:>5.2f} {bm['max_dd']*100:>6.1f}%", flush=True)
    print(f"{'='*130}", flush=True)
    pd.DataFrame(out_rows).to_csv(f"{WORK_DIR}/v16_strategy_results.csv", sep='\t', index=False)
    print("\n[+] 结果已保存: v16_factor_layers.csv, v16_strategy_results.csv", flush=True)


if __name__ == "__main__":
    run()
