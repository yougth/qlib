"""
V18 上线前最终验证: 不含金融 value_comp Top20  完整风险指标
=================================================================
数据已补齐(131只缺失行情回填, 25/26池100%覆盖)。PIT三层已核验干净:
  1) 估值ep/bp/cfp/sp: valuation_cache为真·每日序列, compute_factors用 loc[:sig].iloc[-1]
     严格取信号日当日快照(price_t/eps_ttm_known_at_t), 仅ffill历史值, 无未来数据。
  2) ROE(finind): avail=披露日(Q1→05-01/H1→09-01/Q3→11-01/年报→次年05-01), 保守滞后, 无穿越
     (且value_comp根本不使用roe)。
  3) 池(build_dynamic_universe): 用year-2财务, 避开年报披露前视。
输出图1全套指标: 年化收益/年化波动/Sharpe/最大回撤/Calmar/IC/RankIC/换手率/交易次数 + 逐年。
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

from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code
from v14_ablation import (portfolio_backtest, quarter_signal_exec_dates,
                          load_finind_table, WORK_DIR, DATA_DIR)
from v16_value_rules import load_raw_valuation
from v16_fin_compare import FIN_SET, prep_year
from v15_model_fixed import get_bench_returns_ak

YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
IS_YEARS = {2019, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
K = 20


def full_metrics(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return dict(ar=0, vol=0, sharpe=0, max_dd=0, calmar=0)
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    calmar = ar / abs(max_dd) if max_dd < 0 else 0
    return dict(ar=ar, vol=vol, sharpe=sharpe, max_dd=max_dd, calmar=calmar)


def holding_fwd(price_ext, exec_dt, sig_exec, qi, bt_end):
    """持有期(exec_dt→下一调仓执行日/期末)各票收益, Series(index=instrument)"""
    if qi + 1 < len(sig_exec):
        nxt = sig_exec[qi + 1][1]
    else:
        after = [d for d in price_ext.index if d > pd.Timestamp(bt_end)]
        nxt = after[0] if after else price_ext.index[-1]
    if exec_dt not in price_ext.index or nxt not in price_ext.index:
        return None
    return price_ext.loc[nxt] / price_ext.loc[exec_dt] - 1


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t'); fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t'); profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    name_map = {format_qlib_code(c): n for c, n in fcf_df.drop_duplicates("code")[["code", "name"]].values}
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31"); cal_list = list(cal)
    val_piv = load_raw_valuation(); fin = load_finind_table()

    # 沪深300基准
    bench_all = []
    for y in YEARS:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns_ak(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index(); bench = bench[~bench.index.duplicated(keep="last")]

    # 逐年准备 (不含金融)
    print("\n>>> 准备 不含金融 池 ...", flush=True)
    ctx = {}
    for y in YEARS:
        ctx[y] = prep_year(y, fcf_df, profit_df, cal, cal_list, val_piv, fin, exclude_fin=True)
        print(f"    [{y}] 池{len(ctx[y][0])}只", flush=True)

    # 回测 + IC/RankIC + 交易次数
    all_rets = []
    annual = {}
    to_list = []
    ic_list, ric_list = [], []
    n_rebalances = 0
    n_buys = 0            # 累计买入笔数(新进持仓)
    prev_holdings = set()
    picks_2526 = []
    for y in YEARS:
        universe, sig_exec, price_ext, fac_store = ctx[y]
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        price_mat = price_ext.loc[sig_exec[0][0]:pd.Timestamp(bt_end)]
        rb = []
        for qi, (sig, exec_dt) in enumerate(sig_exec):
            f, _, _ = fac_store[qi]
            s = f["value_comp"].dropna()
            top = s.nlargest(K)
            rb.append((exec_dt, top.index.tolist()))
            # 交易次数
            n_rebalances += 1
            cur = set(top.index)
            n_buys += len(cur - prev_holdings)
            prev_holdings = cur
            # IC/RankIC: 全可交易截面 value_comp vs 持有期收益
            fwd = holding_fwd(price_ext, exec_dt, sig_exec, qi, bt_end)
            if fwd is not None:
                common = s.index.intersection(fwd.dropna().index)
                if len(common) >= 20:
                    x = s.reindex(common); yv = fwd.reindex(common)
                    ic_list.append(x.corr(yv))                       # Pearson IC
                    ric_list.append(x.rank().corr(yv.rank()))        # Spearman RankIC
            if y in (2025, 2026):
                for rk, (inst, sc) in enumerate(top.items(), 1):
                    picks_2526.append({"period": f"{y}Q{qi+1}", "exec_date": exec_dt.date(),
                                       "rank": rk, "instrument": inst,
                                       "name": name_map.get(inst, "?"), "value_comp": round(float(sc), 4)})
        rets, to = portfolio_backtest(rb, price_mat, cal_list)
        all_rets.append(rets); to_list.append(to)
        annual[y] = (1 + rets).prod() - 1

    s = pd.concat(all_rets).sort_index(); s = s[~s.index.duplicated(keep="last")]
    m = full_metrics(s)
    m_is = full_metrics(s[s.index.year.isin(IS_YEARS)])
    m_oos = full_metrics(s[s.index.year.isin(OOS_YEARS)])
    bm = full_metrics(bench)

    ic = np.nanmean(ic_list); ic_ir = np.nanmean(ic_list) / (np.nanstd(ic_list) + 1e-9)
    ric = np.nanmean(ric_list); ric_ir = np.nanmean(ric_list) / (np.nanstd(ric_list) + 1e-9)
    ic_pos = np.mean([1 for v in ric_list if v > 0]) if ric_list else 0
    avg_to = np.mean(to_list)

    # ============ 输出 ============
    line = "=" * 78
    print(f"\n{line}", flush=True)
    print("  V18 上线前最终验证 —— 不含金融 value_comp Top20 (修复数据后重跑)", flush=True)
    print(f"  引擎: T+1 / 往返成本0.4% / 流动性≥500万 / 涨停停牌过滤 / 季频调仓", flush=True)
    print(line, flush=True)

    print("\n【逐年收益】(2026为半年 1-7月)", flush=True)
    print(f"  {'年份':<8}{'策略收益':>12}{'沪深300':>12}{'超额':>12}", flush=True)
    for y in YEARS:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        b_y = (1 + bench[bench.index.year == y]).prod() - 1
        print(f"  {y:<8}{annual[y]*100:>11.2f}%{b_y*100:>11.2f}%{(annual[y]-b_y)*100:>11.2f}%", flush=True)

    print(f"\n【图1 全套风险指标】(2019-2026全期)", flush=True)
    print(f"  {'指标':<14}{'本策略':>14}{'沪深300':>14}{'意义':>10}", flush=True)
    rows = [
        ("年化收益", f"{m['ar']*100:.2f}%", f"{bm['ar']*100:.2f}%", "赚钱能力"),
        ("年化波动", f"{m['vol']*100:.2f}%", f"{bm['vol']*100:.2f}%", "风险"),
        ("Sharpe", f"{m['sharpe']:.2f}", f"{bm['sharpe']:.2f}", "风险收益比"),
        ("最大回撤", f"{m['max_dd']*100:.2f}%", f"{bm['max_dd']*100:.2f}%", "抗风险"),
        ("Calmar", f"{m['calmar']:.2f}", f"{bm['calmar']:.2f}", "回撤收益比"),
        ("IC", f"{ic:.4f}", "-", "预测能力"),
        ("RankIC", f"{ric:.4f}", "-", "排序能力"),
        ("换手率(季均单边)", f"{avg_to*100:.1f}%", "-", "交易成本"),
        ("交易次数(累计买入)", f"{n_buys}笔/{n_rebalances}次调仓", "-", "容量压力"),
    ]
    for k_, a, b, mean in rows:
        print(f"  {k_:<14}{a:>14}{b:>14}{mean:>10}", flush=True)
    print(f"\n  [附] IC_IR={ic_ir:.2f}  RankIC_IR={ric_ir:.2f}  RankIC>0占比={ic_pos*100:.0f}%  "
          f"(共{len(ric_list)}个调仓截面)", flush=True)
    print(f"  [附] IS(19/21/22)年化={m_is['ar']*100:.2f}%  OOS(23-26)年化={m_oos['ar']*100:.2f}%  "
          f"IS夏普={m_is['sharpe']:.2f}  OOS夏普={m_oos['sharpe']:.2f}", flush=True)

    # 保存
    out = {"metric": [r[0] for r in rows], "strategy": [r[1] for r in rows], "hs300": [r[2] for r in rows]}
    pd.DataFrame(out).to_csv(f"{WORK_DIR}/v18_launch_metrics.csv", sep='\t', index=False)
    ann = pd.DataFrame({"year": YEARS, "strategy_ret": [annual[y] for y in YEARS]})
    ann.to_csv(f"{WORK_DIR}/v18_annual.csv", sep='\t', index=False)
    s.to_csv(f"{WORK_DIR}/v18_daily_returns.csv", sep='\t', header=False)
    pd.DataFrame(picks_2526).to_csv(f"{WORK_DIR}/v18_picks_2526.csv", sep='\t', index=False)
    print(f"\n[+] 已保存: v18_launch_metrics.csv / v18_annual.csv / v18_daily_returns.csv / v18_picks_2526.csv", flush=True)
    print(line, flush=True)


if __name__ == "__main__":
    main()
