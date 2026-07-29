"""
终检回测 —— 只跑上线这一个策略 value_comp_M20
==================================================================
作用: 上线目录的**端到端回归测试**。它用 core/ 里的代码从零重算 2019-2026 全部
月度换仓与净值曲线, 然后与 golden/backtest_baseline.csv 逐字段比对。

判据 (打印在末尾, 任一不满足就不要交易):
  · 2019/2021/2022/2023/2024/2025 各年收益 **必须完全一致** (小数位都不能差)
    —— 这些年的数据早已定格, 变了就说明历史被改写
  · 只有 2026 允许变 (每月会补进新交易日)
  · trades 只允许增加, 且增量 <= 20 (每月最多新进20只)
  · ar/sharpe/oos_ar 允许由 2026 传导的小幅变动

与 v18_full_stack 的差别: 去掉了 XGB 残差模型那两个对照策略 (V18_full_Top20/Top15)。
value_comp_M20 的持仓只由 value_comp 决定, 与模型无关, 因此结果必须逐位一致 —— 这一点
本身就是"重构没改口径"的证明。

用法:
  /usr/bin/python3 run_backtest.py            # 回测 + 与基线比对
  /usr/bin/python3 run_backtest.py --relock   # 用当前结果重建基线(仅有意改策略时)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from core import config
from core.universe import load_fin_caches
from core.valuation import load_raw_valuation, compute_value_comp
from core.calendar_rules import get_calendar
from core.selection import liquidity_at, tradable_universe, pick_topk
from core.signal import year_context, year_bt_end
from core.backtest import portfolio_backtest, full_metrics, trade_count
from core.market import load_bench_returns

NAME = "value_comp_M20"
BASE_CSV = f"{config.GOLD_DIR}/backtest_baseline.csv"
OUT_CSV = f"{config.ONLINE_DIR}/backtest_results.csv"
RET_CSV = f"{config.ONLINE_DIR}/backtest_returns_{NAME}.csv"

# 逐年判据: 这些年必须完全定格; LIVE_YEAR 是仍在推进的年份, 允许变化
# 容差只放到 1e-12 (=1e-10pp), 用于吸收浮点末位(1ULP)噪声, 不足以掩盖任何真实变化:
# 换掉一只持仓带来的年收益差异在 1e-3 量级, 相差 9 个数量级。
FROZEN_TOL = 1e-12


def fwd_excess(close, exec_dt, tradable, index):
    """未来约60日的池内超额收益 —— 只用于 IC 诊断, 不影响持仓"""
    fut = [d for d in close.index if d > exec_dt]
    if exec_dt in close.index and len(fut) >= 1:
        target_day = min(fut, key=lambda d: abs((d - exec_dt).days - config.FWD_DAYS))
        if (target_day - exec_dt).days >= config.FWD_DAYS * 0.6:
            fwd = close.loc[target_day] / close.loc[exec_dt] - 1
            pool_ret = fwd.reindex(tradable).mean()
            return (fwd - pool_ret).reindex(index)
    return pd.Series(np.nan, index=index)


def run():
    config.init_qlib()
    print("=" * 100, flush=True)
    print(f"  终检回测 {NAME}  (月度纯估值Top20等权)   回测末端={config.BT_END}", flush=True)
    print("=" * 100, flush=True)

    # 日历显式截到 BT_END: 保证换仓点数量冻结, 与基线可比
    cal_list = get_calendar(end=config.BT_END)
    val_piv = load_raw_valuation()
    fcf_df, profit_df = load_fin_caches()

    all_rets, annual, to_stats, ic_pairs = [], {}, [], []
    trades = 0
    for y in config.YEARS:
        ctx = year_context(y, cal_list, fcf_df, profit_df)
        close = ctx["close"]
        rebal = []
        for sig, exec_dt in ctx["sig_exec"]:
            vc = compute_value_comp(ctx["universe"], sig, val_piv)
            liq = liquidity_at(ctx["liq_table"], sig, vc.index)
            trd = tradable_universe(vc.index, liq, exec_dt, ctx["lu"], ctx["sus"])
            hold = pick_topk(vc, trd)
            rebal.append((exec_dt, hold.index.tolist()))
            # 因子IC诊断: 该期打分(池内rank) vs 未来超额
            score = vc.reindex(trd).rank(pct=True)
            fwd = fwd_excess(close, exec_dt, trd, vc.index).reindex(trd)
            pair = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
            if len(pair) >= 10:
                ic_pairs.append((pair["s"].corr(pair["f"]),
                                 pair["s"].rank().corr(pair["f"].rank())))
        price_mat = close.loc[ctx["sig_exec"][0][0]:ctx["bt_end"]]
        rets, avg_to = portfolio_backtest(rebal, price_mat)
        all_rets.append(rets); to_stats.append(avg_to)
        annual[y] = (1 + rets).prod() - 1
        trades += trade_count(rebal)
        print(f"      → {y} 收益 {annual[y]*100:+.2f}%  单边换手 {avg_to*100:.1f}%", flush=True)

    s = pd.concat(all_rets).sort_index(); s = s[~s.index.duplicated(keep="last")]
    m = full_metrics(s)
    m_is = full_metrics(s[s.index.year.isin(config.IS_YEARS)])
    m_oos = full_metrics(s[s.index.year.isin(config.OOS_YEARS)])
    ic = float(np.nanmean([p[0] for p in ic_pairs])) if ic_pairs else 0.0
    ric = float(np.nanmean([p[1] for p in ic_pairs])) if ic_pairs else 0.0

    row = dict(strategy=NAME, ar=m["ar"], vol=m["vol"], sharpe=m["sharpe"],
               max_dd=m["max_dd"], calmar=m["calmar"], ic=ic, rank_ic=ric,
               turnover=float(np.mean(to_stats)), trades=trades,
               is_ar=m_is["ar"], oos_ar=m_oos["ar"],
               **{str(y): annual[y] for y in config.YEARS})
    cur = pd.DataFrame([row])
    cur.to_csv(OUT_CSV, sep='\t', index=False)
    s.to_csv(RET_CSV, sep='\t', header=False)

    bench = pd.concat([load_bench_returns(f"{y}-01-01", str(year_bt_end(y).date()))
                       for y in config.YEARS]).sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    bm = full_metrics(bench)
    yhdr = "  ".join(f"{y:>8}" for y in config.YEARS)
    print(f"\n{'-'*130}\n{'策略':<15} | {yhdr} | {'年化':>7} {'波动':>6} {'夏普':>5} "
          f"{'回撤':>7} {'Calmar':>6} {'换手':>6} {'交易数':>5} {'IS':>7} {'OOS':>7}", flush=True)
    yrow = "  ".join(f"{annual[y]*100:>7.2f}%" for y in config.YEARS)
    print(f"{NAME:<15} | {yrow} | {m['ar']*100:>6.2f}% {m['vol']*100:>5.1f}% {m['sharpe']:>5.2f} "
          f"{m['max_dd']*100:>6.1f}% {m['calmar']:>6.2f} {np.mean(to_stats)*100:>5.1f}% "
          f"{trades:>5d} {m_is['ar']*100:>6.2f}% {m_oos['ar']*100:>6.2f}%", flush=True)
    print(f"{'沪深300':<15} | {'':>{len(yrow)}} | {bm['ar']*100:>6.2f}% {bm['vol']*100:>5.1f}% "
          f"{bm['sharpe']:>5.2f} {bm['max_dd']*100:>6.1f}% {bm['calmar']:>6.2f}", flush=True)
    print("-" * 130, flush=True)
    return cur


def compare(cur, relock):
    if relock or not os.path.exists(BASE_CSV):
        os.makedirs(config.GOLD_DIR, exist_ok=True)
        cur.to_csv(BASE_CSV, sep='\t', index=False)
        print(f"\n[{'重建' if relock else '首次生成'}] golden/backtest_baseline.csv", flush=True)
        return 0
    # float_precision="round_trip": pandas C 引擎默认的快速解析器对 17 位有效数字会差 1ULP,
    # 会让"其实一模一样"的年份被误判成变了 —— 假警报比漏报更危险, 必须在根因上消掉。
    base = pd.read_csv(BASE_CSV, sep='\t', float_precision="round_trip").iloc[0]
    now = cur.iloc[0]
    live = str(max(config.YEARS))          # 仍在推进的年份, 允许变化
    frozen = [str(y) for y in config.YEARS if str(y) != live]

    print(f"\n{'='*100}\n  与冻结基线逐字段比对 (golden/backtest_baseline.csv)\n{'='*100}", flush=True)
    bad = []
    for y in frozen:
        d = float(now[y]) - float(base[y])
        ok = abs(d) <= FROZEN_TOL
        print(f"  [{'OK  ' if ok else 'FAIL'}] {y:>6} 年收益  {float(base[y])*100:>8.4f}% → "
              f"{float(now[y])*100:>8.4f}%   差 {d:+.3e}   (必须完全一致)", flush=True)
        if not ok:
            bad.append(f"{y}年收益变了({d*100:+.6f}pp) → 历史数据被改写, 查清才能上线")

    dt = int(now["trades"]) - int(base["trades"])
    ok_t = 0 <= dt <= 20
    print(f"  [{'OK  ' if ok_t else 'FAIL'}] trades   {int(base['trades'])} → {int(now['trades'])}"
          f"   增量 {dt:+d}   (只允许增加且<=20)", flush=True)
    if not ok_t:
        bad.append(f"trades 增量 {dt:+d} 越界 → 历史换仓记录变了")

    print(f"  [    ] {live:>6} 年收益  {float(base[live])*100:>8.4f}% → {float(now[live])*100:>8.4f}%"
          f"   (当年仍在推进, 允许变化)", flush=True)
    for k in ["ar", "sharpe", "max_dd", "is_ar", "oos_ar", "turnover"]:
        print(f"  [    ] {k:>8}  {float(base[k]):>14.10f} → {float(now[k]):>14.10f}"
              f"   差 {float(now[k])-float(base[k]):+.10f}", flush=True)

    print("=" * 100, flush=True)
    if bad:
        print("  回测结论: 【不通过, 禁止交易】", flush=True)
        for b in bad:
            print(f"    - {b}", flush=True)
    else:
        print(f"  回测结论: 通过。{len(frozen)}个已定格年份逐位一致, 仅 {live} 年随新数据变动。", flush=True)
    print("=" * 100, flush=True)
    return len(bad)


def main():
    relock = "--relock" in sys.argv
    if relock:
        print("!! --relock: 将用当前结果重建回测基线。仅当你有意改策略并确认 OOS 不下降时使用。", flush=True)
    cur = run()
    sys.exit(0 if compare(cur, relock) == 0 else 1)


if __name__ == "__main__":
    main()
