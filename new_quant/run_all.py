#!/usr/bin/env python3
"""run_all —— new_quant 入口: 共享数据只读加载 → 策略回测 → 标准报告

用法:
    cd qlib/new_quant
    /usr/bin/python3 run_all.py [--skip-etf-fetch]
产出: outputs/yearly_returns.csv, outputs/metrics.csv (报告正文建议重定向保存)
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import config, data  # noqa: E402
from core.backtest import (run_portfolio, calc_metrics, yearly_returns,  # noqa: E402
                           rank_ic, print_yearly_table, print_metric_table)
from strategies import stock_strategies as ss  # noqa: E402
from strategies import trend_etf as te  # noqa: E402


def _run_tag(adj_close, rebals, fee, last_date=None):
    rets, to, buys, frozen = run_portfolio(adj_close, rebals, fee, last_date)
    return dict(rets=rets, to=to, buys=buys, frozen=frozen,
                nhold=[len(w) for _, w in rebals])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-etf-fetch", action="store_true")
    a = ap.parse_args()
    os.makedirs(config.OUT_DIR, exist_ok=True)

    print("=" * 100)
    print(f"  new_quant 回测 | 区间 {config.BT_START} ~ {config.BT_END} | "
          f"A股往返成本 {config.FEE_RT_STOCK:.1%}, ETF {config.FEE_RT_ETF:.1%}")
    print("=" * 100)
    cal = data.load_calendar(config.BT_START, config.BT_END)
    panels = data.load_panels(config.BT_START, config.BT_END)
    fin = data.load_financials()
    bench = data.load_benchmark()
    psym, fsym = set(panels["adj_close"].columns), set(fin["sym"])
    print(f"[CHECK] 覆盖交叉: 行情{len(psym)}只 × 财务{len(fsym)}只 | "
          f"财务无行情(多为已退市→存活偏差上限披露): {len(fsym - psym)} | "
          f"行情无财务: {len(psym - fsym)}", flush=True)
    signals = data.month_end_signals(cal, config.BT_START, config.BT_END)
    print(f"[CHECK] 信号日 {len(signals)} 个: "
          f"{signals[0][0].date()} ~ {signals[-1][0].date()}", flush=True)

    specs = [("QUAL_EW", "质量池等权(基线)", "qual_ew"),
             ("LOWATT", "低关注质量Top15(方向4代理)", "lowatt"),
             ("ACCEL", "盈利改善Top15(方向5代理)", "accel")]
    res, stress = {}, {}
    for tag, label, mode in specs:
        rebals, scores, picks = ss.build_rebalances(panels, fin, signals, mode)
        r = _run_tag(panels["adj_close"], rebals,
                     config.FEE_RT_STOCK, panels["last_date"])
        if mode == "qual_ew":
            ic = ric = float("nan")
        else:
            ic, ric, _ = rank_ic(panels["adj_close"], scores)
        r.update(label=label, ic=ic, ric=ric, picks=picks)
        res[tag] = r
        nh = r["nhold"]
        print(f"[{tag}] 月均持仓 {np.mean(nh):.0f} (min {min(nh)} / max {max(nh)}) | "
              f"月均单边换手 {r['to']:.1%} | 买入笔数 {r['buys']} | "
              f"退市冻结事件 {len(r['frozen'])}", flush=True)
        if mode != "qual_ew":
            r2, _, _, _ = run_portfolio(panels["adj_close"], rebals,
                                        config.FEE_RT_STOCK * config.COST_STRESS,
                                        panels["last_date"])
            stress[tag] = r2

    # ---- ETF: 跨资产趋势 ----
    try:
        if not a.skip_etf_fetch:
            need = [s for s in config.ETF_SYMS
                    if not os.path.exists(os.path.join(config.ETF_DIR, s + ".parquet"))]
            if need:
                subprocess.run([sys.executable, os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "tools", "fetch_etf_ohlcv.py")], check=True)
        epx = te.load_etf_panel("2017-01-01", config.BT_END)
        for tag, label, rebals in [
                ("TREND", "跨资产趋势(方向7)", te.trend_rebalances(epx)),
                ("ETF_EW", "ETF篮子等权(基线)", te.ew_rebalances(epx))]:
            r = _run_tag(epx, rebals, config.FEE_RT_ETF)
            r.update(label=label, ic=float("nan"), ric=float("nan"), picks=None)
            res[tag] = r
            print(f"[{tag}] 月均持仓 {np.mean(r['nhold']):.1f} | "
                  f"月均单边换手 {r['to']:.1%} | 买入笔数 {r['buys']}", flush=True)
        r2, _, _, _ = run_portfolio(epx, te.trend_rebalances(epx),
                                    config.FEE_RT_ETF * config.COST_STRESS)
        stress["TREND"] = r2
    except Exception as e:
        print(f"[WARN] ETF部分跳过: {e!r}", flush=True)

    # ==================== 标准报告 ====================
    order = [t for t in ["QUAL_EW", "LOWATT", "ACCEL", "TREND", "ETF_EW"]
             if t in res] + ["BENCH"]
    labels = {t: res[t]["label"] for t in res}
    labels["BENCH"] = "沪深300"
    yearly_all = {t: yearly_returns(res[t]["rets"]).to_dict() for t in res}
    yearly_all["BENCH"] = yearly_returns(bench).to_dict()
    print_yearly_table(yearly_all, order, labels)

    rows = []
    for t in order:
        if t == "BENCH":
            m = calc_metrics(bench)
            rows.append({"策略": "沪深300", **m, "IC": np.nan, "RankIC": np.nan,
                         "换手率": np.nan, "交易次数": 0})
        else:
            m = calc_metrics(res[t]["rets"], bench)
            rows.append({"策略": res[t]["label"], **m, "IC": res[t]["ic"],
                         "RankIC": res[t]["ric"], "换手率": res[t]["to"],
                         "交易次数": res[t]["buys"]})
    print_metric_table(rows)

    print(f"\n{'='*100}\n  三、成本压力测试 (往返成本×{config.COST_STRESS:.0f})\n{'='*100}",
          flush=True)
    for tag, r2 in stress.items():
        m1 = calc_metrics(res[tag]["rets"])
        m2 = calc_metrics(r2)
        print(f"  {res[tag]['label']}: 年化 {m1['年化收益']:.2%} → "
              f"{m2['年化收益']:.2%} ({(m2['年化收益']-m1['年化收益'])*100:+.2f}pp)",
              flush=True)

    print(f"\n{'='*100}\n  四、相关性与持仓重叠 (名字不同≠收益来源不同)\n{'='*100}",
          flush=True)
    tags = [t for t in ["LOWATT", "ACCEL", "TREND"] if t in res]
    if len(tags) >= 2:
        mret = pd.concat({t: (1 + res[t]["rets"]).resample("M").prod() - 1
                          for t in tags}, axis=1).dropna()
        print("  月度收益相关系数:\n" + mret.corr().round(3).to_string(), flush=True)
    if "LOWATT" in res and "ACCEL" in res:
        ov = [len(x & y) / max(1, len(x | y)) for x, y in
              zip(res["LOWATT"]["picks"], res["ACCEL"]["picks"])]
        print(f"  LOWATT vs ACCEL 月度持仓 Jaccard 重叠均值: {np.mean(ov):.1%}",
              flush=True)

    print(f"\n{'='*100}\n  五、对比结论\n{'='*100}", flush=True)
    def _cmp(a, b):
        ma, mb = calc_metrics(res[a]["rets"]), calc_metrics(res[b]["rets"])
        print(f"  {res[a]['label']} vs {res[b]['label']}: 年化 "
              f"{ma['年化收益']:.2%} vs {mb['年化收益']:.2%} | Sharpe "
              f"{ma['Sharpe']:.2f} vs {mb['Sharpe']:.2f} | 回撤 "
              f"{ma['最大回撤']:.1%} vs {mb['最大回撤']:.1%}", flush=True)
    if "LOWATT" in res:
        _cmp("LOWATT", "QUAL_EW")
    if "ACCEL" in res:
        _cmp("ACCEL", "QUAL_EW")
    if "TREND" in res:
        _cmp("TREND", "ETF_EW")

    pd.DataFrame(yearly_all).T.to_csv(
        os.path.join(config.OUT_DIR, "yearly_returns.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(config.OUT_DIR, "metrics.csv"),
                              index=False)
    print(f"\n[+] 已保存 {config.OUT_DIR}/yearly_returns.csv, metrics.csv",
          flush=True)


if __name__ == "__main__":
    main()
