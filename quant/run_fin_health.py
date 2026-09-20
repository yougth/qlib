#!/usr/bin/env python3
"""
run_fin_health —— 财务健康过滤对照实验 (原池 vs 过滤池)
================================================================================
问题: "十年双正"底池存在两类漏网之鱼 —— 低效繁荣(ROIC<资本成本)与杠杆走钢丝
      (有息负债高企)。极简减法能否提升纯价值策略的表现?
实验: 同口径 (同可交易门禁/同成本/同打分) 下, 对照
  A) 原池 (十年双正)
  B) 过滤池 (十年双正 + ROIC>8% + [利息保障>3 或 负债率<60%])
策略: VGH Top15 / VGH Top10 / VG Top10 / VAL20 (纯价值系, 无 ML 依赖, 快)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import logging

from core import config
from core import data as datalayer
from core import strategy
from core.universe import (build_windows, build_dynamic_universe,
                           format_qlib_code, load_pit_caches)
from core.valuation import (load_valuation, value_comp_score,
                            value_growth_score, value_hk_fcf_score)
from core.tradability import build_tradability
from core.backtest import portfolio_backtest, calc_metrics
from core.fin_health import load_fin_health, load_fin_codes, filter_pool

logging.getLogger("qlib").setLevel(logging.ERROR)

FIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data_cache", "fin_health_pit.parquet")
STRATS = [("VGH15", 15, "vgh"), ("VGH10", 10, "vgh"),
          ("VG10", 10, "vg"), ("VAL20", 20, "val")]


def make_score(kind, val_piv, fcf_df, profit_df):
    if kind == "vgh":
        return lambda cand, dt: value_hk_fcf_score(
            cand, dt, val_piv, fcf_df, profit_df,
            hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W).reindex(cand)
    if kind == "vg":
        return lambda cand, dt: value_growth_score(
            cand, dt, val_piv, fcf_df, profit_df,
            hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W,
            q_w=config.VALUE_QUALITY_W).reindex(cand)
    return lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand)


def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    val_piv = load_valuation()
    fin_df = load_fin_health(FIN_PATH)
    fin_codes = load_fin_codes()
    windows = build_windows()

    rebalances = {(tag, pool): [] for tag, _, _ in STRATS
                  for pool in ("orig", "filt")}
    drop_stats = []

    for win in windows:
        y = win["year"]
        xs, xe = win["test"]
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        kept, stats = filter_pool(codes, y, fin_df, fin_codes)
        drop_stats.append({"window": win["name"], "pool": len(codes),
                           "kept": stats["kept"],
                           "roic_fail": stats["roic_fail"],
                           "debt_fail": stats["debt_fail"]})
        print(f"\n[{win['name']}] 池 {len(codes)} → 过滤后 {stats['kept']} "
              f"(剔 ROIC {stats['roic_fail']} + 债务/缺数据 {stats['debt_fail']})",
              flush=True)

        for pool, pool_codes in (("orig", codes), ("filt", kept)):
            universe = [format_qlib_code(c) for c in pool_codes]
            limit_up, susp, liq, close_px = build_tradability(universe, xs, xe)
            fwd_mat = None
            for tag, k, kind in STRATS:
                score_fn = make_score(kind, val_piv, fcf_df, profit_df)
                strategy.emit_window_signals(
                    f"{tag}_{pool}", None, universe, cal, val_piv, xs, xe,
                    limit_up, susp, liq, fwd_mat, config.TOPK,
                    ic_records=None, rebalances={f"{tag}_{pool}": rebalances[(tag, pool)]},
                    win_name=win["name"], strict=True, min_cand=config.MIN_CAND,
                    score_fn=score_fn, topk_pick=k)
            del limit_up, susp, liq

    all_insts = sorted({i for tag in rebalances for _, tops in rebalances[tag]
                        for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()
    BT_START = pd.Timestamp(config.BT_START)

    rows, yearly = [], {}
    for (tag, pool), reb in rebalances.items():
        if not reb:
            continue
        rets, avg_to, n_buys = portfolio_backtest(reb, price_mat)
        rets = rets[rets.index >= BT_START]
        m = calc_metrics(rets, bench)
        label = f"{tag}_{'原池' if pool == 'orig' else '过滤池'}"
        rows.append({"策略": label, "年化收益": m["ar"], "年化波动": m["vol"],
                     "Sharpe": m["sharpe"], "最大回撤": m["mdd"],
                     "Calmar": m["calmar"], "换手率": avg_to,
                     "超额vsHS300": m.get("excess_ar", float("nan"))})
        yearly[label] = {yr: (1 + g).prod() - 1 for yr, g in rets.groupby(rets.index.year)}

    mdf = pd.DataFrame(rows).set_index("策略")
    print(f"\n{'='*100}\n  财务健康过滤对照 (原池 vs 过滤池, 2020~2026H1)\n{'='*100}")
    print(mdf.round(4).to_string())
    print(f"\n{'='*100}\n  分年收益\n{'='*100}")
    ydf = pd.DataFrame(yearly).T
    print(ydf.applymap(lambda x: f"{x:+.1%}").to_string())
    print(f"\n{'='*100}\n  每窗口剔除统计\n{'='*100}")
    print(pd.DataFrame(drop_stats).to_string(index=False))

    mdf.to_csv(f"{config.OUT_DIR}/fin_health_summary.csv", sep="\t")
    ydf.to_csv(f"{config.OUT_DIR}/fin_health_yearly.csv", sep="\t")
    print(f"\n[+] 已保存 {config.OUT_DIR}/fin_health_summary.csv / fin_health_yearly.csv",
          flush=True)


if __name__ == "__main__":
    main()
