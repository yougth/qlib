#!/usr/bin/env python3
"""
run_vgh_topk —— VGH 持仓数量敏感性实验 (无 ML 训练, 纯价值打分)
================================================================================
目的: 回答"Top10 是否太少"。对 TOPK ∈ {10,15,20,30,50} 各跑一遍完整 7 窗口
滚动回测 (与 run_rolling.py 同口径: 同股票池/同一可交易门禁/同一成本模型),
比较 年化/Sharpe/回撤/换手/单票集中度, 找收益-分散化的最优持仓数。

无 ML 依赖: VGH 打分只用估值/FCF/利润缓存, 跳过 XGB/LGB/DE 训练 → 全程约 10 分钟。
输出: outputs/vgh_topk_summary.csv + 控制台对照表。
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
from core.valuation import load_valuation, value_hk_fcf_score
from core.tradability import build_tradability
from core.backtest import portfolio_backtest, calc_metrics

logging.getLogger("qlib").setLevel(logging.ERROR)

TOPK_LIST = [10, 15, 20, 30, 50]
OUT_DIR = config.OUT_DIR


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    val_piv = load_valuation()
    windows = build_windows()

    rebalances = {f"VGH{k}": [] for k in TOPK_LIST}
    holdings_log = []

    for win in windows:
        y = win["year"]
        xs, xe = win["test"]
        print(f"\n[{win['name']}] 池:{y-11}~{y-2} | 信号 {xs}~{xe}", flush=True)
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = [format_qlib_code(c) for c in codes]
        print(f"  股票池: {len(universe)} 只", flush=True)

        limit_up, susp, liq, close_px = build_tradability(universe, xs, xe)
        fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)

        def vgh_score(cand, dt):
            return value_hk_fcf_score(
                cand, dt, val_piv, fcf_df, profit_df,
                hk_factor=config.VHF_FACTOR,
                grow_w=config.VALUE_GROWTH_W).reindex(cand)

        # 各持仓数共用同一候选集与打分, 只改 topk_pick → 纯粹的"分散化"对照
        for k in TOPK_LIST:
            tag = f"VGH{k}"
            strategy.emit_window_signals(
                tag, None, universe, cal, val_piv, xs, xe,
                limit_up, susp, liq, fwd_mat, config.TOPK,
                ic_records=None, rebalances=rebalances,
                win_name=win["name"], strict=True, min_cand=config.MIN_CAND,
                holdings_log=holdings_log, score_fn=vgh_score,
                topk_pick=k)
        # fwd_mat 仅事后 IC 评估用, 及时释放
        del fwd_mat, limit_up, susp, liq

    # ---- 全期连续回测 ----
    all_insts = sorted({i for tag in rebalances for _, tops in rebalances[tag]
                        for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()

    BT_START = pd.Timestamp(config.BT_START)
    rows = []
    for k in TOPK_LIST:
        tag = f"VGH{k}"
        rets, avg_to, n_buys = portfolio_backtest(rebalances[tag], price_mat)
        rets = rets[rets.index >= BT_START]
        m = calc_metrics(rets, bench)
        rows.append({
            "策略": f"VGH Top{k}", "持仓数": k,
            "年化收益": m["ar"], "年化(剔2020)": None, "年化波动": m["vol"],
            "Sharpe": m["sharpe"], "最大回撤": m["mdd"], "Calmar": m["calmar"],
            "换手率": avg_to, "交易次数": n_buys,
            "单票权重": 1.0 / k,
            "超额vsHS300": m.get("excess_ar", float("nan"))})
    mdf = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n{'='*110}\n  VGH 持仓数量敏感性 (2020~2026H1, 同口径同成本)\n{'='*110}")
    print(mdf.round(3).to_string(index=False))

    # 分年收益对照
    print(f"\n{'='*110}\n  分年收益\n{'='*110}")
    yearly = {}
    for k in TOPK_LIST:
        rets, _, _ = portfolio_backtest(rebalances[f"VGH{k}"], price_mat)
        rets = rets[rets.index >= BT_START]
        yearly[f"Top{k}"] = {yr: (1 + g).prod() - 1
                            for yr, g in rets.groupby(rets.index.year)}
    ydf = pd.DataFrame(yearly).T
    print(ydf.applymap(lambda x: f"{x:+.1%}").to_string())

    mdf.to_csv(f"{OUT_DIR}/vgh_topk_summary.csv", sep="\t", index=False)
    pd.DataFrame(holdings_log).to_csv(f"{OUT_DIR}/vgh_topk_holdings.csv",
                                      sep="\t", index=False)
    print(f"\n[+] 已保存: {OUT_DIR}/vgh_topk_summary.csv / vgh_topk_holdings.csv", flush=True)


if __name__ == "__main__":
    main()
