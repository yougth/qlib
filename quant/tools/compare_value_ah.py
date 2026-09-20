"""对照实验: value_comp 纯A股 vs A+H (不训练模型, 只跑 value 系)
直接回答用户质疑: 加入港股池 value 收益是否提升?
跑两种模式:
  模式 A: INCLUDE_HK=False (纯A股)  → VAL20 / VAL10
  模式 B: INCLUDE_HK=True  (A+H)    → VAL20 / VAL10 / VHF / VHF10
复用 run_rolling 的 emit/backtest 逻辑, 但跳过模型训练.
"""
import os, sys, gc
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import core.config as config
from core import data as datalayer
from core import strategy
from core.universe import (build_windows, build_dynamic_universe,
                           format_qlib_code, load_pit_caches)
from core.dataset import build_dataset
from core.valuation import load_valuation, value_comp_score
from core.tradability import build_tradability
from core.backtest import portfolio_backtest, calc_metrics

OUT = config.OUT_DIR


def run_value_only(include_hk, sig_tag, hk_factor=None):
    """只跑 value 系策略 (VAL20/VAL10 + 可选 VHF), 跳过模型训练."""
    config.INCLUDE_HK = include_hk
    if not include_hk:
        config.INJECT_MARKET = False   # 纯A股下 is_hk 是常数列, 会触发死特征检查

    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    windows = build_windows()
    val_piv = load_valuation()

    # 仅 value 策略
    strats = ["VAL20", "VAL10"] + (["VHF", "VHF10"] if hk_factor else [])
    rebalances = {s: [] for s in strats}
    holdings_log = []
    ic_records = []

    for win in windows:
        y = win["year"]
        codes = build_dynamic_universe(y, fcf_df, profit_df)
        universe = [format_qlib_code(c) for c in codes]
        n_hk = sum(1 for c in universe if c.startswith("hk"))
        print(f"[{win['name']}] 池: {len(universe)} 只 (港股 {n_hk})", flush=True)
        xs, xe = win["test"]

        # value 系不需要特征矩阵, 直接打分
        limit_up, susp, liq, close_px = build_tradability(universe, xs, xe)
        fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)
        score_fns = {
            "VAL20": lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand),
            "VAL10": lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand),
            "VHF": lambda cand, dt: value_comp_score(cand, dt, val_piv,
                                                     hk_factor=config.VHF_FACTOR).reindex(cand),
            "VHF10": lambda cand, dt: value_comp_score(cand, dt, val_piv,
                                                       hk_factor=config.VHF_FACTOR).reindex(cand),
        }
        topk_of = {"VAL20": 20, "VAL10": 10, "VHF": 20, "VHF10": 10}
        for tag in strats:
            strategy.emit_window_signals(
                tag, None, universe, cal, val_piv, xs, xe,
                limit_up, susp, liq, fwd_mat, config.TOPK,
                ic_records=ic_records if tag in ("VAL20", "VHF") else None,
                rebalances=rebalances, win_name=win["name"], strict=True,
                min_cand=config.MIN_CAND,
                holdings_log=holdings_log,
                score_fn=score_fns[tag],
                topk_pick=topk_of[tag])
        gc.collect()

    # ---- 回测 ----
    all_insts = sorted({i for tag in rebalances for _, tops in rebalances[tag]
                        for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()
    BT_START = pd.Timestamp(config.BT_START)
    tag_lbl = {"VAL20": "value20", "VAL10": "value10", "VHF": "vhf20", "VHF10": "vhf10"}
    rows = []
    for tag in strats:
        rets, avg_to, n_buys = portfolio_backtest(rebalances[tag], price_mat)
        rets = rets[rets.index >= BT_START]
        m = calc_metrics(rets, bench)
        # 剔2020年化
        no20 = rets[rets.index >= pd.Timestamp("2021-01-01")]
        n20 = len(no20) / 244
        ar_no20 = (1 + no20).prod() ** (1 / n20) - 1 if n20 > 0 else np.nan
        rows.append({"策略": tag_lbl[tag], "年化": m.get("ar", np.nan),
                     "年化剔20": ar_no20,
                     "Sharpe": m.get("sharpe", np.nan),
                     "回撤": m.get("mdd", np.nan),
                     "换手": m.get("turnover", avg_to)})
        print(f"  [{tag_lbl[tag]}] 年化={m.get('ar',np.nan)*100:.2f}% "
              f"剔20={ar_no20*100:.2f}% "
              f"Sharpe={m.get('sharpe',np.nan):.2f} 回撤={m.get('mdd',np.nan)*100:.1f}% "
              f"换手={avg_to*100:.1f}%", flush=True)

    # 港股占比
    hk_share = {}
    for tag in strats:
        sub = [h for h in holdings_log if h["model"] == tag]
        tot = hk = 0
        for rec in sub:
            holds = str(rec["holdings"]).split(",")
            tot += len(holds)
            hk += sum(1 for c in holds if c.startswith("hk"))
        hk_share[tag_lbl[tag]] = hk / max(tot, 1)
    mode = "A+H" if include_hk else "纯A"
    print(f"\n=== [{mode}] 港股占比: { {k: f'{v*100:.1f}%' for k,v in hk_share.items()} }")
    return rows, hk_share


if __name__ == "__main__":
    import json
    result = {}
    for include_hk in (False, True):
        print(f"\n{'#'*70}\n# 模式: {'纯A股 (INCLUDE_HK=False)' if not include_hk else 'A+H (INCLUDE_HK=True)'}\n{'#'*70}")
        rows, hk_share = run_value_only(include_hk,
                                        "VAL", hk_factor=config.VHF_FACTOR if include_hk else None)
        result["A" if not include_hk else "AH"] = {"rows": rows, "hk_share": hk_share}
    print("\n\n===== 最终对照 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
