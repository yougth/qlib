#!/usr/bin/env python3
"""
run_rolling —— 滚动十年双正 × XGB/LGB/ENS 双模型 × 月频Top10 回测 (产出 24.88%)
================================================================================
本脚本仅做编排, 全部无穿越逻辑复用 core/ 公共层 (信号层见 core.strategy)。
行为与重构前 archive/legacy/rolling10y_dual_model.py 等价。
不穿越设计见 core.dataset / core.universe / core.features 各模块 docstring。
"""
import os
import sys
import gc
import random
import warnings
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=42):
    random.seed(seed)
    import numpy as _np
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
warnings.filterwarnings("ignore")

import pandas as pd
import xgboost as xgb

from core import config
from core import data as datalayer
from core import strategy
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.valuation import load_valuation, value_comp_score
from core.dataset import build_dataset
from core.models import RankICEval, train_xgb, train_lgb, check_convergence
from core.tradability import build_tradability
from core.backtest import (portfolio_backtest, calc_metrics, annualized_since,
                           print_yearly_table, print_metric_table)

logging.getLogger("qlib").setLevel(logging.ERROR)

TOPK = config.TOPK
MIN_CAND = config.MIN_CAND
LABEL_HORIZON = config.LABEL_HORIZON
STRATS = config.STRATS
OUT_DIR = config.OUT_DIR


def run_window(win, fcf_df, profit_df, cal, results, rebalances, holdings_log,
               val_piv, ic_records):
    y = win["year"]
    ts, te = win["train"]
    vs, ve = win["valid"]
    xs, xe = win["test"]
    print(f"\n{'='*70}\n[{win['name']}] 池:{y-11}~{y-2}十年双正 | train {ts}~{te} | "
          f"valid {vs}~{ve}(embargo) | 信号 {xs}~{xe}\n{'='*70}", flush=True)

    codes = build_dynamic_universe(y, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    print(f"  股票池: {len(universe)} 只", flush=True)
    # 行情覆盖率门禁: 池成员若无行情会在 build_candidates 被静默剔除 → 选择偏差
    datalayer.assert_market_coverage(universe, xs, win["bt_end"], tag=win["name"])

    ds = build_dataset(win, universe, fcf_df, profit_df, cal)
    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_va, y_va = ds["X_va"], ds["y_va"]
    test_X = ds["test_X"]
    n_inst = ds["n_inst"]

    va_dates = X_va.index.get_level_values(0).values
    ic_eval = RankICEval(va_dates, y_va.values)

    models = {}
    for tag, trainer in [("XGB", train_xgb), ("LGB", train_lgb)]:
        m, best_iter, valid_ic = trainer(X_tr, y_tr, X_va, y_va, ic_eval)
        check_convergence(f"{win['name']}-{tag}", best_iter, valid_ic)
        models[tag] = m
        results.append({"window": win["name"], "year": y, "model": tag,
                        "pool_size": len(universe), "n_inst": n_inst,
                        "best_iter": best_iter, "valid_rank_ic": round(valid_ic, 4)})

    # 测试段预测
    preds = {}
    preds["XGB"] = pd.Series(models["XGB"].predict(xgb.DMatrix(test_X.values)),
                             index=test_X.index)
    preds["LGB"] = pd.Series(models["LGB"].predict(
        test_X.values, num_iteration=models["LGB"].best_iteration), index=test_X.index)
    del X_tr, X_va, y_tr, y_va, test_X, models
    gc.collect()

    # 可交易性与流动性 (仅测试段)
    limit_up, susp, liq = build_tradability(universe, xs, xe)

    # ---- OOS IC 用: 信号段 20日前瞻真实收益矩阵 (date × instrument) ----
    fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)

    # ENS = XGB/LGB 截面 rank 均值; VALUE 用 value_comp。
    # 六个策略共用同一可交易候选集 (model-agnostic 过滤), 因此逐策略调用
    # core.strategy 时得到的候选完全一致, 差别只在打分。
    score_fns = {
        "XGB": lambda cand, dt: preds["XGB"].xs(dt, level=0).reindex(cand),
        "LGB": lambda cand, dt: preds["LGB"].xs(dt, level=0).reindex(cand),
        "ENS": lambda cand, dt: (
            preds["XGB"].xs(dt, level=0).reindex(cand).rank(pct=True)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True)) / 2,
        "VAL20": lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand),
        "VAL10": lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand),
        "POOL_EW": lambda cand, dt: pd.Series(1.0, index=cand),
    }
    TOPK_OF = {"XGB": TOPK, "LGB": TOPK, "ENS": TOPK, "VAL20": 20, "VAL10": 10,
               "POOL_EW": TOPK}
    # IC 聚合时 VAL10/VAL20 是同一打分, 统一记为 VALUE (与旧版 ic_of 映射一致);
    # VAL10 与 POOL_EW 不再重复记 IC (VAL10 打分与 VAL20 完全相同, POOL_EW 无排序)
    IC_MODEL = {"VAL20": "VALUE"}
    IC_ON = ("XGB", "LGB", "ENS", "VAL20")

    # 预测截面完整性: 缺任一信号日截面就直接失败, 不允许静默少跑一个月
    for tag in ("XGB", "LGB"):
        miss = [d for d in strategy.signal_days(cal, xs, xe)
                if d not in preds[tag].index.get_level_values(0)]
        if miss:
            raise RuntimeError(f"[CHECK] {win['name']} {tag} 缺 {len(miss)} 个信号日"
                               f"预测截面, 首个 {miss[0].date()}!")

    for tag in STRATS:
        # 候选集门禁按 TOPK 判定 (与旧版一致: 所有策略共用 TOPK 下限),
        # 真正取几只由 TOPK_OF 决定
        strategy.emit_window_signals(
            tag, preds["XGB"], universe, cal, val_piv, xs, xe,
            limit_up, susp, liq, fwd_mat, TOPK,
            ic_records=ic_records if tag in IC_ON else None,
            rebalances=rebalances, win_name=win["name"], strict=True,
            min_cand=MIN_CAND,
            holdings_log=holdings_log if tag in ("XGB", "LGB", "ENS", "VAL20") else None,
            score_fn=score_fns[tag], ic_model=IC_MODEL.get(tag),
            topk_pick=TOPK_OF[tag])
    del preds
    gc.collect()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()

    windows = build_windows()
    results, holdings_log, ic_records = [], [], []
    rebalances = {s: [] for s in STRATS}
    val_piv = load_valuation()
    for win in windows:
        run_window(win, fcf_df, profit_df, cal, results, rebalances, holdings_log,
                   val_piv, ic_records)

    # ---- 全期连续回测 (跨窗口换手自然衔接) ----
    all_insts = sorted({i for tag in rebalances for _, tops in rebalances[tag]
                        for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()

    # ---- OOS 信号期 IC / RankIC 聚合 (逐信号日截面, 跨窗口取均值) ----
    ic_df = pd.DataFrame(ic_records)
    ic_agg = ic_df.groupby("model")[["ic", "rank_ic"]].mean().to_dict("index") if len(ic_df) else {}

    def ic_of(strat):
        key = "VALUE" if strat in ("VAL10", "VAL20") else strat
        d = ic_agg.get(key)
        return (d["ic"], d["rank_ic"]) if d else (float("nan"), float("nan"))

    BT_START = pd.Timestamp(config.BT_START)
    ALIGN_START = pd.Timestamp(config.ALIGN_START)
    LABEL = {"XGB": "XGB Top10", "LGB": "LGB Top10", "ENS": "集成ENS Top10",
             "VAL20": "value_comp Top20", "VAL10": "value_comp Top10", "POOL_EW": "十年双正池等权"}

    # ---- 逐策略回测 + 全指标 ----
    nav_out, metric_rows, yearly_all = {}, [], {}
    for tag in STRATS:
        rets, avg_to, n_buys = portfolio_backtest(rebalances[tag], price_mat)
        rets = rets[rets.index >= BT_START]
        m = calc_metrics(rets, bench)
        ic, rankic = ic_of(tag)
        nav_out[tag] = (1 + rets).cumprod()
        metric_rows.append({
            "策略": LABEL[tag], "年化收益": m["ar"], "年化(剔2020)": annualized_since(rets, ALIGN_START),
            "年化波动": m["vol"], "Sharpe": m["sharpe"], "最大回撤": m["mdd"],
            "Calmar": m["calmar"], "IC": ic, "RankIC": rankic,
            "换手率": avg_to, "交易次数": n_buys, "超额vsHS300": m.get("excess_ar", float("nan"))})
        yearly_all[tag] = {yr: (1 + g).prod() - 1 for yr, g in rets.groupby(rets.index.year)}

    bench_bt = bench[bench.index >= BT_START]
    mb = calc_metrics(bench_bt)
    yearly_all["HS300"] = {yr: (1 + g).prod() - 1 for yr, g in bench_bt.groupby(bench_bt.index.year)}
    metric_rows.append({
        "策略": "沪深300基准", "年化收益": mb["ar"], "年化(剔2020)": annualized_since(bench_bt, ALIGN_START),
        "年化波动": mb["vol"], "Sharpe": mb["sharpe"], "最大回撤": mb["mdd"],
        "Calmar": mb["calmar"], "IC": float("nan"), "RankIC": float("nan"),
        "换手率": 0.0, "交易次数": 0, "超额vsHS300": 0.0})

    # ==================== 标准输出 (分年 + 图1九指标 + 对照结论) ====================
    order = list(STRATS) + ["HS300"]
    labels = {**LABEL, "HS300": "沪深300基准"}
    print_yearly_table(yearly_all, order, labels,
                       title="一、分年收益 (2020~2026H1)")
    mdf = pd.DataFrame(metric_rows)
    print_metric_table(mdf, title="二、整体指标 (图1口径; 年化(剔2020)对齐 value_comp 冻结基线)")

    ens = mdf.set_index("策略").loc["集成ENS Top10"]
    v20 = mdf.set_index("策略").loc["value_comp Top20"]
    print(f"\n{'='*96}\n  三、模型 vs value_comp 对照\n{'='*96}", flush=True)
    print(f"  · 全期(含2020)  : ENS年化 {ens['年化收益']*100:.2f}%  vs  value_comp20 {v20['年化收益']*100:.2f}%", flush=True)
    print(f"  · 对齐(剔2020)  : ENS年化 {ens['年化(剔2020)']*100:.2f}%  vs  value_comp20 {v20['年化(剔2020)']*100:.2f}%"
          f"   ← 与冻结基线(25.75%)同口径", flush=True)
    print(f"  · 风险调整      : ENS Sharpe {ens['Sharpe']:.2f}/Calmar {ens['Calmar']:.2f}  vs  "
          f"value_comp20 Sharpe {v20['Sharpe']:.2f}/Calmar {v20['Calmar']:.2f}", flush=True)
    print(f"  · ENS 是 Top10(集中度2倍于Top20); OOS RankIC ENS={ens['RankIC']:.4f} value={v20['RankIC']:.4f}", flush=True)

    # ---- 保存 ----
    pd.DataFrame(results).to_csv(f"{OUT_DIR}/rolling10y_train_diag.csv", sep="\t", index=False)
    mdf.to_csv(f"{OUT_DIR}/rolling10y_summary.csv", sep="\t", index=False)
    pd.DataFrame(holdings_log).to_csv(f"{OUT_DIR}/rolling10y_holdings.csv", sep="\t", index=False)
    pd.DataFrame(yearly_all).T.to_csv(f"{OUT_DIR}/rolling10y_yearly.csv", sep="\t")
    if len(ic_df):
        ic_df.to_csv(f"{OUT_DIR}/rolling10y_ic.csv", sep="\t", index=False)
    nav_df = pd.DataFrame(nav_out)
    nav_df["HS300"] = (1 + bench_bt).cumprod().reindex(nav_df.index)
    nav_df.to_csv(f"{OUT_DIR}/rolling10y_nav.csv", sep="\t")
    print(f"\n[+] 结果已保存至 {OUT_DIR}: rolling10y_summary.csv / _yearly.csv / _ic.csv / "
          f"_train_diag.csv / _holdings.csv / _nav.csv", flush=True)


if __name__ == "__main__":
    main()
