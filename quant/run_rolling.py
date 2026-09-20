#!/usr/bin/env python3
"""
run_rolling —— 滚动十年双正 × XGB/LGB/DE 三模型 × 月频Top10 回测
================================================================================
本脚本仅做编排, 全部无穿越逻辑复用 core/ 公共层 (信号层见 core.strategy)。
行为与重构前 archive/legacy/rolling10y_dual_model.py 等价 (DE 为新增)。
不穿越设计见 core.dataset / core.universe / core.features 各模块 docstring。
ENS = XGB + LGB + DE 三模型截面 rank 百分位均值融合 (降噪 + 多样性)。
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

import numpy as np
import pandas as pd
import xgboost as xgb

from core import config
from core import data as datalayer
from core import strategy
from core.strategy import emit_vghx_signals
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.valuation import (load_valuation, value_comp_score, value_growth_score,
                            value_hk_fcf_score)
from core.dataset import build_dataset
from core.models import RankICEval, train_xgb, train_lgb, train_de, check_convergence
from core.tradability import build_tradability
from core.backtest import (portfolio_backtest, calc_metrics, annualized_since,
                           print_yearly_table, print_metric_table, attrib_returns)

logging.getLogger("qlib").setLevel(logging.ERROR)


def _pct2(x):
    return "   n/a  " if pd.isna(x) else f"{x*100:7.2f}%"


TOPK = config.TOPK
MIN_CAND = config.MIN_CAND
LABEL_HORIZON = config.LABEL_HORIZON
STRATS = config.STRATS
OUT_DIR = config.OUT_DIR


def search_valf_w(models, X_va, universe, cal, win, val_piv):
    """融合版 VALF 的 w 在 valid 段寻优 (纪律: w 不在 OOS 上扫, OOS 只验一次)。

    用各窗口自己的模型对 valid 段做推理 (早停用的同一段, 模型没见过 valid 标签之外
    的未来), 对每个 w 算月度截面 RankIC((1-w)*value_rank + w*model_rank vs 前瞻收益),
    取均值最高者; 平手取更小 w (value 主导更保守)。
    返回 (best_w, [(w, mean_ic, n_months), ...])。
    """
    vs, ve = win["valid"]
    va_pred = {}
    va_pred["XGB"] = pd.Series(
        models["XGB"].predict(xgb.DMatrix(X_va.values),
                              iteration_range=(0, models["XGB"]._rankic_best_iter + 1)),
        index=X_va.index)
    va_pred["LGB"] = pd.Series(
        models["LGB"].predict(X_va.values, num_iteration=models["LGB"].best_iteration),
        index=X_va.index)
    va_pred["DE"] = pd.Series(models["DE"].predict(X_va), index=X_va.index)
    va_ens = (va_pred["XGB"].groupby(level=0).rank(pct=True)
              + va_pred["LGB"].groupby(level=0).rank(pct=True)
              + va_pred["DE"].groupby(level=0).rank(pct=True)) / 3
    fwd = datalayer.forward_return_matrix(universe, vs, ve)
    days = [d for d in strategy.signal_days(cal, vs, ve)
            if d in va_ens.index.get_level_values(0) and d in fwd.index]
    scores = []
    for w in config.VALF_W_GRID:
        ics = []
        for dt in days:
            ens = va_ens.xs(dt, level=0)
            v = value_comp_score(list(ens.index), dt, val_piv)
            fused = (1 - w) * v + w * ens.rank(pct=True)
            f = fwd.loc[dt].reindex(fused.index)
            pair = pd.concat([fused.rename("s"), f.rename("r")], axis=1).dropna()
            if len(pair) >= 30:
                ics.append(pair["s"].corr(pair["r"], method="spearman"))
        scores.append((w, float(np.mean(ics)) if ics else float("nan"), len(ics)))
    ok = [(w, ic, n) for w, ic, n in scores if not pd.isna(ic)]
    if not ok:
        return config.VALF_W_GRID[0], scores
    best = sorted(ok, key=lambda t: (-t[1], t[0]))[0][0]
    return best, scores


def run_window(win, fcf_df, profit_df, cal, results, rebalances, holdings_log,
               val_piv, ic_records, x15t_prev=None, icw15_prev=None,
               fi_records=None, bw_prev=None):
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
    valid_ics = {}  # 路径1: 存各模型 valid RankIC, 用于 ICW 动态加权
    for tag, trainer in [("XGB", train_xgb), ("LGB", train_lgb), ("DE", train_de)]:
        m, best_iter, valid_ic = trainer(X_tr, y_tr, X_va, y_va, ic_eval)
        check_convergence(f"{win['name']}-{tag}", best_iter, valid_ic)
        models[tag] = m
        valid_ics[tag] = valid_ic
        results.append({"window": win["name"], "year": y, "model": tag,
                        "pool_size": len(universe), "n_inst": n_inst,
                        "best_iter": best_iter, "valid_rank_ic": round(valid_ic, 4)})

    # 测试段预测
    preds = {}
    # ---- Feature Importance (XGB gain): 跨窗口聚合, 查噪声因子是否带偏权重 ----
    # DMatrix 未传特征名 → key 是 f0/f1/..., 用 feat_cols 映射回列名
    if fi_records is not None:
        try:
            gain = models["XGB"].get_score(importance_type="gain")
            total = sum(gain.values()) or 1.0
            fc = ds["feat_cols"]
            for k, v in gain.items():
                if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(fc):
                    fi_records.append({"year": y, "feature": fc[int(k[1:])],
                                       "gain_share": v / total})
        except Exception as e:  # 诊断信息, 不允许阻断回测主线
            print(f"    [FI] 特征重要性采集失败: {e}", flush=True)
    xgb_best = models["XGB"]._rankic_best_iter
    preds["XGB"] = pd.Series(
        models["XGB"].predict(xgb.DMatrix(test_X.values),
                              iteration_range=(0, xgb_best + 1)),
        index=test_X.index)
    preds["LGB"] = pd.Series(models["LGB"].predict(
        test_X.values, num_iteration=models["LGB"].best_iteration), index=test_X.index)
    preds["DE"] = pd.Series(models["DE"].predict(test_X), index=test_X.index)

    # ---- 块2实验A: 融合版 VALF 的 w 在 valid 段寻优 (models 删除前做 valid 推理) ----
    valf_w, valf_scores = config.VALF_W_GRID[0], []
    if "VALF" in STRATS:
        try:
            valf_w, valf_scores = search_valf_w(models, X_va, universe, cal, win, val_piv)
            diag = "  ".join(f"w={w:.2f}:{ic if pd.isna(ic) else round(ic, 4)}({n}月)"
                             for w, ic, n in valf_scores)
            print(f"    [VALF] {win['name']} valid 段 w 寻优 → w={valf_w}  [{diag}]",
                  flush=True)
            results.append({"window": win["name"], "year": y, "model": "VALF",
                            "pool_size": len(universe), "n_inst": n_inst,
                            "best_iter": valf_w, "valid_rank_ic": round(
                                max((ic for _, ic, _ in valf_scores
                                     if not pd.isna(ic)), default=float("nan")), 4)})
        except Exception as e:  # 寻优失败不阻断主线, 回退保守 w
            valf_w = config.VALF_W_GRID[0]
            print(f"    [VALF] {win['name']} w 寻优失败回退 w={valf_w}: {e}", flush=True)
    del X_tr, X_va, y_tr, y_va, test_X, models
    gc.collect()

    # 可交易性与流动性 (仅测试段)
    limit_up, susp, liq, close_px = build_tradability(universe, xs, xe)

    # ---- OOS IC 用: 信号段 20日前瞻真实收益矩阵 (date × instrument) ----
    fwd_mat = datalayer.forward_return_matrix(universe, xs, xe)

    # ENS = XGB/LGB/DE 截面 rank 等权均值; DL = DE+LGB 等权融合;
    # ICW = XGB/LGB/DE 按 valid RankIC 动态加权 (路径1: 某窗口谁强给谁更高权重);
    # DL_T = DL打分 + 择时仓位 (路径3, 熊市降仓位, 仓位在回测层实现);
    # VALUE 用 value_comp。所有策略共用同一可交易候选集。
    ic_xgb = max(valid_ics.get("XGB", 0), 0.01)
    ic_lgb = max(valid_ics.get("LGB", 0), 0.01)
    ic_de = max(valid_ics.get("DE", 0), 0.01)
    ic_sum = ic_xgb + ic_lgb + ic_de

    # ENS 预测序列 (逐日截面三模型 rank 均值), 供级联策略 VGC/VHC 用
    ens_pred = (preds["XGB"].groupby(level=0).rank(pct=True)
                + preds["LGB"].groupby(level=0).rank(pct=True)
                + preds["DE"].groupby(level=0).rank(pct=True)) / 3

    # ICW 预测序列 (逐日截面三模型 rank 按 valid IC 加权), 供 ICV 横向融合用
    icw_pred = (preds["XGB"].groupby(level=0).rank(pct=True) * (ic_xgb / ic_sum)
                + preds["LGB"].groupby(level=0).rank(pct=True) * (ic_lgb / ic_sum)
                + preds["DE"].groupby(level=0).rank(pct=True) * (ic_de / ic_sum))

    score_fns = {
        "XGB": lambda cand, dt: preds["XGB"].xs(dt, level=0).reindex(cand),
        "LGB": lambda cand, dt: preds["LGB"].xs(dt, level=0).reindex(cand),
        "DE": lambda cand, dt: preds["DE"].xs(dt, level=0).reindex(cand),
        "ENS": lambda cand, dt: (
            preds["XGB"].xs(dt, level=0).reindex(cand).rank(pct=True)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True)
            + preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True)) / 3,
        "DL": lambda cand, dt: (
            preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True)) / 2,
        "ICW": lambda cand, dt: (
            preds["XGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_xgb / ic_sum)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_lgb / ic_sum)
            + preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_de / ic_sum)),
        "DL_T": lambda cand, dt: (
            preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True)) / 2,
        "X15T": lambda cand, dt: preds["XGB"].xs(dt, level=0).reindex(cand),
        "ICW15": lambda cand, dt: (
            preds["XGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_xgb / ic_sum)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_lgb / ic_sum)
            + preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_de / ic_sum)),
        "ICW_T": lambda cand, dt: (
            preds["XGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_xgb / ic_sum)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_lgb / ic_sum)
            + preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_de / ic_sum)),
        "ICW_BW": lambda cand, dt: (
            preds["XGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_xgb / ic_sum)
            + preds["LGB"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_lgb / ic_sum)
            + preds["DE"].xs(dt, level=0).reindex(cand).rank(pct=True) * (ic_de / ic_sum)),
        "VAL20": lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand),
        "VAL10": lambda cand, dt: value_comp_score(cand, dt, val_piv).reindex(cand),
        # 块2实验C: 排雷版 VALX —— value 打分后, 模型分最低 30% 候选被否决重选
        "VALX": lambda cand, dt: strategy.veto_score(
            value_comp_score(cand, dt, val_piv).reindex(cand),
            ens_pred, dt, config.VALX_VETO),
        # 块2实验A: 融合版 VALF —— (1-w)*value_rank + w*model_rank (w=valid寻优)
        "VALF": lambda cand, dt: strategy.blend_score(
            value_comp_score(cand, dt, val_piv).reindex(cand),
            ens_pred, dt, valf_w),
        "VHF": lambda cand, dt: value_comp_score(cand, dt, val_piv,
                                                 hk_factor=config.VHF_FACTOR).reindex(cand),
        "VHF10": lambda cand, dt: value_comp_score(cand, dt, val_piv,
                                                   hk_factor=config.VHF_FACTOR).reindex(cand),
        "VG": lambda cand, dt: value_growth_score(
            cand, dt, val_piv, fcf_df, profit_df,
            hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W,
            q_w=config.VALUE_QUALITY_W).reindex(cand),
        "VGF": lambda cand, dt: value_growth_score(
            cand, dt, val_piv, fcf_df, profit_df,
            hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W,
            q_w=config.VALUE_QUALITY_W).reindex(cand),
        "VGH": lambda cand, dt: value_hk_fcf_score(
            cand, dt, val_piv, fcf_df, profit_df,
            hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W).reindex(cand),
        # 级联策略: 基本面底仓 Top30 → ENS 精排 → Top10 (持仓 ⊆ 基本面安全集)
        "VGC": lambda cand, dt: strategy.cascade_score(
            value_growth_score(cand, dt, val_piv, fcf_df, profit_df,
                               hk_factor=config.VHF_FACTOR,
                               grow_w=config.VALUE_GROWTH_W,
                               q_w=config.VALUE_QUALITY_W).reindex(cand),
            ens_pred, dt, config.VGC_BASE_TOPK),
        "VHC": lambda cand, dt: strategy.cascade_score(
            value_hk_fcf_score(cand, dt, val_piv, fcf_df, profit_df,
                               hk_factor=config.VHF_FACTOR,
                               grow_w=config.VALUE_GROWTH_W).reindex(cand),
            ens_pred, dt, config.VGC_BASE_TOPK),
        "POOL_EW": lambda cand, dt: pd.Series(1.0, index=cand),
    }
    TOPK_OF = {"XGB": TOPK, "LGB": TOPK, "DE": TOPK, "ENS": TOPK, "DL": TOPK,
               "ICW": TOPK, "DL_T": TOPK, "X15T": config.X15T_TOPK,
               "VAL20": 20, "VAL10": 10, "VALX": 10, "VALF": 10,
               "VHF": 20, "VHF10": 10,
               "VG": 10, "VGF": 20, "VGH": 10, "VGHX": 10,
               "VGC": 10, "VHC": 10, "ICV": 10, "ICW_T": TOPK,
               "ICW15": config.ICW15_TOPK, "ICW_BW": TOPK, "POOL_EW": TOPK}
    # IC 聚合时 VAL10/VAL20 是同一打分, 统一记为 VALUE (与旧版 ic_of 映射一致);
    # VAL10 与 POOL_EW 不再重复记 IC (VAL10 打分与 VAL20 完全相同, POOL_EW 无排序)
    IC_MODEL = {"VAL20": "VALUE", "VGF": "VG"}
    IC_ON = ("XGB", "LGB", "DE", "ENS", "DL", "ICW", "DL_T", "X15T",
             "VAL20", "VALX", "VALF", "VG", "VGH", "VGHX", "VGC", "VHC",
             "ICV", "ICW_T", "ICW15", "ICW_BW")

    # 预测截面完整性: 缺任一信号日截面就直接失败, 不允许静默少跑一个月
    for tag in ("XGB", "LGB", "DE"):
        miss = [d for d in strategy.signal_days(cal, xs, xe)
                if d not in preds[tag].index.get_level_values(0)]
        if miss:
            raise RuntimeError(f"[CHECK] {win['name']} {tag} 缺 {len(miss)} 个信号日"
                               f"预测截面, 首个 {miss[0].date()}!")

    for tag in STRATS:
        # 候选集门禁按 TOPK 判定 (与旧版一致: 所有策略共用 TOPK 下限),
        # 真正取几只由 TOPK_OF 决定
        kwargs = {}
        if tag in ("X15T", "ICW15"):
            buf = config.X15T_BUFFER if tag == "X15T" else config.ICW15_BUFFER
            kwargs = {"turnover_buffer": buf,
                      "hysteresis_band": config.X15T_HYSTERESIS,
                      "prev_holdings": x15t_prev if tag == "X15T" else icw15_prev}
        if tag == "ICW_BW":
            kwargs = {"freq": "biweekly"}
            if config.ICW_BW_HYSTERESIS > 0:
                kwargs["hysteresis_band"] = config.ICW_BW_HYSTERESIS
                kwargs["prev_holdings"] = bw_prev
        if tag == "VGHX":
            # 方向2(改): VGH 与 XGB 横向 Rank 融合 (Z-Score 0.5/0.5, 不截断)
            vghx_score = lambda cand, dt: value_hk_fcf_score(
                cand, dt, val_piv, fcf_df, profit_df,
                hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W)
            emit_vghx_signals(
                tag, preds["XGB"], universe, cal, val_piv, xs, xe,
                limit_up, susp, liq, fwd_mat, TOPK,
                vgh_score_fn=vghx_score, base_topk=config.VGHX_BASE_TOPK,
                ic_records=ic_records if tag in IC_ON else None,
                rebalances=rebalances, win_name=win["name"], strict=True,
                min_cand=MIN_CAND, holdings_log=holdings_log,
                ic_model=IC_MODEL.get(tag), blend_w=config.VGHX_BLEND_W, px=close_px)
            continue
        if tag == "ICV":
            # ICV: VGH + ICW 横向 Rank 融合 (Z-Score, 不截断, 复用 emit_vghx_signals)
            icv_score = lambda cand, dt: value_hk_fcf_score(
                cand, dt, val_piv, fcf_df, profit_df,
                hk_factor=config.VHF_FACTOR, grow_w=config.VALUE_GROWTH_W)
            emit_vghx_signals(
                tag, icw_pred, universe, cal, val_piv, xs, xe,
                limit_up, susp, liq, fwd_mat, TOPK,
                vgh_score_fn=icv_score, base_topk=config.VGHX_BASE_TOPK,
                ic_records=ic_records if tag in IC_ON else None,
                rebalances=rebalances, win_name=win["name"], strict=True,
                min_cand=MIN_CAND, holdings_log=holdings_log,
                ic_model=IC_MODEL.get(tag), blend_w=config.ICV_BLEND_W, px=close_px)
            continue
        # value 系策略(VG/VGF/VGH/VAL/VHF)及级联系(VGC/VHC)用全池候选(pred=None),
        # 不被 XGB 预测截面限制. 级联底仓来自全池 value 打分, ML 预测在 cascade_score
        # 内部按需 reindex (底仓成员缺预测给最低 rank, 不会被静默剔除)。
        # 修正(2026-08-21): 之前全传 preds["XGB"] 导致 value 系候选被模型截面裁剪,
        # VGH 全池 19.32% vs XGB截面 15.81%, 差异即来自候选集大小
        pred_arg = None if tag in ("VGH", "VG", "VGF", "VAL10", "VAL20",
                                   "VHF", "VHF10", "VGC", "VHC",
                                   "VALX", "VALF") else preds["XGB"]
        # ICW15 用 ICW 打分 (score_fn 已设), pred_arg 用 XGB 截面作为候选基础
        strategy.emit_window_signals(
            tag, pred_arg, universe, cal, val_piv, xs, xe,
            limit_up, susp, liq, fwd_mat, TOPK,
            ic_records=ic_records if tag in IC_ON else None,
            rebalances=rebalances, win_name=win["name"], strict=True,
            min_cand=MIN_CAND,
            holdings_log=holdings_log if tag in ("XGB", "LGB", "DE", "ENS", "DL", "ICW", "DL_T", "X15T", "VAL20", "VHF", "VHF10", "VG", "VGF", "VGH", "VGHX", "VGC", "VHC", "ICV", "ICW_T", "ICW15", "ICW_BW", "VALX", "VALF") else None,
            score_fn=score_fns[tag], ic_model=IC_MODEL.get(tag),
            topk_pick=TOPK_OF[tag], px=close_px, **kwargs)
    del preds
    gc.collect()

    # 返回 X15T/ICW15/BW 期末持仓, 供下一窗口延续换手缓冲
    icw15_ret = rebalances.get("ICW15", [None])
    icw15_ret = icw15_ret[-1][1] if icw15_ret and icw15_ret[0] is not None else icw15_prev
    bw_ret = rebalances.get("ICW_BW", [None])
    bw_ret = bw_ret[-1][1] if bw_ret and bw_ret[0] is not None else bw_prev
    return (rebalances["X15T"][-1][1] if rebalances["X15T"] else x15t_prev,
            icw15_ret, bw_ret)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()

    windows = build_windows()
    results, holdings_log, ic_records = [], [], []
    fi_records = []  # XGB gain 重要性 (逐窗口), 聚合输出见报表"七"
    rebalances = {s: [] for s in STRATS}
    val_piv = load_valuation()
    x15t_prev, icw15_prev, bw_prev = None, None, None
    for win in windows:
        x15t_prev, icw15_prev, bw_prev = run_window(win, fcf_df, profit_df, cal, results,
                                            rebalances, holdings_log, val_piv,
                                            ic_records, x15t_prev, icw15_prev,
                                            fi_records, bw_prev)

    # ---- 全期连续回测 (跨窗口换手自然衔接) ----
    all_insts = sorted({i for tag in rebalances for _, tops in rebalances[tag]
                        for i in tops})
    price_mat = datalayer.load_price_matrix(all_insts)
    bench = datalayer.load_benchmark()

    # ---- OOS 信号期 IC / RankIC 聚合 (逐信号日截面, 跨窗口取均值) ----
    ic_df = pd.DataFrame(ic_records)
    ic_agg = ic_df.groupby("model")[["ic", "rank_ic"]].mean().to_dict("index") if len(ic_df) else {}

    def ic_of(strat):
        key = ("VALUE" if strat in ("VAL10", "VAL20")
               else "VG" if strat in ("VGF",) else strat)
        d = ic_agg.get(key)
        return (d["ic"], d["rank_ic"]) if d else (float("nan"), float("nan"))

    BT_START = pd.Timestamp(config.BT_START)
    ALIGN_START = pd.Timestamp(config.ALIGN_START)
    LABEL = {"XGB": "XGB Top10", "LGB": "LGB Top10", "DE": "DE Top10",
             "ENS": "集成ENS Top10", "DL": "DE+LGB Top10",
             "ICW": "IC加权 Top10", "DL_T": "DL+择时 Top10", "X15T": "XGB15+缓冲",
             "VHF": "VHF Top20", "VHF10": "VHF Top10",
             "VG": "value+盈利 Top10", "VGF": "value+盈利 Top20",
             "VGH": "结构化剥离 Top10", "VGHX": "VGH+XGB Top10",
             "VGC": "VG底仓+ENS精排", "VHC": "VGH底仓+ENS精排",
             "ICV": "VGH+ICW融合", "ICW_T": "ICW+择时 Top10", "ICW15": "ICW15+缓冲",
             "ICW_BW": "ICW双周 Top10",
             "VAL20": "value_comp Top20", "VAL10": "value_comp Top10", "POOL_EW": "十年双正池等权",
             "VALX": "排雷版VAL10", "VALF": "融合版VAL10"}

    # ---- 路径3: 择时仓位信号 (沪深300 MA60 趋势, PIT: 仅用 <= 信号日的数据) ----
    # 信号日 D: 调仓执行日的趋势判断决定该调仓区间的仓位 (熊市半仓)
    # MA60 用 csi300_cache 的日收盘价计算, 不含未来信息
    bench_close = (1 + bench).cumprod()  # 从日收益还原收盘价序列
    bench_ma = bench_close.rolling(config.TIMING_MA, min_periods=20).mean()

    def band_bear(hs, ma, enter_pct, hyst_days):
        """带宽过滤 + 迟滞确认的熊市状态序列 (True=熊市)。

        纯日频 MA60 交叉噪声大 (6.5年98次切换), 用两条规则压频:
          入熊: 收盘价 < MA*(1-enter_pct) 连续 hyst_days 日 (跌破均线下沿3%才算)
          出熊: 收盘价重新站上 MA (回到带宽上沿)
        """
        below = hs < ma
        deep = hs < ma * (1 - enter_pct)
        bear = pd.Series(False, index=hs.index)
        state = False
        run = 0  # 连续 deep 日计数
        for i, dt in enumerate(hs.index):
            if not state:
                run = run + 1 if deep.iloc[i] else 0
                if run >= hyst_days:
                    state = True
            elif not below.iloc[i]:
                state = False
                run = 0
            bear.iloc[i] = state
        return bear

    # 逐日择时信号: True=牛市满仓, False=熊市半仓
    timing_signal = ~band_bear(bench_close, bench_ma,
                               config.TIMING_BANDWIDTH, config.TIMING_HYSTERESIS)

    def apply_timing(rets, rebal_dates, timing_sig, bear_pos):
        """按调仓区间应用择时仓位: 每个执行日用当日趋势信号, 持续到下一执行日。

        仓位在整段持有期内生效 (1-pos) 为现金收益 0; 信号只用执行日及之前数据, 不穿越。
        """
        adjusted = rets.copy()
        for i, exec_dt in enumerate(rebal_dates):
            if exec_dt not in timing_sig.index:
                continue
            is_bull = timing_sig.loc[exec_dt]
            pos = 1.0 if is_bull else bear_pos
            next_dt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
            mask = adjusted.index >= exec_dt
            if next_dt is not None:
                mask &= adjusted.index < next_dt
            adjusted.loc[mask] = adjusted.loc[mask] * pos
        return adjusted

    def bear_switch(rets_bw, rets_vgh, rebal_dates, timing_sig, w_bear):
        """熊市防御切换: 牛市全仓 ICW双周, 熊市持 w_bear*BW + (1-w_bear)*VGH。

        与持现金择时 (apply_timing) 的区别: 熊市仓位不闲置, 切到低相关价值策略 VGH
        (相关性仅 0.40, 熊市年显著抗跌)。切换发生在调仓执行日, 区间内保持不变。
        """
        out = rets_bw.copy()
        for i, exec_dt in enumerate(rebal_dates):
            if exec_dt not in timing_sig.index:
                continue
            is_bull = timing_sig.loc[exec_dt]
            next_dt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
            mask = out.index >= exec_dt
            if next_dt is not None:
                mask &= out.index < next_dt
            if not is_bull:
                out.loc[mask] = w_bear * rets_bw.loc[mask] + (1 - w_bear) * rets_vgh.loc[mask]
        return out

    # ---- 逐策略回测 + 全指标 ----
    nav_out, metric_rows, yearly_all = {}, [], {}
    attrib_rows = []
    rets_all = {}
    for tag in STRATS:
        rets, avg_to, n_buys = portfolio_backtest(rebalances[tag], price_mat)
        rets = rets[rets.index >= BT_START]
        rets_all[tag] = rets
        # 路径3: DL_T / ICW_T 策略应用择时仓位 (熊市半仓)
        if tag in ("DL_T", "ICW_T"):
            rebal_dts = [d for d, _ in rebalances[tag]]
            rets = apply_timing(rets, rebal_dts, timing_signal, config.TIMING_BEAR_POS)
        # 收益归因: beta / 择时 / 选股+因子 (rets_all[tag] 为未择时版本)
        att = attrib_returns(rets, bench,
                             rets_all[tag] if tag in ("DL_T", "ICW_T") else None)
        if att:
            attrib_rows.append({"策略": LABEL[tag], **att})
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

    # ---- 块2对照: 双腿分离 (形态B, monthly_signal 生产形态的收益层近似) ----
    # 生产 = DE腿Top5×40%资金 + value_comp腿Top10×60%资金; 回测层用 0.4*DE + 0.6*VAL10
    # 近似 (腿内持仓数差异对结论方向无影响, 两列都在表里单独可查)
    leg_rets = 0.4 * rets_all["DE"] + 0.6 * rets_all["VAL10"]
    m_leg = calc_metrics(leg_rets, bench)
    nav_out["DUAL_LEG"] = (1 + leg_rets).cumprod()
    metric_rows.append({
        "策略": "双腿40/60(DE+VAL)", "年化收益": m_leg["ar"],
        "年化(剔2020)": annualized_since(leg_rets, ALIGN_START),
        "年化波动": m_leg["vol"], "Sharpe": m_leg["sharpe"], "最大回撤": m_leg["mdd"],
        "Calmar": m_leg["calmar"], "IC": float("nan"), "RankIC": float("nan"),
        "换手率": float("nan"), "交易次数": 0, "超额vsHS300": m_leg.get("excess_ar", float("nan"))})
    yearly_all["DUAL_LEG"] = {yr: (1 + g).prod() - 1 for yr, g in leg_rets.groupby(leg_rets.index.year)}

    # ---- ICW_SW: ICW双周 + 熊市防御切换 VGH (最终推荐策略) ----
    # 复用 ICW_BW / VGH 的信号与回测, 在收益层做熊市切换 (切换仅24次/6.5年, 额外成本可忽略)
    sw_rets = bear_switch(rets_all["ICW_BW"], rets_all["VGH"],
                          [d for d, _ in rebalances["ICW_BW"]],
                          timing_signal, config.SWITCH_W_BEAR)
    att_sw = attrib_returns(sw_rets, bench, rets_all["ICW_BW"])
    if att_sw:
        attrib_rows.append({"策略": "ICW双周+熊市切VGH", **att_sw})
    m_sw = calc_metrics(sw_rets, bench)
    ic_sw, rankic_sw = ic_of("ICW_BW")
    nav_out["ICW_SW"] = (1 + sw_rets).cumprod()
    metric_rows.append({
        "策略": "ICW双周+熊市切VGH", "年化收益": m_sw["ar"],
        "年化(剔2020)": annualized_since(sw_rets, ALIGN_START),
        "年化波动": m_sw["vol"], "Sharpe": m_sw["sharpe"], "最大回撤": m_sw["mdd"],
        "Calmar": m_sw["calmar"], "IC": ic_sw, "RankIC": rankic_sw,
        "换手率": metric_rows[[r["策略"] for r in metric_rows].index("ICW双周 Top10")]["换手率"],
        "交易次数": metric_rows[[r["策略"] for r in metric_rows].index("ICW双周 Top10")]["交易次数"],
        "超额vsHS300": m_sw.get("excess_ar", float("nan"))})
    yearly_all["ICW_SW"] = {yr: (1 + g).prod() - 1 for yr, g in sw_rets.groupby(sw_rets.index.year)}

    # ==================== 标准输出 (分年 + 图1九指标 + 对照结论) ====================
    order = list(STRATS) + ["ICW_SW", "DUAL_LEG", "HS300"]
    labels = {**LABEL, "ICW_SW": "ICW双周+熊市切VGH", "HS300": "沪深300基准",
              "DUAL_LEG": "双腿40/60(DE+VAL)"}
    print_yearly_table(yearly_all, order, labels,
                       title="一、分年收益 (2020~2026H1)")
    mdf = pd.DataFrame(metric_rows)
    print_metric_table(mdf, title="二、整体指标 (图1口径; 年化(剔2020)对齐 value_comp 冻结基线)")

    ens = mdf.set_index("策略").loc["集成ENS Top10"]
    de = mdf.set_index("策略").loc["DE Top10"]
    icw = mdf.set_index("策略").loc["IC加权 Top10"]
    x15t = mdf.set_index("策略").loc["XGB15+缓冲"]
    v20 = mdf.set_index("策略").loc["value_comp Top20"]
    print(f"\n{'='*96}\n  三、模型 vs value_comp 对照\n{'='*96}", flush=True)
    print(f"  · 全期(含2020)  : DE {de['年化收益']*100:.2f}%  "
          f"ICW {icw['年化收益']*100:.2f}%  XGB15+缓冲 {x15t['年化收益']*100:.2f}%  "
          f"vs  value_comp20 {v20['年化收益']*100:.2f}%", flush=True)
    print(f"  · 对齐(剔2020)  : DE {de['年化(剔2020)']*100:.2f}%  "
          f"ICW {icw['年化(剔2020)']*100:.2f}%  XGB15+缓冲 {x15t['年化(剔2020)']*100:.2f}%  "
          f"vs  value_comp20 {v20['年化(剔2020)']*100:.2f}%", flush=True)
    print(f"  · 风险调整      : DE Sharpe {de['Sharpe']:.2f}  "
          f"ICW Sharpe {icw['Sharpe']:.2f}  XGB15+缓冲 Sharpe {x15t['Sharpe']:.2f}  "
          f"vs  value_comp20 Sharpe {v20['Sharpe']:.2f}", flush=True)
    print(f"  · 回撤           : DE {de['最大回撤']*100:.1f}%  "
          f"ICW {icw['最大回撤']*100:.1f}%  XGB15+缓冲 {x15t['最大回撤']*100:.1f}%  "
          f"vs  value_comp20 {v20['最大回撤']*100:.1f}%", flush=True)
    print(f"  · 换手           : DE {de['换手率']*100:.1f}%  "
          f"ICW {icw['换手率']*100:.1f}%  XGB15+缓冲 {x15t['换手率']*100:.1f}%  "
          f"vs  value_comp20 {v20['换手率']*100:.1f}%", flush=True)
    print(f"  · OOS RankIC    : DE={de['RankIC']:.4f}  "
          f"ICW={icw['RankIC']:.4f}  XGB15+缓冲={x15t['RankIC']:.4f}  value={v20['RankIC']:.4f}", flush=True)

    # ---- 级联策略 vs value 底仓对照 (方向: ML 精排是否在安全集内增值) ----
    _ix = mdf.set_index("策略")
    print(f"\n{'='*96}\n  四、级联精排 vs value 底仓对照\n{'='*96}", flush=True)
    for c_tag, b_tag in [("VG底仓+ENS精排", "value+盈利 Top10"),
                         ("VGH底仓+ENS精排", "结构化剥离 Top10")]:
        c, b = _ix.loc[c_tag], _ix.loc[b_tag]
        print(f"  · {c_tag} vs {b_tag}", flush=True)
        print(f"      全期: {c['年化收益']*100:.2f}% vs {b['年化收益']*100:.2f}%  |  "
              f"剔20: {c['年化(剔2020)']*100:.2f}% vs {b['年化(剔2020)']*100:.2f}%  |  "
              f"Sharpe: {c['Sharpe']:.2f} vs {b['Sharpe']:.2f}  |  "
              f"回撤: {c['最大回撤']*100:.1f}% vs {b['最大回撤']*100:.1f}%", flush=True)

    # ---- 新策略 vs 基线对照 (ICW_BW/ICW_SW vs ICW/VGH) ----
    print(f"\n{'='*96}\n  五、新策略 vs 基线对照 (ICW_BW/ICW_SW/ICV/ICW_T/ICW15)\n{'='*96}", flush=True)
    bw_r = mdf.set_index("策略").loc["ICW双周 Top10"]
    print(f"  · ICW双周 Top10 : 年化 {bw_r['年化收益']*100:.2f}%  "
          f"剔20 {bw_r['年化(剔2020)']*100:.2f}%  "
          f"Sharpe {bw_r['Sharpe']:.2f}  "
          f"回撤 {bw_r['最大回撤']*100:.1f}%  "
          f"换手 {bw_r['换手率']*100:.1f}%", flush=True)
    sw_r = mdf.set_index("策略").loc["ICW双周+熊市切VGH"]
    print(f"  · ICW双周+熊市切VGH (最终推荐): 年化 {sw_r['年化收益']*100:.2f}%  "
          f"剔20 {sw_r['年化(剔2020)']*100:.2f}%  "
          f"Sharpe {sw_r['Sharpe']:.2f}  "
          f"回撤 {sw_r['最大回撤']*100:.1f}%", flush=True)
    for tag_name in ("VGH+ICW融合", "ICW+择时 Top10", "ICW15+缓冲"):
        if tag_name in _ix.index:
            r = _ix.loc[tag_name]
            print(f"  · {tag_name:12s}: 年化 {r['年化收益']*100:.2f}%  "
                  f"剔20 {r['年化(剔2020)']*100:.2f}%  "
                  f"Sharpe {r['Sharpe']:.2f}  "
                  f"回撤 {r['最大回撤']*100:.1f}%  "
                  f"换手 {r['换手率']*100:.1f}%", flush=True)
    print(f"  vs ICW       : 年化 {icw['年化收益']*100:.2f}%  "
          f"剔20 {icw['年化(剔2020)']*100:.2f}%  "
          f"Sharpe {icw['Sharpe']:.2f}  "
          f"回撤 {icw['最大回撤']*100:.1f}%  "
          f"换手 {icw['换手率']*100:.1f}%", flush=True)

    # ---- 六、收益归因: 你赚的是什么钱? (基准单一指数: 沪深300) ----
    if attrib_rows:
        print(f"\n{'='*96}\n  六、收益归因 (年化拆分, 基准=沪深300)\n{'='*96}", flush=True)
        print("  总收益 ≈ beta贡献 + 择时贡献 + 选股/因子贡献  "
              "(DFA/AQR尽调第一问; 拆开看是在收溢价还是吃风格行情)", flush=True)
        adf = pd.DataFrame(attrib_rows)
        disp = pd.DataFrame({
            "策略": adf["策略"],
            "总年化": adf["total"].map(_pct2),
            "beta贡献": adf["beta"].map(_pct2),
            "择时贡献": adf["timing"].map(_pct2),
            "选股+因子": adf["alpha"].map(_pct2),
            "β系数": adf["beta_coef"].map(lambda x: f"{x:.2f}"),
            "基准年化": adf["bench_ar"].map(_pct2)})
        print(disp.to_string(index=False), flush=True)
        adf.to_csv(f"{OUT_DIR}/rolling10y_attrib.csv", sep="\t", index=False)

    # ---- 七、XGB Feature Importance: 查高频噪声因子是否带偏权重 + CMA贡献 ----
    if fi_records:
        print(f"\n{'='*96}\n  七、XGB Feature Importance (gain, 跨窗口平均)\n{'='*96}", flush=True)
        fi_df = pd.DataFrame(fi_records)
        agg = (fi_df.groupby("feature")["gain_share"]
               .agg(["mean", "std", "count"]).sort_values("mean", ascending=False))
        agg.columns = ["平均gain占比", "std", "窗口数"]
        print("  [Top 30]", flush=True)
        for f, r in agg.head(30).iterrows():
            print(f"    {f:<24} {r['平均gain占比']*100:6.2f}%  "
                  f"(std {r['std']*100:.2f}pp, {int(r['窗口数'])}窗口)", flush=True)
        fund_feats = agg[agg.index.str.startswith("F_")]
        if len(fund_feats):
            print("  [基本面 PIT 因子] (CMA=F_asset_growth 新增; 验证增量不显著则放弃)",
                  flush=True)
            for f, r in fund_feats.iterrows():
                print(f"    {f:<24} {r['平均gain占比']*100:6.2f}%  "
                      f"(std {r['std']*100:.2f}pp, {int(r['窗口数'])}窗口)", flush=True)
        agg.to_csv(f"{OUT_DIR}/xgb_feature_importance.csv", sep="\t",
                   encoding="utf-8-sig")
        print(f"  [+] 完整表已保存: {OUT_DIR}/xgb_feature_importance.csv", flush=True)

    # ---- 八、块2实验: VAL10 × 模型打分 (原版 / 双腿B / 排雷C / 融合A 四列对照) ----
    print(f"\n{'='*96}\n  八、VAL10 × 模型打分对照实验 (C排雷/A融合 vs B双腿 vs 原版)\n{'='*96}",
          flush=True)
    print("  判定纪律: OOS年化/Sharpe/回撤/换手 四项同时不劣化, 且增量扣完成本 ≥1pp 才算赢;",
          flush=True)
    print("            靠某一两年极端行情赢的, 不算赢 (看分年稳定性)。", flush=True)
    _m = mdf.set_index("策略")
    valf_w_ser = pd.DataFrame(results)
    valf_w_ser = (valf_w_ser[valf_w_ser["model"] == "VALF"][["window", "best_iter"]]
                  .values.tolist() if len(results) else [])
    print("  · VALF 逐窗口 w (valid段寻优): "
          + "  ".join(f"{w}:{g}" for w, g in valf_w_ser), flush=True)
    base = _m.loc["value_comp Top10"]
    for exp_name in ("双腿40/60(DE+VAL)", "排雷版VAL10", "融合版VAL10"):
        if exp_name not in _m.index:
            continue
        r = _m.loc[exp_name]
        print(f"  · {exp_name:14s}: 年化 {r['年化收益']*100:6.2f}%  "
              f"剔20 {r['年化(剔2020)']*100:6.2f}%  Sharpe {r['Sharpe']:5.2f}  "
              f"回撤 {r['最大回撤']*100:6.1f}%  "
              f"换手 {r['换手率']*100 if pd.notna(r['换手率']) else float('nan'):5.1f}%"
              f"  | 增量(剔20) {(r['年化(剔2020)']-base['年化(剔2020)'])*100:+.2f}pp",
              flush=True)
    print(f"  · 原版VAL10基准: 年化 {base['年化收益']*100:.2f}%  "
          f"剔20 {base['年化(剔2020)']*100:.2f}%  Sharpe {base['Sharpe']:.2f}  "
          f"回撤 {base['最大回撤']*100:.1f}%  换手 {base['换手率']*100:.1f}%", flush=True)
    # 分年稳定性: 排雷/融合 vs 原版, 输赢年份分布 (不靠个别年份)
    for exp_tag in ("VALX", "VALF"):
        if exp_tag not in yearly_all or "VAL10" not in yearly_all:
            continue
        diffs = {yr: (yearly_all[exp_tag].get(yr, 0) - yearly_all["VAL10"].get(yr, 0)) * 100
                 for yr in yearly_all["VAL10"]}
        win_yrs = sum(1 for v in diffs.values() if v > 0)
        print(f"  · {LABEL[exp_tag]} 分年 vs 原版: {win_yrs}/{len(diffs)} 年跑赢, "
              + "  ".join(f"{yr}:{v:+.1f}pp" for yr, v in sorted(diffs.items())), flush=True)

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
