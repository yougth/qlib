"""
V15 模型修复版消融: 修复V14诊断出的模型问题后重跑 E1/E2/E3 (E0不变, 直接复用)
=================================================================
V14诊断结论 (v14_diag.py):
  1. 27窗口中E1有6个best_iter=0 → 模型输出常数, nlargest(20)退化为伪随机选股
  2. 根因: RMSE早停与排序任务错配 —— validRMSE在第0轮就"最优"(任何学习都增大RMSE),
     与lr/正则强度无关 (4种参数组合全部best_iter=0)
  3. IC早停(maximize RankIC)可避免退化: 2021_Q2 testIC +0.0197→+0.0486, 截面std放大48倍

V15修复:
  F1. 早停指标: RMSE → valid RankIC (maximize), 早停轮数50
  F2. 参数: lr 0.005→0.02, reg_alpha 10→1, reg_lambda 50→5 (对zscore标签过强)
  F3. 退化保护: validIC < 0.005 时该窗口fallback到等权持池 ("没把握时不下注")
  F4. 沪深300基准: qlib SH000300只到2019 → 改用akshare csi300_cache.csv
继承V14全部: 数据管线/T+1/0.4%成本/流动性/embargo/PIT对齐/季频Top20
"""
import os, sys, random
import warnings, logging

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
import xgboost as xgb
import joblib
warnings.filterwarnings("ignore")
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set, filter_pred_by_tradability
import v14_ablation as v14
from v14_ablation import (generate_quarterly_windows, prepare_window_data,
                          build_exp_matrices, build_liquidity_table,
                          load_price_matrix, portfolio_backtest, calc_metrics,
                          quarter_signal_exec_dates,
                          WORK_DIR, DATA_DIR, TOPK, LIQ_THRESHOLD,
                          TOP_N_FEATURES, FS_ROUNDS, FS_EARLY_STOP, FS_LR)

# ---- F2: 修复后的参数 ----
XGB_PARAMS_V15 = {
    "objective": "reg:squarederror", "learning_rate": 0.02,
    "max_depth": 4, "colsample_bytree": 0.8879, "subsample": 0.8789,
    "reg_alpha": 1.0, "reg_lambda": 5.0,
    "tree_method": "hist", "nthread": 4, "seed": 42,
    "disable_default_eval_metric": 1,
}
N_ROUNDS, EARLY_STOP = 500, 50
IC_DEGENERATE_TH = 0.005     # F3: validIC低于此值 → fallback等权


def rank_ic(pred, label):
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    ics = df.groupby(level=0).apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 10 else np.nan)
    return float(ics.mean())


def make_ic_feval(va_index):
    def feval(preds, dmat):
        y = pd.Series(dmat.get_label(), index=va_index)
        p = pd.Series(preds, index=va_index)
        ic = rank_ic(p, y)
        return "rank_ic", ic if not np.isnan(ic) else 0.0
    return feval


def select_and_train_v15(X_tr, y_tr, X_va, y_va):
    """两阶段: stage1 RMSE快速取Top80 (不变) → stage2 IC早停正式训练"""
    feats_all = list(X_tr.columns)
    p1 = {k: v for k, v in XGB_PARAMS_V15.items() if k != "disable_default_eval_metric"}
    p1["learning_rate"] = FS_LR
    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feats_all)
    dva = xgb.DMatrix(X_va.values, label=y_va.values, feature_names=feats_all)
    m1 = xgb.train(p1, dtr, num_boost_round=FS_ROUNDS, evals=[(dva, "valid")],
                   early_stopping_rounds=FS_EARLY_STOP, verbose_eval=False)
    imp = pd.Series(m1.get_score(importance_type="gain")).reindex(feats_all).fillna(0.0)
    feats = imp.nlargest(TOP_N_FEATURES).index.tolist()

    dtr2 = xgb.DMatrix(X_tr[feats].values, label=y_tr.values, feature_names=feats)
    dva2 = xgb.DMatrix(X_va[feats].values, label=y_va.values, feature_names=feats)
    m2 = xgb.train(XGB_PARAMS_V15, dtr2, num_boost_round=N_ROUNDS,
                   evals=[(dva2, "valid")], early_stopping_rounds=EARLY_STOP,
                   custom_metric=make_ic_feval(X_va.index), maximize=True,
                   verbose_eval=False)
    va_pred = pd.Series(m2.predict(dva2, iteration_range=(0, m2.best_iteration + 1)),
                        index=X_va.index)
    va_ic = rank_ic(va_pred, y_va)
    return m2, feats, m2.best_iteration, va_ic


def get_bench_returns_ak(start, end):
    """F4: akshare缓存的沪深300 (qlib SH000300只到2019)"""
    df = pd.read_csv(f"{DATA_DIR}/csi300_cache.csv", sep='\t', parse_dates=["date"])
    s = df.set_index("date")["close"].sort_index().loc[start:end]
    return s.pct_change().fillna(0)


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")
    cal_list = list(cal)

    print(f"\n{'='*100}", flush=True)
    print("  V15 模型修复版: IC早停 + lr0.02/弱正则 + 退化保护fallback等权  (季频Top20)", flush=True)
    print(f"{'='*100}", flush=True)

    windows = generate_quarterly_windows()
    years = sorted(set(w["year"] for w in windows))
    preds = {e: {} for e in ["E1", "E2", "E3"]}
    fallback_flags = {e: {} for e in ["E1", "E2", "E3"]}   # (year, sig顺序) → 是否fallback
    year_universe = {}
    n_fallback = {e: 0 for e in ["E1", "E2", "E3"]}

    for win in windows:
        print(f"\n[+] 窗口 {win['name']}", flush=True)
        X_tr, y_tr, X_va, y_va, test_X, universe, ext = prepare_window_data(win, fcf_df, profit_df, cal)
        year_universe[win["year"]] = universe

        for exp in ["E1", "E2", "E3"]:
            Xt, Xv, Xe = build_exp_matrices(exp, X_tr, X_va, test_X, ext)
            model, feats, best_it, va_ic = select_and_train_v15(Xt, y_tr, Xv, y_va)
            degenerate = (va_ic < IC_DEGENERATE_TH) or (best_it <= 0)
            scores = model.predict(xgb.DMatrix(Xe[feats].values, feature_names=feats),
                                   iteration_range=(0, best_it + 1))
            pred = pd.DataFrame({"score": scores}, index=Xe.index)
            preds[exp].setdefault(win["year"], []).append(pred)
            fallback_flags[exp].setdefault(win["year"], []).append(degenerate)
            if degenerate:
                n_fallback[exp] += 1
            print(f"    [{exp}] best_iter={best_it} validIC={va_ic:+.4f}"
                  f"{'  →退化,fallback等权' if degenerate else ''}", flush=True)
            if win["name"] == "2026_Q3":
                joblib.dump({"model": model, "features": feats, "valid_ic": va_ic,
                             "degenerate": degenerate},
                            f"{WORK_DIR}/v15_{exp}_latest_Q.pkl")

    # ==================== 回测 ====================
    print(f"\n{'='*100}\n  回测 (季频, T+1, 往返0.4%, 500万流动性, 退化季fallback等权)\n{'='*100}", flush=True)
    exp_list = ["E0", "E1", "E2", "E3"]
    all_rets = {e: [] for e in exp_list}
    annual = {e: {} for e in exp_list}
    to_stats = {e: [] for e in exp_list}

    for y in years:
        universe = year_universe[y]
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        sig_exec = quarter_signal_exec_dates(y, cal_list)
        liq_table = build_liquidity_table(universe, sig_exec[0][0] - pd.Timedelta(days=10), bt_end)
        price_mat = load_price_matrix(universe, sig_exec[0][0], bt_end)
        limit_up_set, suspension_set = build_limit_up_set(universe, cal)

        row = f"  {y}:"
        for exp in exp_list:
            rebalances = []
            for qi, (sig, exec_dt) in enumerate(sig_exec):
                ew_holdings = [i for i in universe if i in price_mat.columns
                               and pd.notna(price_mat.loc[exec_dt, i])]
                if exp == "E0" or fallback_flags.get(exp, {}).get(y, [False]*4)[qi]:
                    holdings = ew_holdings          # 等权全池 (基准 或 退化保护)
                else:
                    pred_y = pd.concat(preds[exp][y]).sort_index()
                    pred_y = pred_y[~pred_y.index.duplicated(keep="last")]
                    pred_f = filter_pred_by_tradability(pred_y, limit_up_set, suspension_set)
                    if sig not in pred_f.index.get_level_values(0):
                        raise RuntimeError(f"[{exp}] 信号日{sig}不在pred中!")
                    day_pred = pred_f.xs(sig, level=0)["score"].copy()
                    day_liq = liq_table.xs(sig, level=0).reindex(day_pred.index)
                    day_pred[day_liq.isna() | (day_liq < LIQ_THRESHOLD)] = -np.inf
                    day_pred = day_pred[day_pred > -np.inf]
                    holdings = day_pred.nlargest(TOPK).index.tolist()
                rebalances.append((exec_dt, holdings))

            rets, avg_to = portfolio_backtest(rebalances, price_mat, cal_list)
            all_rets[exp].append(rets)
            to_stats[exp].append(avg_to)
            annual[exp][y] = (1 + rets).prod() - 1
            row += f"  {exp} {annual[exp][y]*100:>7.2f}%"
        print(row, flush=True)

    # ==================== 汇总 (基准用akshare) ====================
    bench_all = []
    for y in years:
        be = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        bench_all.append(get_bench_returns_ak(f"{y}-01-01", be))
    bench = pd.concat(bench_all).sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    bench_m = calc_metrics(bench)

    print(f"\n{'='*118}", flush=True)
    header = "  ".join(f"{y:>7}" for y in years)
    print(f"{'实验':<16} | {header} | {'全期年化':>8} {'夏普':>5} {'回撤':>7} {'季换手':>6} {'vs沪深300超额':>10}", flush=True)
    print(f"{'-'*118}", flush=True)
    results = {}
    labels = {"E0": "E0 等权基准", "E1": "E1 量价60d", "E2": "E2 +估值", "E3": "E3 +基本面"}
    for exp in exp_list:
        s = pd.concat(all_rets[exp]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        m = calc_metrics(s)
        results[exp] = m
        excess_ar = m["ar"] - bench_m["ar"]
        row = "  ".join(f"{annual[exp][y]*100:>6.2f}%" for y in years)
        print(f"{labels[exp]:<14} | {row} | {m['ar']*100:>7.2f}% {m['sharpe']:>5.2f} {m['max_dd']*100:>6.1f}% "
              f"{np.mean(to_stats[exp])*100:>5.1f}% {excess_ar*100:>+9.2f}pp", flush=True)
        s.to_csv(f"{WORK_DIR}/v15_returns_{exp}.csv", sep='\t', header=False)
    row_b = "  ".join(f"{(1+bench[bench.index.year==y]).prod()-1:>7.2%}" for y in years)
    print(f"{'沪深300(ak)':<14} | {row_b} | {bench_m['ar']*100:>7.2f}% {bench_m['sharpe']:>5.2f} {bench_m['max_dd']*100:>6.1f}%", flush=True)
    print(f"{'='*118}", flush=True)

    print("\n消融增量 (全期年化):", flush=True)
    print(f"  模型排序增量  (E1-E0): {(results['E1']['ar']-results['E0']['ar'])*100:+.2f}pp", flush=True)
    print(f"  估值因子增量  (E2-E1): {(results['E2']['ar']-results['E1']['ar'])*100:+.2f}pp", flush=True)
    print(f"  基本面因子增量(E3-E2): {(results['E3']['ar']-results['E2']['ar'])*100:+.2f}pp", flush=True)
    print(f"\n退化保护触发次数 (27窗口): " +
          ", ".join(f"{e}={n_fallback[e]}" for e in ["E1", "E2", "E3"]), flush=True)

    rows = []
    for exp in exp_list:
        for y in years:
            rows.append({"exp": exp, "year": y, "annual_ret": annual[exp][y]})
        rows.append({"exp": exp, "year": "ALL", "annual_ret": results[exp]["ar"],
                     "sharpe": results[exp]["sharpe"], "max_dd": results[exp]["max_dd"]})
    pd.DataFrame(rows).to_csv(f"{WORK_DIR}/v15_fixed_results.csv", sep='\t', index=False)
    print("\n[+] 结果已保存: v15_fixed_results.csv", flush=True)


if __name__ == "__main__":
    run()
