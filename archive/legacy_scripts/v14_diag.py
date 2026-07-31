"""
V14 模型退化诊断: 单窗口(2021_Q2, E1量价)参数敏感性
假设: (a) reg_alpha=10/reg_lambda=50 对zscore标签过强 → 第一棵树就不分裂
      (b) lr=0.005 x 100轮早停 总步长不足
      (c) RMSE早停与排序任务错配 → 应用RankIC早停
输出: 各参数组合的 best_iter / valid RankIC / test RankIC
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
import xgboost as xgb
warnings.filterwarnings("ignore")
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v14_ablation import (generate_quarterly_windows, prepare_window_data,
                          DATA_DIR)

def rank_ic(pred, label):
    """按日截面 Spearman RankIC 均值"""
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    ics = df.groupby(level=0).apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 10 else np.nan)
    return ics.mean(), ics.std()

def make_ic_feval(dva_index):
    def feval(preds, dmat):
        y = dmat.get_label()
        s = pd.Series(preds, index=dva_index)
        l = pd.Series(y, index=dva_index)
        ic, _ = rank_ic(s, l)
        return "rank_ic", ic if not np.isnan(ic) else 0.0
    return feval

def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv(f"{DATA_DIR}/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv(f"{DATA_DIR}/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")

    wins = {w["name"]: w for w in generate_quarterly_windows()}
    results = []
    for wname in ["2021_Q2", "2023_Q4"]:   # 两个退化窗口
        win = wins[wname]
        print(f"\n=== 窗口 {wname} ===", flush=True)
        X_tr, y_tr, X_va, y_va, test_X, universe, ext = prepare_window_data(win, fcf_df, profit_df, cal)

        # test标签: 60天前向收益(仅诊断用, 不影响任何训练)
        te_label_raw = D.features(universe, ["Ref($close,-60)/$close-1"],
                                  start_time=win["test"][0], end_time=win["test"][1])
        te_label = te_label_raw.iloc[:, 0]
        te_label.index = te_label.index.swaplevel(0, 1)  # (inst,dt)→(dt,inst)
        te_label = te_label.sort_index()

        configs = [
            ("当前参数(基线)",   {"learning_rate":0.005,"reg_alpha":10.0,"reg_lambda":50.0}, "rmse"),
            ("lr=0.02",          {"learning_rate":0.02, "reg_alpha":10.0,"reg_lambda":50.0}, "rmse"),
            ("弱正则(a1,l5)",    {"learning_rate":0.005,"reg_alpha":1.0, "reg_lambda":5.0},  "rmse"),
            ("lr=0.02+弱正则",   {"learning_rate":0.02, "reg_alpha":1.0, "reg_lambda":5.0},  "rmse"),
            ("lr=0.02+弱正则+IC早停", {"learning_rate":0.02,"reg_alpha":1.0,"reg_lambda":5.0}, "ic"),
        ]
        feats = list(X_tr.columns)
        dtr = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feats)
        dva = xgb.DMatrix(X_va.values, label=y_va.values, feature_names=feats)
        dte = xgb.DMatrix(test_X.values, feature_names=feats)

        for name, over, stop_metric in configs:
            p = {"objective":"reg:squarederror","max_depth":4,
                 "colsample_bytree":0.8879,"subsample":0.8789,
                 "tree_method":"hist","nthread":4,"seed":42}
            p.update(over)
            kw = dict(num_boost_round=1000, evals=[(dva,"valid")],
                      early_stopping_rounds=100, verbose_eval=False)
            if stop_metric == "ic":
                p["disable_default_eval_metric"] = 1
                kw["custom_metric"] = make_ic_feval(X_va.index)
                kw["maximize"] = True
            m = xgb.train(p, dtr, **kw)
            va_pred = pd.Series(m.predict(dva, iteration_range=(0, m.best_iteration+1)), index=X_va.index)
            te_pred = pd.Series(m.predict(dte, iteration_range=(0, m.best_iteration+1)), index=test_X.index)
            va_ic, _ = rank_ic(va_pred, y_va)
            te_ic, _ = rank_ic(te_pred, te_label.reindex(test_X.index))
            cs_std = te_pred.groupby(level=0).std().mean()
            print(f"  {name:<24} best_iter={m.best_iteration:>4}  validIC={va_ic:+.4f}  testIC={te_ic:+.4f}  预测截面std={cs_std:.5f}", flush=True)
            results.append({"window":wname,"config":name,"best_iter":m.best_iteration,
                            "valid_ic":va_ic,"test_ic":te_ic})
    pd.DataFrame(results).to_csv("/Users/11164591/Documents/Qoder目录/qlib/v14_diag_results.csv", sep='\t', index=False)
    print("\n[+] 诊断完成", flush=True)

if __name__ == "__main__":
    run()
