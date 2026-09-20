#!/usr/bin/env python3
"""
诊断脚本: 对比 qlib_cn_tencent vs cn_data_fixed 的特征/标签管线
1. 特征IC (单因子与label的RankIC)
2. 模型valid RankIC
3. train/valid/test 标签分布偏移
4. 股票池大小差异
"""
import os, sys, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import xgboost as xgb

def run_diag(provider_name, provider_path):
    """在指定数据源上运行诊断"""
    os.environ["QLIB_PROVIDER"] = provider_path
    # 强制重新初始化
    import importlib
    from core import data as datalayer
    datalayer._INITED = False
    importlib.reload(datalayer)
    from core import config
    datalayer.init_qlib()

    from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
    from core.dataset import build_dataset
    from core.models import RankICEval, train_xgb

    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    wins = build_windows()

    # 用 W2024 做诊断 (train 2019-2022, valid 2023, test 2024)
    win = [w for w in wins if w["year"] == 2024][0]
    y = win["year"]
    print(f"\n{'='*60}")
    print(f"[{provider_name}] W{y}: train {win['train']} | valid {win['valid']} | test {win['test']}")
    print(f"{'='*60}")

    codes = build_dynamic_universe(y, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    print(f"  股票池: {len(universe)} 只")

    ds = build_dataset(win, universe, fcf_df, profit_df, cal)
    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_va, y_va = ds["X_va"], ds["y_va"]
    test_X = ds["test_X"]

    print(f"  特征矩阵: train {X_tr.shape}, valid {X_va.shape}, test {test_X.shape}")
    print(f"  标签分布: train mean={y_tr.mean():.4f} std={y_tr.std():.4f} "
          f"median={y_tr.median():.4f} | valid mean={y_va.mean():.4f} std={y_va.std():.4f}")

    # 1. 单因子IC (top 15 by abs IC)
    print(f"\n--- 单因子RankIC (train段, top 15 by |IC|) ---")
    feat_ics = {}
    for col in X_tr.columns:
        if X_tr[col].std() < 1e-12:
            continue
        df_tmp = pd.DataFrame({"x": X_tr[col].values, "y": y_tr.values})
        df_tmp = df_tmp.dropna()
        if len(df_tmp) < 100:
            continue
        # 按日期分组计算RankIC
        idx = X_tr.index
        dates = idx.get_level_values(0)
        ics = []
        for d in pd.DatetimeIndex(dates).unique():
            mask = dates == d
            if mask.sum() < 5:
                continue
            xv = X_tr[col].values[mask]
            yv = y_tr.values[mask]
            valid = ~(np.isnan(xv) | np.isnan(yv))
            if valid.sum() < 5:
                continue
            from scipy.stats import spearmanr
            r, _ = spearmanr(xv[valid], yv[valid])
            if not np.isnan(r):
                ics.append(r)
        feat_ics[col] = np.mean(ics) if ics else 0

    ic_series = pd.Series(feat_ics).sort_values(key=lambda x: x.abs(), ascending=False)
    for i, (name, ic) in enumerate(ic_series.head(15).items()):
        print(f"  {i+1:2d}. {name:25s} IC={ic:+.4f}")

    # 2. 基本面因子IC
    print(f"\n--- 基本面因子IC ---")
    for col in [c for c in X_tr.columns if c.startswith("F_")]:
        if col in feat_ics:
            print(f"  {col:25s} IC={feat_ics[col]:+.4f}")

    # 3. 训练XGB
    va_dates = X_va.index.get_level_values(0).values
    ic_eval = RankICEval(va_dates, y_va.values)
    print(f"\n--- 训练XGB ---")
    m, best_iter, valid_ic = train_xgb(X_tr, y_tr, X_va, y_va, ic_eval)
    print(f"  best_iter={best_iter}, valid RankIC={valid_ic:.4f}")

    # 4. test段IC (OOS)
    test_idx = test_X.index
    test_dates = test_idx.get_level_values(0)
    # 需要test标签来计算OOS IC
    # 从dataset handler获取
    print(f"\n  (OOS IC需要test标签, 跳过)")

    del X_tr, y_tr, X_va, y_va, test_X, m
    gc.collect()

    return {"pool_size": len(universe), "valid_rank_ic": valid_ic,
            "best_iter": best_iter, "train_label_mean": y_tr.mean(),
            "train_label_std": y_tr.std()}


if __name__ == "__main__":
    providers = [
        ("qlib_cn_tencent", "/Users/11164591/Documents/Qoder目录/qlib/data_cache/qlib_cn_tencent"),
        ("cn_data_fixed", os.path.expanduser("~/.qlib/qlib_data/cn_data_fixed")),
    ]

    results = {}
    for name, path in providers:
        if not os.path.exists(f"{path}/features"):
            print(f"\n[SKIP] {name}: 数据目录不存在 {path}")
            continue
        try:
            results[name] = run_diag(name, path)
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("对比汇总:")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:20s}: pool={r['pool_size']:4d}  valid_RankIC={r['valid_rank_ic']:.4f}  "
              f"best_iter={r['best_iter']}  label_mean={r['train_label_mean']:.4f}  "
              f"label_std={r['train_label_std']:.4f}")
