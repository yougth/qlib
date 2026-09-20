"""诊断: is_hk 市场标识为何降低 XGB 收益 (单窗口 W2026)
对比 with/without is_hk:
  1. 特征重要性中 is_hk 的排名
  2. 港股预测得分的分布 vs A股
  3. 港股整体排名 (beta 开关?) vs 个股内差异
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb

import core.config as config
from core import data as datalayer
from core.universe import build_windows, build_dynamic_universe, format_qlib_code, load_pit_caches
from core.dataset import build_dataset
from core.models import RankICEval, train_xgb

def main():
    datalayer.init_qlib()
    fcf_df, profit_df = load_pit_caches()
    cal = datalayer.get_calendar()
    windows = build_windows()
    win = [w for w in windows if w["year"] == 2026][0]
    y = win["year"]
    print(f"[DIAG] 窗口: {win['name']}", flush=True)

    codes = build_dynamic_universe(y, fcf_df, profit_df)
    universe = [format_qlib_code(c) for c in codes]
    n_hk = sum(1 for c in universe if c.startswith("hk"))
    print(f"[DIAG] 池: {len(universe)} 只 (港股 {n_hk}, {n_hk/len(universe)*100:.1f}%)", flush=True)

    # ---- 版本 A: 带 is_hk (当前配置) ----
    ds = build_dataset(win, universe, fcf_df, profit_df, cal)
    X_tr, y_tr, X_va, y_va, test_X = ds["X_tr"], ds["y_tr"], ds["X_va"], ds["y_va"], ds["test_X"]
    print(f"[DIAG] A(带is_hk) 特征列: {X_tr.shape[1]}  最后5列: {list(X_tr.columns[-5:])}", flush=True)
    va_dates = X_va.index.get_level_values(0).values
    ic_eval = RankICEval(va_dates, y_va.values)
    m_a, iter_a, ic_a = train_xgb(X_tr, y_tr, X_va, y_va, ic_eval)
    imp_a = m_a.get_score(importance_type="gain")
    imp_a = pd.Series(imp_a).sort_values(ascending=False)
    pred_a = pd.Series(m_a.predict(xgb.DMatrix(test_X.values),
                                   iteration_range=(0, m_a._rankic_best_iter + 1)),
                       index=test_X.index)
    del X_tr, y_tr, X_va, y_va, test_X, m_a
    import gc; gc.collect()

    # ---- 版本 B: 不带 is_hk ----
    config.INJECT_MARKET = False
    ds = build_dataset(win, universe, fcf_df, profit_df, cal)
    X_tr, y_tr, X_va, y_va, test_X = ds["X_tr"], ds["y_tr"], ds["X_va"], ds["y_va"], ds["test_X"]
    print(f"[DIAG] B(不带is_hk) 特征列: {X_tr.shape[1]}", flush=True)
    ic_eval = RankICEval(va_dates, y_va.values)
    m_b, iter_b, ic_b = train_xgb(X_tr, y_tr, X_va, y_va, ic_eval)
    pred_b = pd.Series(m_b.predict(xgb.DMatrix(test_X.values),
                                   iteration_range=(0, m_b._rankic_best_iter + 1)),
                       index=test_X.index)
    del X_tr, y_tr, X_va, y_va, test_X, m_b
    config.INJECT_MARKET = True

    # ---- 分析 ----
    print(f"\n[RESULT] A(带is_hk): valid RankIC={ic_a:.4f}, best_iter={iter_a}", flush=True)
    print(f"[RESULT] B(不带is_hk): valid RankIC={ic_b:.4f}, best_iter={iter_b}", flush=True)

    print(f"\n[RESULT] A is_hk 特征重要性排名: {list(imp_a.index).index('is_hk')+1 if 'is_hk' in imp_a.index else 'N/A'} / {len(imp_a)}")
    print(f"[RESULT] A is_hk gain: {imp_a.get('is_hk', 0):.6f}")
    print(f"[RESULT] A Top10 特征: {list(imp_a.index[:10])}")

    # 港股 vs A股预测分布
    for name, pred in [("A(带is_hk)", pred_a), ("B(不带is_hk)", pred_b)]:
        inst = pred.index.get_level_values(1)
        hk = inst.str.startswith("hk")
        print(f"\n[RESULT] {name} 预测分布:")
        print(f"  A股: 均值={pred[~hk].mean():.4f} 中位={pred[~hk].median():.4f} 样本={int((~hk).sum())}")
        print(f"  港股: 均值={pred[hk].mean():.4f} 中位={pred[hk].median():.4f} 样本={int(hk.sum())}")
        # 港股整体排位: 每日港股预测均值在全池百分位
        df = pred.to_frame('score')
        df['inst'] = df.index.get_level_values(1)
        df['is_hk'] = df['inst'].str.startswith('hk')
        daily = df.groupby(level=0).apply(
            lambda g: g['score'].rank(pct=True)[g['is_hk']].mean(), include_groups=False)
        print(f"  港股每日平均百分位: 均值={daily.mean():.3f} (0.5=无系统性偏差)")

    # 额外: 验证 B 全期收益应该更高 (通过预测截面 rank 反推)
    # 这里只打印 IC 差异, 完整回测另跑
    print("\n[DONE] 诊断完成")

if __name__ == "__main__":
    main()
