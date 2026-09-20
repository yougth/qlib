"""DE 优化正确性+性能测试: numpy rank vs pandas rank 等价性 + SR/FS 流程冒烟测试"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib/quant")

# ---- 1) numpy rank vs pandas rank(axis=0, pct=True) 等价性 ----
rng = np.random.default_rng(42)
a = rng.normal(size=(50000, 80)).astype(np.float32)

t0 = time.time()
pd_rank = pd.DataFrame(a).rank(axis=0, pct=True).values.astype(np.float32)
t_pd = time.time() - t0

n, t = a.shape
ranks = np.empty((n, t), dtype=np.float32)
pct = (np.arange(n, dtype=np.float32) + 1.0) / n
t0 = time.time()
for c0 in range(0, t, 64):
    c1 = min(c0 + 64, t)
    order = a[:, c0:c1].argsort(axis=0)
    ranks[order, np.arange(c0, c1)[None, :]] = pct[:, None]
t_np = time.time() - t0

diff = np.abs(ranks - pd_rank)
assert diff.max() < 1e-4, f"rank mismatch: {diff.max()}"
print(f"[PASS] numpy rank == pandas rank (max diff {diff.max():.2e}); "
      f"pandas {t_pd:.2f}s vs numpy {t_np:.3f}s, 加速 {t_pd / max(t_np, 1e-9):.0f}x")

# ---- 2) DoubleEnsembleRankIC 冒烟测试 (含 SR + FS) ----
import lightgbm as lgbm
from core.models import DoubleEnsembleRankIC

n_rows, n_feat = 8000, 25
X = pd.DataFrame(rng.normal(size=(n_rows, n_feat)),
                 index=[f"r{i}" for i in range(n_rows)],
                 columns=[f"f{j}" for j in range(n_feat)])
beta = rng.normal(size=n_feat)
y = pd.Series(X.values @ beta + rng.normal(scale=0.5, size=n_rows), index=X.index)
X_va = pd.DataFrame(rng.normal(size=(2000, n_feat)), columns=X.columns)
y_va = pd.Series(X_va.values @ beta + rng.normal(scale=0.5, size=2000), index=X_va.index)

def ic_eval(predt):
    return float(pd.Series(predt).corr(pd.Series(y_va.values), method="spearman"))

de_cfg = {"num_models": 3, "sub_weights": [1, 1, 1], "enable_sr": True,
          "enable_fs": True, "alpha1": 0.3, "alpha2": 0.3, "bins_sr": 5,
          "bins_fs": 5, "decay": 0.5, "sample_ratios": [0.8, 0.6, 0.4]}
params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31,
          "verbosity": -1, "num_threads": 4}

t0 = time.time()
de = DoubleEnsembleRankIC(X, y, X_va, y_va, ic_eval, params, de_cfg).fit()
t_fit = time.time() - t0

assert len(de.ensemble) == 3, "应有 3 个子模型"
pred = de.predict(X_va)
ic = ic_eval(pred)
assert np.isfinite(pred).all(), "预测含 NaN"
print(f"[PASS] DE 冒烟测试: 3 子模型 fit {t_fit:.1f}s, 集成 valid RankIC={ic:.4f}, "
      f"best_iters={de.best_iters}, 选中特征数={[len(f) for f in de.sub_features]}")

# ---- 3) _sample_reweight 权重语义检查: 高损失样本应获更高权重 ----
m1 = de.ensemble[0]
lc = de._retrieve_loss_curve(m1, de.sub_features[0])
assert isinstance(lc, np.ndarray) and lc.dtype == np.float32
assert lc.shape[0] == n_rows, f"行数 {lc.shape[0]} != {n_rows}"
part = max(int(m1.num_trees() * 0.1), 1)
exp_cols = 2 * part if 2 * part < m1.num_trees() else m1.num_trees()
assert lc.shape[1] == exp_cols, f"列数 {lc.shape[1]} != 预期 {exp_cols}"
loss_values = pd.Series((y.values - m1.predict(
    X.iloc[:, de.sub_features[0]].values)) ** 2)  # 与 fit() 一致: RangeIndex
weights = de._sample_reweight(lc, loss_values, 1)
assert len(weights) == n_rows and (weights >= 0).all()
top_loss = loss_values.nlargest(1000).index
bottom_loss = loss_values.nsmallest(1000).index
assert weights[top_loss].mean() > weights[bottom_loss].mean(), \
    "高损失样本权重均值应更高"
print(f"[PASS] SR 权重语义: 高损失均值 {weights[top_loss].mean():.3f} > "
      f"低损失均值 {weights[bottom_loss].mean():.3f}, 权重范围 "
      f"[{weights.min():.3f}, {weights.max():.3f}]")
print("\n全部通过 ✓")
