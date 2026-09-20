"""验证: pred_leaf 索引 ↔ trees_to_dataframe 叶值 (DFS序) 映射是否正确"""
import numpy as np
import pandas as pd
import lightgbm as lgbm

rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 20))
y = X[:, 0] * 2 - X[:, 3] + rng.normal(scale=0.1, size=3000)
ds = lgbm.Dataset(X, label=y)
m = lgbm.train({"objective": "regression", "learning_rate": 0.1, "num_leaves": 15,
                "verbosity": -1}, ds, num_boost_round=50)

# 方案A: trees_to_dataframe 的叶子行序 (DFS)
tdf = m.trees_to_dataframe()
leaf_rows = tdf[tdf["node_type"] == "Leaf"]
leaf_vals_dfs = [leaf_rows[leaf_rows["tree_index"] == t]["value"].values.astype(np.float64)
                 for t in range(50)]

leaf_idx = m.predict(X, pred_leaf=True)  # (n, n_trees)
pred_direct = m.predict(X)

# 用 DFS 叶值重建预测
ok = True
max_err = 0.0
for t in range(50):
    li = leaf_idx[:, t].astype(int)
    if li.max() >= len(leaf_vals_dfs[t]):
        print(f"[FAIL] tree {t}: leaf_idx max {li.max()} >= 叶数 {len(leaf_vals_dfs[t])}")
        ok = False
        break
recon = np.zeros(len(X))
for t in range(50):
    recon += leaf_vals_dfs[t][leaf_idx[:, t].astype(int)]
    err = np.abs(recon - m.predict(X, num_iteration=t + 1)).max()
    max_err = max(max_err, err)
if ok:
    print(f"DFS 叶值映射: 逐步累计预测 max err = {max_err:.2e}")
    print("[PASS] DFS 顺序正确" if max_err < 1e-6 else "[FAIL] DFS 顺序不匹配!")

# 方案B对照: 逐树 predict (慢路径, 验证基准)
slow = np.zeros(len(X))
for t in range(50):
    slow += m.predict(X, start_iteration=t, num_iteration=1)
print(f"慢路径 vs 直接预测: {np.abs(slow - pred_direct).max():.2e}")
print(f"快路径 vs 慢路径:   {np.abs(recon - slow).max():.2e}")
