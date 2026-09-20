"""
core.models —— RankIC 早停训练器 (XGB/LGB/DE + 收敛 fallback) + 收敛检查 + 通用 qlib 训练器
================================================================================
- RankICEval: 预计算分组, 每轮 boosting 高速评估验证集日度 RankIC
- train_xgb / train_lgb: 验证集 RankIC 最大化早停 (严禁 RMSE 早停退化);
  best_iter<30 时用备选超参重训, 按 valid RankIC 择优 (仅用valid集, 不碰test, 无穿越)
- train_de: DoubleEnsemble (SR样本重加权 + FS特征选择), 每个子模型用 RankIC 早停,
  与 train_xgb/train_lgb 同口径接受 numpy 矩阵 → 可注入 FCF 因子 (基准 DE 走 qlib 原生
  不注入 FCF, 这是 8% vs 24% 差距根源之一)
- check_convergence: best_iter==0 报错; best_iter<30 或 valid RankIC<0.01 高亮告警
- train_qlib_model: 用 init_instance_by_config 加载 qlib 内置模型 (基准 sweep 用),
  按实际特征数注入 d_feat/input_dim, fit(train/valid) 后对 test 段 predict, 全程无穿越。
"""
import numpy as np
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd

from . import config


class RankICEval:
    def __init__(self, dates, labels):
        dates = np.asarray(dates)
        order = np.argsort(dates, kind="stable")
        self.order = order
        d_sorted = dates[order]
        _, starts = np.unique(d_sorted, return_index=True)
        bounds = np.append(starts, len(d_sorted))
        self.slices = [(bounds[i], bounds[i + 1]) for i in range(len(starts))
                       if bounds[i + 1] - bounds[i] > 5]
        y_sorted = np.asarray(labels)[order]
        self.y_ranks = []
        for lo, hi in self.slices:
            seg = y_sorted[lo:hi]
            r = np.empty(len(seg))
            r[np.argsort(seg, kind="stable")] = np.arange(len(seg))
            self.y_ranks.append(r)

    def __call__(self, preds):
        p_sorted = np.asarray(preds)[self.order]
        ics = []
        for (lo, hi), yr in zip(self.slices, self.y_ranks):
            seg = p_sorted[lo:hi]
            pr = np.empty(len(seg))
            pr[np.argsort(seg, kind="stable")] = np.arange(len(seg))
            ics.append(np.corrcoef(pr, yr)[0, 1])
        return float(np.nanmean(ics))


class WarmupEarlyStopping(xgb.callback.TrainingCallback):
    """RankIC 早停 with warmup: 前 warmup 轮不触发早停, 避开初始噪声峰值。

    xgboost 3.x 的 maximize=True early_stopping 会在第0轮的 rank_ic 噪声峰值处
    立即停止 (rank_ic 先降后升, 第0轮恰好是局部高点)。旧版 xgboost 2.x 的
    disable_default_eval_metric=1 隐含了 warmup 行为, 3.x 去掉了这个隐含行为。
    本 callback 显式实现 warmup, 复现旧版行为 (W2020 best_iter=686, RankIC=0.0818)。
    """

    def __init__(self, rounds, warmup=100):
        self.rounds = rounds
        self.warmup = warmup
        self.best_score = -np.inf
        self.best_iter = 0
        self.no_improve = 0

    def after_iteration(self, model, epoch, evals_log):
        for dname, metrics in evals_log.items():
            if "rank_ic" in metrics and len(metrics["rank_ic"]) > 0:
                score = metrics["rank_ic"][-1]
                if epoch < self.warmup:
                    return False
                if score > self.best_score:
                    self.best_score = score
                    self.best_iter = epoch
                    self.no_improve = 0
                else:
                    self.no_improve += 1
                if self.no_improve >= self.rounds:
                    return True
        return False


def train_xgb(X_tr, y_tr, X_va, y_va, ic_eval):
    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dva = xgb.DMatrix(X_va.values, label=y_va.values)

    def feval(predt, dmat):
        return "rank_ic", ic_eval(predt)

    def _fit(params):
        p = {k: v for k, v in params.items() if k != "disable_default_eval_metric"}
        p["eval_metric"] = "rmse"
        es = WarmupEarlyStopping(rounds=config.EARLY_STOP, warmup=100)
        m = xgb.train(p, dtr, num_boost_round=config.N_ROUNDS,
                      evals=[(dva, "valid")], custom_metric=feval,
                      callbacks=[es], verbose_eval=False)
        m._rankic_best_iter = es.best_iter
        return m, es.best_iter, float(es.best_score)

    m, bi, ic = _fit(config.XGB_PARAMS)
    # 收敛不足 fallback: 极小 best_iter 说明该窗口下超参不匹配(早停撞上噪声峰值),
    # 用备选超参重训, 按 valid RankIC 择优(仍只用 valid 集, 不碰 test, 无穿越)
    if bi < 30:
        for alt in ({"learning_rate": 0.02, "max_depth": 6, "reg_lambda": 20.0},
                    {"learning_rate": 0.01, "max_depth": 5, "reg_alpha": 3.0}):
            p2 = {**config.XGB_PARAMS, **alt}
            m2, bi2, ic2 = _fit(p2)
            print(f"    [XGB-fallback] {alt} → best_iter={bi2}, RankIC={ic2:.4f}", flush=True)
            if ic2 > ic and bi2 >= 30:
                m, bi, ic = m2, bi2, ic2
                break
            if ic2 > ic:
                m, bi, ic = m2, bi2, ic2
    return m, bi, ic


def train_lgb(X_tr, y_tr, X_va, y_va, ic_eval):
    tr = lgbm.Dataset(X_tr.values, label=y_tr.values)
    va = lgbm.Dataset(X_va.values, label=y_va.values, reference=tr)

    def feval(predt, ds):
        return "rank_ic", ic_eval(predt), True

    def _fit(params):
        m = lgbm.train(params, tr, num_boost_round=config.N_ROUNDS,
                       valid_sets=[va], valid_names=["valid"], feval=feval,
                       callbacks=[lgbm.early_stopping(config.EARLY_STOP, first_metric_only=True,
                                                      verbose=False)])
        return m, m.best_iteration, float(m.best_score["valid"]["rank_ic"])

    m, bi, ic = _fit(config.LGB_PARAMS)
    # 收敛不足 fallback: 同 XGB, 用更慢学习率/更浅树重训, 按 valid RankIC 择优(无穿越)
    if bi < 30:
        for alt in ({"learning_rate": 0.01, "max_depth": 6, "num_leaves": 48},
                    {"learning_rate": 0.005, "max_depth": 5, "num_leaves": 32}):
            p2 = {**config.LGB_PARAMS, **alt}
            m2, bi2, ic2 = _fit(p2)
            print(f"    [LGB-fallback] {alt} → best_iter={bi2}, RankIC={ic2:.4f}", flush=True)
            if ic2 > ic and bi2 >= 30:
                m, bi, ic = m2, bi2, ic2
                break
            if ic2 > ic:
                m, bi, ic = m2, bi2, ic2
    return m, bi, ic


class DoubleEnsembleRankIC:
    """DoubleEnsemble (SR + FS) with RankIC early stopping per sub-model.

    与 qlib DEnsembleModel 的区别:
    1. 每个子模型用 RankIC 早停 (qlib 原生用 RMSE), 与主引擎 XGB/LGB 口径一致
    2. 接受 numpy 矩阵 (与 train_xgb/train_lgb 同口径), 可注入 FCF 因子
       (qlib 原生走 DatasetH, 不注入 FCF)
    3. predict 直接对 numpy 矩阵输出, 无需 DatasetH

    SR (Sample Reweighting) 和 FS (Feature Selection) 逻辑与 qlib 原版一致:
    - SR: 按当前集成损失 + 训练曲线趋势给高损失样本更高权重
    - FS: shuffle 各特征后按损失增量分箱采样, 保留重要特征
    """

    def __init__(self, X_tr, y_tr, X_va, y_va, ic_eval, params, de_config):
        self.X_tr = X_tr
        self.y_tr = pd.Series(np.asarray(y_tr).ravel(), index=X_tr.index)
        self.X_va = X_va
        self.y_va = y_va
        self.ic_eval = ic_eval
        self.params = params
        self.cfg = de_config
        self.n_models = de_config["num_models"]
        self.sub_weights = de_config["sub_weights"]
        self.ensemble = []
        self.sub_features = []
        self.best_iters = []
        self.valid_ic = 0.0

    def _train_submodel(self, weights, features):
        X_tr_sub = self.X_tr.iloc[:, features] if isinstance(features, list) else self.X_tr
        X_va_sub = self.X_va.iloc[:, features] if isinstance(features, list) else self.X_va
        tr = lgbm.Dataset(X_tr_sub.values, label=self.y_tr.values, weight=weights)
        va = lgbm.Dataset(X_va_sub.values, label=np.asarray(self.y_va).ravel(), reference=tr)

        def feval(predt, ds):
            return "rank_ic", self.ic_eval(predt), True

        m = lgbm.train(self.params, tr, num_boost_round=config.N_ROUNDS,
                       valid_sets=[va], valid_names=["valid"], feval=feval,
                       callbacks=[lgbm.early_stopping(config.EARLY_STOP, first_metric_only=True,
                                                      verbose=False)])
        return m, m.best_iteration, float(m.best_score["valid"]["rank_ic"])

    @staticmethod
    def _get_loss(labels, preds):
        return (labels - preds) ** 2

    def _retrieve_loss_curve(self, model, features):
        """每棵树的累计损失曲线 (仅保留 SR 实际使用的前/后 10% 树)。

        性能: _sample_reweight 只用前 10% 和后 10% 树的均值 (l_start/l_end),
        中间 80% 树的列从未被使用 → 只在 checkpoint 树上存损失。
        用 float32 numpy 替代 pandas DataFrame (避免 205K×2000 大表 +
        pandas rank(axis=0) 的分钟级开销; 实测单子模型 30~60min → ~1.5min)。
        """
        X_tr_sub = self.X_tr.iloc[:, features] if isinstance(features, list) else self.X_tr
        n_trees = model.num_trees()
        n = len(self.X_tr)
        part = max(int(n_trees * 0.1), 1)
        # checkpoint 列: 前 part 棵 + 后 part 棵 (与原 l_start/l_end 窗口一致)
        if 2 * part >= n_trees:
            ckpts = list(range(n_trees))
        else:
            ckpts = list(range(part)) + list(range(n_trees - part, n_trees))
        col_of = {t: i for i, t in enumerate(ckpts)}
        loss_curve = np.zeros((n, len(ckpts)), dtype=np.float32)
        pred_tree = np.zeros(n, dtype=float)
        x_vals = X_tr_sub.values
        y_vals = self.y_tr.values
        for t in range(n_trees):
            pred_tree += model.predict(x_vals, start_iteration=t, num_iteration=1)
            i = col_of.get(t)
            if i is not None:
                loss_curve[:, i] = (y_vals - pred_tree) ** 2
        return loss_curve

    def _sample_reweight(self, loss_curve, loss_values, k_th):
        """SR: 高损失样本加权。loss_curve 为 (n, ckpts) float32 数组
        (前半列=前10%树, 后半列=后10%树), 与原 pandas 版语义一致。"""
        n, t = loss_curve.shape
        # 每列 pct-rank (numpy argsort, 分块控制峰值内存)
        ranks = np.empty((n, t), dtype=np.float32)
        pct = (np.arange(n, dtype=np.float32) + 1.0) / n
        for c0 in range(0, t, 64):
            c1 = min(c0 + 64, t)
            order = loss_curve[:, c0:c1].argsort(axis=0)
            ranks[order, np.arange(c0, c1)[None, :]] = pct[:, None]
        # ckpts 前半=前10%树, 后半=后10%树 → 各取前半/后半均值
        part = max(t // 2, 1)
        l_start = ranks[:, :part].mean(axis=1)
        l_end = ranks[:, -part:].mean(axis=1)
        h1 = (-loss_values).rank(pct=True).values
        h2 = pd.Series(l_end / l_start).rank(pct=True).values
        h = pd.DataFrame({"h_value": self.cfg["alpha1"] * h1 + self.cfg["alpha2"] * h2})
        h["bins"] = pd.cut(h["h_value"], self.cfg["bins_sr"])
        h_avg = h.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
        weights = pd.Series(np.zeros(n, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.cfg["decay"] ** k_th * h_avg[b] + 0.1)
        return weights

    def _feature_selection(self, loss_values, features_idx, features_names):
        x_train = self.X_tr
        y_vals = self.y_tr.values
        n, f = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(f, dtype=float)})
        m = len(self.ensemble)
        x_tmp = x_train.copy()
        for i_f in range(f):
            x_tmp.iloc[:, i_f] = np.random.permutation(x_tmp.iloc[:, i_f].values)
            pred = np.zeros(n)
            for i_s, (submodel, sub_feat) in enumerate(zip(self.ensemble, self.sub_features)):
                pred += submodel.predict(x_tmp.iloc[:, sub_feat].values) / m
            loss_feat = self._get_loss(y_vals, pred)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values.values) / \
                (np.std(loss_feat - loss_values.values) + 1e-7)
            x_tmp.iloc[:, i_f] = x_train.iloc[:, i_f].values
        g["g_value"] = g["g_value"].fillna(0)
        g["bins"] = pd.cut(g["g_value"], self.cfg["bins_fs"])
        res_feat = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            if i_b >= len(self.cfg["sample_ratios"]):
                break
            b_mask = g["bins"] == b
            b_feats = [j for j in range(f) if b_mask[j]]
            if not b_feats:
                continue
            num_feat = int(np.ceil(self.cfg["sample_ratios"][i_b] * len(b_feats)))
            sampled = np.random.choice(b_feats, size=min(num_feat, len(b_feats)), replace=False)
            res_feat.extend(sampled.tolist())
        return list(set(res_feat))

    def fit(self):
        n, f = self.X_tr.shape
        weights = pd.Series(np.ones(n, dtype=float))
        features = list(range(f))
        pred_sub = pd.DataFrame(np.zeros((n, self.n_models), dtype=float),
                                index=self.X_tr.index)
        valid_ics = []
        for k in range(self.n_models):
            self.sub_features.append(list(features))
            m_k, bi, vic = self._train_submodel(weights, list(features))
            self.ensemble.append(m_k)
            self.best_iters.append(bi)
            valid_ics.append(vic)
            if k + 1 == self.n_models:
                break
            loss_curve = self._retrieve_loss_curve(m_k, list(features))
            pred_k = pd.Series(m_k.predict(self.X_tr.iloc[:, features].values),
                               index=self.X_tr.index)
            pred_sub.iloc[:, k] = pred_k
            pred_ens = (pred_sub.iloc[:, :k + 1] *
                        self.sub_weights[:k + 1]).sum(axis=1) / sum(self.sub_weights[:k + 1])
            loss_values = pd.Series(self._get_loss(self.y_tr.values, pred_ens.values))
            if self.cfg["enable_sr"]:
                weights = self._sample_reweight(loss_curve, loss_values, k + 1)
            if self.cfg["enable_fs"]:
                features = self._feature_selection(loss_values, features, None)
        self.valid_ic = float(np.mean(valid_ics))
        return self

    def predict(self, X_test):
        n = len(X_test)
        pred = np.zeros(n, dtype=float)
        for i_s, (submodel, sub_feat) in enumerate(zip(self.ensemble, self.sub_features)):
            pred += submodel.predict(X_test.iloc[:, sub_feat].values) * self.sub_weights[i_s]
        return pred / sum(self.sub_weights)


def train_de(X_tr, y_tr, X_va, y_va, ic_eval):
    """DoubleEnsemble (SR + FS) with RankIC early stopping.

    与 train_xgb/train_lgb 同口径: 接受 DataFrame (含 FCF 因子), RankIC 早停,
    best_iter<30 时 fallback. 返回 (fitted_model, best_iter, valid_rank_ic)。
    """
    de = DoubleEnsembleRankIC(X_tr, y_tr, X_va, y_va, ic_eval,
                              config.DE_LGB_PARAMS, config.DE_CONFIG)
    de.fit()
    mean_bi = float(np.mean(de.best_iters))
    valid_ic = de.valid_ic

    if mean_bi < 30:
        for alt in ({"learning_rate": 0.02, "max_depth": 6, "num_leaves": 48},
                    {"learning_rate": 0.01, "max_depth": 5, "num_leaves": 32}):
            p2 = {**config.DE_LGB_PARAMS, **alt}
            de2 = DoubleEnsembleRankIC(X_tr, y_tr, X_va, y_va, ic_eval, p2, config.DE_CONFIG)
            de2.fit()
            print(f"    [DE-fallback] {alt} → mean_best_iter={np.mean(de2.best_iters):.0f}, "
                  f"RankIC={de2.valid_ic:.4f}", flush=True)
            if de2.valid_ic > valid_ic and np.mean(de2.best_iters) >= 30:
                de, mean_bi, valid_ic = de2, float(np.mean(de2.best_iters)), de2.valid_ic
                break
            if de2.valid_ic > valid_ic:
                de, mean_bi, valid_ic = de2, float(np.mean(de2.best_iters)), de2.valid_ic
    return de, int(mean_bi), valid_ic


def check_convergence(tag, best_iter, valid_ic):
    if best_iter is None or best_iter <= 0:
        raise RuntimeError(f"[CHECK] {tag} best_iter={best_iter}, 模型退化为常数输出!")
    warn = []
    if best_iter < 30:
        warn.append(f"best_iter={best_iter}<30 收敛轮数偏少")
    if valid_ic < 0.01:
        warn.append(f"valid RankIC={valid_ic:.4f}<0.01 预测力弱")
    msg = " | ".join(warn) if warn else "OK"
    print(f"    [{tag}] best_iter={best_iter}, valid RankIC={valid_ic:.4f} → {msg}", flush=True)


# ==================== 通用 qlib 内置模型训练器 (基准 sweep 用) ====================
def _n_features(dataset):
    """从 handler 取实际特征列数 (Alpha158Enhanced ≈ 172), 用于注入 d_feat/input_dim"""
    from qlib.data.dataset.handler import DataHandlerLP
    cols = dataset.handler.get_cols(col_set="feature")
    return len(cols)


def train_qlib_model(name, dataset, seed_key=None):
    """加载并训练 qlib 内置模型, 返回 (fitted_model, pred_test_series)。
    - 按 config.MODEL_CONFIGS[name] 构造模型; DL/NN 按实际特征数注入维度;
    - fit(dataset) 仅用 train/valid 段 (dataset 已按 embargo 分段, 无穿越);
    - predict 得信号段截面预测 Series。
    - seed_key: 复现性锚点 (如 "DoubleEnsemble:W2026"), fit 前按其哈希重置全局
      随机流, 保证结果与执行路径(续跑/单跑/全跑)无关。DE 的特征采样用裸
      np.random, 不重置则每次重训结果都不同。"""
    import random as _random
    import hashlib
    from qlib.utils import init_instance_by_config
    if seed_key is not None:
        s = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) % (2 ** 31)
        _random.seed(s)
        np.random.seed(s)
        try:
            import torch
            torch.manual_seed(s)
        except ImportError:
            pass
    cfg = config.MODEL_CONFIGS[name]
    kwargs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg["kwargs"].items()}
    dfeat = cfg.get("dfeat")
    if dfeat:
        n_feat = _n_features(dataset)
        if dfeat == "d_feat":
            kwargs["d_feat"] = n_feat
        elif dfeat == "mlp":
            kwargs.setdefault("pt_model_kwargs", {})["input_dim"] = n_feat
        print(f"    [{name}] 注入特征维度 = {n_feat}", flush=True)
    model = init_instance_by_config({"class": cfg["class"],
                                     "module_path": cfg["module_path"], "kwargs": kwargs})
    model.fit(dataset)
    # qlib DatasetH 模型 predict(dataset, segment="test"); 而 pytorch TS 序列模型
    # (GRU/LSTM/ALSTM/GATs/TCN/Localformer/Transformer) 的 predict 无 segment 形参, 默认预测 test 段.
    try:
        pred = model.predict(dataset, segment="test")
    except TypeError:
        pred = model.predict(dataset)
    import pandas as pd
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    return model, pred
