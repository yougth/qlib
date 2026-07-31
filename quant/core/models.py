"""
core.models —— RankIC 早停训练器 (XGB/LGB + 收敛 fallback) + 收敛检查 + 通用 qlib 训练器
================================================================================
- RankICEval: 预计算分组, 每轮 boosting 高速评估验证集日度 RankIC
- train_xgb / train_lgb: 验证集 RankIC 最大化早停 (严禁 RMSE 早停退化);
  best_iter<30 时用备选超参重训, 按 valid RankIC 择优 (仅用valid集, 不碰test, 无穿越)
- check_convergence: best_iter==0 报错; best_iter<30 或 valid RankIC<0.01 高亮告警
- train_qlib_model: 用 init_instance_by_config 加载 qlib 内置模型 (基准 sweep 用),
  按实际特征数注入 d_feat/input_dim, fit(train/valid) 后对 test 段 predict, 全程无穿越。
"""
import numpy as np
import xgboost as xgb
import lightgbm as lgbm

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


def train_xgb(X_tr, y_tr, X_va, y_va, ic_eval):
    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dva = xgb.DMatrix(X_va.values, label=y_va.values)

    def feval(predt, dmat):
        return "rank_ic", ic_eval(predt)

    def _fit(params):
        m = xgb.train(params, dtr, num_boost_round=config.N_ROUNDS,
                      evals=[(dva, "valid")], custom_metric=feval, maximize=True,
                      early_stopping_rounds=config.EARLY_STOP, verbose_eval=False)
        return m, m.best_iteration, float(m.best_score)

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


def train_qlib_model(name, dataset):
    """加载并训练 qlib 内置模型, 返回 (fitted_model, pred_test_series)。
    - 按 config.MODEL_CONFIGS[name] 构造模型; DL/NN 按实际特征数注入维度;
    - fit(dataset) 仅用 train/valid 段 (dataset 已按 embargo 分段, 无穿越);
    - predict 得信号段截面预测 Series。"""
    from qlib.utils import init_instance_by_config
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
