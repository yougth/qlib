"""
core.dataset —— 无穿越数据层 (滚动引擎与27模型基准共用的唯一入口)
================================================================================
单点保证无穿越:
  · train/valid/test 分段: valid 截至 (Y-1)-11-30 embargo 留白, 防20日label偷看测试期
  · label 仅后向 Ref($close,-20); infer 用 train 段 fit 的 RobustZScoreNorm(无未来fit)
  · learn 段 CSZScoreNorm 按截面; 基本面 PIT 因子 Y+1-05-01 起生效, 按日截面MAD-zscore
  · 行情覆盖率 / 样本量 / NaN 比例三重 check, 拒绝静默异常
"""
import gc
import numpy as np
import pandas as pd
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

from . import config
from .features import load_fund_features

DEFAULT_LABEL = ["Ref($close, -20) / $close - 1"]


def _handler_config(universe, ts, te, xe, label):
    return {"start_time": ts, "end_time": xe, "fit_start_time": ts, "fit_end_time": te,
            "instruments": universe,
            "infer_processors": [
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
            "label": label}


def build_dataset(win, universe, fcf_df, profit_df, cal, label=None, inject_fund=True):
    """构造无穿越训练矩阵. 返回 dict:
    X_tr,y_tr,X_va,y_va,test_X,n_inst,cov,feat_cols"""
    ts, te = win["train"]
    vs, ve = win["valid"]
    xs, xe = win["test"]
    label = label or DEFAULT_LABEL

    dhc = _handler_config(universe, ts, te, xe, label)
    dsc = {"class": "DatasetH", "module_path": "qlib.data.dataset",
           "kwargs": {"handler": {"class": "Alpha158Enhanced",
                                  "module_path": "core.features", "kwargs": dhc},
                      "segments": {"train": (ts, te), "valid": (vs, ve),
                                   "test": (xs, xe)}}}
    dataset = init_instance_by_config(dsc)
    train_df = dataset.prepare("train", col_set=["feature", "label"],
                               data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature", "label"],
                               data_key=DataHandlerLP.DK_L)
    test_X = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    del dataset
    gc.collect()

    # ---- check: 行情覆盖率 ----
    n_inst = train_df.index.get_level_values(1).nunique()
    cov = n_inst / len(universe)
    print(f"  [CHECK] 行情覆盖: {n_inst}/{len(universe)} ({cov*100:.0f}%)", flush=True)
    if cov < 0.6:
        raise RuntimeError(f"[CHECK] 行情覆盖率仅{cov*100:.0f}%, qlib数据异常!")

    X_tr, y_tr = train_df["feature"], train_df["label"].iloc[:, 0]
    X_va, y_va = valid_df["feature"], valid_df["label"].iloc[:, 0]
    mask = y_va.notna()
    X_va, y_va = X_va[mask], y_va[mask]
    if len(X_tr) < 10000 or len(X_va) < 1000:
        raise RuntimeError(f"[CHECK] 样本量异常: train={len(X_tr)}, valid={len(X_va)}!")

    if inject_fund:
        fund = load_fund_features(universe, fcf_df, profit_df, cal, ts, xe)
        X_tr = pd.concat([X_tr, fund.reindex(X_tr.index).fillna(0)], axis=1).astype(np.float32)
        X_va = pd.concat([X_va, fund.reindex(X_va.index).fillna(0)], axis=1).astype(np.float32)
        test_X = pd.concat([test_X, fund.reindex(test_X.index).fillna(0)], axis=1).astype(np.float32)
        del fund
    else:
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)
        test_X = test_X.astype(np.float32)

    nan_pct = X_tr.isna().values.mean() * 100
    print(f"  特征矩阵: train {X_tr.shape}, valid {X_va.shape}, test {test_X.shape}, "
          f"NaN={nan_pct:.2f}%", flush=True)
    if nan_pct > 5:
        raise RuntimeError(f"[CHECK] 特征NaN比例{nan_pct:.1f}%过高!")
    del train_df, valid_df
    gc.collect()

    return {"X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va,
            "test_X": test_X, "n_inst": n_inst, "cov": cov,
            "feat_cols": list(X_tr.columns)}


def build_datasetH(seg, universe, label=None, ds_class="DatasetH", step_len=20):
    """构造 qlib 原生 Dataset (DatasetH / TSDatasetH), 供 27 模型基准统一调用。
    与滚动引擎共用同一 handler 口径 (Alpha158Enhanced + 同 processors + 同 embargo 分段 +
    同 20 日后向 label), 单点保证无穿越。DL 序列模型需 TSDatasetH(step_len)。
    注: 基准走 qlib 原生 model API, 不注入 build_dataset 的 5 个 FCF 因子。"""
    ts, te = seg["train"]
    vs, ve = seg["valid"]
    xs, xe = seg["test"]
    label = label or DEFAULT_LABEL
    dhc = _handler_config(universe, ts, te, xe, label)
    kwargs = {"handler": {"class": "Alpha158Enhanced", "module_path": "core.features",
                          "kwargs": dhc},
              "segments": {"train": (ts, te), "valid": (vs, ve), "test": (xs, xe)}}
    if ds_class == "TSDatasetH":
        kwargs["step_len"] = step_len
    dsc = {"class": ds_class, "module_path": "qlib.data.dataset", "kwargs": kwargs}
    return init_instance_by_config(dsc)


def build_datasetH(seg, universe, label=None, ds_class="DatasetH", step_len=20):
    """构造 qlib 原生 Dataset (DatasetH / TSDatasetH), 供 27 模型基准统一调用。
    与滚动引擎共用同一 handler 口径 (Alpha158Enhanced + 同 processors + 同 embargo 分段 +
    同 20 日后向 label), 单点保证无穿越。DL 序列模型需 TSDatasetH(step_len)。
    注: 基准走 qlib 原生 model API, 不注入 build_dataset 的 5 个 FCF 因子。"""
    ts, te = seg["train"]
    vs, ve = seg["valid"]
    xs, xe = seg["test"]
    label = label or DEFAULT_LABEL
    dhc = _handler_config(universe, ts, te, xe, label)
    kwargs = {"handler": {"class": "Alpha158Enhanced", "module_path": "core.features",
                          "kwargs": dhc},
              "segments": {"train": (ts, te), "valid": (vs, ve), "test": (xs, xe)}}
    if ds_class == "TSDatasetH":
        kwargs["step_len"] = step_len
    dsc = {"class": ds_class, "module_path": "qlib.data.dataset", "kwargs": kwargs}
    return init_instance_by_config(dsc)


def build_datasetH(seg, universe, label=None, ds_class="DatasetH", step_len=20):
    """构造 qlib 原生 Dataset (DatasetH / TSDatasetH), 供 27 模型基准统一调用。
    与滚动引擎共用同一 handler 口径 (Alpha158Enhanced + 同 processors + 同 embargo 分段 +
    同 20 日后向 label), 单点保证无穿越。DL 序列模型需 TSDatasetH(step_len)。

    seg: dict(train=(ts,te), valid=(vs,ve), test=(xs,xe))
    返回: qlib Dataset 实例 (喂给 init_instance_by_config 出来的模型 .fit/.predict)

    注: 基准走 qlib 原生 model API, 不注入 build_dataset 的 5 个 FCF 因子
    (那是滚动引擎 XGB/LGB 的专属增强); 基准比较"同一 Alpha158Enhanced 特征下模型架构优劣"。
    """
    ts, te = seg["train"]
    vs, ve = seg["valid"]
    xs, xe = seg["test"]
    label = label or DEFAULT_LABEL
    dhc = _handler_config(universe, ts, te, xe, label)
    kwargs = {"handler": {"class": "Alpha158Enhanced", "module_path": "core.features",
                          "kwargs": dhc},
              "segments": {"train": (ts, te), "valid": (vs, ve), "test": (xs, xe)}}
    if ds_class == "TSDatasetH":
        kwargs["step_len"] = step_len
    dsc = {"class": ds_class, "module_path": "qlib.data.dataset", "kwargs": kwargs}
    return init_instance_by_config(dsc)


def build_datasetH(seg, universe, label=None, ds_class="DatasetH", step_len=20):
    """构造 qlib 原生 Dataset (DatasetH / TSDatasetH), 供 27 模型基准统一调用。
    与滚动引擎共用同一 handler 口径 (Alpha158Enhanced + 同 processors + 同 embargo 分段 +
    同 20 日后向 label), 单点保证无穿越。DL 序列模型需 TSDatasetH(step_len)。

    seg: dict(train=(ts,te), valid=(vs,ve), test=(xs,xe))
    返回: qlib Dataset 实例 (可直接喂给 init_instance_by_config 出来的模型 .fit/.predict)

    注: 基准走 qlib 原生 model API, 不注入 build_dataset 的 5 个 FCF 因子
    (那是滚动引擎 XGB/LGB 的专属增强); 基准比较"同一 Alpha158Enhanced 特征下模型架构优劣"。
    """
    ts, te = seg["train"]
    vs, ve = seg["valid"]
    xs, xe = seg["test"]
    label = label or DEFAULT_LABEL
    dhc = _handler_config(universe, ts, te, xe, label)
    kwargs = {"handler": {"class": "Alpha158Enhanced", "module_path": "core.features",
                          "kwargs": dhc},
              "segments": {"train": (ts, te), "valid": (vs, ve), "test": (xs, xe)}}
    if ds_class == "TSDatasetH":
        kwargs["step_len"] = step_len
    dsc = {"class": ds_class, "module_path": "qlib.data.dataset", "kwargs": kwargs}
    return init_instance_by_config(dsc)


def build_datasetH(seg, universe, label=None, ds_class="DatasetH", step_len=20):
    """构造 qlib 原生 Dataset (DatasetH / TSDatasetH), 供 27 模型基准统一调用。
    与滚动引擎共用同一 handler 口径 (Alpha158Enhanced + 同 processors + 同 embargo 分段 +
    同 20 日后向 label), 单点保证无穿越。DL 序列模型需 TSDatasetH(step_len)。

    seg: dict(train=(ts,te), valid=(vs,ve), test=(xs,xe))
    返回: qlib Dataset 实例 (可直接喂给 init_instance_by_config 出来的模型 .fit/.predict)

    注: 基准走 qlib 原生 model API, 不注入 core.dataset.build_dataset 的 5 个 FCF 因子
    (那是滚动引擎 XGB/LGB 的专属增强); 基准比较的是"同一 Alpha158Enhanced 特征下模型架构优劣"。
    """
    ts, te = seg["train"]
    vs, ve = seg["valid"]
    xs, xe = seg["test"]
    label = label or DEFAULT_LABEL
    dhc = _handler_config(universe, ts, te, xe, label)
    kwargs = {"handler": {"class": "Alpha158Enhanced", "module_path": "core.features",
                          "kwargs": dhc},
              "segments": {"train": (ts, te), "valid": (vs, ve), "test": (xs, xe)}}
    if ds_class == "TSDatasetH":
        kwargs["step_len"] = step_len
    dsc = {"class": ds_class, "module_path": "qlib.data.dataset", "kwargs": kwargs}
    return init_instance_by_config(dsc)
