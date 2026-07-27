"""诊断inject_features: 检查fund/vf是否真正进入handler._data"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
import warnings
warnings.filterwarnings("ignore")

from v5_validation import Alpha158Enhanced, load_value_factors, load_fundamental_features

MODEL_CONFIG = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421,
        "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
        "max_depth": 8, "num_leaves": 210, "num_threads": 4}}

def format_qlib_code(code):
    c = str(code).zfill(6)
    return f"SH{c}" if c.startswith("6") else f"SZ{c}"

def build_dynamic_universe(backtest_year, fcf_df, profit_df):
    fcf_start = backtest_year - 11; fcf_end = backtest_year - 2
    profit_start = max(backtest_year - 11, 2016); profit_end = backtest_year - 2
    target_fcf_years = list(range(fcf_start, fcf_end + 1))
    target_profit_years = list(range(profit_start, profit_end + 1))
    fcf_filtered = fcf_df[fcf_df["year"].isin(target_fcf_years)]
    fcf_positive = fcf_filtered.groupby("code").filter(
        lambda g: len(g) >= len(target_fcf_years) * 0.8 and (g["fcf"] > 0).all())
    fcf_codes = set(fcf_positive["code"].unique())
    if profit_end >= 2016 and len(target_profit_years) > 0:
        profit_filtered = profit_df[profit_df["year"].isin(target_profit_years)]
        profit_positive = profit_filtered.groupby("code").filter(
            lambda g: len(g) >= len(target_profit_years) * 0.8 and (g["net_profit"] > 0).all())
        profit_codes = set(profit_positive["code"].unique())
    else:
        profit_codes = fcf_codes
    return fcf_codes & profit_codes

def inject_features_v2(dataset, feature_df, label="feature"):
    """修复版inject_features: 带完整诊断"""
    handler = dataset.handler
    data = handler.fetch()

    print(f"  [{label}] handler data: shape={data.shape}, columns type={type(data.columns)}")
    feat_cols_before = [c for c in data.columns if (isinstance(c, tuple) and c[0] == "feature") or (isinstance(c, str) and c.startswith("feature"))]
    print(f"  [{label}] feature columns before: {len(feat_cols_before)}")

    # 检查feature_df
    print(f"  [{label}] feature_df: shape={feature_df.shape}, index names={feature_df.index.names}")
    print(f"  [{label}] feature_df columns: {list(feature_df.columns)}")
    print(f"  [{label}] feature_df non-null: {feature_df.notna().sum().sum()}/{feature_df.size}")

    # 检查index对齐
    handler_index = data.index
    feature_index = feature_df.index
    common = handler_index.intersection(feature_index)
    print(f"  [{label}] handler index: {len(handler_index)}, feature index: {len(feature_index)}, common: {len(common)}")

    if len(common) == 0:
        print(f"  [{label}] ERROR: No common index! Handler index sample: {handler_index[:3]}, Feature index sample: {feature_index[:3]}")
        return False

    # Reindex
    aligned = feature_df.reindex(handler_index)
    non_null_after = aligned.notna().sum().sum()
    print(f"  [{label}] after reindex: non-null={non_null_after}/{aligned.size} ({non_null_after/aligned.size*100:.1f}%)")

    # 设置列格式
    if isinstance(data.columns, pd.MultiIndex):
        aligned.columns = pd.MultiIndex.from_tuples([("feature", c) for c in aligned.columns])
    else:
        aligned.columns = [("feature", c) for c in aligned.columns]

    # Join
    handler._data = data.join(aligned).fillna(0)

    # 验证
    data_after = handler.fetch()
    feat_cols_after = [c for c in data_after.columns if (isinstance(c, tuple) and c[0] == "feature") or (isinstance(c, str) and c.startswith("feature"))]
    print(f"  [{label}] feature columns after: {len(feat_cols_after)}")
    new_cols = [c for c in feat_cols_after if c not in feat_cols_before]
    print(f"  [{label}] NEW feature columns: {len(new_cols)} → {[c[1] if isinstance(c, tuple) else c for c in new_cols[:10]]}")

    # 检查注入的列是否非零
    for col in new_cols[:5]:
        vals = data_after[col]
        non_zero = (vals != 0).sum()
        print(f"  [{label}] {col}: non-zero={non_zero}/{len(vals)} ({non_zero/len(vals)*100:.1f}%)")

    return True


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv")
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv")

    codes = build_dynamic_universe(2026, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"Universe: {len(universe)} stocks")

    cal = D.calendar(start_time="2021-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    ts, te = "2021-01-01", "2024-12-31"
    vs, ve = "2025-01-01", "2025-12-31"
    bs, be = "2026-01-01", "2026-07-21"

    # Create dataset
    dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
        "instruments": universe,
        "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                              {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label": ["Ref($close, -20) / $close - 1"]}
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"v5_validation","kwargs":dhc},
                  "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}

    print("\n=== Creating dataset ===")
    dataset = init_instance_by_config(dsc)

    # 检查handler数据
    handler = dataset.handler
    data = handler.fetch()
    print(f"Handler data shape: {data.shape}")
    print(f"Columns sample: {list(data.columns[:5])}")
    feat_cols = [c for c in data.columns if isinstance(c, tuple) and c[0] == "feature"]
    print(f"Feature columns: {len(feat_cols)}")

    # 加载fund
    print("\n=== Loading fundamental features ===")
    fund = load_fundamental_features(universe, cal)
    if fund is not None:
        print(f"Fund shape: {fund.shape}")
        print(f"Fund index names: {fund.index.names}")
        print(f"Fund index sample: {fund.index[:3].tolist()}")
        print(f"Fund columns: {list(fund.columns)}")
        # 检查handler index vs fund index
        handler_idx_sample = data.index[:3].tolist()
        fund_idx_sample = fund.index[:3].tolist()
        print(f"Handler index sample: {handler_idx_sample}")
        print(f"Fund index sample: {fund_idx_sample}")
        # 检查数据类型是否匹配
        h_dt = type(handler_idx_sample[0][0])
        f_dt = type(fund_idx_sample[0][0])
        print(f"Handler datetime type: {h_dt}, Fund datetime type: {f_dt}")
    else:
        print("Fund is None!")

    # 加载vf
    print("\n=== Loading value factors ===")
    vf = load_value_factors(universe, cal, cal_set)
    if vf is not None:
        print(f"VF shape: {vf.shape}")
        print(f"VF index names: {vf.index.names}")
        print(f"VF index sample: {vf.index[:3].tolist()}")
        print(f"VF columns: {list(vf.columns)}")
    else:
        print("VF is None!")

    # 注入测试
    print("\n=== Inject test: fund ===")
    if fund is not None:
        inject_features_v2(dataset, fund, "fund")

    print("\n=== Inject test: vf ===")
    if vf is not None:
        inject_features_v2(dataset, vf, "vf")

    # 检查model能看到的特征
    print("\n=== Model feature check ===")
    train_data = dataset.prepare("train", col_set="feature")
    print(f"Train data shape: {train_data.shape}")
    print(f"Train columns: {list(train_data.columns[:10])}...{list(train_data.columns[-10:])}")

    # 检查基本面列是否在训练数据中
    fund_cols = [c for c in train_data.columns if isinstance(c, tuple) and c[1] in
                 ["fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"]]
    vf_cols = [c for c in train_data.columns if isinstance(c, tuple) and c[1] in
               ["roe_annual", "pb_pct_3y", "div_yield_est", "pe_pct_3y"]]
    print(f"Fundamental columns in train: {len(fund_cols)} → {[c[1] for c in fund_cols]}")
    print(f"Value columns in train: {len(vf_cols)} → {[c[1] for c in vf_cols]}")

    # 检查非零比例
    for col in fund_cols + vf_cols:
        vals = train_data[col]
        non_zero = (vals != 0).sum()
        print(f"  {col[1]}: non-zero={non_zero}/{len(vals)} ({non_zero/len(vals)*100:.1f}%), mean={vals.mean():.4f}, std={vals.std():.4f}")


if __name__ == "__main__":
    run()
