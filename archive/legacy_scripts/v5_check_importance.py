"""
特征重要性归因审查脚本 (XGBoost版)
=========================================
提取指定窗口训练好的XGBoost模型，按Gain（分裂增益）打印Top 30特征。
"""
import os
import sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
import xgboost as xgb

# 复用之前的配置与Top80特征
from v5_new_baseline_xgb import WINDOWS, TOP80_FEATURES, MODEL_CONFIG, build_dynamic_universe, format_qlib_code
from v5_validation import Alpha158Enhanced, load_value_factors
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed

def run_importance_check():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    cal = qlib.data.D.calendar(start_time="2013-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    # 以 W4 窗口（回测 2024 年）为例进行特征归因审查
    win = [w for w in WINDOWS if w["name"] == "W4"][0]
    print(f"\n[+] 正在加载 {win['name']} 窗口数据并训练模型以提取特征重要性...")

    by = win["year"]
    codes = build_dynamic_universe(by, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)

    ts, te = win["train"]
    vs, ve = win["valid"]
    bs, be = win["backtest"]

    dhc = {"start_time": ts, "end_time": be, "fit_start_time": ts, "fit_end_time": te,
        "instruments": universe,
        "infer_processors": [{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                              {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
        "learn_processors": [{"class":"DropnaLabel"}, {"class":"CSZScoreNorm","kwargs":{"fields_group":"label"}}],
        "label": ["Ref($close, -20) / $close - 1"]}
    dsc = {"class":"DatasetH","module_path":"qlib.data.dataset",
        "kwargs":{"handler":{"class":"Alpha158Enhanced","module_path":"v5_validation","kwargs":dhc},
                  "segments":{"train":(ts,te),"valid":(vs,ve),"test":(bs,be)}}}
    dataset = init_instance_by_config(dsc)

    vf = load_value_factors(universe, cal, cal_set)
    fund = load_fundamental_features_fixed(universe, cal)
    if fund is not None:
        inject_features_fixed(dataset, fund, "fund")
    if vf is not None:
        inject_features_fixed(dataset, vf, "vf")

    # 训练模型
    model = init_instance_by_config(MODEL_CONFIG)
    model.fit(dataset)

    # 提取 XGBoost Booster 的特征 Gain 重要性
    booster = model.model
    importance_dict = booster.get_score(importance_type='gain')
    
    # 转换为 DataFrame 排序
    imp_df = pd.DataFrame(list(importance_dict.items()), columns=["feature", "gain"])
    imp_df = imp_df.sort_values(by="gain", ascending=False).reset_index(drop=True)
    imp_df["gain_pct"] = imp_df["gain"] / imp_df["gain"].sum() * 100

    print(f"\n{'='*50}")
    print(f"  XGBoost 特征重要性 Top 30 (按 Gain 排序)")
    print(f"{'='*50}")
    print(f"{'排名':<6} {'特征名称':<25} {'Gain占比':<10}")
    print(f"{'-'*45}")
    
    for idx, row in imp_df.head(30).iterrows():
        is_fundamental = row["feature"] in ["roe_annual", "pb_pct_3y", "pe_pct_3y", "div_yield_est", "fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"]
        tag = " [基本面]" if is_fundamental else ""
        print(f"{(idx+1):<6} {row['feature']:<25} {row['gain_pct']:>6.2f}%{tag}")

    print(f"\n[+] 检查完毕。如果基本面因子在前列，说明模型确实在利用价值逻辑做判断。")

if __name__ == "__main__":
    run_importance_check()
