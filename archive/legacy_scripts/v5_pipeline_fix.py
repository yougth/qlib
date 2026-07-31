"""
V5 修复注入 + 特征重要性 + 双层拦截管线
========================================
1. 修复inject_features: 用pd.concat替代join, 保持列格式一致
2. 修复load_fundamental_features: code统一zfill(6), 日频映射
3. 重训W6, 打出Feature Importance报告
4. 双层拦截: 宏观200日MA→70%仓位 → 30%基本面滑坡剔除 → LGBM双通道 → α=0.3价值融合
"""
import os, sys
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from scipy import stats
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.backtest import backtest as qlib_backtest
import warnings
warnings.filterwarnings("ignore")

from v5_validation import Alpha158Enhanced, MonthlyTopkStrategy, \
    apply_value_fusion, build_limit_up_set, filter_pred_by_tradability, \
    load_value_factors

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


# ==================== 修复版: load_fundamental_features ====================
def load_fundamental_features_fixed(universe, cal):
    """修复版: code统一zfill(6), 年报数据按披露滞后映射到日频"""
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    # 关键修复: 统一code格式
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    universe_codes = {c[2:] for c in universe}
    fcf_df = fcf_df[fcf_df["code"].isin(universe_codes)]
    profit_df = profit_df[profit_df["code"].isin(universe_codes)]
    merged = fcf_df[["code", "year", "fcf"]].merge(
        profit_df[["code", "year", "net_profit"]], on=["code", "year"], how="inner")
    merged = merged.dropna(subset=["fcf", "net_profit"]).sort_values(["code", "year"])

    cal_set = set(cal)
    rows = []
    for code, grp in merged.groupby("code"):
        qc = format_qlib_code(code)
        grp = grp.copy().sort_values("year")
        grp["fcf_growth"] = grp["fcf"].pct_change()
        grp["profit_growth"] = grp["net_profit"].pct_change()
        grp["fcf_profit_ratio"] = grp["fcf"] / (grp["net_profit"].abs() + 1e-8)
        grp["fcf_avg_3y"] = grp["fcf"].rolling(3, min_periods=1).mean()
        grp["fcf_cv_3y"] = grp["fcf"].rolling(3, min_periods=2).std() / (grp["fcf"].rolling(3, min_periods=2).mean().abs() + 1e-8)
        years_list = sorted(grp["year"].unique())
        for i, (_, r) in enumerate(grp.iterrows()):
            if pd.isna(r["fcf_growth"]): continue
            year = int(r["year"])
            # 年报披露滞后: Y年报在Y+1年4月底前披露
            avail_from = pd.Timestamp(f"{year + 1}-05-01")
            if i + 1 < len(years_list):
                avail_to = pd.Timestamp(f"{years_list[i+1] + 1}-04-30")
            else:
                avail_to = pd.Timestamp(f"{year + 2}-04-30")
            mask = (cal >= avail_from) & (cal <= avail_to)
            dates = cal[mask]
            for d in dates:
                if d not in cal_set: continue
                rows.append({"instrument": qc, "datetime": d,
                             "fcf_growth": r["fcf_growth"],
                             "profit_growth": r["profit_growth"],
                             "fcf_profit_ratio": r["fcf_profit_ratio"],
                             "fcf_avg_3y_norm": r["fcf_avg_3y"] / 1e8,
                             "fcf_cv_3y": r["fcf_cv_3y"]})
    if not rows:
        print("  [WARNING] 基本面特征为空!")
        return None
    fdf = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    fdf = fdf[~fdf.index.duplicated(keep="last")]
    # 去极值+标准化: 截面归一化(按日期), 避免全局归一化的时间泄露
    for col in fdf.columns:
        grp = fdf[col].groupby(level=0)  # level=0 = datetime
        median = grp.transform("median")
        mad = grp.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
        fdf[col] = ((fdf[col] - median) / (1.4826 * mad)).clip(-3, 3).fillna(0)
    print(f"  基本面特征: {fdf.shape}, {len(fdf.index.get_level_values(1).unique())}只股票")
    return fdf


# ==================== 修复版 V3: inject_features ====================
# 根因: DataHandlerLP有三层数据 (_data → _infer → _learn)
#   - fetch() 默认返回 _infer (DK_I)
#   - model.fit() 使用 _learn (DK_L)
#   - model.predict() 使用 _infer (DK_I)
# 之前修改 _data 无效, 因为 _infer/_learn 在 __init__ 时已从 _data 处理生成
# 修复: 直接修改 _infer 和 _learn
def inject_features_fixed(dataset, feature_df, label=""):
    """V3: 直接修改 _infer 和 _learn, 确保基本面/价值因子进入模型训练和预测"""
    handler = dataset.handler
    success_count = 0

    for attr_name in ["_infer", "_learn", "_data"]:
        if not hasattr(handler, attr_name):
            continue
        df = getattr(handler, attr_name)
        if df is None:
            continue

        n_before = df.shape[1]
        is_multi = isinstance(df.columns, pd.MultiIndex)

        # Reindex 到当前数据的index
        aligned = feature_df.reindex(df.index).fillna(0)

        if is_multi:
            # MultiIndex列: 添加为 ("feature", name) 元组
            aligned.columns = pd.MultiIndex.from_tuples(
                [("feature", c) for c in aligned.columns])
        # else: flat列保持纯字符串

        # 用concat合并
        new_df = pd.concat([df, aligned], axis=1)
        setattr(handler, attr_name, new_df)

        # 验证
        df_after = getattr(handler, attr_name)
        n_after = df_after.shape[1]
        n_new = n_after - n_before
        col_type = "MultiIndex" if is_multi else "flat"
        print(f"  [{label}] {attr_name}: {n_before}→{n_after}列 (新增{n_new}, {col_type})")

        if n_new > 0 and attr_name in ["_infer", "_learn"]:
            success_count += 1
            # 检查非零比例
            for col in aligned.columns:
                col_key = col
                vals = df_after[col_key]
                non_zero = (vals != 0).sum()
                name = col[1] if isinstance(col, tuple) else col
                pct = non_zero / len(vals) * 100 if len(vals) > 0 else 0
                print(f"    {name}: non-zero={non_zero}/{len(vals)} ({pct:.1f}%)")

    return success_count > 0


# ==================== 基本面滑坡硬过滤 ====================
def filter_by_fundamental_deterioration(pred, fcf_df, profit_df, threshold=0.30, signal_date=None):
    """剔除最新年报FCF或净利润同比下降超过threshold的股票"""
    pred_filtered = pred.copy()
    # 获取信号日时可用的最新年报年份
    if signal_date is not None:
        avail_year = signal_date.year - 2 if signal_date.month < 5 else signal_date.year - 1
    else:
        avail_year = pred.index.get_level_values(0)[-1].year - 1

    excluded = []
    dates = pred.index.get_level_values(0).unique()
    for dt in dates:
        # 每个日期对应可用的最新年报
        ay = dt.year - 2 if dt.month < 5 else dt.year - 1
        day_pred = pred.loc[[dt]] if dt in pred.index.get_level_values(0) else None
        if day_pred is None: continue
        instruments = day_pred.index.get_level_values(1)
        for inst in instruments:
            code = inst[2:]
            # 获取ay年和ay-1年的FCF/净利润
            fcf_cur = fcf_df[(fcf_df["code"] == code) & (fcf_df["year"] == ay)]
            fcf_prev = fcf_df[(fcf_df["code"] == code) & (fcf_df["year"] == ay - 1)]
            profit_cur = profit_df[(profit_df["code"] == code) & (profit_df["year"] == ay)]
            profit_prev = profit_df[(profit_df["code"] == code) & (profit_df["year"] == ay - 1)]

            should_exclude = False
            reason = ""
            if len(fcf_cur) > 0 and len(fcf_prev) > 0 and fcf_prev.iloc[0]["fcf"] > 0:
                fcf_decline = (fcf_prev.iloc[0]["fcf"] - fcf_cur.iloc[0]["fcf"]) / abs(fcf_prev.iloc[0]["fcf"])
                if fcf_decline > threshold:
                    should_exclude = True
                    reason = f"FCF下降{fcf_decline*100:.0f}%"
            if len(profit_cur) > 0 and len(profit_prev) > 0 and profit_prev.iloc[0]["net_profit"] > 0:
                profit_decline = (profit_prev.iloc[0]["net_profit"] - profit_cur.iloc[0]["net_profit"]) / abs(profit_prev.iloc[0]["net_profit"])
                if profit_decline > threshold:
                    should_exclude = True
                    reason = f"利润下降{profit_decline*100:.0f}%"

            if should_exclude:
                pred_filtered.loc[(dt, inst), "score"] = -999
                if dt == dates[-1]:
                    excluded.append((inst, reason))

    if excluded:
        print(f"  基本面硬过滤 (信号日{dates[-1].date()}): 剔除{len(excluded)}只")
        for inst, reason in excluded[:5]:
            print(f"    {inst}: {reason}")
    return pred_filtered


# ==================== 流动性过滤器 ====================
def filter_pred_by_liquidity(pred, universe, threshold_wan=2000):
    """过滤掉日均成交额低于threshold_wan(万元)的股票
    使用每个调仓日前20个交易日的日均成交额
    防止微盘股流动性陷阱: 回测假设可按收盘价成交, 但实盘微盘股可能无法全额买卖
    """
    dates = sorted(pred.index.get_level_values(0).unique())
    if len(dates) == 0:
        return pred, 0

    # 一次性获取所有需要的数据
    lookback = pd.Timedelta(days=60)  # 日历日60天, 确保有20个交易日
    start = dates[0] - lookback
    end = dates[-1]

    try:
        vol_data = D.features(list(universe), ["$close", "$volume"],
                               start_time=start, end_time=end)
    except Exception as e:
        print(f"  [WARNING] 流动性数据获取失败: {e}")
        return pred, 0

    if vol_data is None or len(vol_data) == 0:
        return pred, 0

    vol_data = vol_data.reset_index()
    vol_data.columns = ["instrument", "datetime", "close", "volume"]
    vol_data["amount"] = vol_data["close"] * vol_data["volume"]
    vol_data = vol_data.sort_values(["instrument", "datetime"])
    vol_data["avg_amount_20d"] = vol_data.groupby("instrument")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    vol_data = vol_data.set_index(["datetime", "instrument"])

    filtered = pred.copy()
    n_excluded = 0
    threshold = threshold_wan * 1e4  # 转为元

    for dt in dates:
        if dt not in vol_data.index.get_level_values(0):
            continue
        day_liq = vol_data.loc[dt, "avg_amount_20d"]
        illiquid_stocks = day_liq[day_liq < threshold].index
        for inst in illiquid_stocks:
            if (dt, inst) in filtered.index:
                filtered.loc[(dt, inst), "score"] = -999
                n_excluded += 1

    print(f"  流动性过滤: 剔除{n_excluded}条 (阈值{threshold_wan}万/日)")
    return filtered, n_excluded


# ==================== 宏观200日均线阀门 ====================
def compute_macro_signal(universe, signal_date, lookback_days=300):
    """计算全A等权200日均线信号: 1.0=满仓, 0.7=降仓"""
    start = signal_date - pd.Timedelta(days=lookback_days)
    prices = D.features(universe, ["$close"], start_time=start, end_time=signal_date)
    if prices is None or len(prices) == 0:
        return 1.0
    prices = prices.reset_index()
    prices.columns = ["instrument", "datetime", "close"]
    prices = prices.sort_values(["instrument", "datetime"])
    prices["ret"] = prices.groupby("instrument")["close"].pct_change()
    ew_ret = prices.groupby("datetime")["ret"].mean().dropna()
    ew_cum = (1 + ew_ret).cumprod()
    ew_ma200 = ew_cum.rolling(200, min_periods=60).mean()

    if len(ew_cum) == 0 or pd.isna(ew_ma200.iloc[-1]):
        return 1.0

    current_cum = ew_cum.iloc[-1]
    current_ma = ew_ma200.iloc[-1]
    signal = 0.7 if current_cum < current_ma else 1.0
    trend = "熊市(低于200日线)→70%仓" if signal < 1.0 else "牛市(高于200日线)→100%仓"
    print(f"  宏观阀门: 全A等权={current_cum:.4f}, 200日均线={current_ma:.4f} → {trend}")
    return signal


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)

    codes = build_dynamic_universe(2026, fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    print(f"W6股票池: {len(universe)}只")

    cal = D.calendar(start_time="2021-01-01", end_time="2026-07-23")
    cal_set = set(cal)

    ts, te = "2021-01-01", "2024-12-31"
    vs, ve = "2025-01-01", "2025-12-31"
    bs, be = "2026-01-01", "2026-07-21"

    # ====== 1. 创建Dataset + 修复注入 ======
    print(f"\n{'='*70}")
    print(f"  1. 修复注入 + 重训W6")
    print(f"{'='*70}")

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

    # 加载特征
    print("加载价值因子...")
    vf = load_value_factors(universe, cal, cal_set)
    print("加载基本面特征(修复版)...")
    fund = load_fundamental_features_fixed(universe, cal)

    # 注入
    print("\n注入基本面特征:")
    fund_ok = inject_features_fixed(dataset, fund, "fund")
    print("注入价值因子:")
    vf_ok = inject_features_fixed(dataset, vf, "vf")

    # 验证模型可见特征数
    train_data = dataset.prepare("train", col_set="feature")
    print(f"\n模型训练特征数: {train_data.shape[1]}")

    # 提取特征名 (兼容MultiIndex和flat两种格式)
    def extract_feature_names(columns):
        names = []
        for c in columns:
            if isinstance(c, tuple):
                names.append(c[1] if len(c) > 1 else str(c[0]))
            else:
                names.append(str(c))
        return names

    all_names = extract_feature_names(train_data.columns)
    fund_name_set = {"fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"}
    vf_name_set = {"roe_annual", "pb_pct_3y", "div_yield_est", "pe_pct_3y"}
    fund_cols = [n for n in all_names if n in fund_name_set]
    vf_cols = [n for n in all_names if n in vf_name_set]
    print(f"  Alpha158+增强: {len(all_names) - len(fund_cols) - len(vf_cols)}个")
    print(f"  基本面: {len(fund_cols)}个 → {fund_cols}")
    print(f"  价值: {len(vf_cols)}个 → {vf_cols}")

    # 如果直接修改_infer/_learn失败, 用monkey-patch prepare()作为后备
    if len(fund_cols) == 0:
        print("\n  ⚠ 直接注入未生效! 启用 monkey-patch prepare() 后备方案...")
        extra_combined = pd.concat([fund, vf], axis=1).sort_index() if vf is not None else fund
        original_prepare = dataset.prepare

        def patched_prepare(segments, col_set=None, data_key=None, **kwargs):
            seg_kwargs = {"col_set": col_set}
            if data_key is not None:
                seg_kwargs["data_key"] = data_key
            seg_kwargs.update(kwargs)
            result = original_prepare(segments, **seg_kwargs)
            if result is not None and not result.empty:
                should_add = False
                if col_set == "feature":
                    should_add = True
                elif isinstance(col_set, list) and "feature" in col_set:
                    should_add = True
                if should_add:
                    extra_aligned = extra_combined.reindex(result.index).fillna(0)
                    if isinstance(result.columns, pd.MultiIndex):
                        extra_aligned.columns = pd.MultiIndex.from_tuples(
                            [("feature", c) for c in extra_aligned.columns])
                    result = pd.concat([result, extra_aligned], axis=1)
            return result

        dataset.prepare = patched_prepare
        train_data = dataset.prepare("train", col_set="feature")
        all_names = extract_feature_names(train_data.columns)
        fund_cols = [n for n in all_names if n in fund_name_set]
        vf_cols = [n for n in all_names if n in vf_name_set]
        print(f"  monkey-patch后: 特征数={train_data.shape[1]}, 基本面={len(fund_cols)}, 价值={len(vf_cols)}")

    # ====== 2. 训练模型 ======
    print(f"\n  训练LightGBM...")
    model = init_instance_by_config(MODEL_CONFIG)
    with R.start(experiment_name="v5_fixed_inject"):
        rec = R.get_recorder()
        model.fit(dataset)
        sig_rec = SignalRecord(model, dataset, rec)
        sig_rec.generate()
        pred = rec.load_object("pred.pkl")

    # ====== 3. Feature Importance ======
    print(f"\n{'='*70}")
    print(f"  2. Feature Importance (修复后)")
    print(f"{'='*70}")

    booster = model.model
    importance_gain = booster.feature_importance(importance_type='gain')
    importance_split = booster.feature_importance(importance_type='split')
    # 获取特征名: 用兼容MultiIndex的方式提取
    feature_names = all_names  # 已通过extract_feature_names提取
    n_model_features = len(importance_gain)

    print(f"  模型特征数: {n_model_features}, 数据集特征数: {len(feature_names)}")

    if n_model_features == len(feature_names):
        names = feature_names
    else:
        # LightGBM可能用Column_N, 需要映射
        print(f"  ⚠ 特征数不匹配! 模型={n_model_features}, 数据集={len(feature_names)}")
        names = [f"Column_{i}" for i in range(n_model_features)]
        if len(feature_names) >= n_model_features:
            names = feature_names[:n_model_features]

    fi_df = pd.DataFrame({"feature": names, "gain": importance_gain, "split": importance_split})
    fi_df = fi_df.sort_values("gain", ascending=False)

    print(f"\n  Top 25 特征 (按Gain排序):")
    print(f"  {'排名':<6} {'特征名':<24} {'Gain':>12} {'Split':>8}")
    print(f"  {'-'*52}")
    for i, (_, r) in enumerate(fi_df.head(25).iterrows()):
        print(f"  {i+1:<6} {r['feature']:<24} {r['gain']:>12.1f} {r['split']:>8}")

    # 按类别统计
    alpha158_base = ["KMID","KMID2","KLEN","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2",
                     "OPEN0","OPEN1","OPEN2","HIGH0","HIGH1","HIGH2","LOW0","LOW1","LOW2",
                     "VWAP0","VWAP1","VWAP2","ROC","MA","STD","BOLL","RSI","WR","CORR","CORD",
                     "CNTN","CNTP","IMBA","SUM","ABS","VSTD","WMA","KDJ","MACD","CCI",
                     "VRATIO","VMA","CORR_PV","PEAK","TREND","ROC120","ROC240","MA120","MA240",
                     "STD120","STD240","BOLL60","BOLL120","VMA120","VMA240","CORR_PV20",
                     "CORR_PV60","VRATIO_5_60","VRATIO_5_120"]
    fund_names_set = {"fcf_growth", "profit_growth", "fcf_profit_ratio", "fcf_avg_3y_norm", "fcf_cv_3y"}
    vf_names_set = {"roe_annual", "pb_pct_3y", "div_yield_est", "pe_pct_3y"}

    categories = {"Alpha158技术(短周期)": [], "Alpha158增强(长周期)": [], "基本面": [], "价值因子": [], "其他": []}
    enhanced_set = {"ROC120","ROC240","MA120","MA240","STD120","STD240","BOLL60","BOLL120",
                    "VMA120","VMA240","CORR_PV20","CORR_PV60","VRATIO_5_60","VRATIO_5_120"}

    for _, r in fi_df.iterrows():
        fn = r["feature"]
        if fn in fund_names_set:
            categories["基本面"].append(r)
        elif fn in vf_names_set:
            categories["价值因子"].append(r)
        elif fn in enhanced_set:
            categories["Alpha158增强(长周期)"].append(r)
        elif any(fn.startswith(c) or fn == c for c in alpha158_base):
            categories["Alpha158技术(短周期)"].append(r)
        else:
            categories["其他"].append(r)

    total_gain = fi_df["gain"].sum()
    print(f"\n  按类别统计Gain占比:")
    print(f"  {'类别':<24} {'Gain占比':>10} {'特征数':>8} {'Top特征':>16} {'Top Gain':>12}")
    print(f"  {'-'*70}")
    for cat, items in categories.items():
        if items:
            cat_gain = sum(r["gain"] for r in items)
            top_name = items[0]["feature"]
            top_gain = items[0]["gain"]
            print(f"  {cat:<24} {cat_gain/total_gain*100:>9.1f}% {len(items):>8} {top_name:>16} {top_gain:>12.1f}")

    fi_df.to_csv("/Users/11164591/Documents/Qoder目录/qlib/w6_fi_fixed.csv", sep='\t', index=False)
    print(f"\n  已保存: w6_fi_fixed.csv")

    # ====== 4. 双层拦截管线 ======
    print(f"\n{'='*70}")
    print(f"  3. 双层拦截管线: 宏观阀门 → 基本面硬过滤 → LGBM → 价值融合")
    print(f"{'='*70}")

    # Step 1: 价值融合 (α=0.3)
    pred_fused = apply_value_fusion(pred.copy(), vf, alpha=0.3)

    # Step 2: 涨跌停/停牌过滤
    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred_fused = filter_pred_by_tradability(pred_fused, limit_up_set, suspension_set)

    # Step 3: 基本面滑坡硬过滤 (30%阈值)
    print("\n  [Layer 2] 基本面滑坡硬过滤 (30%阈值):")
    pred_filtered = filter_by_fundamental_deterioration(pred_fused, fcf_df, profit_df, threshold=0.30)

    # Step 4: 获取最新信号日
    latest_dates = sorted(pred_filtered.index.get_level_values(0).unique())
    latest_dt = latest_dates[-1]
    day_pred = pred_filtered.xs(latest_dt, level=0)
    top10 = day_pred["score"].nlargest(10)

    # Step 5: 宏观阀门
    print(f"\n  [Layer 1] 宏观200日均线阀门:")
    position_ratio = compute_macro_signal(universe, latest_dt)

    # Step 6: 输出最终结果
    name_map = {}
    if "name" in fcf_df.columns:
        name_map = dict(zip(fcf_df["code"], fcf_df["name"]))

    print(f"\n  {'='*60}")
    print(f"  最终调仓列表 (信号日: {latest_dt.date()}, 仓位: {position_ratio*100:.0f}%)")
    print(f"  {'='*60}")
    print(f"  {'排名':<6} {'代码':<10} {'名称':<12} {'融合分':>10} {'建议仓位':>10}")
    print(f"  {'-'*50}")
    for i, (inst, score) in enumerate(top10.items()):
        code = inst[2:]
        name = name_map.get(code, "—")
        pos = f"{position_ratio*10:.1f}%"  # 每只股票占总资金的position_ratio*10%
        print(f"  {i+1:<6} {inst:<10} {name:<12} {score:>10.4f} {pos:>10}")

    # 获取价格
    top10_codes = top10.index.tolist()
    prices = D.features(top10_codes, ["$close", "$volume"],
                       start_time=latest_dt - pd.Timedelta(days=10), end_time=latest_dt)
    if prices is not None and len(prices) > 0:
        prices = prices.reset_index()
        prices.columns = ["instrument", "datetime", "close", "volume"]
        print(f"\n  {'代码':<10} {'名称':<12} {'最新价':>10} {'5日均量':>12}")
        print(f"  {'-'*44}")
        for inst in top10_codes:
            sp = prices[prices["instrument"] == inst].sort_values("datetime")
            if len(sp) > 0:
                code = inst[2:]
                name = name_map.get(code, "—")
                print(f"  {inst:<10} {name:<12} {sp.iloc[-1]['close']:>10.2f} {sp['volume'].tail(5).mean():>12.0f}")

    # 保存
    output = pd.DataFrame({
        "rank": range(1, len(top10) + 1),
        "instrument": top10.index,
        "name": [name_map.get(i[2:], "—") for i in top10.index],
        "fused_score": top10.values,
        "signal_date": latest_dt,
        "position_ratio": position_ratio,
    })
    output.to_csv("/Users/11164591/Documents/Qoder目录/qlib/paper_trading_pipeline.csv", sep='\t', index=False)
    print(f"\n  已保存: paper_trading_pipeline.csv")
    print(f"\n  管线流程: 宏观阀门({position_ratio*100:.0f}%) → 基本面硬过滤 → LGBM({len(train_data.columns)}特征) → α=0.3价值融合")


if __name__ == "__main__":
    run()
