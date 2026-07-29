"""
V13 修复版模型对比 (LightGBM vs XGBoost)
=========================================================
修复审计发现的全部问题:
1. 流动性过滤: 改用 close*volume 口径 (qlib $amount 字段为空), 阈值500万/日,
   数据拉取失败时直接报错, 禁止静默跳过。
2. 交易成本: 往返 0.4% (买0.15% + 卖0.25%含印花税), 按换手率计。
3. T+1执行: T日收盘出信号, T+1日收盘价建仓, 消除同日成交前视。
4. 特征选择去前视: 每个窗口内用训练集数据两阶段动态选 Top80,
   禁止使用全局/未来窗口的特征重要性。
5. valid embargo: valid结束日提前1个月, 防止20日forward标签泄漏进测试期。
6. 速度优化: XGB tree_method=hist + n_jobs=4, LGB num_threads=4。
"""
import os, sys, random
import warnings, logging

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, "/Users/11164591/Documents/Qoder目录/qlib")

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
import lightgbm as lgb
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")
logging.getLogger('qlib.data.data').setLevel(logging.ERROR)

from v5_validation import build_limit_up_set, filter_pred_by_tradability
from v5_pipeline_fix import load_fundamental_features_fixed, inject_features_fixed
from v5_xgb_turnover_comparative import build_dynamic_universe, format_qlib_code, get_month_end_dates

WORK_DIR = "/Users/11164591/Documents/Qoder目录/qlib"

# ==================== 模型参数 (含图示速度优化: hist + 4线程) ====================
XGB_PARAMS = {
    "objective": "reg:squarederror", "learning_rate": 0.005,
    "max_depth": 4, "colsample_bytree": 0.8879, "subsample": 0.8789,
    "reg_alpha": 10.0, "reg_lambda": 50.0,
    "tree_method": "hist", "nthread": 4, "seed": 42,
}
LGB_PARAMS = {
    "objective": "mse", "learning_rate": 0.005,
    "max_depth": 7, "num_leaves": 15,
    "colsample_bytree": 0.8879, "subsample": 0.8789,
    "lambda_l1": 10.0, "lambda_l2": 50.0,
    "num_threads": 4, "seed": 42, "verbose": -1,
    "force_col_wise": True,
}
N_ROUNDS, EARLY_STOP = 1000, 100
# stage1 快速特征选择参数 (高学习率快速收敛, 仅用于窗口内取Top80)
FS_ROUNDS, FS_EARLY_STOP, FS_LR = 200, 30, 0.05
TOP_N_FEATURES = 80

FEE_ROUNDTRIP = 0.004        # 往返 0.4%
LIQ_THRESHOLD = 5_000_000    # 500万/日 (close*volume 20日均)


def generate_quarterly_windows():
    windows = []
    backtest_years = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
    for year in backtest_years:
        periods = [(1, 3), (4, 6), (7, 9)] if year == 2026 else [(1, 3), (4, 6), (7, 9), (10, 12)]
        for start_m, end_m in periods:
            bt_start = pd.Timestamp(f"{year}-{start_m:02d}-01")
            bt_end = pd.Timestamp(f"{year}-{end_m:02d}-01") + relativedelta(months=1) - pd.Timedelta(days=1)
            train_end = bt_start - relativedelta(years=1) - pd.Timedelta(days=1)
            train_start = train_end - relativedelta(years=4)
            valid_start = train_end + pd.Timedelta(days=1)
            # 修复5: embargo — valid提前1个月结束, 20日forward标签不碰测试期
            valid_end = bt_start - relativedelta(months=1) - pd.Timedelta(days=1)
            windows.append({
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "backtest": (bt_start.strftime("%Y-%m-%d"), bt_end.strftime("%Y-%m-%d")),
                "name": f"{year}_Q{start_m//3 + 1}", "year": year})
    return windows


def prepare_window_data(win, fcf_df, profit_df, cal):
    """构建窗口数据集, 返回 (X_train, y_train, X_valid, y_valid, X_test, universe)"""
    codes = build_dynamic_universe(win["year"], fcf_df, profit_df)
    universe = sorted(format_qlib_code(c) for c in codes)
    ts, te = win["train"]; vs, ve = win["valid"]; bs, be = win["backtest"]

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

    fund = load_fundamental_features_fixed(universe, cal)
    if fund is not None:
        inject_features_fixed(dataset, fund, "fund")

    train_df = dataset.prepare("train", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    test_X = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)

    X_tr, y_tr = train_df["feature"], train_df["label"].iloc[:, 0]
    X_va, y_va = valid_df["feature"], valid_df["label"].iloc[:, 0]
    # valid里label为NaN的行剔除 (数据末端forward窗口不足)
    va_mask = y_va.notna()
    X_va, y_va = X_va[va_mask], y_va[va_mask]
    del dataset
    return X_tr, y_tr, X_va, y_va, test_X, universe


# ==================== 修复4: 窗口内两阶段动态特征选择 ====================
def select_features_lgb(X_tr, y_tr, X_va, y_va):
    p = dict(LGB_PARAMS); p["learning_rate"] = FS_LR
    dtr = lgb.Dataset(X_tr.values, label=y_tr.values, feature_name=list(X_tr.columns))
    dva = lgb.Dataset(X_va.values, label=y_va.values, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=FS_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(FS_EARLY_STOP, verbose=False)])
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    return imp.nlargest(TOP_N_FEATURES).index.tolist()

def select_features_xgb(X_tr, y_tr, X_va, y_va):
    p = dict(XGB_PARAMS); p["learning_rate"] = FS_LR
    dtr = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=list(X_tr.columns))
    dva = xgb.DMatrix(X_va.values, label=y_va.values, feature_names=list(X_tr.columns))
    m = xgb.train(p, dtr, num_boost_round=FS_ROUNDS, evals=[(dva, "valid")],
                  early_stopping_rounds=FS_EARLY_STOP, verbose_eval=False)
    score = m.get_score(importance_type="gain")
    imp = pd.Series(score).reindex(X_tr.columns).fillna(0.0)
    return imp.nlargest(TOP_N_FEATURES).index.tolist()


def train_lgb(X_tr, y_tr, X_va, y_va, feats):
    dtr = lgb.Dataset(X_tr[feats].values, label=y_tr.values, feature_name=feats)
    dva = lgb.Dataset(X_va[feats].values, label=y_va.values, reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=N_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m, m.best_iteration

def train_xgb(X_tr, y_tr, X_va, y_va, feats):
    dtr = xgb.DMatrix(X_tr[feats].values, label=y_tr.values, feature_names=feats)
    dva = xgb.DMatrix(X_va[feats].values, label=y_va.values, feature_names=feats)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=N_ROUNDS, evals=[(dva, "valid")],
                  early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return m, m.best_iteration


# ==================== 修复1: 流动性过滤 close*volume 口径, 失败报错 ====================
def build_liquidity_table(universe, start, end):
    """20日均成交额 (close*volume), 截至各交易日收盘 — 无前视"""
    lookback_start = pd.Timestamp(start) - pd.Timedelta(days=60)
    px = D.features(list(universe), ["$close", "$volume"],
                    start_time=lookback_start, end_time=end)
    if px is None or len(px) == 0:
        raise RuntimeError(f"[流动性过滤] 价格数据拉取失败 ({start}~{end}), 拒绝静默跳过!")
    px = px.reset_index()
    px.columns = ["instrument", "datetime", "close", "volume"]
    px["amount"] = px["close"] * px["volume"]
    px = px.sort_values(["instrument", "datetime"])
    px["avg20"] = px.groupby("instrument")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    return px.set_index(["datetime", "instrument"])["avg20"]


# ==================== 修复2+3: T+1执行 + 往返0.4%成本 回测引擎 ====================
def backtest_fixed(pred, universe, cal, month_ends, liq_table):
    """T日收盘信号 → T+1收盘价建仓; 换手成本=turnover*0.4%在执行日扣"""
    if pred is None or len(pred) == 0:
        return pd.Series(dtype=float)

    limit_up_set, suspension_set = build_limit_up_set(universe, cal)
    pred = filter_pred_by_tradability(pred, limit_up_set, suspension_set)

    all_dates = sorted(pred.index.get_level_values(0).unique())
    cal_list = list(cal)
    insts = list(set(pred.index.get_level_values(1)))
    price_data = D.features(insts, ["$close"],
        start_time=all_dates[0] - pd.Timedelta(days=10),
        end_time=all_dates[-1] + pd.Timedelta(days=40)).reset_index()
    price_data.columns = ["instrument", "datetime", "close"]
    price_dict = {inst: grp.set_index("datetime")["close"] for inst, grp in price_data.groupby("instrument")}

    def next_trading_day(dt):
        for d in cal_list:
            if d > dt:
                return d
        return None

    net_returns, portfolio_dates = [], []
    prev_topk = []
    turnover_log = []

    for i, dt in enumerate(month_ends):
        if dt not in pred.index.get_level_values(0):
            earlier = [d for d in all_dates if d <= dt]
            if not earlier:
                continue
            dt = earlier[-1]

        # 信号日选股: 流动性过滤 (20日均<500万 或 无数据 → 剔除)
        day_pred = pred.xs(dt, level=0)["score"].copy()
        if dt in liq_table.index.get_level_values(0):
            day_liq = liq_table.xs(dt, level=0).reindex(day_pred.index)
            day_pred[day_liq.isna() | (day_liq < LIQ_THRESHOLD)] = -np.inf
        else:
            raise RuntimeError(f"[流动性过滤] 调仓日{dt}无流动性数据, 拒绝静默跳过!")
        day_pred = day_pred[day_pred > -np.inf]
        curr_topk = day_pred.nlargest(10).index.tolist()

        # T+1执行日
        exec_dt = next_trading_day(dt)
        if exec_dt is None or exec_dt > all_dates[-1] + pd.Timedelta(days=40):
            break

        intersection = len(set(curr_topk).intersection(set(prev_topk)))
        turnover = 1.0 if not prev_topk else 1.0 - (intersection / 10.0)
        turnover_log.append(turnover)

        # 下一期执行日 (持有到下期换仓的收盘)
        if i + 1 < len(month_ends):
            next_exec = next_trading_day(month_ends[i + 1])
            if next_exec is None:
                next_exec = all_dates[-1]
        else:
            next_exec = all_dates[-1]

        # exec_dt收盘价建仓, 当日记换手成本
        prev_prices = {inst: price_dict[inst][exec_dt] for inst in curr_topk
                       if inst in price_dict and exec_dt in price_dict[inst].index}
        net_returns.append(-turnover * FEE_ROUNDTRIP)
        portfolio_dates.append(exec_dt)

        period_dates = [d for d in cal_list if exec_dt < d <= next_exec]
        for pd_dt in period_dates:
            day_ret, n_valid = 0, 0
            for inst in curr_topk:
                if inst in price_dict and pd_dt in price_dict[inst].index and inst in prev_prices:
                    cur_price = price_dict[inst][pd_dt]
                    if pd.notna(cur_price) and prev_prices[inst] > 0:
                        day_ret += (cur_price / prev_prices[inst] - 1)
                        prev_prices[inst] = cur_price
                        n_valid += 1
            net_returns.append(day_ret / 10.0 if n_valid > 0 else 0.0)
            portfolio_dates.append(pd_dt)

        prev_topk = curr_topk

    s = pd.Series(net_returns, index=pd.DatetimeIndex(portfolio_dates))
    s = s.groupby(s.index).sum()  # 执行日成本与前期收益同日合并
    avg_turnover = np.mean(turnover_log) if turnover_log else 0
    return s, avg_turnover


def calc_metrics(returns):
    if len(returns) == 0:
        return {"ar": 0, "sharpe": 0, "max_dd": 0}
    n_years = len(returns) / 252
    ar = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    max_dd = ((nav / nav.cummax()) - 1).min()
    return {"ar": ar, "sharpe": sharpe, "max_dd": max_dd}


def run():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    fcf_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/fcf_cache.csv", sep='\t')
    profit_df = pd.read_csv("/Users/11164591/Documents/Qoder目录/profit_cache.csv", sep='\t')
    fcf_df["code"] = fcf_df["code"].astype(str).str.zfill(6)
    profit_df["code"] = profit_df["code"].astype(str).str.zfill(6)
    cal = D.calendar(start_time="2013-01-01", end_time="2026-07-31")

    print(f"\n{'='*95}", flush=True)
    print(f"  V13 修复版对比: LightGBM vs XGBoost (T+1执行 | 往返0.4% | 500万流动性 | 窗口内特征选择)", flush=True)
    print(f"{'='*95}", flush=True)

    windows = generate_quarterly_windows()
    # {year: {"universe": ..., "lgb": [pred...], "xgb": [pred...]}}
    year_data = {}
    latest_preds = {}

    for win in windows:
        print(f"\n[+] 窗口 {win['name']}: train{win['train']} valid{win['valid']} bt{win['backtest']}", flush=True)
        X_tr, y_tr, X_va, y_va, test_X, universe = prepare_window_data(win, fcf_df, profit_df, cal)
        print(f"    数据: train{X_tr.shape} valid{X_va.shape} test{test_X.shape}, 股票池{len(universe)}只", flush=True)

        y = win["year"]
        if y not in year_data:
            year_data[y] = {"universe": universe, "lgb": [], "xgb": []}

        for tag, sel_fn, tr_fn in [("lgb", select_features_lgb, train_lgb),
                                    ("xgb", select_features_xgb, train_xgb)]:
            feats = sel_fn(X_tr, y_tr, X_va, y_va)
            model, best_it = tr_fn(X_tr, y_tr, X_va, y_va, feats)
            if tag == "lgb":
                scores = model.predict(test_X[feats].values)
            else:
                scores = model.predict(xgb.DMatrix(test_X[feats].values, feature_names=feats))
            pred = pd.DataFrame({"score": scores}, index=test_X.index)
            year_data[y][tag].append(pred)
            print(f"    [{tag.upper()}] Top{len(feats)}特征, best_iter={best_it}", flush=True)

            if win["name"] == "2026_Q3":
                latest_preds[tag] = pred
                joblib.dump({"model": model, "features": feats},
                            f"{WORK_DIR}/v13_{tag}_latest_Q.pkl")

    # ==================== 按年回测 (年内跨季度连续换手) ====================
    print(f"\n{'='*95}\n  回测阶段 (T+1执行, 往返0.4%, 500万流动性过滤)\n{'='*95}", flush=True)
    all_rets = {"lgb": [], "xgb": []}
    annual = {"lgb": {}, "xgb": {}}
    turnover_stats = {"lgb": [], "xgb": []}

    for y in sorted(year_data.keys()):
        yd = year_data[y]
        bt_start = f"{y}-01-01"
        bt_end = f"{y}-07-31" if y == 2026 else f"{y}-12-31"
        month_ends = get_month_end_dates(cal, bt_start, bt_end)
        liq_table = build_liquidity_table(yd["universe"], bt_start, bt_end)

        for tag in ["lgb", "xgb"]:
            pred_y = pd.concat(yd[tag]).sort_index()
            pred_y = pred_y[~pred_y.index.duplicated(keep="last")]
            ret, avg_to = backtest_fixed(pred_y, yd["universe"], cal, month_ends, liq_table)
            all_rets[tag].append(ret)
            turnover_stats[tag].append(avg_to)
            ann_ret = (1 + ret).prod() - 1
            annual[tag][y] = ann_ret
        print(f"  {y}: LGB {annual['lgb'][y]*100:>7.2f}%  |  XGB {annual['xgb'][y]*100:>7.2f}%", flush=True)

    # ==================== 汇总 ====================
    print(f"\n{'='*105}", flush=True)
    years = sorted(year_data.keys())
    header = "  ".join(f"{y:>7}" for y in years)
    print(f"{'模型':<10} | {header} | {'全期净年化':>9} {'净夏普':>6} {'最大回撤':>8} {'月均换手':>8}", flush=True)
    print(f"{'-'*105}", flush=True)
    results = {}
    for tag, label in [("lgb", "LightGBM"), ("xgb", "XGBoost")]:
        s = pd.concat(all_rets[tag]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        m = calc_metrics(s)
        results[tag] = m
        row = "  ".join(f"{annual[tag][y]*100:>6.2f}%" for y in years)
        avg_to = np.mean(turnover_stats[tag]) * 100
        print(f"{label:<10} | {row} | {m['ar']*100:>8.2f}% {m['sharpe']:>6.2f} {m['max_dd']*100:>7.1f}% {avg_to:>7.1f}%", flush=True)
        s.to_csv(f"{WORK_DIR}/v13_returns_{tag}.csv", sep='\t', header=False)
    print(f"{'='*105}", flush=True)

    # 实盘信号 (两个模型各出Top10, 含流动性+可交易过滤)
    for tag in ["lgb", "xgb"]:
        if tag not in latest_preds:
            continue
        pred = latest_preds[tag]
        last_date = pred.index.get_level_values(0).max()
        day_scores = pred.xs(last_date, level=0)["score"].copy()
        insts = day_scores.index.tolist()
        liq = build_liquidity_table(insts, last_date - pd.Timedelta(days=60), last_date)
        if last_date in liq.index.get_level_values(0):
            day_liq = liq.xs(last_date, level=0).reindex(day_scores.index)
            day_scores[day_liq.isna() | (day_liq < LIQ_THRESHOLD)] = -np.inf
        else:
            raise RuntimeError(f"[实盘信号] {last_date}无流动性数据!")
        limit_up_set, suspension_set = build_limit_up_set(insts, cal)
        for inst in list(day_scores.index):
            if (last_date, inst) in suspension_set or (last_date, inst) in limit_up_set:
                day_scores[inst] = -np.inf
        top10 = day_scores[day_scores > -np.inf].nlargest(10)
        out = top10.reset_index()
        out.columns = ["instrument", "score"]
        out.to_csv(f"{WORK_DIR}/v13_live_signals_{tag}.csv", sep='\t', index=False)
        print(f"[+] {tag.upper()} 实盘信号已生成: v13_live_signals_{tag}.csv (信号日{last_date.date()})", flush=True)

    # 保存汇总
    rows = []
    for tag in ["lgb", "xgb"]:
        for y in years:
            rows.append({"model": tag, "year": y, "annual_ret": annual[tag][y]})
        rows.append({"model": tag, "year": "ALL",
                     "annual_ret": results[tag]["ar"],
                     "sharpe": results[tag]["sharpe"], "max_dd": results[tag]["max_dd"]})
    pd.DataFrame(rows).to_csv(f"{WORK_DIR}/v13_compare_results.csv", sep='\t', index=False)
    print(f"[+] 结果已保存: v13_compare_results.csv", flush=True)


if __name__ == "__main__":
    run()
